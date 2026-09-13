#!/usr/bin/env python3
"""Step 2 validation: look at the tensors on both sides of the transformer.

Two stages, matching the two commands in the plan:

  --stage tokenizer    what the frozen Wan VAE produces, i.e. the input to the
                       transformer: name / shape / dtype / range for the raw
                       batch, the normalised video, and the latents. Also prints
                       how much of each pointmap channel the frozen bounds clip.

  --stage transformer  the token layout (num_rgb_tokens / num_pointmap_tokens /
                       num_prog_tokens and the exact block order), the
                       block-causal mask drawn as a matrix, and then the forward
                       pass output.

Data comes either from the real Step 1 cache (`--dataset`) or from synthetic
tensors (`--synthetic`), so this runs before Step 1 has been executed.

Examples:
    # no cache, no GPU needed for the layout; still downloads the VAE (2.8 GB, once)
    uv run python dev_scripts/test_precise_model.py --synthetic --stage all

    # on the real cache
    export ROBOMETER_PROCESSED_DATASETS_PATH=/path/to/processed_datasets
    uv run python dev_scripts/test_precise_model.py \
        --dataset precise_square_rbm_square_d1_train --stage all
"""

from __future__ import annotations

# unsloth patches transformers before diffusers reaches this venv's broken torchao
# (`ImportError: cannot import name 'ScalingType' from 'torch.nn.functional'`).
try:  # noqa: SIM105
    import unsloth  # noqa: F401
except Exception:
    pass

import argparse
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch


def describe(name: str, value, indent: str = "    ") -> None:
    """The one line this script exists to print: name / shape / dtype / range."""
    if value is None:
        print(f"{indent}{name:<22} None")
        return
    if isinstance(value, (list, tuple)):
        print(f"{indent}{name:<22} {type(value).__name__}(len={len(value)})  e.g. {value[:3]}")
        return
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    numeric = tensor.is_floating_point() or not tensor.is_complex()
    head = f"{indent}{name:<22} shape={str(tuple(tensor.shape)):<22} dtype={str(tensor.dtype).replace('torch.', ''):<9}"
    if tensor.numel() and numeric:
        as_float = tensor.detach().float()
        print(
            f"{head} range=[{as_float.min():+.4g}, {as_float.max():+.4g}] "
            f"mean={as_float.mean():+.4g} std={as_float.std():.4g} device={tensor.device}"
        )
    else:
        print(f"{head} device={tensor.device}")


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ----------------------------------------------------------------------- inputs


def synthetic_batch(batch_size: int, num_frames: int, size: int) -> Dict[str, torch.Tensor]:
    """A batch shaped exactly like the real one, with plausible camera-frame metres."""
    generator = torch.Generator().manual_seed(0)
    frames = torch.randint(0, 256, (batch_size, num_frames, 3, size, size), generator=generator, dtype=torch.uint8)

    # A table plane in front of the camera plus a bit of structure, so the frozen
    # bounds see values in the range they were chosen for rather than noise.
    grid = torch.linspace(-1, 1, size)
    yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    x = xx * 0.5
    y = yy * 0.5 - 0.4
    z = 0.75 + 0.25 * yy
    pointmap = torch.stack([x, y, z])[None, None].repeat(batch_size, num_frames, 1, 1, 1)
    pointmap = pointmap + 0.01 * torch.randn(pointmap.shape, generator=generator)
    return {
        "pixel_values_videos": frames,
        "pointmap_values": pointmap.to(torch.float16),
        "source": "synthetic",
    }


def cache_batch(dataset_name: str, batch_size: int, num_frames: int, cutoff_file: str) -> Dict[str, torch.Tensor]:
    """One real batch, straight through PreciseRBMDataset + PreciseBatchCollator."""
    from robometer.configs.experiment_configs import DataConfig
    from robometer.data.collators.precise import PreciseBatchCollator
    from robometer.data.datasets.precise_data import PreciseRBMDataset

    config = DataConfig(
        train_datasets=[dataset_name],
        eval_datasets=[dataset_name],
        max_frames=num_frames,
        min_frames_per_trajectory=5,
        sample_type_ratio=[0, 1, 0],
        progress_strategy_ratio=[0, 1, 1, 1],
        progress_pred_type="absolute_first_frame",
        dataset_success_cutoff_file=cutoff_file,
        load_embeddings=False,
        shuffle=False,
    )
    dataset = PreciseRBMDataset(config, is_evaluation=False)
    collator = PreciseBatchCollator(
        processor=None, base_model_id="precise_transformer", load_embeddings=False
    )
    batch = collator([dataset[i] for i in range(batch_size)])["progress_inputs"]
    batch["source"] = f"cache:{dataset_name}"

    print("  sample-level check (this is the alignment guarantee, made visible):")
    for i in range(min(2, batch_size)):
        indices = batch["metadata"][i]["frame_indices"]
        print(f"    sample {i}: frame_indices={np.asarray(indices).tolist()}")
    print("    RGB and pointmap were both sliced with that one vector -- see PreciseProgressSampler")
    return batch


# ------------------------------------------------------------------ the stages


def stage_tokenizer(model, batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    from robometer.models.precise.layout import POINTMAP, RGB

    tokenizer = model.latent_tokenizer
    section("STAGE 1 -- after the tokenizer, before the transformer")
    print(f"  {tokenizer.describe()}")
    print(f"  batch source: {batch['source']}")

    print("\n  raw batch (exactly what the collator produced -- no normalisation yet):")
    describe("pixel_values_videos", batch.get("pixel_values_videos"))
    describe("pointmap_values", batch.get("pointmap_values"))
    print("    pointmap_values is RAW CAMERA-FRAME METRES; the only clamp is normalize_pointmap, below")

    frames = batch.get("pixel_values_videos")
    pointmap = batch.get("pointmap_values")
    if frames is not None:
        frames = frames.to(device)
    if pointmap is not None:
        pointmap = pointmap.to(device)

    print("\n  after normalisation (on GPU, immediately before the frozen VAE):")
    if RGB in model.streams:
        describe("normalized rgb", tokenizer.normalize_rgb(frames))
        print("    rgb: x / 127.5 - 1, no clamp needed")
    if POINTMAP in model.streams:
        describe("normalized pointmap", tokenizer.normalize_pointmap(pointmap))
        clipped = tokenizer.clip_fraction(pointmap)
        bounds = tokenizer.bounds
        for channel, fraction in clipped.items():
            print(f"    {channel}: {fraction * 100:6.2f}% of pixels saturate at {list(bounds[channel])} m")
        print("    (saturating background is intended -- see Flex-pi's fixed workspace bounds)")

    print("\n  latents (the transformer's actual input):")
    latents = {}
    if RGB in model.streams:
        latents[RGB] = tokenizer.encode_rgb(frames)
    if POINTMAP in model.streams:
        latents[POINTMAP] = tokenizer.encode_pointmap(pointmap)
    for name, latent in latents.items():
        describe(f"latent[{name}]", latent)
    num_frames = (frames if frames is not None else pointmap).shape[1]
    size = (frames if frames is not None else pointmap).shape[-1]
    print(
        f"    [B, z_dim={tokenizer.z_dim}, T_latent, H/{tokenizer.spatial_ratio}, W/{tokenizer.spatial_ratio}]  "
        f"<- {num_frames} frames at {size}x{size}"
    )
    print(
        f"    T_latent = 1 + ({num_frames} - 1) // {tokenizer.temporal_ratio} = "
        f"{tokenizer.latent_frame_count(num_frames)}"
    )

    patch = model.config.latent_patch_size
    per_frame = model.grid_h * model.grid_w
    print(f"\n  after patchify (latent_patch_size={patch}) and the per-stream Linear:")
    for name, latent in latents.items():
        describe(f"tokens[{name}]", model._tokenize_stream(name, latent))
    print(f"    {model.grid_h}x{model.grid_w} = {per_frame} tokens per latent frame per stream")
    return latents


def draw_block_mask(layout, prog_block_causal: bool) -> None:
    """Block-level view of the mask. Rows are queries, columns are keys."""
    from robometer.models.precise.layout import build_block_causal_mask

    blocked = build_block_causal_mask(layout, prog_block_causal)
    num_blocks = len(layout.blocks)
    header = "        " + " ".join(f"B{i + 1:<2}" for i in range(num_blocks))
    print("\n  block-level attention (rows = query block, columns = key block)")
    print("    ██ = attends,  ·· = blocked")
    print(header)
    for query in layout.blocks:
        cells = []
        for key in layout.blocks:
            window = blocked[query.start : query.end, key.start : key.end]
            cells.append("··" if window.all() else ("██" if not window.any() else "▒▒"))
        print(f"    B{query.index + 1:<3} " + "  ".join(cells))
    if prog_block_causal:
        print("    ▒▒ on a progress block's diagonal = causal inside the block (prog_2 -> prog_3 -> ...)")
    else:
        print("    progress blocks are bidirectional inside themselves (prog_block_causal=false)")


def check_mask(layout, prog_block_causal: bool) -> bool:
    """Assert the properties the design claims, rather than trusting the picture."""
    from robometer.models.precise.layout import POINTMAP, RGB, build_block_causal_mask

    blocked = build_block_causal_mask(layout, prog_block_causal)
    checks = []

    first_prog = int(layout.prog_index[0])
    later_visual = [b for b in layout.blocks if b.kind == "visual" and b.latent_frame > 0]
    if later_visual:
        window = blocked[first_prog, later_visual[0].start : later_visual[0].end]
        checks.append(("prog_1 cannot see latent frame 2", bool(window.all())))

    first_visual = layout.blocks[0]
    checks.append(
        (
            "latent frame 1 is bidirectional inside itself",
            not blocked[first_visual.start : first_visual.end, first_visual.start : first_visual.end].any(),
        )
    )
    checks.append(
        (
            "every progress token sees its own latent frame",
            all(
                not blocked[int(layout.prog_index[f]), b.start : b.end].any()
                for f in range(layout.num_frames)
                for b in layout.blocks
                if b.kind == "visual" and b.latent_frame == layout.frame_to_latent[f]
            ),
        )
    )
    prog_blocks = [b for b in layout.blocks if b.kind == "prog" and b.size > 1]
    if prog_blocks:
        block = prog_blocks[0]
        window = blocked[block.start : block.end, block.start : block.end]
        if prog_block_causal:
            checks.append(("progress tokens are ordered inside their block", bool(blocked[block.start, block.start + 1])))
        else:
            checks.append(("progress blocks are bidirectional inside themselves", not window.any()))
    if layout.modality == "rgb_pointmap":
        rgb = layout.positions[layout.kinds == RGB]
        pointmap = layout.positions[layout.kinds == POINTMAP]
        checks.append(("rgb and pointmap share RoPE positions", np.array_equal(rgb, pointmap)))

    print("\n  mask / position checks")
    ok = True
    for label, passed in checks:
        print(f"    {'✅' if passed else '❌'} {label}")
        ok = ok and passed
    return ok


def stage_transformer(model, batch: Dict[str, torch.Tensor], device: str, check_leakage: bool) -> bool:
    from robometer.models.precise.layout import describe_positions, format_layout

    section("STAGE 2 -- token layout and transformer output")

    reference = batch.get("pixel_values_videos")
    if reference is None:
        reference = batch["pointmap_values"]
    num_frames = reference.shape[1]
    layout = model.get_layout(num_frames)

    print(format_layout(layout, model.config.prog_block_causal))
    print()
    print(describe_positions(layout))
    draw_block_mask(layout, model.config.prog_block_causal)
    ok = check_mask(layout, model.config.prog_block_causal)

    inputs = {
        key: value.to(device)
        for key, value in batch.items()
        if key in ("pixel_values_videos", "pointmap_values") and value is not None
    }
    model.eval()
    with torch.no_grad():
        output, _ = model(**inputs, sample_type="progress")

    print("\n  transformer output")
    describe("hidden_states", output.hidden_states)
    print(f"    -> [B, L={layout.seq_len}, D={model.config.hidden_dim}], one vector per token")
    describe("prog_hidden", output.prog_hidden)
    print(f"    -> the {layout.num_frames} progress tokens gathered out of hidden_states")
    describe("progress_logits[A]", output.progress_logits["A"])
    describe("success_logits[A]", output.success_logits["A"])
    print(f"    progress_logits[B] / success_logits[B]: {output.progress_logits['B']} (progress-only model)")
    print(f"    last_token_index = {output.last_token_index}  <- the only leak-free frame; log it separately")

    values = output.progress_logits["A"][0].float().cpu().numpy()
    print("\n  sample 0 per-frame progress prediction (untrained, so these are noise -- shape is the point)")
    for frame, value in enumerate(values):
        latent = int(layout.frame_to_latent[frame]) + 1
        print(f"    prog_{frame + 1:<2} = {value:6.4f}   (reads latent frame {latent})")

    if "target_progress" in batch:
        describe("target_progress", batch["target_progress"])
        describe("success_label", batch.get("success_label"))
        describe("padding_mask", batch.get("padding_mask"))
        print("    these come from the stock RBMBaseSampler, unchanged")

    if check_leakage:
        ok = demonstrate_leakage(model, inputs, layout, device) and ok
    return ok


def demonstrate_leakage(model, inputs, layout, device: str) -> bool:
    """Show the VAE-induced leak concretely: perturb a late frame, watch an early prediction move.

    This is not a bug to fix -- it is a property of a 4x temporally-compressed VAE,
    and no attention mask can remove it. Worth measuring so training metrics are
    read with it in mind.
    """
    section("Leakage probe -- perturb one input frame, see which predictions move")
    with torch.no_grad():
        base, _ = model(**inputs, sample_type="progress")
        base_values = base.progress_logits["A"][0].float().cpu()

        target_frame = min(5, layout.num_frames - 1)  # frame 5 shares latent frame 2 with frame 2
        perturbed = {key: value.clone() for key, value in inputs.items()}
        for key in perturbed:
            if perturbed[key].dtype == torch.uint8:
                perturbed[key][:, target_frame] = 255 - perturbed[key][:, target_frame]
            else:
                perturbed[key][:, target_frame] = perturbed[key][:, target_frame] + 0.25
        changed, _ = model(**perturbed, sample_type="progress")
        delta = (changed.progress_logits["A"][0].float().cpu() - base_values).abs()

    print(f"  perturbed input frame {target_frame + 1} (latent frame {int(layout.frame_to_latent[target_frame]) + 1})")
    for frame in range(layout.num_frames):
        latent = int(layout.frame_to_latent[frame]) + 1
        expected = "leak" if (layout.frame_to_latent[frame] == layout.frame_to_latent[target_frame] and frame < target_frame) else ""
        print(f"    |Δ prog_{frame + 1:<2}| = {delta[frame]:.5f}   (latent frame {latent}) {expected}")
    print("  A non-zero Δ on a prog_k with k < the perturbed frame is the VAE's temporal")
    print("  compression, not a mask bug. Only the last frame of a clip is leak-free,")
    print("  which is exactly what the Step 4 eval protocol reads.")
    return True


# -------------------------------------------------------------------- entrypoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", help="Processed cache name, e.g. precise_square_rbm_square_d1_train")
    source.add_argument("--synthetic", action="store_true", help="Use generated tensors; no Step 1 cache needed")

    parser.add_argument("--stage", choices=("tokenizer", "transformer", "all"), default="all")
    parser.add_argument("--modality", choices=("rgb", "pointmap", "rgb_pointmap"), default="rgb_pointmap")
    parser.add_argument("--frames", type=int, default=9, help="data.max_frames")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--frame-size", type=int, default=256)
    parser.add_argument("--latent-patch-size", type=int, default=2, help="1 = no patchify (sequence 393 -> 1545)")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--check-leakage", action="store_true", help="Perturb a late frame and measure the VAE leak")
    parser.add_argument(
        "--prog-block-causal",
        action="store_true",
        help="Order progress tokens inside their block (default: bidirectional, like the visual blocks)",
    )
    parser.add_argument("--cache-dir", default=None, help="Default: $ROBOMETER_PROCESSED_DATASETS_PATH")
    parser.add_argument("--cutoff-file", default="robometer/data/dataset_success_cutoff_precise.txt")
    args = parser.parse_args()

    from robometer.utils.precise_setup_utils import build_precise_model, precise_config_from_overrides

    config = precise_config_from_overrides(
        {
            "modality": args.modality,
            "max_len": args.frames,
            "frame_size": args.frame_size,
            "latent_patch_size": args.latent_patch_size,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "prog_block_causal": args.prog_block_causal,
        }
    )

    section("Building PreciseTransformer")
    print(f"  device={args.device}")
    for key, value in sorted(config.model.precise.items()):
        print(f"    model.precise.{key:<22} {value}")
    model = build_precise_model(config.model, device=args.device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  trainable parameters: {trainable / 1e6:.2f} M")
    print(f"  frozen VAE parameters: {sum(p.numel() for p in model.latent_tokenizer.vae.parameters()) / 1e6:.2f} M")
    print("  (the VAE is a plain attribute, not a submodule, so it is absent from state_dict)")
    assert not any("latent_tokenizer" in name for name in model.state_dict()), "VAE leaked into state_dict"

    if args.synthetic:
        batch = synthetic_batch(args.batch_size, args.frames, args.frame_size)
    else:
        cache_root = args.cache_dir or os.environ.get("ROBOMETER_PROCESSED_DATASETS_PATH", "")
        if not cache_root:
            raise ValueError("Set ROBOMETER_PROCESSED_DATASETS_PATH or pass --cache-dir")
        os.environ["ROBOMETER_PROCESSED_DATASETS_PATH"] = cache_root
        batch = cache_batch(args.dataset, args.batch_size, args.frames, args.cutoff_file)

    if args.stage in ("tokenizer", "all"):
        stage_tokenizer(model, batch, args.device)

    ok = True
    if args.stage in ("transformer", "all"):
        ok = stage_transformer(model, batch, args.device, args.check_leakage)

    print("\n" + ("✅ all checks passed" if ok else "❌ some checks failed -- see above"))


if __name__ == "__main__":
    main()
