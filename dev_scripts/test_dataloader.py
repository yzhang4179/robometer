#!/usr/bin/env python3
"""Step 1 validation: look at what the processed cache actually contains.

Prints name / shape / dtype / range for every column and every npz key, then runs
three checks that would each catch a silent corruption:

  * **alignment** -- RGB and pointmap must be the *same* frames in the same order.
    Checked by correlating their per-frame spatial edge maps over shifts in
    [-3, +3]; the peak has to sit at shift 0. A k-frame slip passes every shape
    check ever written, so this is the only thing standing between us and
    training on quietly mismatched modalities.
  * **raw metres** -- the cache must hold unnormalised camera-frame XYZ. A cache
    that accidentally stored `[-1, 1]` values instead would look fine to every
    downstream shape check but train on garbage, so the ranges are printed and
    the z channel is asserted positive.
  * **clip coverage** -- how much of the data the frozen
    `pointmap_norm_bounds` actually saturate, per channel.

Two sampler checks sit on top of that, and the contrast between them is the point:

  --with-sampler     the **stock, unmodified** `RBMDataset` + `ProgressSampler`.
                     It returns RGB and no pointmap, because `load_frames_from_npz`
                     reads only the `frames` key. That absence is the positive
                     result: it is what "the original RGB path runs on our cache
                     with zero changes" actually looks like.
  --precise-sampler  `PreciseRBMDataset` + `PreciseProgressSampler`, which do
                     surface the pointmap, and `PreciseBatchCollator` on top. Also
                     re-checks RGB/pointmap alignment on the *sampled* frames, and
                     prints raw pointmap min/max so a cache that accidentally holds
                     normalised values instead of metres is obvious immediately.

Example:
    export ROBOMETER_PROCESSED_DATASETS_PATH=/path/to/processed_datasets

    uv run python dev_scripts/test_dataloader.py \
        --dataset precise_square_rbm_square_d1_train \
        --with-sampler
"""

from __future__ import annotations

# unsloth patches transformers before the broken torchao import path is reached.
# Without this, importing RBMDataset dies with
# `ImportError: cannot import name 'ScalingType' from 'torch.nn.functional'`.
try:  # noqa: SIM105
    import unsloth  # noqa: F401
except Exception:
    pass

import argparse
import json
import os
from typing import Dict, Optional, Tuple

import numpy as np

# Frozen for the square task -- see the Step 2 section of precise_robometer_research.md.
FROZEN_BOUNDS: Dict[str, Tuple[float, float]] = {
    "x": (-0.6, 0.6),
    "y": (-1.0, 0.25),
    "z": (0.2, 1.3),
}
CHANNELS = ("x", "y", "z")


def describe(name: str, array: np.ndarray) -> None:
    """One line of name / shape / dtype / range, the format this script exists for."""
    array = np.asarray(array)
    if array.size == 0:
        print(f"    {name:<16} shape={str(array.shape):<22} dtype={array.dtype}  (empty)")
        return
    if np.issubdtype(array.dtype, np.number):
        finite = array[np.isfinite(array)] if np.issubdtype(array.dtype, np.floating) else array
        lo, hi = (finite.min(), finite.max()) if finite.size else (float("nan"), float("nan"))
        print(f"    {name:<16} shape={str(array.shape):<22} dtype={str(array.dtype):<9} range=[{lo:.4g}, {hi:.4g}]")
    else:
        print(f"    {name:<16} shape={str(array.shape):<22} dtype={array.dtype}")


def edge_magnitude(array: np.ndarray) -> np.ndarray:
    """Per-frame spatial gradient magnitude, flattened to (T, H*W)."""
    array = array.astype(np.float32)
    grad_y = np.abs(np.diff(array, axis=1))[:, :, :-1]
    grad_x = np.abs(np.diff(array, axis=2))[:, :-1, :]
    return np.hypot(grad_x, grad_y).reshape(array.shape[0], -1)


def best_alignment_shift(frames: np.ndarray, pointmap: np.ndarray, max_shift: int = 3) -> Tuple[int, float, float]:
    """Find the frame offset at which RGB and pointmap agree best.

    RGB and depth come from the same render, so their *spatial* edges coincide:
    the arm's silhouette sits in the same pixels in both. Correlating per-frame
    edge maps and scanning the offset therefore peaks sharply at 0 when the two
    are aligned, and at k when the pointmap is k frames off -- which is exactly
    the silent corruption worth catching, since every shape check still passes.

    Returns (best shift, its score, margin over the best non-zero shift).
    """
    rgb_edges = edge_magnitude(frames.mean(axis=-1))
    depth_edges = edge_magnitude(pointmap[..., 2])

    def correlate(shift: int) -> float:
        if shift >= 0:
            left, right = rgb_edges[shift:], depth_edges[: len(depth_edges) - shift]
        else:
            left, right = rgb_edges[: len(rgb_edges) + shift], depth_edges[-shift:]
        if len(left) == 0:
            return float("nan")
        per_frame = [
            np.corrcoef(a, b)[0, 1] for a, b in zip(left, right) if a.std() > 0 and b.std() > 0
        ]
        return float(np.mean(per_frame)) if per_frame else float("nan")

    scores = {shift: correlate(shift) for shift in range(-max_shift, max_shift + 1)}
    valid = {k: v for k, v in scores.items() if not np.isnan(v)}
    if not valid:
        return 0, float("nan"), float("nan")
    best = max(valid, key=valid.get)
    others = [v for k, v in valid.items() if k != 0]
    margin = valid.get(0, float("nan")) - max(others) if others else float("nan")
    return best, valid[best], margin


def inspect_cache(cache_dir: str, num_trajectories: int, out_dir: Optional[str]) -> None:
    from datasets import Dataset

    from robometer.data.datasets.helpers import load_frames_from_npz

    dataset_dir = os.path.join(cache_dir, "processed_dataset")
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"No processed dataset at {dataset_dir} -- run preprocess_precise.py first")

    dataset = Dataset.load_from_disk(dataset_dir)
    info_path = os.path.join(cache_dir, "dataset_info.json")
    if os.path.exists(info_path):
        with open(info_path) as handle:
            print(f"  dataset_info: {json.load(handle)}")

    print(f"\n=== Processed dataset: {len(dataset)} trajectories ===")
    print("  columns:")
    for column, feature in dataset.features.items():
        print(f"    {column:<22} {feature}")

    row = dataset[0]
    print("\n  row[0]:")
    for key, value in row.items():
        shown = value
        if isinstance(value, list):
            shown = f"list(len={len(value)}) e.g. {value[:4]}"
        elif isinstance(value, str) and len(value) > 90:
            shown = "..." + value[-87:]
        print(f"    {key:<22} {type(value).__name__:<8} {shown}")

    mappings_path = os.path.join(cache_dir, "index_mappings.json")
    if os.path.exists(mappings_path):
        with open(mappings_path) as handle:
            mappings = json.load(handle)
        print("\n  index_mappings:")
        for key, value in mappings.items():
            size = len(value) if isinstance(value, (list, dict)) else value
            print(f"    {key:<26} {size}")

    print(f"\n=== npz contents (first {num_trajectories} trajectories) ===")
    all_ok = True
    for index in range(min(num_trajectories, len(dataset))):
        row = dataset[index]
        npz_path = row["frames"]
        print(f"\n  [{index}] id={row['id']}  quality={row['quality_label']}  source={row['data_source']}")
        with np.load(npz_path) as cached:
            keys = list(cached.keys())
            print(f"    keys: {keys}")
            for key in keys:
                describe(key, cached[key])
            frames = cached["frames"]
            pointmap = cached["pointmap"] if "pointmap" in keys else None

        # The stock RGB reader must see exactly the RGB array and nothing else.
        via_helper = load_frames_from_npz(npz_path)
        assert via_helper.shape == frames.shape and via_helper.dtype == frames.dtype, "load_frames_from_npz changed"
        assert np.array_equal(via_helper, frames), "load_frames_from_npz did not return the RGB array"
        print(f"    load_frames_from_npz -> {via_helper.shape} {via_helper.dtype}  (RGB only, unchanged ✓)")

        if pointmap is None:
            print("    ⚠️  no 'pointmap' key -- this cache is RGB-only")
            all_ok = False
            continue

        if frames.shape[0] != pointmap.shape[0] or frames.shape[1:3] != pointmap.shape[1:3]:
            print(f"    ❌ shape mismatch: frames {frames.shape} vs pointmap {pointmap.shape}")
            all_ok = False
            continue

        for axis, channel in enumerate(CHANNELS):
            values = pointmap[..., axis].astype(np.float32)
            lo, hi = FROZEN_BOUNDS[channel]
            clipped = float(np.mean((values < lo) | (values > hi)) * 100)
            print(
                f"    {channel}: [{values.min():+.3f}, {values.max():+.3f}] m   "
                f"median={np.median(values):+.3f}   {clipped:5.2f}% outside {list(FROZEN_BOUNDS[channel])}"
            )

        depth = pointmap[..., 2].astype(np.float32)
        if depth.min() <= 0:
            print(f"    ❌ z has non-positive values (min {depth.min():.4f}) -- not raw camera-frame depth")
            all_ok = False
        elif depth.max() <= 1.01:
            print(f"    ❌ z max is {depth.max():.4f} -- looks normalised, the cache must hold raw metres")
            all_ok = False

        shift, score, margin = best_alignment_shift(frames, pointmap)
        if np.isnan(score):
            # Every frame had a flat edge map, so the test could not run. Never call
            # that a pass -- it is exactly the case where a slip would go unnoticed.
            print("    alignment: ⚠️  inconclusive (no spatial gradient to correlate)")
        else:
            verdict = "✓" if shift == 0 else "❌"
            print(f"    alignment: best shift={shift:+d}  edge-corr={score:.3f}  margin={margin:+.3f} {verdict}")
            if shift != 0:
                all_ok = False

        if out_dir and index == 0:
            write_preview(frames, pointmap, out_dir, row["id"])

    print("\n" + ("✅ cache looks correct" if all_ok else "❌ problems found above"))


def write_preview(frames: np.ndarray, pointmap: np.ndarray, out_dir: str, traj_id: str) -> None:
    """RGB above, pointmap-under-frozen-bounds below, for a handful of frames."""
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)
    picks = np.linspace(0, frames.shape[0] - 1, min(6, frames.shape[0]), dtype=int)

    lo = np.array([FROZEN_BOUNDS[c][0] for c in CHANNELS], dtype=np.float32)
    hi = np.array([FROZEN_BOUNDS[c][1] for c in CHANNELS], dtype=np.float32)
    clipped = np.clip(pointmap[picks].astype(np.float32), lo, hi)
    normalised = ((clipped - lo) / (hi - lo) * 255).astype(np.uint8)

    top = np.concatenate(list(frames[picks]), axis=1)
    bottom = np.concatenate(list(normalised), axis=1)
    path = os.path.join(out_dir, f"preview_{traj_id}.png")
    Image.fromarray(np.concatenate([top, bottom], axis=0)).save(path)
    print(f"    preview -> {path}  (top: RGB, bottom: XYZ under the frozen bounds)")


def run_sampler(dataset_name: str, max_frames: int, cutoff_file: str) -> None:
    """Pull one real sample through the stock robometer dataset + sampler."""
    from robometer.configs.experiment_configs import DataConfig
    from robometer.data.datasets.rbm_data import RBMDataset

    config = DataConfig(
        train_datasets=[dataset_name],
        eval_datasets=[dataset_name],
        max_frames=max_frames,
        min_frames_per_trajectory=5,
        sample_type_ratio=[0, 1, 0],  # progress samples only
        progress_strategy_ratio=[0, 1, 1, 1],  # no different-task: single-task setting
        progress_pred_type="absolute_first_frame",
        dataset_success_cutoff_file=cutoff_file,
        load_embeddings=False,  # must stay false, see finding 5 in the plan
        shuffle=False,
    )

    print(f"\n=== Stock RBMDataset + ProgressSampler (max_frames={max_frames}) ===")
    dataset = RBMDataset(config, is_evaluation=False)
    print(f"  dataset length: {len(dataset)}")

    sample = dataset[0]
    trajectory = sample.trajectory
    print(f"  sample_type={sample.sample_type}  data_gen_strategy={sample.data_gen_strategy}")
    print("  Trajectory fields:")
    for field_name, value in trajectory.model_dump().items():
        if value is None:
            continue
        if isinstance(value, np.ndarray):
            describe(field_name, value)
        elif isinstance(value, list) and value and isinstance(value[0], (int, float)):
            describe(field_name, np.asarray(value))
        elif isinstance(value, dict):
            print(f"    {field_name:<16} dict keys={list(value.keys())}")
        else:
            text = str(value)
            print(f"    {field_name:<16} {type(value).__name__:<8} {text[:80]}")

    frames = np.asarray(trajectory.frames)
    print(f"\n  ✅ stock sampler returned {frames.shape} {frames.dtype} RGB with no code changes")
    print("  (pointmaps are not in this Trajectory -- that is PreciseProgressSampler's job in Step 2)")


def run_precise_sampler(dataset_name: str, max_frames: int, cutoff_file: str, batch_size: int) -> bool:
    """Pull a real sample and a real batch through the precise dataset + collator."""
    from robometer.configs.experiment_configs import DataConfig
    from robometer.data.collators.precise import PreciseBatchCollator
    from robometer.data.datasets.precise_data import PreciseRBMDataset

    config = DataConfig(
        train_datasets=[dataset_name],
        eval_datasets=[dataset_name],
        max_frames=max_frames,
        min_frames_per_trajectory=5,
        sample_type_ratio=[0, 1, 0],
        progress_strategy_ratio=[0, 1, 1, 1],
        progress_pred_type="absolute_first_frame",
        dataset_success_cutoff_file=cutoff_file,
        load_embeddings=False,
        shuffle=False,
    )

    print(f"\n=== PreciseRBMDataset + PreciseProgressSampler (max_frames={max_frames}) ===")
    dataset = PreciseRBMDataset(config, is_evaluation=False)
    sample = dataset[0]
    trajectory = sample.trajectory
    metadata = trajectory.metadata or {}

    print(f"  sample_type={sample.sample_type}  data_gen_strategy={sample.data_gen_strategy}")
    frames = np.asarray(trajectory.frames)
    pointmap = np.asarray(metadata["pointmap"])
    indices = np.asarray(metadata["frame_indices"])
    describe("frames", frames)
    describe("metadata[pointmap]", pointmap)
    print(f"    metadata[frame_indices] {indices.tolist()}")
    print("    RGB and pointmap were sliced with that one vector -- see PreciseProgressSampler")
    describe("target_progress", np.asarray(trajectory.target_progress))

    ok = True
    if frames.shape[0] != pointmap.shape[0]:
        print(f"    ❌ {frames.shape[0]} rgb frames but {pointmap.shape[0]} pointmap frames")
        ok = False
    shift, score, margin = best_alignment_shift(frames, pointmap, max_shift=min(3, max_frames - 1))
    if np.isnan(score):
        print("    alignment on the sampled frames: ⚠️  inconclusive (no spatial gradient to correlate)")
    else:
        verdict = "✓" if shift == 0 else "❌"
        print(
            f"    alignment on the sampled frames: best shift={shift:+d}  edge-corr={score:.3f}  "
            f"margin={margin:+.3f} {verdict}"
        )
        ok = ok and shift == 0

    collator = PreciseBatchCollator(processor=None, base_model_id="precise_transformer", load_embeddings=False)
    batch = collator([dataset[i] for i in range(batch_size)])["progress_inputs"]
    print(f"\n  collated batch (batch_size={batch_size}) -- what the model receives:")
    for key, value in batch.items():
        if isinstance(value, np.ndarray) or hasattr(value, "shape"):
            describe(key, np.asarray(value))
        else:
            print(f"    {key:<16} {type(value).__name__}(len={len(value)})")

    pm_batch = np.asarray(batch["pointmap_values"]).astype(np.float32)
    print("\n  raw pointmap range per channel (must be METRES, not [-1, 1])")
    for axis, channel in enumerate(CHANNELS):
        values = pm_batch[:, :, axis]
        lo, hi = FROZEN_BOUNDS[channel]
        outside = float(np.mean((values < lo) | (values > hi)) * 100)
        print(
            f"    {channel}: [{values.min():+.3f}, {values.max():+.3f}] m  median={np.median(values):+.3f}  "
            f"{outside:5.2f}% outside {list(FROZEN_BOUNDS[channel])}"
        )
    depth = pm_batch[:, :, 2]
    if depth.max() <= 1.01:
        print(f"    ❌ z max is {depth.max():.4f} -- this looks normalised; the cache must hold raw metres")
        ok = False
    else:
        print("    ✅ z is in metres; the clamp to pointmap_norm_bounds happens later, on GPU")

    print(f"\n  {'✅' if ok else '❌'} precise sampler path {'looks correct' if ok else 'has problems'}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="Cache dir name, e.g. precise_square_rbm_square_d1_train")
    parser.add_argument("--cache-dir", default=None, help="Default: $ROBOMETER_PROCESSED_DATASETS_PATH")
    parser.add_argument("--num-trajectories", type=int, default=3, help="How many npz files to open")
    parser.add_argument("--with-sampler", action="store_true", help="Also run the stock RBMDataset + ProgressSampler")
    parser.add_argument(
        "--precise-sampler",
        action="store_true",
        help="Also run PreciseRBMDataset + PreciseProgressSampler + PreciseBatchCollator (Step 2)",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for --precise-sampler")
    parser.add_argument("--max-frames", type=int, default=9, help="data.max_frames for the sampler check")
    parser.add_argument(
        "--cutoff-file",
        default="robometer/data/dataset_success_cutoff_precise.txt",
        help="Repo-relative, so run from the repo root",
    )
    parser.add_argument("--out-dir", default="dev_scripts/out/dataloader", help="Where to write the preview png")
    args = parser.parse_args()

    cache_root = args.cache_dir or os.environ.get("ROBOMETER_PROCESSED_DATASETS_PATH", "")
    if not cache_root:
        raise ValueError("Set ROBOMETER_PROCESSED_DATASETS_PATH or pass --cache-dir")
    os.environ["ROBOMETER_PROCESSED_DATASETS_PATH"] = cache_root

    print(f"cache root: {cache_root}")
    inspect_cache(os.path.join(cache_root, args.dataset), args.num_trajectories, args.out_dir)

    if args.with_sampler:
        run_sampler(args.dataset, args.max_frames, args.cutoff_file)

    if args.precise_sampler:
        run_precise_sampler(args.dataset, args.max_frames, args.cutoff_file, args.batch_size)


if __name__ == "__main__":
    main()
