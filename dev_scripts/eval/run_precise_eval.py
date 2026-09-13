#!/usr/bin/env python3
"""Step 4 part 1: mimicgen test hdf5 -> one progress/success video per demo per model.

Reads the raw mimicgen hdf5 directly, so the qualitative test splits never need to
be converted to the RBM format or preprocessed into a cache.  Both streams come
out of the same file at full frame rate (~134-171 frames), which gives a much
smoother curve than the 32-frame training cache would.

The three modality checkpoints are loaded into **one** process, so each demo is
read from disk once and scored by all three.  Output layout is exactly the tree
from the Step 4 spec, with the file name taken from each checkpoint's own
``model.precise.modality`` so it cannot drift from what the weights actually are:

    <save-dir>/<task-name>/<demo key>/rgb_only.mp4
                                     pointmap_only.mp4
                                     rgb_pointmap.mp4
                                     predictions.json
    <save-dir>/<task-name>/metrics.json

Each video frame is three panels, left to right: the RGB frame, the predicted
progress curve, the predicted success probability.  Geometry is identical across
the three models so they stack visually.

**The metrics assume every demo in the file is a failure.**  That is true of the
three Step 4 test splits, and it is what makes a success metric possible without
ground-truth progress: every success label is 0, so the success head can be scored
on how reliably it says "not done".  ``--has-success-labels`` turns the metrics off
for a mixed or successful split rather than reporting something false.

Example (one split, all three models, 2 demos as a smoke test):

    uv run python dev_scripts/eval/run_precise_eval.py \\
        --hdf5 $SQUARE/demo_src_square_task_D1_validation/demo_failed.hdf5 \\
        --task-name square_d1_val_failed \\
        --model ./logs/precise/precise_rgb/final \\
        --model ./logs/precise/precise_pointmap/final \\
        --model ./logs/precise/precise_rgb_pointmap/final \\
        --save-dir ./dev_scripts/out/eval --max-demos 2

`dev_scripts/eval/run_all_eval.sh` drives every split and both bounds settings.
"""

# Same import-order preamble as train_precise.py; see precise_eval_server.py.
import sys
import types

from transformers.utils import import_utils as transformers_import_utils

transformers_import_utils._torchao_available = False

if "unsloth" not in sys.modules:

    class _UnusedFastVisionModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            del args, kwargs
            raise RuntimeError("Precise does not support the stock Unsloth model path")

    _stub = types.ModuleType("unsloth")
    _stub.FastVisionModel = _UnusedFastVisionModel
    sys.modules["unsloth"] = _stub

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

from robometer.evals.precise_eval_server import PreciseInferenceEngine

# modality -> the file name the Step 4 spec asks for.
MODALITY_FILENAME = {
    "rgb": "rgb_only",
    "pointmap": "pointmap_only",
    "rgb_pointmap": "rgb_pointmap",
}
CHANNELS = ("x", "y", "z")

PROGRESS_COLOR = "#1f6feb"
SUCCESS_COLOR = "#d1660f"
PLAYHEAD_COLOR = (255, 255, 255)
PROGRESS_MARKER = (31, 111, 235)
SUCCESS_MARKER = (209, 102, 15)


# ------------------------------------------------------------------------ hdf5


def find_camera(demo_group: h5py.Group, want_pointmap: bool = True, camera: Optional[str] = None) -> str:
    """The camera prefix to read frames from.

    Splits disagree on the name (``agentview`` vs ``sideview``), so this is detected
    rather than passed in -- the same rule ``hdf5_to_rbm.py`` uses. An rgb-only
    checkpoint needs no ``_pointmap``, and requiring one made rgb-only evaluation
    raise ``KeyError`` on a file that was perfectly usable.
    """
    obs = demo_group["obs"]
    cameras = sorted(key[: -len("_image")] for key in obs.keys() if key.endswith("_image"))
    if want_pointmap:
        cameras = [name for name in cameras if f"{name}_pointmap" in obs]
    if camera is not None:
        if camera not in cameras:
            raise KeyError(f"camera {camera!r} is not usable here; available: {cameras}")
        return camera
    if not cameras:
        need = "an _image and a _pointmap" if want_pointmap else "an _image"
        raise KeyError(f"no camera with {need} in obs; got {sorted(obs.keys())}")
    if len(cameras) > 1:
        print(f"    note: multiple cameras {cameras}, using {cameras[0]} (pass --camera to choose)")
    return cameras[0]


def demo_keys(handle: h5py.File) -> List[str]:
    """hdf5 demo keys in numeric order (``demo_0``, ``demo_1``, ... not ``demo_10`` first)."""
    keys = list(handle["data"].keys())

    def order(key: str):
        suffix = key.rsplit("_", 1)[-1]
        return (0, int(suffix)) if suffix.isdigit() else (1, key)

    return sorted(keys, key=order)


def load_display_rgb(handle: h5py.File, key: str, camera: Optional[str], fallback: np.ndarray) -> np.ndarray:
    """Frames for the video's RGB panel, which need not be what the model is fed.

    Exp 3 renders a goal-marker variant (``agentview_goal``) of the same rollout. A
    visual geom is rasterised into the depth pass too, so those frames would shift the
    pointmap out of distribution -- they are for the viewer only, never for the model.
    """
    if camera is None:
        return fallback
    obs = handle["data"][key]["obs"]
    frames = np.asarray(obs[f"{camera}_image"][:], dtype=np.uint8)
    if frames.shape[0] != fallback.shape[0]:
        raise ValueError(f"{key}: display camera has {frames.shape[0]} frames but the model camera has {fallback.shape[0]}")
    return frames


def load_demo(handle: h5py.File, key: str, camera: str, want_pointmap: bool):
    """``(rgb [T,H,W,3] uint8, pointmap [T,H,W,3] float16 | None)`` for one demo."""
    obs = handle["data"][key]["obs"]
    rgb = np.asarray(obs[f"{camera}_image"][:], dtype=np.uint8)
    pointmap = None
    if want_pointmap:
        # float16 matches the training cache exactly (Step 1); the VAE consumes bf16
        # afterwards, so this loses nothing the model could have used.
        pointmap = np.asarray(obs[f"{camera}_pointmap"][:], dtype=np.float32).astype(np.float16)
        if pointmap.shape[0] != rgb.shape[0]:
            raise ValueError(f"{key}: {rgb.shape[0]} rgb frames but {pointmap.shape[0]} pointmap frames")
    return rgb, pointmap


# ------------------------------------------------------------------- rendering


def _resize_nearest(frame: np.ndarray, size: int) -> np.ndarray:
    """Upscale a square-ish frame with no interpolation, so pixels stay honest."""
    return np.asarray(Image.fromarray(frame).resize((size, size), Image.Resampling.NEAREST))


def render_curve_panel(
    values: np.ndarray,
    label: str,
    color: str,
    title: str,
    width: int,
    height: int,
    ylabel: str,
    threshold: Optional[float] = None,
    dpi: int = 100,
):
    """Render one curve once; the playhead is composited per frame.

    Rendering matplotlib for every frame would mean tens of thousands of draws over
    a full sweep. Drawing once and compositing the playhead with numpy keeps the run
    dominated by the VAE rather than by plotting.
    """
    num_frames = values.shape[0]
    figure = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    axes = figure.add_subplot(111)

    axes.plot(np.arange(num_frames), values, label=label, color=color, linewidth=1.8)
    if threshold is not None:
        axes.axhline(threshold, color="#999999", linewidth=0.9, linestyle=":", zorder=0)
        axes.text(
            0.99, threshold + 0.02, f"threshold {threshold:g}", transform=axes.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=6, color="#777777",
        )

    axes.set_xlim(0, max(num_frames - 1, 1))
    axes.set_ylim(-0.03, 1.03)
    axes.set_xlabel("frame")
    axes.set_ylabel(ylabel, fontsize=8)
    axes.set_title(title, fontsize=9)
    axes.grid(alpha=0.25, linewidth=0.6)
    figure.tight_layout(pad=0.6)
    figure.canvas.draw()

    panel = np.asarray(figure.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
    return panel, axes, figure


def data_to_pixels(axes, xs: np.ndarray, ys: np.ndarray, panel_height: int) -> Tuple[np.ndarray, np.ndarray]:
    """Data coordinates -> (column, row) in the rendered panel array."""
    points = axes.transData.transform(np.column_stack([xs, ys]))
    columns = np.rint(points[:, 0]).astype(int)
    # Matplotlib's display origin is bottom-left; image row 0 is the top.
    rows = np.rint(panel_height - points[:, 1]).astype(int)
    return columns, rows


def _draw_playhead(panel: np.ndarray, column: int, marker_row: int, color: Tuple[int, int, int]) -> np.ndarray:
    """A vertical line at the current frame plus a dot on the curve."""
    out = panel.copy()
    height, width = out.shape[:2]
    column = int(np.clip(column, 0, width - 1))
    out[:, max(column - 1, 0) : column + 1] = PLAYHEAD_COLOR
    marker_column = int(np.clip(column, 2, width - 3))
    marker_row = int(np.clip(marker_row, 2, height - 3))
    out[marker_row - 2 : marker_row + 3, marker_column - 2 : marker_column + 3] = color
    return out


def compose_frame(rgb_panel: np.ndarray, curve_panels: Sequence[np.ndarray], caption: str) -> np.ndarray:
    """RGB frame, then each curve panel, left to right."""
    canvas = np.concatenate([rgb_panel, *curve_panels], axis=1)
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    draw.rectangle([(0, 0), (rgb_panel.shape[1] - 1, 13)], fill=(0, 0, 0))
    draw.text((4, 2), caption, fill=(255, 255, 255))
    return np.asarray(image)


def write_demo_video(
    path: str,
    rgb: np.ndarray,
    progress: np.ndarray,
    success: Optional[np.ndarray],
    title: str,
    frame_size: int,
    plot_width: int,
    fps: int,
    threshold: float,
) -> None:
    """One mp4 for one (demo, model): rgb | progress | success prediction."""
    num_frames = rgb.shape[0]
    frames_axis = np.arange(num_frames)

    progress_panel, progress_axes, progress_figure = render_curve_panel(
        progress, "progress", PROGRESS_COLOR, "predicted progress", plot_width, frame_size, "progress"
    )
    columns, progress_rows = data_to_pixels(progress_axes, frames_axis, progress, frame_size)
    plt.close(progress_figure)

    success_panel = success_rows = None
    if success is not None:
        success_panel, success_axes, success_figure = render_curve_panel(
            success, "success prob", SUCCESS_COLOR, "predicted success", plot_width, frame_size,
            "P(success)", threshold=threshold,
        )
        _, success_rows = data_to_pixels(success_axes, frames_axis, success, frame_size)
        plt.close(success_figure)

    resized = [_resize_nearest(rgb[t], frame_size) for t in range(num_frames)]

    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
        for t in range(num_frames):
            panels = [_draw_playhead(progress_panel, columns[t], progress_rows[t], PROGRESS_MARKER)]
            caption = f"{t + 1}/{num_frames}  prog={progress[t]:.2f}"
            if success_panel is not None:
                panels.append(_draw_playhead(success_panel, columns[t], success_rows[t], SUCCESS_MARKER))
                caption += f"  succ={success[t]:.2f}"
            writer.append_data(compose_frame(resized[t], panels, caption))


# --------------------------------------------------------------------- metrics


def summarise_failure_split(
    demo_records: Dict[str, Dict[str, list]],
    threshold: float,
) -> Dict[str, object]:
    """Success metrics for a split in which **every demo is a failure**.

    Every success label is therefore 0, which is exactly the case
    ``negative_success_acc`` covers in
    ``compile_results.run_reward_alignment_eval_per_trajectory`` -- the fraction of
    label-0 predictions correctly placed at or below the threshold.  The same
    function's ``success_auprc`` and ``positive_success_acc`` are **not** reported
    here: both need positives, and robometer's AUPRC branch falls through to a
    hardcoded ``0.0`` when only one class is present, which would look like a real
    (and terrible) score rather than an undefined one.

    Scored endpoints are used, not per-frame expanded values, so a stride > 1 does
    not weight a prediction by how many frames it happens to cover.

    Progress is reported only as context: these trajectories never succeed, so
    there is no honest ground-truth progress curve to correlate against.
    """
    per_frame_success: List[np.ndarray] = []
    peak_success: List[float] = []
    final_success: List[float] = []
    final_progress: List[float] = []
    peak_progress: List[float] = []
    alarms = 0
    has_success = False

    for record in demo_records.values():
        progress = np.asarray(record["progress"], dtype=np.float64)
        final_progress.append(float(progress[-1]))
        peak_progress.append(float(progress.max()))
        if record.get("success") is None:
            continue
        has_success = True
        success = np.asarray(record["success"], dtype=np.float64)
        per_frame_success.append(success)
        peak_success.append(float(success.max()))
        final_success.append(float(success[-1]))
        alarms += int(success.max() > threshold)

    num_demos = len(demo_records)
    summary: Dict[str, object] = {
        "num_demos": num_demos,
        "success_threshold": threshold,
        "progress_context": {
            "mean_final_progress": float(np.mean(final_progress)) if final_progress else None,
            "mean_peak_progress": float(np.mean(peak_progress)) if peak_progress else None,
            "max_final_progress": float(np.max(final_progress)) if final_progress else None,
        },
    }
    if not has_success:
        summary["note"] = "checkpoint has no success head; success metrics omitted"
        return summary

    pooled = np.concatenate(per_frame_success)
    correct = float((pooled <= threshold).mean())
    summary["per_frame"] = {
        "num_scored_endpoints": int(pooled.size),
        # Robometer's definition, specialised to all-zero labels.
        "negative_success_acc": correct,
        "false_positive_rate": 1.0 - correct,
        "mean_success_prob": float(pooled.mean()),
        "max_success_prob": float(pooled.max()),
    }
    summary["per_demo"] = {
        "false_alarm_demo_rate": alarms / num_demos if num_demos else None,
        "num_demos_with_false_alarm": alarms,
        "mean_peak_success_prob": float(np.mean(peak_success)),
        "mean_final_success_prob": float(np.mean(final_success)),
        "max_peak_success_prob": float(np.max(peak_success)),
    }
    summary["success_auprc"] = None
    summary["positive_success_acc"] = None
    summary["undefined_metrics_note"] = (
        "success_auprc and positive_success_acc need both classes; this split is failures only, "
        "so every label is 0. Robometer would emit 0.0 for AUPRC here, which is a sentinel, not a score."
    )
    return summary


# ------------------------------------------------------------------- inference


def expand_to_frames(end_indices: np.ndarray, values: np.ndarray, num_frames: int) -> np.ndarray:
    """Scored endpoints -> one value per original frame.

    Holds the value of the most recent endpoint at or before each frame.  Rounding
    to the *nearest* endpoint instead would let a frame display a value computed
    from frames after it, which is exactly the lookahead the growing-prefix
    protocol exists to avoid.  Endpoint 0 is always present, so every frame has one.
    """
    positions = np.searchsorted(end_indices, np.arange(num_frames), side="right") - 1
    positions = np.clip(positions, 0, len(end_indices) - 1)
    return values[positions]


def parse_model_arg(item: str) -> Tuple[Optional[str], str]:
    """``PATH`` or ``LABEL=PATH``; the label is only a display name.

    Checkpoint directories contain '=' themselves
    (``ckpt-pearson_..._val=0.9958_step=18250``), so a bare "contains '='" test
    would tear a real path in half. A label is a plain word: no path separator.
    """
    head, sep, tail = item.partition("=")
    if sep and "/" not in head and os.sep not in head:
        return head.strip(), tail.strip()
    return None, item


def parse_bounds(items: Sequence[str]) -> Optional[Dict[str, List[float]]]:
    """``x=-1.0,1.0 y=-1.2,0.5 z=0.9,2.2`` -> the model.precise bounds dict."""
    if not items:
        return None
    bounds: Dict[str, List[float]] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--pointmap-bounds entry must look like z=0.9,2.2 (got {item!r})")
        key, value = item.split("=", 1)
        key = key.strip().lower()
        if key not in CHANNELS:
            raise ValueError(f"unknown channel {key!r}, expected one of {CHANNELS}")
        low, high = (float(part) for part in value.split(","))
        bounds[key] = [low, high]
    missing = [channel for channel in CHANNELS if channel not in bounds]
    if missing:
        raise ValueError(f"--pointmap-bounds must give all three channels; missing {missing}")
    return bounds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score mimicgen test demos with the Precise models and write per-demo videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--hdf5", required=True, help="Source mimicgen hdf5 (demo.hdf5 or demo_failed.hdf5)")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        dest="models",
        metavar="[LABEL=]PATH",
        help="Precise checkpoint dir; repeat once per modality",
    )
    parser.add_argument("--save-dir", default="./dev_scripts/out/eval", help="Root of the output tree")
    parser.add_argument(
        "--task-name",
        default=None,
        help="Directory under --save-dir. Defaults to the hdf5's parent folder plus its stem",
    )
    parser.add_argument(
        "--num-subsampled-frames",
        type=int,
        default=5,
        help="Frames per growing-prefix clip; must be 4k+1 and <= the checkpoint's max_len",
    )
    parser.add_argument("--frame-stride", type=int, default=1, help="Score every Nth frame (1 = every frame)")
    parser.add_argument("--batch-size", type=int, default=32, help="Clips per forward pass")
    parser.add_argument("--max-demos", type=int, default=-1, help="Cap for smoke tests; -1 for all")
    parser.add_argument("--fps", type=int, default=15, help="Playback fps of the written mp4s")
    parser.add_argument("--frame-size", type=int, default=384, help="Pixel size of the video panel (square)")
    parser.add_argument("--plot-width", type=int, default=448, help="Pixel width of EACH curve panel")
    parser.add_argument(
        "--success-threshold", type=float, default=0.5, help="Success probability counted as a positive"
    )
    parser.add_argument(
        "--has-success-labels",
        action="store_true",
        help="This split is NOT failures-only, so skip the failure metrics instead of reporting false ones",
    )
    parser.add_argument(
        "--pointmap-bounds",
        nargs="*",
        default=[],
        metavar="C=LO,HI",
        help="Override the checkpoints' pointmap clip bounds, e.g. x=-1.0,1.0 y=-1.2,0.5 z=0.9,2.2",
    )
    parser.add_argument(
        "--quality-label",
        default=None,
        choices=["successful", "failure"],
        help="Write under <task>/<quality_label>/. 'successful' also skips the failure-only metrics",
    )
    parser.add_argument("--camera", default=None, help="Camera prefix fed to the models; auto-detected when omitted")
    parser.add_argument(
        "--task",
        default=None,
        help="Instruction for checkpoints trained with use_lang_token=true, e.g. 'lift the cube to 5 cm'. "
        "Embedded with the same all-MiniLM-L6-v2 the converter used",
    )
    parser.add_argument(
        "--display-camera",
        default=None,
        help="Camera used ONLY for the video's RGB panel, e.g. agentview_goal. Models still see --camera",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true", help="Re-render demos whose mp4s already exist")
    args = parser.parse_args()

    # A successes split has real success labels, so the failure-only summary is wrong there.
    if args.quality_label == "successful":
        args.has_success_labels = True

    clip_length = args.num_subsampled_frames
    bounds = parse_bounds(args.pointmap_bounds)

    task_name = args.task_name
    if task_name is None:
        parent = os.path.basename(os.path.dirname(os.path.abspath(args.hdf5)))
        task_name = f"{parent}__{os.path.splitext(os.path.basename(args.hdf5))[0]}"
    task_dir = os.path.join(args.save_dir, task_name)
    if args.quality_label is not None:
        task_dir = os.path.join(task_dir, args.quality_label)
    os.makedirs(task_dir, exist_ok=True)

    print("\n=== Precise qualitative eval ===")
    print(f"  hdf5      : {args.hdf5}")
    print(f"  output    : {task_dir}")
    print(f"  clip len  : {clip_length}")
    print(f"  bounds    : {'as trained' if bounds is None else bounds}")

    engines: List[PreciseInferenceEngine] = []
    for item in args.models:
        label, path = parse_model_arg(item)
        engines.append(
            PreciseInferenceEngine(
                model_path=path,
                device=args.device,
                pointmap_norm_bounds=bounds,
                num_subsampled_frames=clip_length,
                batch_size=args.batch_size,
                label=label,
            )
        )
    modalities = {engine.modality for engine in engines}
    if len(modalities) != len(engines):
        raise ValueError(f"two checkpoints share a modality, so they would overwrite each other: {modalities}")
    if any(engine.needs_lang for engine in engines):
        if args.task is None:
            raise ValueError("a checkpoint has use_lang_token=true; pass --task 'lift the cube to N cm'")
        from dataset_upload.helpers import load_sentence_transformer_model

        embedding = load_sentence_transformer_model().encode(args.task).astype(np.float32)
        for engine in engines:
            engine.set_instruction(embedding if engine.needs_lang else None)
        print(f"  task      : {args.task!r} -> lang_vector[{embedding.shape[0]}]")
    want_pointmap = any(engine.needs_pointmap for engine in engines)
    if bounds is not None and not want_pointmap:
        print("  note: --pointmap-bounds given but no loaded checkpoint consumes pointmaps; it has no effect")

    # modality -> {demo key -> {"progress": [...], "success": [...]}}, for the split summary.
    collected: Dict[str, Dict[str, Dict[str, list]]] = {engine.modality: {} for engine in engines}

    started = time.time()
    with h5py.File(args.hdf5, "r") as handle:
        keys = demo_keys(handle)
        if args.max_demos > 0:
            keys = keys[: args.max_demos]
        camera = find_camera(handle["data"][keys[0]], want_pointmap, args.camera)
        print(f"  camera    : {camera}   demos: {len(keys)}\n")

        for position, key in enumerate(keys):
            demo_dir = os.path.join(task_dir, key)
            predictions_path = os.path.join(demo_dir, "predictions.json")
            expected = [os.path.join(demo_dir, f"{MODALITY_FILENAME[e.modality]}.mp4") for e in engines]

            if not args.overwrite and os.path.exists(predictions_path) and all(map(os.path.exists, expected)):
                # Reuse the stored curves so the split metrics still cover every demo.
                with open(predictions_path) as stream:
                    cached = json.load(stream)
                if cached.get("clip_length") == clip_length and set(cached.get("models", {})) == modalities:
                    for modality, record in cached["models"].items():
                        collected[modality][key] = record
                    print(f"  [{position + 1}/{len(keys)}] {key}: already rendered, reusing predictions.json")
                    continue

            os.makedirs(demo_dir, exist_ok=True)
            rgb, pointmap = load_demo(handle, key, camera, want_pointmap)
            display_rgb = load_display_rgb(handle, key, args.display_camera, rgb)
            num_frames = rgb.shape[0]
            demo_started = time.time()
            demo_record: Dict[str, object] = {
                "num_frames": int(num_frames),
                "camera": camera,
                "clip_length": clip_length,
                "frame_stride": args.frame_stride,
                "models": {},
            }

            for engine in engines:
                out = engine.predict_trajectory(
                    rgb if engine.needs_rgb else None,
                    pointmap if engine.needs_pointmap else None,
                    frame_stride=args.frame_stride,
                    num_subsampled_frames=clip_length,
                )
                ends = out["end_indices"]
                progress = expand_to_frames(ends, out["progress"], num_frames)
                success = expand_to_frames(ends, out["success"], num_frames) if "success" in out else None

                write_demo_video(
                    path=os.path.join(demo_dir, f"{MODALITY_FILENAME[engine.modality]}.mp4"),
                    rgb=display_rgb,
                    progress=progress,
                    success=success,
                    title=f"{engine.label}  |  {task_name}/{key}",
                    frame_size=args.frame_size,
                    plot_width=args.plot_width,
                    fps=args.fps,
                    threshold=args.success_threshold,
                )

                record = {
                    "end_indices": ends.tolist(),
                    "progress": np.round(out["progress"], 5).tolist(),
                    "success": np.round(out["success"], 5).tolist() if "success" in out else None,
                }
                demo_record["models"][engine.modality] = record
                collected[engine.modality][key] = record

            with open(predictions_path, "w") as stream:
                json.dump(demo_record, stream, indent=2)

            finals = "  ".join(
                f"{e.modality}: p={demo_record['models'][e.modality]['progress'][-1]:.3f}" for e in engines
            )
            print(
                f"  [{position + 1}/{len(keys)}] {key}  {num_frames} frames  "
                f"{time.time() - demo_started:5.1f}s   {finals}"
            )

    metrics: Dict[str, object] = {
        "task_name": task_name,
        "hdf5": os.path.abspath(args.hdf5),
        "camera": camera,
        "display_camera": args.display_camera,
        "quality_label": args.quality_label,
        "task": args.task,
        "clip_length": clip_length,
        "frame_stride": args.frame_stride,
        "pointmap_norm_bounds": bounds,
        "checkpoints": {e.modality: os.path.abspath(e.model_path) for e in engines},
        "wall_clock_seconds": round(time.time() - started, 1),
    }
    if args.has_success_labels:
        metrics["note"] = "--has-success-labels was set: this split is not failures-only, so no metrics were computed"
    else:
        metrics["assumption"] = (
            "every demo in this file is a failure, so every success label is 0; "
            "no ground-truth progress curve exists, so progress is reported as context only"
        )
        metrics["models"] = {
            modality: summarise_failure_split(records, args.success_threshold)
            for modality, records in collected.items()
        }

    with open(os.path.join(task_dir, "metrics.json"), "w") as stream:
        json.dump(metrics, stream, indent=2)

    if "models" in metrics:
        print(f"\n  {'model':<16}{'neg_succ_acc':>14}{'FP rate':>10}{'demos w/ alarm':>16}{'mean final prog':>17}")
        print("  " + "-" * 73)
        for modality, values in metrics["models"].items():
            frame_stats = values.get("per_frame")
            demo_stats = values.get("per_demo")
            context = values["progress_context"]
            if frame_stats is None:
                print(f"  {modality:<16}{'(no success head)':>14}")
                continue
            print(
                f"  {modality:<16}{frame_stats['negative_success_acc']:>14.4f}"
                f"{frame_stats['false_positive_rate']:>10.4f}"
                f"{demo_stats['num_demos_with_false_alarm']:>10d}/{values['num_demos']:<5d}"
                f"{context['mean_final_progress']:>17.4f}"
            )

    print(f"\nDone in {metrics['wall_clock_seconds']}s -> {task_dir}")


if __name__ == "__main__":
    main()
