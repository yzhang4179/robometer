#!/usr/bin/env python3
"""Step 1, stage A: mimicgen hdf5 -> published RBM dataset (RGB mp4 + pointmap sidecar).

The published schema keeps robometer's eight columns untouched and adds exactly one:

    pointmap_path   str   "<subset>/pointmaps/<batch>/trajectory_XXXX.npz"

`frames` still points at an MP4 holding **RGB only**, so every existing robometer
loader keeps working on this dataset with no changes. The pointmap rides alongside
as a float16 npz in **raw camera-frame metres** -- it is never clamped or normalised
here, because the clip bounds are a runtime knob
(`model.precise.pointmap_norm_bounds`) and baking them into ~700 files on disk would
mean a full re-conversion every time they change.

RGB and pointmap are sliced with the *same* index vector, computed once by
robometer's own `downsample_frames`, so the two modalities cannot drift apart.

Layout produced under --output-dir (which is the dataset *repo* directory, i.e.
$ROBOMETER_DATASET_PATH/precise_square_rbm):

    <subset>/                             <- arrow dataset (save_to_disk)
    <subset>/batch_0000/trajectory_0000.mp4
    <subset>/pointmaps/batch_0000/trajectory_0000.npz

Example:
    export SQUARE=/home/reward/Desktop/precise_robometer/environments/mimicgen/datasets/robometer_core_datasets/square
    export ROBOMETER_DATASET_PATH=/path/to/datasets

    uv run python -m dataset_upload.data_scripts.mimicgen_precise.hdf5_to_rbm \
        --hdf5 $SQUARE/demo_src_square_task_D1/demo.hdf5 \
        --output-dir $ROBOMETER_DATASET_PATH/precise_square_rbm \
        --subset square_d1_train \
        --data-source precise_square_d1_train \
        --max-frames 32
"""

from __future__ import annotations

# `dataset_upload.helpers` imports sentence_transformers -> transformers -> torchao,
# and this venv's torchao is broken (`cannot import name 'ScalingType'`). Importing
# unsloth first gets transformers loaded before that path is reached. Needs a visible
# GPU, same as every other entry point in this repo.
try:  # noqa: SIM105
    import unsloth  # noqa: F401
except Exception:
    pass

import argparse
import multiprocessing as mp
import os
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
from tqdm import tqdm

import datasets
from datasets import Dataset

# The mp4 encoder and the subsample rule are imported, not reimplemented, so our
# videos are bit-identical to what the stock converter would have written.
from dataset_upload.helpers import create_trajectory_video_optimized, downsample_frames

# all-MiniLM-L6-v2, the model `load_sentence_transformer_model()` returns.
LANG_VECTOR_DIM = 384

# Mirrors `generate_hf_dataset.BASE_FEATURES` with use_video=true, plus our one extra
# column. Spelled out here rather than imported because that module pulls in
# tensorflow and huggingface_hub, which this converter has no use for.
FEATURES = datasets.Features(
    {
        "id": datasets.Value("string"),
        "task": datasets.Value("string"),
        "lang_vector": datasets.Sequence(datasets.Value("float32")),
        "data_source": datasets.Value("string"),
        "frames": datasets.Value("string"),  # mp4 path, RGB only
        "is_robot": datasets.Value("bool"),
        "quality_label": datasets.Value("string"),
        "partial_success": datasets.Value("float32"),
        "pointmap_path": datasets.Value("string"),  # <- the only addition
    }
)

# With --no-pointmap the column is dropped rather than left empty, so
# `preprocess_precise.py` takes its existing "no pointmap_path column" branch
# instead of trying to load a sidecar that was never written.
RGB_ONLY_FEATURES = datasets.Features({k: v for k, v in FEATURES.items() if k != "pointmap_path"})


def get_trajectory_subdir_path(trajectory_idx: int, files_per_subdir: int = 1000) -> str:
    """Same batching rule as `generate_hf_dataset.get_trajectory_subdir_path`."""
    return f"batch_{trajectory_idx // files_per_subdir:04d}"


def ensure_ffmpeg(explicit: Optional[str] = None) -> str:
    """Make sure a binary literally named `ffmpeg` is on PATH.

    `create_trajectory_video_optimized` shells out to the bare name `ffmpeg`, but
    this machine has no system ffmpeg -- only the one imageio-ffmpeg bundles inside
    the venv under a versioned filename. Rather than edit the stock helper, link a
    correctly-named shim into a cache dir and prepend it to PATH. Spawned workers
    inherit the environment, so doing this once in the parent covers them all.
    """
    import shutil

    if explicit:
        if not os.path.exists(explicit):
            raise FileNotFoundError(f"--ffmpeg {explicit} does not exist")
        binary = explicit
    else:
        found = shutil.which("ffmpeg")
        if found:
            return found
        try:
            import imageio_ffmpeg

            binary = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "No `ffmpeg` on PATH and imageio-ffmpeg is unavailable, so the mp4s cannot be encoded. "
                "Install ffmpeg (`sudo apt install ffmpeg`) or pass --ffmpeg /path/to/ffmpeg."
            ) from exc

    shim_dir = os.path.join(os.path.expanduser("~/.cache/precise_robometer"), "bin")
    os.makedirs(shim_dir, exist_ok=True)
    shim = os.path.join(shim_dir, "ffmpeg")
    if not (os.path.islink(shim) and os.readlink(shim) == binary):
        if os.path.lexists(shim):
            os.remove(shim)
        os.symlink(binary, shim)
    os.environ["PATH"] = shim_dir + os.pathsep + os.environ.get("PATH", "")
    print(f"   ffmpeg: {binary}\n           (linked as {shim} and prepended to PATH)")
    return shim


def find_camera(obs_group: h5py.Group, camera: Optional[str], want_pointmap: bool = True) -> str:
    """Pick the camera to convert.

    The sideview split names them `sideview_*`, every other split `agentview_*`,
    so this is auto-detected rather than hardcoded. With `--no-pointmap` (push-T,
    or any rgb-only source) a camera needs only an `<cam>_image`.
    """
    available = sorted(
        key[: -len("_image")] for key in obs_group.keys() if key.endswith("_image")
    )
    if want_pointmap:
        available = [cam for cam in available if f"{cam}_pointmap" in obs_group]
    if camera is not None:
        if camera not in available:
            raise ValueError(f"camera {camera!r} has no image+pointmap pair; available: {available}")
        return camera
    if len(available) != 1:
        raise ValueError(f"expected exactly one camera with an image+pointmap pair, found {available}; pass --camera")
    return available[0]


def _worker(task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert one demo: write the mp4 and the pointmap npz, return the dataset row."""
    try:
        with h5py.File(task["hdf5"], "r") as handle:
            obs = handle["data"][task["demo_key"]]["obs"]
            image_ds = obs[f"{task['camera']}_image"]

            num_source_frames = image_ds.shape[0]
            # The one index vector that both modalities are sliced with. Computed by
            # robometer's own helper so our subsampling is bit-identical to the stock
            # converter's, rather than a reimplementation that could drift.
            indices = np.asarray(downsample_frames(np.arange(num_source_frames), task["max_frames"]))

            rgb = image_ds[indices]  # (T, H, W, 3) uint8
            pointmap = None
            if task["pointmap_rel"]:
                pointmap = obs[f"{task['camera']}_pointmap"][indices]  # float32, camera-frame metres

        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)
        if pointmap is not None and rgb.shape[:3] != pointmap.shape[:3]:
            raise ValueError(f"rgb {rgb.shape} and pointmap {pointmap.shape} disagree on (T, H, W)")

        height, width = rgb.shape[1:3]
        if height % 2 or width % 2:
            raise ValueError(f"H.264 needs even dimensions, got {height}x{width}")

        os.makedirs(os.path.dirname(task["video_path"]), exist_ok=True)
        if pointmap is not None:
            os.makedirs(os.path.dirname(task["pointmap_path"]), exist_ok=True)

        # max_frames=-1 and shortest_edge_size=None: the frames are already subsampled
        # above, and resizing here would desynchronise the mp4 from the pointmap, which
        # is not resized.
        written = create_trajectory_video_optimized(
            rgb,
            task["video_path"],
            max_frames=-1,
            fps=task["fps"],
            shortest_edge_size=None,
            center_crop=False,
        )
        if written is None:
            return None

        # float16 raw metres. Sub-millimetre over the whole workspace, and finer than
        # the bf16 the VAE consumes one step later, at half the bytes of float32.
        if pointmap is not None:
            save = np.savez_compressed if task["compress"] else np.savez
            save(
                task["pointmap_path"],
                pointmap=pointmap.astype(np.float16),
                shape=np.asarray(pointmap.shape),
                num_frames=len(indices),
                frame_indices=indices.astype(np.int32),  # provenance: which hdf5 frames these are
            )

        row = {
            "id": task["id"],
            "task": task["task"],
            "lang_vector": task["lang_vector"],
            "data_source": task["data_source"],
            "frames": task["video_rel"],
            "is_robot": True,
            "quality_label": task["quality_label"],
            "partial_success": None,
        }
        if task["pointmap_rel"]:
            row["pointmap_path"] = task["pointmap_rel"]
        return row
    except Exception as exc:  # noqa: BLE001 - one bad demo must not kill the run
        print(f"❌ {task['demo_key']}: {exc}")
        return None


def build_tasks(args: argparse.Namespace, lang_vector: List[float]) -> Tuple[List[Dict[str, Any]], str]:
    with h5py.File(args.hdf5, "r") as handle:
        demo_keys = sorted(handle["data"].keys(), key=lambda k: int(k.split("_")[-1]))
        camera = find_camera(handle["data"][demo_keys[0]]["obs"], args.camera, not args.no_pointmap)
    if args.max_trajectories > 0:
        demo_keys = demo_keys[: args.max_trajectories]

    tasks = []
    for idx, demo_key in enumerate(demo_keys):
        batch = get_trajectory_subdir_path(idx)
        video_rel = os.path.join(args.subset, batch, f"trajectory_{idx:04d}.mp4")
        pointmap_rel = (
            "" if args.no_pointmap else os.path.join(args.subset, "pointmaps", batch, f"trajectory_{idx:04d}.npz")
        )
        tasks.append(
            {
                "hdf5": args.hdf5,
                "demo_key": demo_key,
                "camera": camera,
                "max_frames": args.max_frames,
                "fps": args.fps,
                "compress": not args.no_compress_pointmaps,
                # Deterministic, readable ids: re-running is idempotent and the cache
                # files stay traceable back to a source demo.
                "id": f"{args.subset}_{demo_key}",
                "task": args.task,
                "lang_vector": lang_vector,
                "data_source": args.data_source,
                "quality_label": args.quality_label,
                "video_rel": video_rel,
                "pointmap_rel": pointmap_rel,
                "video_path": os.path.join(args.output_dir, video_rel),
                "pointmap_path": os.path.join(args.output_dir, pointmap_rel) if pointmap_rel else "",
            }
        )
    return tasks, camera


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hdf5", required=True, help="Source mimicgen hdf5 (demo.hdf5 or demo_failed.hdf5)")
    parser.add_argument("--output-dir", required=True, help="Dataset repo dir, e.g. $ROBOMETER_DATASET_PATH/precise_square_rbm")
    parser.add_argument("--subset", required=True, help="Subset name, e.g. square_d1_train")
    parser.add_argument("--data-source", default=None, help="data_source column (default: precise_<subset>)")
    parser.add_argument("--task", default="put the square nut on the square peg", help="Language instruction")
    parser.add_argument("--quality-label", default="successful", choices=["successful", "failure", "suboptimal"])
    parser.add_argument("--camera", default=None, help="Camera prefix; auto-detected when omitted")
    parser.add_argument("--max-frames", type=int, default=32, help="Frames kept per trajectory (-1 keeps all)")
    parser.add_argument("--fps", type=int, default=10, help="Playback fps written into the mp4")
    parser.add_argument("--max-trajectories", type=int, default=-1, help="Cap for smoke tests")
    parser.add_argument("--num-workers", type=int, default=-1, help="-1 for cpu_count")
    parser.add_argument("--no-compress-pointmaps", action="store_true", help="np.savez instead of savez_compressed")
    parser.add_argument(
        "--no-pointmap",
        action="store_true",
        help="RGB only: do not read or write pointmaps (push-T, or any source that has none)",
    )
    parser.add_argument(
        "--lang-embedding",
        action="store_true",
        help="Fill lang_vector with a real all-MiniLM-L6-v2 embedding of --task instead of 384 zeros. "
        "Required for model.precise.use_lang_token=true (exp 3's lift-to-N-cm)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Re-encode mp4s that already exist")
    parser.add_argument("--ffmpeg", default=None, help="ffmpeg binary; auto-detected when omitted")
    args = parser.parse_args()

    if args.data_source is None:
        args.data_source = f"precise_{args.subset}"

    ensure_ffmpeg(args.ffmpeg)

    # Exp 1 and 2 are single-instruction, so the column stays a placeholder of the right
    # dtype and width. Exp 3 asks the model to tell "lift to 5 cm" from "lift to 20 cm",
    # which only works if the instruction is actually in there.
    if args.lang_embedding:
        from dataset_upload.helpers import load_sentence_transformer_model

        lang_vector = load_sentence_transformer_model().encode(args.task).astype("float32").tolist()
        if len(lang_vector) != LANG_VECTOR_DIM:
            raise ValueError(f"expected a {LANG_VECTOR_DIM}-d embedding, got {len(lang_vector)}")
        print(f"   lang_vector: all-MiniLM-L6-v2({args.task!r})")
    else:
        lang_vector = [0.0] * LANG_VECTOR_DIM

    tasks, camera = build_tasks(args, lang_vector)
    print(f"📥 {args.hdf5}")
    print(
        f"   camera={camera}  demos={len(tasks)}  max_frames={args.max_frames}  "
        f"quality={args.quality_label}  pointmaps={'no' if args.no_pointmap else 'yes'}"
    )
    print(f"   data_source={args.data_source}  subset={args.subset}")

    if args.overwrite:
        removed = 0
        for task in tasks:
            for path in (task["video_path"], task["pointmap_path"]):
                if path and os.path.exists(path):
                    os.remove(path)
                    removed += 1
        print(f"   --overwrite: removed {removed} existing files")

    num_workers = mp.cpu_count() if args.num_workers == -1 else max(1, args.num_workers)
    num_workers = min(num_workers, len(tasks))

    if num_workers == 1:
        rows = [_worker(task) for task in tqdm(tasks, desc="Converting", unit="traj")]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=num_workers) as pool:
            rows = list(tqdm(pool.imap(_worker, tasks), total=len(tasks), desc="Converting", unit="traj"))

    rows = [row for row in rows if row is not None]
    if not rows:
        raise RuntimeError("every trajectory failed to convert")
    if len(rows) < len(tasks):
        print(f"⚠️  {len(tasks) - len(rows)} trajectories failed and were skipped")

    dataset = Dataset.from_list(rows, features=RGB_ONLY_FEATURES if args.no_pointmap else FEATURES)
    dataset_dir = os.path.join(args.output_dir, args.subset)
    dataset.save_to_disk(dataset_dir)

    print(f"\n✅ {len(dataset)} trajectories -> {dataset_dir}")
    print(f"   columns: {dataset.column_names}")
    print("\nNext (stage B):")
    print(
        "  uv run python -m robometer.data.scripts.preprocess_precise \\\n"
        f"      --train_datasets '[\"{os.path.basename(os.path.normpath(args.output_dir))}\"]' \\\n"
        f"      --train_subsets '[[\"{args.subset}\"]]' \\\n"
        f"      --max_frames_for_preprocessing {args.max_frames} \\\n"
        + ("      --require_pointmap=false \\\n" if args.no_pointmap else "")
        + "      --cache_dir $ROBOMETER_PROCESSED_DATASETS_PATH"
    )


if __name__ == "__main__":
    main()
