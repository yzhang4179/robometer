#!/usr/bin/env python3
"""Smoke client for a running ``precise_eval_server``.

The server itself writes nothing to disk -- it answers with JSON, the same as the
stock ``eval_server.py`` -- so this is what proves it works end to end. It reads a
few demos out of a mimicgen test hdf5, POSTs them over the same multipart ``.npy``
wire format ``scripts/example_inference.py`` uses, and writes the returned curves
plus a plot next to each other.

The one addition to the stock payload is ``trajectory.pointmap``: the stock client
only ships ``frames``, which is fine for an RGB checkpoint and would make a
pointmap checkpoint reject the request.

    # terminal 1
    uv run python robometer/evals/precise_eval_server.py \\
        model_path=./logs/precise/precise_rgb_pointmap/final server_port=8000

    # terminal 2
    uv run python dev_scripts/eval/precise_server_client.py \\
        --hdf5 $SQUARE/demo_src_square_task_D1_validation/demo_failed.hdf5 \\
        --num-demos 2 --out-dir ./dev_scripts/out/eval_server_smoke
"""

import argparse
import io
import json
import os
from typing import Any, Dict, List

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import requests

NUMPY_FIELDS = ("frames", "pointmap")


def find_camera(demo_group: h5py.Group) -> str:
    obs = demo_group["obs"]
    cameras = sorted(
        key[: -len("_pointmap")]
        for key in obs.keys()
        if key.endswith("_pointmap") and f"{key[: -len('_pointmap')]}_image" in obs
    )
    if not cameras:
        raise KeyError(f"no camera with both _image and _pointmap; got {sorted(obs.keys())}")
    return cameras[0]


def build_multipart_payload(samples: List[Dict[str, Any]]):
    """Numpy arrays -> .npy blobs plus ``{"__numpy_file__": key}`` references."""
    files: Dict[str, Any] = {}
    data: Dict[str, str] = {}
    for index, sample in enumerate(samples):
        trajectory = sample["trajectory"]
        shell = {"sample_type": sample.get("sample_type", "progress"), "trajectory": {}}
        for key, value in trajectory.items():
            if key in NUMPY_FIELDS and isinstance(value, np.ndarray):
                file_key = f"sample_{index}_trajectory_{key}"
                buffer = io.BytesIO()
                np.save(buffer, value)
                buffer.seek(0)
                files[file_key] = (f"{file_key}.npy", buffer, "application/octet-stream")
                shell["trajectory"][key] = {"__numpy_file__": file_key}
            else:
                shell["trajectory"][key] = value
        data[f"sample_{index}"] = json.dumps(shell)
    return files, data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="POST mimicgen demos to a running precise_eval_server and save the curves.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the running server")
    parser.add_argument("--hdf5", required=True, help="Source mimicgen hdf5")
    parser.add_argument("--num-demos", type=int, default=2, help="Demos to send (one request each)")
    parser.add_argument("--frame-stride", type=int, default=4, help="Server-side scoring stride; 1 is every frame")
    parser.add_argument("--out-dir", default="./dev_scripts/out/eval_server_smoke")
    parser.add_argument("--timeout-s", type=float, default=600.0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    info = requests.get(f"{args.url.rstrip('/')}/model_info", timeout=30).json()
    print("server /model_info:")
    print(json.dumps(info, indent=2))
    needs_pointmap = "pointmap" in str(info.get("modality", ""))

    with h5py.File(args.hdf5, "r") as handle:
        keys = sorted(handle["data"].keys(), key=lambda k: (0, int(k.rsplit("_", 1)[-1])) if k.rsplit("_", 1)[-1].isdigit() else (1, k))
        keys = keys[: args.num_demos]
        camera = find_camera(handle["data"][keys[0]])
        print(f"\ncamera={camera}  demos={keys}\n")

        for key in keys:
            obs = handle["data"][key]["obs"]
            trajectory: Dict[str, Any] = {
                "id": key,
                "task": "put the square nut on the square peg",
                "frames": np.asarray(obs[f"{camera}_image"][:], dtype=np.uint8),
            }
            if needs_pointmap:
                trajectory["pointmap"] = np.asarray(obs[f"{camera}_pointmap"][:], dtype=np.float32).astype(np.float16)

            files, data = build_multipart_payload([{"sample_type": "progress", "trajectory": trajectory}])
            data["use_frame_steps"] = "true"
            data["frame_stride"] = str(args.frame_stride)
            response = requests.post(
                f"{args.url.rstrip('/')}/evaluate_batch_npy", files=files, data=data, timeout=args.timeout_s
            )
            response.raise_for_status()
            payload = response.json()

            progress = np.asarray(payload["outputs_progress"]["progress_pred"][0], dtype=np.float32)
            meta = payload["outputs_progress"]["metadata"][0]
            ends = np.asarray(meta["end_indices"], dtype=np.int64)
            success = None
            if payload.get("outputs_success"):
                success = np.asarray(payload["outputs_success"]["success_probs"][0], dtype=np.float32)

            print(
                f"  {key}: {trajectory['frames'].shape[0]} frames -> {progress.shape[0]} scored endpoints  "
                f"progress[0]={progress[0]:.3f}  progress[-1]={progress[-1]:.3f}"
                + (f"  success[-1]={success[-1]:.3f}" if success is not None else "")
            )

            figure, axes = plt.subplots(figsize=(7, 3))
            axes.plot(ends, progress, label="progress", color="#1f6feb")
            if success is not None:
                axes.plot(ends, success, label="success prob", color="#d1660f")
            axes.axhline(0.5, color="#999999", linewidth=0.8, linestyle=":")
            axes.set_ylim(-0.03, 1.03)
            axes.set_xlabel("frame")
            axes.set_title(f"{info.get('modality')} | {key}", fontsize=9)
            axes.legend(fontsize=8)
            axes.grid(alpha=0.25)
            figure.tight_layout()
            figure.savefig(os.path.join(args.out_dir, f"{key}.png"), dpi=140)
            plt.close(figure)

            with open(os.path.join(args.out_dir, f"{key}.json"), "w") as stream:
                json.dump(
                    {
                        "id": key,
                        "modality": info.get("modality"),
                        "end_indices": ends.tolist(),
                        "progress": progress.round(5).tolist(),
                        **({"success": success.round(5).tolist()} if success is not None else {}),
                    },
                    stream,
                    indent=2,
                )

    print(f"\nWrote curves and plots to {args.out_dir}")


if __name__ == "__main__":
    main()
