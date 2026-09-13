#!/usr/bin/env python3
"""Eyeball the pointmap XYZ clip bounds before freezing them into the config.

The Wan VAE expects its input in [-1, 1]. Camera-frame XYZ is in metres, so we
clamp each channel to a fixed range and affine-map it -- the same scheme Flex-pi
uses (`flex-pi/src/flexpi/models/pointmap_encoder.py:_normalize_pointmap`).

The bounds should cover the *workspace*, not the whole scene. Setting them to the
data min/max wastes most of the dynamic range on the back wall, which sits ~2.7 m
out while the table is ~0.7 m out. This script shows you where the workspace
actually is so you can choose deliberately.

Reads the mimicgen hdf5 files directly, so it works before Step 1 has run.

Example:
    uv run python dev_scripts/inspect_pointmap_bounds.py \
        --hdf5 $SQUARE/demo_src_square_task_D1/demo.hdf5 \
        --hdf5 $SQUARE/demo_src_square_task_D1_sideview_validation/demo.hdf5 \
        --bounds x=-0.6,0.6 y=-1.0,0.25 z=0.2,1.3 \
        --out-dir dev_scripts/out/pointmap_bounds
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CHANNELS = ("x", "y", "z")

# What Flex-pi ships for every one of its benchmarks, as the reference line.
# flex-pi/configs/model/flexpi.yaml:28
FLEXPI_BOUNDS: Dict[str, Tuple[float, float]] = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (0.0, 1.5),
}


def parse_bounds(items: List[str]) -> Dict[str, Tuple[float, float]]:
    """Parse ``x=-0.6,0.6 y=-1.0,0.25 z=0.2,1.3`` into a dict."""
    bounds = dict(FLEXPI_BOUNDS)
    for item in items:
        if "=" not in item:
            raise ValueError(f"--bounds entry must look like x=-0.6,0.6 (got {item!r})")
        key, value = item.split("=", 1)
        key = key.strip().lower()
        if key not in CHANNELS:
            raise ValueError(f"unknown channel {key!r}, expected one of {CHANNELS}")
        lo, hi = (float(v) for v in value.split(","))
        if not hi > lo:
            raise ValueError(f"channel {key}: hi must be > lo (got {lo}, {hi})")
        bounds[key] = (lo, hi)
    return bounds


def find_camera(demo_group: h5py.Group) -> str:
    """Return the camera prefix that carries a pointmap (agentview / sideview / ...)."""
    obs = demo_group["obs"]
    cams = sorted(k[: -len("_pointmap")] for k in obs.keys() if k.endswith("_pointmap"))
    if not cams:
        raise KeyError(f"no *_pointmap in obs; got {sorted(obs.keys())}")
    if len(cams) > 1:
        print(f"    note: multiple cameras {cams}, using {cams[0]}")
    return cams[0]


def sample_split(path: str, num_demos: int, frame_stride: int):
    """Return (rgb uint8 [N,H,W,3], xyz float32 [N,H,W,3], camera, split_name)."""
    split_name = os.path.basename(os.path.dirname(path))
    rgb_chunks, xyz_chunks = [], []
    with h5py.File(path, "r") as f:
        data = f["data"]
        demo_keys = sorted(data.keys())[:num_demos]
        camera = find_camera(data[demo_keys[0]])
        for key in demo_keys:
            obs = data[key]["obs"]
            xyz_chunks.append(obs[f"{camera}_pointmap"][::frame_stride].astype(np.float32))
            rgb_chunks.append(obs[f"{camera}_image"][::frame_stride])
    xyz = np.concatenate(xyz_chunks, axis=0)
    rgb = np.concatenate(rgb_chunks, axis=0)
    print(f"  {split_name}: {len(demo_keys)} demos, {xyz.shape[0]} frames, camera={camera}")
    return rgb, xyz, camera, split_name


def normalize(xyz: np.ndarray, bounds: Dict[str, Tuple[float, float]]) -> np.ndarray:
    """Clamp per channel to bounds, then affine-map to [-1, 1]. Flex-pi's scheme."""
    lo = np.array([bounds[c][0] for c in CHANNELS], dtype=np.float32)
    hi = np.array([bounds[c][1] for c in CHANNELS], dtype=np.float32)
    clipped = np.clip(xyz, lo, hi)
    return 2.0 * (clipped - lo) / (hi - lo) - 1.0


def to_display(normalized: np.ndarray) -> np.ndarray:
    """[-1,1] float -> uint8, so XYZ can be looked at as if it were RGB."""
    return np.clip((normalized + 1.0) * 127.5, 0, 255).astype(np.uint8)


def print_percentiles(xyz: np.ndarray, split_name: str) -> None:
    flat = xyz.reshape(-1, 3)
    print(f"\n  [{split_name}] per-channel percentiles (metres, camera frame)")
    print(f"    {'ch':>2}  {'min':>8} {'p1':>8} {'p5':>8} {'median':>8} {'p95':>8} {'p99':>8} {'max':>8}")
    for i, name in enumerate(CHANNELS):
        q = np.percentile(flat[:, i], [0, 1, 5, 50, 95, 99, 100])
        print(f"    {name:>2}  " + " ".join(f"{v:8.3f}" for v in q))


def print_bounds_report(xyz: np.ndarray, split_name: str, label: str,
                        bounds: Dict[str, Tuple[float, float]]) -> None:
    """How much gets clipped, and how much of [-1,1] the bulk of the data uses."""
    flat = xyz.reshape(-1, 3)
    print(f"\n  [{split_name}] bounds '{label}': {{" +
          ", ".join(f"{c}:[{bounds[c][0]:g},{bounds[c][1]:g}]" for c in CHANNELS) + "}")
    print(f"    {'ch':>2}  {'clip_lo':>8} {'clip_hi':>8}   {'p5..p95 maps to':>22}  {'range used':>10}")
    for i, name in enumerate(CHANNELS):
        lo, hi = bounds[name]
        col = flat[:, i]
        clip_lo = float(np.mean(col < lo)) * 100.0
        clip_hi = float(np.mean(col > hi)) * 100.0
        p5, p95 = np.percentile(col, [5, 95])
        n5 = 2.0 * (min(max(p5, lo), hi) - lo) / (hi - lo) - 1.0
        n95 = 2.0 * (min(max(p95, lo), hi) - lo) / (hi - lo) - 1.0
        used = (n95 - n5) / 2.0 * 100.0
        print(f"    {name:>2}  {clip_lo:7.2f}% {clip_hi:7.2f}%   "
              f"{f'[{n5:+.2f}, {n95:+.2f}]':>22}  {used:9.1f}%")
    any_clipped = np.any((flat < np.array([bounds[c][0] for c in CHANNELS])) |
                         (flat > np.array([bounds[c][1] for c in CHANNELS])), axis=1)
    print(f"    any channel clipped: {any_clipped.mean() * 100:.2f}% of pixels")
    print("    ('range used' = how much of [-1,1] the central 90% of that channel spans;"
          " higher is better)")


def plot_histograms(splits, bounds, out_path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    colors = plt.cm.tab10.colors
    for i, name in enumerate(CHANNELS):
        ax = axes[i]
        for s_idx, (_, xyz, _, split_name) in enumerate(splits):
            ax.hist(xyz[..., i].ravel(), bins=250, histtype="step", log=True,
                    density=True, color=colors[s_idx % 10], label=split_name)
        f_lo, f_hi = FLEXPI_BOUNDS[name]
        c_lo, c_hi = bounds[name]
        for v in (f_lo, f_hi):
            ax.axvline(v, color="gray", ls="--", lw=1.2)
        for v in (c_lo, c_hi):
            ax.axvline(v, color="crimson", ls="-", lw=1.6)
        ax.set_title(f"{name}   flex-pi [{f_lo:g},{f_hi:g}] (grey)   candidate [{c_lo:g},{c_hi:g}] (red)")
        ax.set_xlabel("metres (camera frame)")
        ax.set_ylabel("density (log)")
        if i == 0:
            ax.legend(fontsize=8)
    fig.suptitle("Pointmap XYZ distribution vs candidate clip bounds", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_preview(rgb, xyz, split_name, bounds, out_path, num_frames=4) -> None:
    """RGB next to XYZ-as-RGB under both bound sets -- the actual eyeball test."""
    idx = np.linspace(0, len(xyz) - 1, num_frames, dtype=int)
    flexpi_img = to_display(normalize(xyz[idx], FLEXPI_BOUNDS))
    cand_norm = normalize(xyz[idx], bounds)
    cand_img = to_display(cand_norm)
    cols = ["RGB", "XYZ @ flex-pi bounds", "XYZ @ candidate bounds", "Z channel @ candidate"]

    fig, axes = plt.subplots(num_frames, 4, figsize=(13, 3.3 * num_frames))
    axes = np.atleast_2d(axes)
    for r in range(num_frames):
        panels = [rgb[idx[r]], flexpi_img[r], cand_img[r], cand_img[r][..., 2]]
        for c, panel in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(panel, cmap="gray" if c == 3 else None, vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=10)
        axes[r, 0].set_ylabel(f"frame {idx[r]}", fontsize=9)
    bstr = ", ".join(f"{c}:[{bounds[c][0]:g},{bounds[c][1]:g}]" for c in CHANNELS)
    fig.suptitle(f"{split_name}   candidate bounds {{{bstr}}}", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=115, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def emit_yaml(bounds: Dict[str, Tuple[float, float]], out_path: str) -> None:
    lines = [
        "# Paste into robometer/configs/precise_transformer.yaml under model.precise",
        "pointmap_norm_bounds:",
    ]
    lines += [f"  {c}: [{bounds[c][0]:g}, {bounds[c][1]:g}]" for c in CHANNELS]
    text = "\n".join(lines) + "\n"
    with open(out_path, "w") as f:
        f.write(text)
    print(f"\n  wrote {out_path}:\n")
    print("    " + text.replace("\n", "\n    ").rstrip())
    override = ",".join(f"{c}:[{bounds[c][0]:g},{bounds[c][1]:g}]" for c in CHANNELS)
    print(f"\n  or override on the CLI without touching the yaml:\n"
          f"    'model.precise.pointmap_norm_bounds={{{override}}}'")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hdf5", action="append", required=True,
                    help="mimicgen demo.hdf5 (repeat to compare splits)")
    ap.add_argument("--num-demos", type=int, default=8)
    ap.add_argument("--frame-stride", type=int, default=10)
    ap.add_argument("--bounds", nargs="*", default=[],
                    help="candidate bounds, e.g. x=-0.6,0.6 y=-1.0,0.25 z=0.2,1.3 "
                         "(unspecified channels fall back to flex-pi's)")
    ap.add_argument("--out-dir", default="dev_scripts/out/pointmap_bounds")
    ap.add_argument("--preview-frames", type=int, default=4)
    args = ap.parse_args()

    bounds = parse_bounds(args.bounds)
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading splits...")
    splits = [sample_split(p, args.num_demos, args.frame_stride) for p in args.hdf5]

    for _, xyz, _, split_name in splits:
        print_percentiles(xyz, split_name)
        print_bounds_report(xyz, split_name, "flex-pi", FLEXPI_BOUNDS)
        print_bounds_report(xyz, split_name, "candidate", bounds)

    print("\nWriting figures...")
    plot_histograms(splits, bounds, os.path.join(args.out_dir, "hist_xyz.png"))
    for rgb, xyz, _, split_name in splits:
        plot_preview(rgb, xyz, split_name, bounds,
                     os.path.join(args.out_dir, f"preview_{split_name}.png"),
                     num_frames=args.preview_frames)

    emit_yaml(bounds, os.path.join(args.out_dir, "bounds.yaml"))

    print("\nWhat to look for:")
    print("  - hist_xyz.png: z should be bimodal -- workspace near ~0.7 m, back wall")
    print("    near ~2.7 m. Put the red lines around the workspace mode and let the")
    print("    wall saturate. Do NOT stretch the bounds to cover the far mode.")
    print("  - preview_*.png: under good bounds the table, nut and gripper show rich")
    print("    colour variation. If the workspace looks flat or washed out, the range")
    print("    is too wide.")
    print("  - the 'range used' column: aim high on every channel. Low means that")
    print("    channel is wasting the VAE's dynamic range.")
    print("  - compare splits: if the sideview split's mass sits outside bounds chosen")
    print("    from train, its pointmaps will saturate and the pointmap model will look")
    print("    worse there for that reason alone.")


if __name__ == "__main__":
    main()
