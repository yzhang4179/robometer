#!/usr/bin/env python3
"""Render one episode's pointmap as a 3D point-cloud video, before vs after clamping.

Clamping does not *drop* out-of-range points, it *squashes* them onto the faces of
the bounds box. This shows you exactly what that costs: whether the workspace
survives intact and how much geometry gets flattened onto the far wall of the box.

Three panels per frame:
  1. RAW, full scene extent   -- the real cloud, bounds box drawn, out-of-bounds
                                 points tinted red so you can see what is at risk
  2. RAW, box extent          -- the workspace as it really is
  3. CLAMPED, box extent      -- what the Wan VAE will actually see

Panels 2 and 3 are the comparison that matters. If they look the same apart from a
thin shell pasted on the box faces, the bounds are good. If the workspace itself is
deformed, the bounds are too tight.

Points are coloured by their RGB pixel, so the table / nut / arm stay recognisable.
Coordinates are camera frame (x right, y down, z forward); the plot shows
(x, z, -y) so the scene is right-side-up.

Example:
    uv run python dev_scripts/render_pointcloud_video.py \
        --hdf5 $SQUARE/demo_src_square_task_D1/demo.hdf5 \
        --demo-index 0 \
        --bounds x=-0.6,0.6 y=-1,0.25 z=0.2,1.3 \
        --out dev_scripts/out/pointmap_bounds/cloud_demo0.mp4
"""

from __future__ import annotations

import argparse
import os

import h5py
import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Share the bound parsing / normalization with the histogram script so the two can
# never disagree about what "clamped" means.
from inspect_pointmap_bounds import CHANNELS, find_camera, parse_bounds


def box_edges(bounds):
    """12 wireframe edges of the bounds box, in plot space (x, z, -y)."""
    xlo, xhi = bounds["x"]
    ylo, yhi = bounds["y"]
    zlo, zhi = bounds["z"]
    corners = np.array([[x, y, z] for x in (xlo, xhi) for y in (ylo, yhi) for z in (zlo, zhi)])
    edges = []
    for i in range(len(corners)):
        for j in range(i + 1, len(corners)):
            # An edge joins corners differing in exactly one coordinate.
            if np.sum(~np.isclose(corners[i], corners[j])) == 1:
                a, b = corners[i], corners[j]
                edges.append(([a[0], b[0]], [a[2], b[2]], [-a[1], -b[1]]))
    return edges


def draw_box(ax, bounds, color="crimson", lw=1.0):
    for xs, zs, ys in box_edges(bounds):
        ax.plot(xs, zs, ys, color=color, lw=lw, alpha=0.8)


def scatter(ax, xyz, rgb, size):
    """xyz [N,3] camera frame -> plot as (x, z, -y)."""
    ax.scatter(xyz[:, 0], xyz[:, 2], -xyz[:, 1], c=rgb / 255.0, s=size,
               marker=".", linewidths=0, depthshade=False)


def set_view(ax, bounds, elev, azim, full_extent=None):
    if full_extent is None:
        ax.set_xlim(bounds["x"])
        ax.set_ylim(bounds["z"])
        ax.set_zlim(-bounds["y"][1], -bounds["y"][0])
    else:
        (xlo, xhi), (ylo, yhi), (zlo, zhi) = full_extent
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(zlo, zhi)
        ax.set_zlim(-yhi, -ylo)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x (m)", fontsize=8, labelpad=-6)
    ax.set_ylabel("z (m, depth)", fontsize=8, labelpad=-6)
    ax.set_zlabel("-y (m, up)", fontsize=8, labelpad=-6)
    ax.tick_params(labelsize=6, pad=-2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hdf5", required=True, help="mimicgen demo.hdf5")
    ap.add_argument("--demo-index", type=int, default=0, help="which demo in the file")
    ap.add_argument("--bounds", nargs="*", default=[],
                    help="x=-0.6,0.6 y=-1,0.25 z=0.2,1.3 (defaults to flex-pi's)")
    ap.add_argument("--out", default="dev_scripts/out/pointmap_bounds/cloud.mp4")
    ap.add_argument("--pixel-stride", type=int, default=1,
                    help="subsample pixels; 4 -> 64x64 = 4096 points/frame")
    ap.add_argument("--frame-stride", type=int, default=1, help="subsample time")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--point-size", type=float, default=2.0)
    # Oblique 3/4 view: shows the depth extent and the box. Use --elev 5 --azim -90
    # for something close to the camera's own viewpoint.
    ap.add_argument("--elev", type=float, default=18.0, help="matplotlib 3d elevation")
    ap.add_argument("--azim", type=float, default=-72.0, help="matplotlib 3d azimuth")
    ap.add_argument("--dpi", type=int, default=100)
    args = ap.parse_args()

    bounds = parse_bounds(args.bounds)
    lo = np.array([bounds[c][0] for c in CHANNELS], dtype=np.float32)
    hi = np.array([bounds[c][1] for c in CHANNELS], dtype=np.float32)

    with h5py.File(args.hdf5, "r") as f:
        data = f["data"]
        demo_key = sorted(data.keys())[args.demo_index]
        camera = find_camera(data[demo_key])
        obs = data[demo_key]["obs"]
        sl = slice(None, None, args.frame_stride)
        xyz_all = obs[f"{camera}_pointmap"][sl].astype(np.float32)
        rgb_all = obs[f"{camera}_image"][sl]

    if args.max_frames is not None:
        xyz_all = xyz_all[: args.max_frames]
        rgb_all = rgb_all[: args.max_frames]

    s = args.pixel_stride
    xyz_all = xyz_all[:, ::s, ::s, :].reshape(len(xyz_all), -1, 3)
    rgb_all = rgb_all[:, ::s, ::s, :].reshape(len(rgb_all), -1, 3)

    print(f"demo={demo_key}  camera={camera}  frames={len(xyz_all)}  "
          f"points/frame={xyz_all.shape[1]}")
    print("bounds: {" + ", ".join(f"{c}:[{bounds[c][0]:g},{bounds[c][1]:g}]"
                                  for c in CHANNELS) + "}")

    flat = xyz_all.reshape(-1, 3)
    oob_all = np.any((flat < lo) | (flat > hi), axis=1)
    print(f"episode-wide: {oob_all.mean() * 100:.2f}% of points fall outside the box")
    for i, c in enumerate(CHANNELS):
        below = np.mean(flat[:, i] < lo[i]) * 100
        above = np.mean(flat[:, i] > hi[i]) * 100
        print(f"  {c}: {below:5.2f}% below {lo[i]:g},  {above:5.2f}% above {hi[i]:g}")

    # Fixed scene extent for panel 1, so the camera does not jitter between frames.
    pad = 0.05
    full_extent = [(float(flat[:, i].min()) - pad, float(flat[:, i].max()) + pad)
                   for i in range(3)]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig = plt.figure(figsize=(15, 5.2), dpi=args.dpi)
    axes = [fig.add_subplot(1, 3, i + 1, projection="3d") for i in range(3)]
    titles = ["RAW · full scene (red = out of bounds)", "RAW · box extent",
              "CLAMPED · box extent  ← what the VAE sees"]

    writer = imageio.get_writer(args.out, fps=args.fps)
    try:
        for t in tqdm(range(len(xyz_all)), desc="rendering", unit="frame"):
            xyz = xyz_all[t]
            rgb = rgb_all[t].astype(np.float32)
            oob = np.any((xyz < lo) | (xyz > hi), axis=1)
            clamped = np.clip(xyz, lo, hi)

            # Panel 1 tints out-of-bounds points red so they are unmistakable.
            rgb_tinted = rgb.copy()
            rgb_tinted[oob] = 0.35 * rgb_tinted[oob] + 0.65 * np.array([255.0, 40.0, 40.0])

            for ax in axes:
                ax.cla()
            scatter(axes[0], xyz, rgb_tinted, args.point_size)
            scatter(axes[1], xyz, rgb, args.point_size)
            scatter(axes[2], clamped, rgb, args.point_size)

            set_view(axes[0], bounds, args.elev, args.azim, full_extent=full_extent)
            set_view(axes[1], bounds, args.elev, args.azim)
            set_view(axes[2], bounds, args.elev, args.azim)
            for ax in axes:
                draw_box(ax, bounds)
            for ax, title in zip(axes, titles):
                ax.set_title(title, fontsize=10)

            fig.suptitle(
                f"{demo_key} · frame {t + 1}/{len(xyz_all)} · "
                f"{oob.mean() * 100:.1f}% of points clamped", fontsize=11)
            fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.02, wspace=0.06)

            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            h, w = frame.shape[:2]
            writer.append_data(frame[: h - h % 2, : w - w % 2])  # libx264 wants even dims
    finally:
        writer.close()
        plt.close(fig)

    print(f"\nwrote {args.out}")
    print("\nWhat to look for:")
    print("  - panel 2 vs panel 3: if they differ only by a thin shell pasted on the")
    print("    box faces, the bounds are good. If the workspace itself is deformed,")
    print("    the bounds are too tight.")
    print("  - panel 1: red should be the back wall / floor only. Red on the table,")
    print("    the nut, or the gripper means real geometry is being destroyed.")
    print("  - if a whole box face turns into a flat sheet of points, that is the far")
    print("    background collapsing -- expected, and harmless.")


if __name__ == "__main__":
    main()
