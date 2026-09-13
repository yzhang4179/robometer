#!/usr/bin/env python3
"""Exp 2 collection: scripted push-T episodes, rendered top-down, written as mimicgen-style hdf5.

Each episode runs *without* rendering (MPPI alone steps the physics thousands of
times per decision), records the joint state at every **physics** step (100 Hz), and
is rendered only if kept -- by replaying the recorded states through `mj_forward` and
the renderer. The picture and the dynamics come from the same XML and the same state,
so they cannot disagree. MPPI averages ~55 control steps (0.05 s each) per success;
`--record-stride 4` stores a frame every 0.04 s, so successes average ~65 frames and
the 64-frame cache is a light downsample rather than a padded stretch.

The hdf5 layout is the subset of mimicgen's that the rest of the pipeline reads:

    data/demo_N/obs/topdown_image   (T, S, S, 3) uint8
    data/demo_N/actions             (T, 2)
    data/demo_N/states              (T, 16)   q and qd, the upstream observation
    data/demo_N/rewards             (T,)      the upstream reward, for later analysis

so `hdf5_to_rbm.py --no-pointmap` and `run_precise_eval.py` consume it unchanged.
Successes go to `demo.hdf5`, failures to `demo_failed.hdf5`, exactly like the square data.

    MUJOCO_GL=egl uv run python dev_scripts/collect/collect_pusht.py \
        --num-success 500 --num-failure 80 --output-dir dev_scripts/out/pusht_collect
"""

import argparse
import json
import os
import sys
import time

import h5py
import imageio
import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pusht_env import PushTEnv  # noqa: E402
from pusht_policy import run_episode  # noqa: E402

IMAGE_KEY = "topdown_image"


class EpisodeRecorder:
    """Collects `(qpos, qvel, ctrl, reward)` after every physics step."""

    def __init__(self):
        self.qpos, self.qvel, self.ctrl, self.rewards = [], [], [], []

    def __call__(self, env):
        self.qpos.append(env.data.qpos.copy())
        self.qvel.append(env.data.qvel.copy())
        self.ctrl.append(env.data.ctrl.copy())
        self.rewards.append(env.reward())


def render_episode(env, qpos_history, stride):
    """Replay recorded states through the renderer; the last frame is always kept."""
    indices = list(range(0, len(qpos_history), stride))
    if indices[-1] != len(qpos_history) - 1:
        indices.append(len(qpos_history) - 1)
    frames = []
    for index in indices:
        env.data.qpos[:] = qpos_history[index]
        mujoco.mj_forward(env.model, env.data)
        frames.append(env.render())
    return np.stack(frames), np.asarray(indices)


def write_episode_video(directory, index, success, frames, fps):
    """One mp4 per kept episode -- the qualitative view of what was collected."""
    os.makedirs(directory, exist_ok=True)
    label = "success" if success else "fail"
    path = os.path.join(directory, f"demo_{index:03d}_{label}.mp4")
    imageio.mimsave(path, list(frames), fps=fps, macro_block_size=1)
    return path


def write_demo(group, index, frames, recorder, kept):
    demo = group.create_group(f"demo_{index}")
    demo.attrs["num_samples"] = len(kept)
    demo.create_dataset(f"obs/{IMAGE_KEY}", data=frames, compression="gzip", compression_opts=4)
    states = np.concatenate([np.asarray(recorder.qpos)[kept, :8], np.asarray(recorder.qvel)[kept, :8]], axis=1)
    demo.create_dataset("states", data=states.astype(np.float32))
    demo.create_dataset("actions", data=np.asarray(recorder.ctrl)[kept].astype(np.float32))
    demo.create_dataset("rewards", data=np.asarray(recorder.rewards)[kept].astype(np.float32))
    demo.create_dataset("source_step", data=kept.astype(np.int32))  # provenance: control step per frame


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True, help="Gets demo.hdf5, demo_failed.hdf5, success_rate.json")
    parser.add_argument("--num-success", type=int, default=500)
    parser.add_argument("--num-failure", type=int, default=80)
    parser.add_argument("--max-attempts", type=int, default=2000)
    parser.add_argument("--iterations", type=int, default=5, help="Report a success rate this many times")
    parser.add_argument("--horizon", type=int, default=300, help="Control steps before an episode is a failure")
    parser.add_argument("--weak-horizon", type=int, default=120, help="Same, for deliberately weak episodes")
    parser.add_argument("--render-size", type=int, default=224, help="Square frame; the latent grid needs square")
    parser.add_argument(
        "--record-stride", type=int, default=4, help="Store every Nth physics step (4 -> 25 fps; ~65 frames per success on average)"
    )
    parser.add_argument(
        "--weak-rate",
        type=float,
        default=0.3,
        help="Fraction of episodes run by the under-powered planner with execution noise, so failures are near misses",
    )
    parser.add_argument("--execution-noise", type=float, default=0.5, help="Action noise in weak episodes")
    parser.add_argument("--video-dir", default=None, help="Write one mp4 per kept episode here")
    parser.add_argument(
        "--video-start-index", type=int, default=0, help="First demo_i, so runs can share one --video-dir"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.RandomState(args.seed)
    env = PushTEnv(seed=args.seed, render_size=args.render_size, horizon=args.horizon)

    success_file = h5py.File(os.path.join(args.output_dir, "demo.hdf5"), "w")
    failure_file = h5py.File(os.path.join(args.output_dir, "demo_failed.hdf5"), "w")
    success_group = success_file.create_group("data")
    failure_group = failure_file.create_group("data")
    for group in (success_group, failure_group):
        group.attrs["env_args"] = json.dumps(
            {"env_name": "PushT", "type": "mujoco", "env_kwargs": {"horizon": args.horizon, "render_size": args.render_size}}
        )

    print(f"\n=== collect push-T ===")
    print(f"  output   : {args.output_dir}")
    print(f"  targets  : {args.num_success} successes + {args.num_failure} failures  (horizon {args.horizon})")
    print(f"  render   : {args.render_size}x{args.render_size} every {args.record_stride} physics step(s)\n")

    started = time.time()
    num_success = num_failure = num_attempts = 0
    per_iteration = []
    iteration_size = max(1, args.max_attempts // max(1, args.iterations))
    iteration_seen = iteration_success = 0
    wrote_iteration_video = False
    video_index = args.video_start_index
    lengths = []
    episodes = []  # one record per kept episode, for the summary json

    while num_attempts < args.max_attempts and (num_success < args.num_success or num_failure < args.num_failure):
        # Once the success tier is full, every remaining attempt is a weak one.
        weak = num_success >= args.num_success or rng.rand() < args.weak_rate
        env.horizon = args.weak_horizon if weak else args.horizon

        env.reset()
        recorder = EpisodeRecorder()
        recorder(env)  # frame 0 is the reset state
        env.on_substep = recorder
        best = run_episode(env, rng, weak=weak, execution_noise=args.execution_noise if weak else 0.0)
        env.on_substep = None
        success = bool(env.is_success())
        num_attempts += 1
        iteration_seen += 1
        iteration_success += success

        keep = (success and num_success < args.num_success) or (not success and num_failure < args.num_failure)
        if keep:
            frames, kept = render_episode(env, recorder.qpos, args.record_stride)
            if success:
                write_demo(success_group, num_success, frames, recorder, kept)
                num_success += 1
                lengths.append(len(kept))
            else:
                write_demo(failure_group, num_failure, frames, recorder, kept)
                num_failure += 1
            video_path = None
            if args.video_dir:
                video_path = write_episode_video(
                    args.video_dir, video_index, success, frames, int(100 / args.record_stride)
                )
            episodes.append(
                {
                    "id": video_index,
                    "planner": "weak" if weak else "strong",
                    "success": success,
                    "control_steps": int(env.steps),
                    "seconds": round(env.steps * 0.05, 2),
                    "frames": int(len(kept)),
                    "best_reward": round(float(best), 4),
                    "final_reward": round(float(env.reward()), 4),
                    "video": video_path,
                }
            )
            video_index += 1
            if not wrote_iteration_video:
                path = os.path.join(args.output_dir, f"iter_{len(per_iteration)}", "demo.mp4")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                imageio.mimsave(path, list(frames), fps=int(100 / args.record_stride), macro_block_size=1)
                wrote_iteration_video = True

        print(
            f"  [{num_attempts:>4}] {'weak  ' if weak else 'strong'} success={str(success):<5} "
            f"best_r={best:+.3f} steps={env.steps:>4}  kept {num_success}/{args.num_success} succ, "
            f"{num_failure}/{args.num_failure} fail"
        )

        if iteration_seen >= iteration_size:
            per_iteration.append(
                {"iteration": len(per_iteration), "attempts": iteration_seen, "successes": iteration_success,
                 "success_rate": round(iteration_success / iteration_seen, 4)}
            )
            iteration_seen = iteration_success = 0
            wrote_iteration_video = False

    if iteration_seen:
        per_iteration.append(
            {"iteration": len(per_iteration), "attempts": iteration_seen, "successes": iteration_success,
             "success_rate": round(iteration_success / max(1, iteration_seen), 4)}
        )

    success_group.attrs["total"] = num_success
    failure_group.attrs["total"] = num_failure
    success_file.close()
    failure_file.close()
    env.close()

    stats = {
        "env_name": "PushT",
        "num_success": num_success,
        "num_failure": num_failure,
        "num_attempts": num_attempts,
        "success_rate": round(num_success / max(1, num_attempts), 4),
        "per_iteration": per_iteration,
        "frames_per_success_episode": {
            "mean": float(np.mean(lengths)) if lengths else None,
            "min": int(np.min(lengths)) if lengths else None,
            "max": int(np.max(lengths)) if lengths else None,
        },
        "per_planner": {
            planner: {
                "attempts": len(rows),
                "successes": sum(r["success"] for r in rows),
                "success_rate": round(sum(r["success"] for r in rows) / len(rows), 4),
                "control_steps_mean": round(float(np.mean([r["control_steps"] for r in rows])), 1),
            }
            for planner in ("strong", "weak")
            for rows in [[e for e in episodes if e["planner"] == planner]]
            if rows
        },
        "episodes": episodes,
        "horizon": args.horizon,
        "weak_horizon": args.weak_horizon,
        "weak_rate": args.weak_rate,
        "record_stride": args.record_stride,
        "frames_per_second": 100 / args.record_stride,
        "render_size": args.render_size,
        "seed": args.seed,
        "wall_clock_seconds": round(time.time() - started, 1),
    }
    with open(os.path.join(args.output_dir, "success_rate.json"), "w") as stream:
        json.dump(stats, stream, indent=2)
    print(f"\n{json.dumps(stats, indent=2)}")
    if num_success < args.num_success or num_failure < args.num_failure:
        print("\n  WARNING: hit --max-attempts before filling both tiers")


if __name__ == "__main__":
    main()
