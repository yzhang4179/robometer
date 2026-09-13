#!/usr/bin/env python3
"""Exp 3 collection: scripted reach -> grasp -> lift-to-N-cm rollouts, written as mimicgen hdf5s.

No motion planner exists in any of the three repos, but mimicgen's `WaypointTrajectory`
already *is* a straight-line OSC planner (`waypoint.py:175` / `:313`), with per-waypoint
action noise built in, so the policy here is a handful of target poses rather than new
control code.

Every episode renders **three** streams from the *same* simulator state:

    agentview_image        rgb, what the model is trained and evaluated on
    agentview_pointmap     camera-frame XYZ metres, from the same depth pass
    agentview_goal_image   rgb WITH the goal-height marker, for the videos only

The marker sits in its own geom group, which is switched off before the scene is
built, so it costs the plain RGB and the pointmap exactly nothing -- measured, not
assumed (see lift_hnn.py). Successes and failures go to `demo.hdf5` and
`demo_failed.hdf5`, the same two-file layout the square data uses.

Run in the mimicgen env, which has robosuite/robomimic/mimicgen:

    MUJOCO_GL=glx ~/miniconda3/envs/mimicgen/bin/python dev_scripts/collect/collect_lift.py \
        --height-cm 5 --num-success 75 --num-failure 15 \
        --output-dir dev_scripts/out/lift_collect/h5
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import imageio
import robosuite
from robosuite.controllers import load_controller_config

import mimicgen.utils.file_utils as MG_FileUtils
import mimicgen.utils.pose_utils as PoseUtils
import mimicgen.robometer.utils.robomimic_utils as RobometerUtils
from mimicgen.datagen.waypoint import WaypointSequence, WaypointTrajectory
from mimicgen.env_interfaces.base import make_interface
from mimicgen.robometer.robosuite import EnvRobometer

from lift_hnn import ENV_NAMES, MG_Lift, TARGET_HEIGHTS_CM  # noqa: F401  (registers envs)

GRIPPER_OPEN = np.array([-1.0])
GRIPPER_CLOSE = np.array([1.0])
GOAL_IMAGE_KEY = "agentview_goal_image"
FAILURE_MODES = ("miss", "under", "drop")


class EnvLiftGoal(EnvRobometer):
    """`EnvRobometer` plus one extra RGB render with the goal marker switched on."""

    def get_observation(self, di=None):
        ret = super().get_observation(di)
        base = self.env
        if hasattr(base, "goal_marker_visible"):
            height, width = base.camera_heights[0], base.camera_widths[0]
            with base.goal_marker_visible():
                frame = base.sim.render(width=width, height=height, camera_name="agentview", depth=False)
            # MuJoCo images come out flipped in height, same as every other camera obs.
            ret[GOAL_IMAGE_KEY] = frame[::-1]
        return ret


def build_env_meta(env_name, camera_size):
    """The env metadata `create_robometer_env` wants, without a source dataset to read it from."""
    return {
        "env_name": env_name,
        "type": 1,  # EB.EnvType.ROBOSUITE_TYPE
        "env_version": robosuite.__version__,
        "env_kwargs": {
            "robots": ["Panda"],
            "controller_configs": load_controller_config(default_controller="OSC_POSE"),
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "ignore_done": True,
            "use_object_obs": True,
            "use_camera_obs": True,
            "control_freq": 20,
            "reward_shaping": False,
            "camera_names": ["agentview"],
            "camera_heights": camera_size,
            "camera_widths": camera_size,
            "camera_depths": True,
        },
    }


def add_fixed_sequence(traj, pose, gripper_action, num_steps, noise=0.0):
    """Hold one target pose for `num_steps`.

    Not `add_waypoint_sequence_for_target_pose(skip_interpolation=True)`: that branch
    is broken upstream (`waypoint.py:211` reads an undefined `gripper`), and mimicgen
    is not ours to edit. `WaypointSequence.from_poses` is the same public API one
    level down and does exactly this.
    """
    poses = np.array([pose for _ in range(num_steps)])
    gripper_actions = np.array([gripper_action for _ in range(num_steps)])
    traj.add_waypoint_sequence(WaypointSequence.from_poses(poses, gripper_actions, float(noise)))


def plan_lift(env_interface, base_env, target_height, rng, noise, failure_mode):
    """Reach above the cube, descend, close, lift to `target_height`, hold.

    Orientation is whatever the arm reset to -- already top-down for the Panda. A cube
    is 4-fold symmetric and 4.2 cm across against an 8 cm gripper opening, so even a
    45-degree yaw mismatch still fits; there is nothing to solve for.
    """
    start_pose = env_interface.get_robot_eef_pose()
    _, rotation = PoseUtils.unmake_pose(start_pose)
    cube_pos = np.array(base_env.sim.data.body_xpos[base_env.cube_body_id], dtype=np.float64)

    # Failures are meant to sit *near* success, not be trivially wrong.
    offset = np.zeros(3)
    lift_target = target_height + 0.03
    release_early = False
    if failure_mode == "miss":
        angle = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(0.030, 0.045)  # past the gripper's half-opening
        offset[:2] = [radius * np.cos(angle), radius * np.sin(angle)]
    elif failure_mode == "under":
        lift_target = target_height * rng.uniform(0.45, 0.92)
    elif failure_mode == "drop":
        release_early = True

    above = cube_pos + offset + np.array([0.0, 0.0, 0.10])
    grasp = cube_pos + offset + np.array([0.0, 0.0, 0.004])
    lifted = cube_pos + offset + np.array([0.0, 0.0, lift_target])

    traj = WaypointTrajectory()
    add_fixed_sequence(traj, start_pose, GRIPPER_OPEN, num_steps=1)
    traj.add_waypoint_sequence_for_target_pose(
        pose=PoseUtils.make_pose(above, rotation), gripper_action=GRIPPER_OPEN, num_steps=25, action_noise=noise
    )
    traj.add_waypoint_sequence_for_target_pose(
        pose=PoseUtils.make_pose(grasp, rotation), gripper_action=GRIPPER_OPEN, num_steps=20, action_noise=noise
    )
    add_fixed_sequence(traj, PoseUtils.make_pose(grasp, rotation), GRIPPER_CLOSE, num_steps=12)
    if release_early:
        # Lift most of the way, then open: the cube falls back short of the marker.
        midway = cube_pos + offset + np.array([0.0, 0.0, lift_target * 0.7])
        traj.add_waypoint_sequence_for_target_pose(
            pose=PoseUtils.make_pose(midway, rotation), gripper_action=GRIPPER_CLOSE, num_steps=22, action_noise=noise
        )
        add_fixed_sequence(traj, PoseUtils.make_pose(midway, rotation), GRIPPER_OPEN, num_steps=8)
        traj.add_waypoint_sequence_for_target_pose(
            pose=PoseUtils.make_pose(lifted, rotation), gripper_action=GRIPPER_OPEN, num_steps=20, action_noise=noise
        )
    else:
        traj.add_waypoint_sequence_for_target_pose(
            pose=PoseUtils.make_pose(lifted, rotation), gripper_action=GRIPPER_CLOSE, num_steps=30, action_noise=noise
        )
        add_fixed_sequence(traj, PoseUtils.make_pose(lifted, rotation), GRIPPER_CLOSE, num_steps=15)
    return traj


def write_video(path, observations, fps=20):
    """Goal-marker view of one episode: the eyeball check."""
    frames = [np.asarray(obs[GOAL_IMAGE_KEY], dtype=np.uint8) for obs in observations]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, macro_block_size=1)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--height-cm", type=int, required=True, choices=list(TARGET_HEIGHTS_CM))
    parser.add_argument("--output-dir", required=True, help="Gets demo.hdf5, demo_failed.hdf5, success_rate.json")
    parser.add_argument("--num-success", type=int, default=75, help="Successful episodes to keep")
    parser.add_argument("--num-failure", type=int, default=15, help="Failed episodes to keep")
    parser.add_argument("--max-attempts", type=int, default=600, help="Give up rather than loop forever")
    parser.add_argument("--iterations", type=int, default=5, help="Report a success rate this many times")
    parser.add_argument("--camera-size", type=int, default=224, help="Square render, both axes")
    parser.add_argument("--noise", type=float, default=0.02, help="Per-waypoint action noise")
    parser.add_argument("--failure-rate", type=float, default=0.25, help="Fraction of attempts deliberately perturbed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video-dir", default=None, help="Write one mp4 per kept episode here (goal-marker view)")
    parser.add_argument(
        "--video-start-index", type=int, default=0, help="First demo_i, so runs can share one --video-dir"
    )
    parser.add_argument(
        "--no-camera-randomization",
        action="store_true",
        help="Keep the camera fixed; exp 1's square data randomised it, so the default matches",
    )
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    np.random.seed(args.seed)
    env_name = ENV_NAMES[args.height_cm]
    target_height = args.height_cm / 100.0

    os.makedirs(args.output_dir, exist_ok=True)
    tmp_success = os.path.join(args.output_dir, "tmp")
    tmp_failure = os.path.join(args.output_dir, "tmp_failed")
    os.makedirs(tmp_success, exist_ok=True)
    os.makedirs(tmp_failure, exist_ok=True)

    env = RobometerUtils.create_robometer_env(
        env_meta=build_env_meta(env_name, args.camera_size),
        env_class=EnvLiftGoal,
        camera_names=["agentview"],
        camera_height=args.camera_size,
        camera_width=args.camera_size,
        use_image_obs=True,
        use_depth_obs=False,  # the pointmap consumes depth; the raw map is not stored
        use_pointmap_obs=True,
        randomize_camera_on_reset=not args.no_camera_randomization,
        random_state=np.random.RandomState(args.seed),
    )
    env_interface = make_interface(name="MG_Lift", interface_type="robosuite", env=env.base_env)

    print(f"\n=== collect {env_name} (lift {args.height_cm} cm) ===")
    print(f"  output   : {args.output_dir}")
    print(f"  targets  : {args.num_success} successes + {args.num_failure} failures")
    print(f"  render   : {args.camera_size}x{args.camera_size}  camera_randomization="
          f"{not args.no_camera_randomization}\n")

    started = time.time()
    num_success = num_failure = num_attempts = 0
    per_iteration = []
    iteration_size = max(1, args.max_attempts // max(1, args.iterations))
    iteration_seen = iteration_success = 0
    wrote_iteration_video = False
    video_index = args.video_start_index
    episodes = []  # one record per kept episode, for the summary json

    while num_attempts < args.max_attempts and (num_success < args.num_success or num_failure < args.num_failure):
        # Aim the remaining attempts at whichever tier is still short.
        want_failure = num_failure < args.num_failure and (
            num_success >= args.num_success or rng.rand() < args.failure_rate
        )
        mode = FAILURE_MODES[rng.randint(len(FAILURE_MODES))] if want_failure else None

        env.reset()
        initial_state = env.get_state()
        traj = plan_lift(env_interface, env.base_env, target_height, rng, args.noise, mode)
        result = traj.execute(env=env, env_interface=env_interface)

        num_attempts += 1
        iteration_seen += 1
        success = bool(result["success"])
        keep = (success and num_success < args.num_success) or (not success and num_failure < args.num_failure)
        if keep:
            MG_FileUtils.write_demo_to_hdf5(
                folder=tmp_success if success else tmp_failure,
                env=env,
                initial_state=initial_state,
                states=result["states"],
                observations=result["observations"],
                datagen_info=result["datagen_infos"],
                actions=result["actions"],
            )
            if not wrote_iteration_video:
                write_video(
                    os.path.join(args.output_dir, f"iter_{len(per_iteration)}", "demo.mp4"), result["observations"]
                )
                wrote_iteration_video = True
            video_path = None
            if args.video_dir:
                label = "success" if success else "fail"
                video_path = write_video(
                    os.path.join(args.video_dir, f"demo_{video_index:03d}_{label}.mp4"), result["observations"]
                )
            episodes.append(
                {
                    "id": video_index,
                    "height_cm": args.height_cm,
                    "mode": mode or "nominal",
                    "success": success,
                    "steps": int(result["actions"].shape[0]),
                    "seconds": round(result["actions"].shape[0] / 20.0, 2),
                    "lift_cm": round(env.base_env.cube_lift_height() * 100.0, 2),
                    "video": video_path,
                }
            )
            video_index += 1
        if success:
            num_success += keep
            iteration_success += 1
        else:
            num_failure += keep

        lift = env.base_env.cube_lift_height()
        print(
            f"  [{num_attempts:>4}] mode={str(mode):<6} success={str(success):<5} lift={lift * 100:5.1f}cm  "
            f"kept {num_success}/{args.num_success} succ, {num_failure}/{args.num_failure} fail"
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

    env.close()

    MG_FileUtils.merge_all_hdf5(
        folder=tmp_success, new_hdf5_path=os.path.join(args.output_dir, "demo.hdf5"), delete_folder=True
    )
    if num_failure:
        MG_FileUtils.merge_all_hdf5(
            folder=tmp_failure, new_hdf5_path=os.path.join(args.output_dir, "demo_failed.hdf5"), delete_folder=True
        )

    stats = {
        "env_name": env_name,
        "target_height_cm": args.height_cm,
        "num_success": num_success,
        "num_failure": num_failure,
        "num_attempts": num_attempts,
        "success_rate": round(num_success / max(1, num_attempts), 4),
        "per_iteration": per_iteration,
        "seed": args.seed,
        "camera_size": args.camera_size,
        "wall_clock_seconds": round(time.time() - started, 1),
        "per_mode": {
            mode: {
                "attempts": len(rows),
                "successes": sum(r["success"] for r in rows),
                "success_rate": round(sum(r["success"] for r in rows) / len(rows), 4),
                "lift_cm_mean": round(float(np.mean([r["lift_cm"] for r in rows])), 2),
            }
            for mode in ("nominal", *FAILURE_MODES)
            for rows in [[e for e in episodes if e["mode"] == mode]]
            if rows
        },
        "episodes": episodes,
    }
    with open(os.path.join(args.output_dir, "success_rate.json"), "w") as stream:
        json.dump(stats, stream, indent=2)

    print(f"\n{json.dumps(stats, indent=2)}")
    if num_success < args.num_success or num_failure < args.num_failure:
        print("\n  WARNING: hit --max-attempts before filling both tiers")


if __name__ == "__main__":
    main()
