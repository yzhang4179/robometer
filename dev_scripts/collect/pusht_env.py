#!/usr/bin/env python3
"""Exp 2 environment: push-T in plain MuJoCo, with rendering.

The upstream task (LeCAR-Lab/model-based-diffusion, `mbd/envs/pushT.py`) is a Brax
`PipelineEnv` over `pushT.xml` with `backend="generalized"` and `n_frames=5`. The XML
is ordinary MuJoCo and the Brax wrapper adds only that frame-skip and the
reset/reward/done functions, all three of which are reproduced here exactly.

Stepping the same XML with `mj_step` therefore gives the same task without jax, brax,
a repo clone or a third conda environment -- and it renders, which the Brax version
cannot do at all (it offers only an interactive `html.render()` page). Brax's
generalized integrator is not bit-identical to MuJoCo's, but nothing is being
transferred between the two: this study generates its own data from its own rollouts,
so what has to match is the task definition, not one particular integrator's rounding.

State layout, straight from the XML's joint order:

    q[0:2]  pusher x, y          qd[0:2]
    q[2:4]  slider (the T) x, y  qd[2:4]
    q[4]    slider z rotation    qd[4]
    q[5:7]  goal x, y            qd[5:7]
    q[7]    goal z rotation      qd[7]
"""

import os

import mujoco
import numpy as np

XML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "pushT.xml")

N_FRAMES = 5  # brax PipelineEnv(n_frames=5): one action is held for five 0.01 s steps
SUCCESS_REWARD = 0.95
PUSHER_SLIDER_SLACK = 0.2  # the reward stops penalising pusher distance inside this


def wrap_angle(angle: float) -> float:
    """To (-pi, pi]. The upstream reward takes a raw difference, which is only correct
    while both angles sit in the same branch -- reset puts the goal near +pi and the
    slider at 0, so wrapping is what keeps a +179-degree and a -179-degree error from
    reading as nearly 2 pi apart."""
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


class PushTEnv:
    """Plain-MuJoCo push-T with the upstream reset, reward and success rule."""

    def __init__(self, seed: int = 0, render_size: int = 224, horizon: int = 120):
        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=render_size, width=render_size)
        self.camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "topdown")
        if self.camera_id < 0:
            raise RuntimeError(f"{XML_PATH} has no camera named 'topdown'")
        self.rng = np.random.RandomState(seed)
        self.render_size = render_size
        self.horizon = horizon
        self.steps = 0
        # Called after every *physics* step (100 Hz). Episodes are only a few dozen
        # control steps long, so the collector records frames at this rate instead.
        self.on_substep = None

    # ------------------------------------------------------------------ state

    @property
    def pusher_pos(self):
        return self.data.qpos[0:2].copy()

    @property
    def pusher_vel(self):
        return self.data.qvel[0:2].copy()

    @property
    def slider_pos(self):
        return self.data.qpos[2:4].copy()

    @property
    def slider_angle(self):
        return float(self.data.qpos[4])

    @property
    def goal_pos(self):
        return self.data.qpos[5:7].copy()

    @property
    def goal_angle(self):
        return float(self.data.qpos[7])

    def get_obs(self):
        """The upstream 16-d observation: `concat(q, qd)`."""
        return np.concatenate([self.data.qpos[:8], self.data.qvel[:8]]).astype(np.float32)

    # ----------------------------------------------------------------- reward

    def reward(self) -> float:
        """`1 - (position error + normalised angle error + pusher-distance penalty)`."""
        position_error = float(np.linalg.norm(self.goal_pos - self.slider_pos))
        angle_error = abs(wrap_angle(self.goal_angle - self.slider_angle)) / np.pi
        pusher_penalty = max(float(np.linalg.norm(self.pusher_pos - self.slider_pos)) - PUSHER_SLIDER_SLACK, 0.0)
        return 1.0 - (position_error + angle_error + pusher_penalty)

    def is_success(self) -> bool:
        return self.reward() > SUCCESS_REWARD

    # ------------------------------------------------------------ step/reset

    def reset(self, seed=None):
        """Upstream reset: pusher parked, slider at the origin, goal near (-0.4, 0.4, pi)."""
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        mujoco.mj_resetData(self.model, self.data)
        qpos = np.zeros(self.model.nq)
        qpos[0:2] = [0.1, -0.15]
        qpos[5:8] = self.rng.uniform(-1.0, 1.0, size=3) * np.array([0.2, 0.2, np.pi / 4]) + np.array(
            [-0.4, 0.4, np.pi]
        )
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        return self.get_obs()

    def step(self, action):
        """One control step: the action is held for `N_FRAMES` physics steps.

        `done` on success is the upstream rule (`_get_done` is `reward > 0.95`), and it
        is also what the progress labels need: `compute_progress_from_segment` treats a
        trajectory's last frame as completion, so a successful episode has to *stop* at
        the moment it succeeds rather than drift afterwards.
        """
        self.data.ctrl[:] = np.clip(action, -1.0, 1.0)
        for _ in range(N_FRAMES):
            mujoco.mj_step(self.model, self.data)
            if self.on_substep is not None:
                self.on_substep(self)
        self.steps += 1
        reward = self.reward()
        return self.get_obs(), reward, (reward > SUCCESS_REWARD) or (self.steps >= self.horizon)

    # ----------------------------------------------------------------- render

    def render(self) -> np.ndarray:
        """`[render_size, render_size, 3]` uint8 from the top-down camera.

        Square by construction: `grid_h == grid_w` is hardcoded in the model's latent
        layout, so a non-square frame could never be tokenised.
        """
        self.renderer.update_scene(self.data, camera=self.camera_id)
        return self.renderer.render()

    def close(self):
        self.renderer.close()
