#!/usr/bin/env python3
"""A scripted push-T policy: MPPI over the true MuJoCo dynamics.

At every control step, sample K action sequences of length H around the current
plan, roll all of them out in `mujoco.rollout` (batched, multithreaded C -- the
physics *is* the model, so there is nothing to learn or fit), score each by the
upstream reward, and take the exponentially-weighted average as the new plan. Execute
its first action, shift, repeat. Standard MPPI; the same family of sampling planners
model-based-diffusion itself builds on.

The pusher stays in contact and pushes continuously, unlike stroke-by-stroke search
which spent ~80% of every episode retreating and walking around the T. That is the
difference between ~1000 control steps and well under 200.

Reward deficit = |position error| + |angle error| / pi + pusher-distance penalty, and
success needs it under 0.05 -- about 9 degrees plus a few centimetres. `weak=True`
runs a deliberately under-powered planner (fewer samples, shorter horizon, execution
noise), so a fraction of episodes fail *near* success rather than trivially.
"""

import os

import mujoco
import numpy as np
from mujoco import rollout

from pusht_env import N_FRAMES, SUCCESS_REWARD, wrap_angle  # noqa: F401  (same rule as the env)

PUSHER_SLIDER_SLACK = 0.2


def batched_reward(qpos):
    """The env's `reward()` over an array of `[..., 8]` joint positions."""
    position_error = np.linalg.norm(qpos[..., 5:7] - qpos[..., 2:4], axis=-1)
    angle = qpos[..., 7] - qpos[..., 4]
    angle_error = np.abs((angle + np.pi) % (2 * np.pi) - np.pi) / np.pi
    penalty = np.maximum(np.linalg.norm(qpos[..., 0:2] - qpos[..., 2:4], axis=-1) - PUSHER_SLIDER_SLACK, 0.0)
    return 1.0 - (position_error + angle_error + penalty)


class MPPIPlanner:
    def __init__(
        self,
        env,
        rng,
        horizon=20,
        samples=64,
        sigma=0.45,
        temperature=0.03,
        terminal_weight=4.0,
        action_cost=0.0,
        max_action=1.0,
        threads=None,
    ):
        self.env = env
        self.rng = rng
        self.horizon = horizon
        self.samples = samples
        self.sigma = sigma
        self.temperature = temperature
        self.terminal_weight = terminal_weight
        self.action_cost = action_cost
        self.max_action = max_action
        threads = threads or min(16, os.cpu_count() or 1)
        self.datas = [mujoco.MjData(env.model) for _ in range(threads)]
        self.nstate = mujoco.mj_stateSize(env.model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        self.nq = env.model.nq
        self.mean = np.zeros((horizon, env.model.nu))

    def reset(self):
        self.mean[:] = 0.0

    def plan(self):
        """One MPPI update from the env's current state; returns the action to execute."""
        model = self.env.model
        initial = np.empty(self.nstate)
        mujoco.mj_getState(model, self.env.data, initial, mujoco.mjtState.mjSTATE_FULLPHYSICS)

        noise = self.rng.normal(0.0, self.sigma, size=(self.samples, self.horizon, model.nu))
        actions = np.clip(self.mean[None] + noise, -self.max_action, self.max_action)
        # One control action is held for N_FRAMES physics steps, exactly as env.step does.
        control = np.repeat(actions, N_FRAMES, axis=1)

        state, _ = rollout.rollout(model, self.datas, np.tile(initial, (self.samples, 1)), control)
        # Full-physics state is [time, qpos, qvel]; read qpos at the end of each control step.
        qpos = state[:, N_FRAMES - 1 :: N_FRAMES, 1 : 1 + self.nq]
        rewards = batched_reward(qpos)  # [K, H]

        # Cost per step is the reward deficit; the episode would end at the first success,
        # so nothing after it counts, and the final state is weighted extra.
        deficit = 1.0 - rewards
        succeeded = rewards > SUCCESS_REWARD
        alive = np.cumprod(~succeeded, axis=1, dtype=bool)
        alive = np.concatenate([np.ones((self.samples, 1), dtype=bool), alive[:, :-1]], axis=1)
        costs = (deficit * alive).sum(axis=1) + self.terminal_weight * deficit[:, -1] * alive[:, -1]
        costs -= 2.0 * succeeded.any(axis=1)  # reaching the threshold at all is worth a lot
        # Without this the planner slams the T across the table in a handful of steps --
        # physically allowed (gear 30 on 1 kg, no gravity) but a useless video.
        costs += self.action_cost * (actions**2).sum(axis=(1, 2))

        weights = np.exp(-(costs - costs.min()) / self.temperature)
        weights /= weights.sum()
        self.mean = np.einsum("k,khu->hu", weights, actions)

        action = self.mean[0].copy()
        self.mean = np.roll(self.mean, -1, axis=0)
        self.mean[-1] = self.mean[-2]
        return action


STRONG = dict(horizon=20, samples=64, sigma=0.3, action_cost=0.05, max_action=0.5)
WEAK = dict(horizon=6, samples=8, sigma=0.5, action_cost=0.05, max_action=0.5)


def run_episode(env, rng, weak=False, execution_noise=0.0, record=None):
    """MPPI until success or the horizon. `record(env)` runs after every control step."""
    planner = MPPIPlanner(env, rng, **(WEAK if weak else STRONG))
    best_reward = env.reward()
    while True:
        action = planner.plan()
        if execution_noise:
            action = np.clip(action + rng.normal(0.0, execution_noise, size=action.shape), -1.0, 1.0)
        _obs, reward, done = env.step(action)
        best_reward = max(best_reward, reward)
        if record is not None:
            record(env)
        if done:
            return best_reward
