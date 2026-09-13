#!/usr/bin/env python3
"""Pointmap-aware reward-alignment sampler for PreciseTransformer.

The stock reward-alignment sampler decides which growing prefixes to evaluate.
``PreciseTrajectoryMixin`` then runs those exact indices through the same integer
carrier used by training, so RGB and pointmap prefixes remain aligned through
uniform thinning and right-padding.
"""

from __future__ import annotations

from robometer.data.samplers.eval.reward_alignment import RewardAlignmentSampler
from robometer.data.samplers.precise_progress import PreciseTrajectoryMixin


class PreciseRewardAlignmentSampler(PreciseTrajectoryMixin, RewardAlignmentSampler):
    """``RewardAlignmentSampler`` with aligned RGB and pointmap materialisation."""

    def __init__(
        self,
        max_trajectories: int | None = None,
        frame_step: int = 1,
        use_frame_steps: bool = True,
        subsample_n_frames: int | None = None,
        require_pointmap: bool = True,
        **kwargs,
    ):
        super().__init__(
            max_trajectories=max_trajectories,
            frame_step=frame_step,
            use_frame_steps=use_frame_steps,
            subsample_n_frames=subsample_n_frames,
            require_pointmap=require_pointmap,
            **kwargs,
        )


__all__ = ["PreciseRewardAlignmentSampler"]
