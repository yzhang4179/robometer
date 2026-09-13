#!/usr/bin/env python3
"""Pointmap-aware policy-ranking sampler for PreciseTransformer.

Two things separate this from the stock ``ProgressPolicyRankingSampler``:

* ``PreciseTrajectoryMixin`` runs the chosen frame indices through the same
  integer carrier training uses, so RGB and pointmap prefixes stay aligned.
* ``subsample_n_frames`` is honoured. The stock sampler only knows ``frame_step``
  (a stride), so on a 32-frame cache at ``frame_step=1`` it emits 32 prefixes per
  trajectory while ``RewardAlignmentSampler`` emits ``subsample_n_frames`` of
  them. ``kendall_avg`` / ``kendall_sum`` average over however many prefixes they
  are given, so the two evaluations would be averaging over different clip sets
  and could not be quoted side by side. The endpoint arithmetic below is copied
  from ``RewardAlignmentSampler`` rather than re-derived, so the two samplers pick
  byte-identical prefixes.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from robometer.data.samplers.eval.progress_policy_ranking import ProgressPolicyRankingSampler
from robometer.data.samplers.precise_progress import PreciseTrajectoryMixin


class PrecisePolicyRankingSampler(PreciseTrajectoryMixin, ProgressPolicyRankingSampler):
    """``ProgressPolicyRankingSampler`` with aligned pointmaps and linspace prefixes."""

    def __init__(
        self,
        num_examples_per_quality_pr: int | None = None,
        num_partial_successes: int | None = None,
        frame_step: int = 1,
        use_frame_steps: bool = True,
        max_tasks: int | None = None,
        subsample_n_frames: int | None = None,
        require_pointmap: bool = True,
        **kwargs,
    ):
        # Read before super().__init__ because the parent builds every sample
        # index inside its constructor, which calls the override below.
        self.subsample_n_frames = subsample_n_frames
        super().__init__(
            num_examples_per_quality_pr=num_examples_per_quality_pr,
            num_partial_successes=num_partial_successes,
            frame_step=frame_step,
            use_frame_steps=use_frame_steps,
            max_tasks=max_tasks,
            require_pointmap=require_pointmap,
            **kwargs,
        )

    def _generate_indices_for_trajectory(self, traj_idx: int, traj: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not (self.use_frame_steps and self.subsample_n_frames):
            return super()._generate_indices_for_trajectory(traj_idx, traj)

        num_frames = traj["num_frames"]
        if self.subsample_n_frames > num_frames:
            end_indices = list(range(num_frames))
        else:
            end_indices = np.linspace(0, num_frames - 1, self.subsample_n_frames)

        return [
            {
                "traj_idx": traj_idx,
                "frame_indices": list(range(int(end_idx) + 1)),
                "num_frames": num_frames,
                "video_path": traj["frames"],
                "id": traj["id"],
                "use_frame_steps": True,
            }
            for end_idx in end_indices
        ]


__all__ = ["PrecisePolicyRankingSampler"]
