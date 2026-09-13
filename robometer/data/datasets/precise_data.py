#!/usr/bin/env python3
"""Precise training and reward-alignment datasets.

`RBMDataset` builds `ProgressSampler` by name, so there is no config knob to point
it elsewhere. Rebuilding just that one sampler after `super().__init__` is a
smaller and more honest change than threading a sampler class through the base
class -- everything else (index building, success cutoffs, strategy ratios,
checkpointable random state) is inherited untouched.

The stock ``CustomEvalDataset`` similarly hardcodes its sampler table *and* the
``filter_quality_labels=["successful"]`` that goes with reward alignment.
``PreciseCustomEvalDataset`` installs the pointmap-aware sampler directly and
picks the quality filter per evaluation type: reward alignment needs a
ground-truth progress curve so it keeps successes only, while policy ranking
exists precisely to rank successes against failures and must keep both.
"""

from __future__ import annotations

from robometer.data.datasets.base import BaseDataset
from robometer.data.datasets.rbm_data import RBMDataset
from robometer.data.samplers.eval.precise_policy_ranking import PrecisePolicyRankingSampler
from robometer.data.samplers.eval.precise_reward_alignment import PreciseRewardAlignmentSampler
from robometer.data.samplers.precise_progress import (
    POINTMAP_STREAM,
    PreciseProgressSampler,
    precise_streams,
)
from robometer.utils.logger import get_logger

logger = get_logger()


def _require_pointmap_for_modality(modality: str, override: bool | None = None) -> bool:
    """Validate ``modality`` and derive whether its samples must contain pointmaps."""
    needs_pointmap = POINTMAP_STREAM in precise_streams(modality)
    require_pointmap = needs_pointmap if override is None else bool(override)
    if needs_pointmap and not require_pointmap:
        raise ValueError(
            f"modality={modality!r} requires pointmaps; require_pointmap cannot be false."
        )
    return require_pointmap


class PreciseRBMDataset(RBMDataset):
    """Progress-only dataset whose samples carry `metadata["pointmap"]`."""

    def __init__(
        self,
        config,
        is_evaluation: bool = False,
        max_samples=None,
        sampler_kwargs=None,
        modality: str = "rgb_pointmap",
        require_pointmap: bool | None = None,
        **kwargs,
    ):
        self.modality = modality
        require_pointmap = _require_pointmap_for_modality(modality, require_pointmap)
        super().__init__(
            config,
            is_evaluation=is_evaluation,
            max_samples=max_samples,
            sampler_kwargs=sampler_kwargs,
            **kwargs,
        )

        if config.sample_type_ratio[0] > 0:
            raise ValueError(
                "PreciseRBMDataset is progress-only; set data.sample_type_ratio=[0, 1, 0]. "
                f"Got {config.sample_type_ratio}."
            )
        if self.progress_sampler is None:
            raise ValueError("data.sample_type_ratio[1] must be > 0 -- there is nothing else to sample")

        self.progress_sampler = PreciseProgressSampler(
            is_evaluation=is_evaluation,
            require_pointmap=require_pointmap,
            config=config,
            dataset=self.dataset,
            combined_indices=self._combined_indices,
            dataset_success_cutoff_map=self.dataset_success_cutoff_map,
            verbose=False,
            **(sampler_kwargs or {}),
        )
        logger.info("[PreciseRBMDataset] progress sampler replaced with PreciseProgressSampler")


# Reward alignment is defined on successful trajectories only -- a failure has no
# honest progress target. Policy ranking needs at least two quality tiers in the
# same task or ``ProgressPolicyRankingSampler`` drops the task entirely, so it
# takes no filter at all.
_EVAL_SAMPLERS = {
    "reward_alignment": (PreciseRewardAlignmentSampler, ["successful"]),
    "policy_ranking": (PrecisePolicyRankingSampler, None),
}


class PreciseCustomEvalDataset(BaseDataset):
    """Reward-alignment / policy-ranking dataset with pointmap-aware samplers."""

    def __init__(
        self,
        sampler_type: str,
        config,
        verbose: bool = True,
        sampler_kwargs: dict | None = None,
        modality: str = "rgb_pointmap",
        require_pointmap: bool | None = None,
    ):
        if sampler_type not in _EVAL_SAMPLERS:
            raise ValueError(
                f"PreciseCustomEvalDataset supports {sorted(_EVAL_SAMPLERS)}; got {sampler_type!r}."
            )
        sampler_cls, filter_quality_labels = _EVAL_SAMPLERS[sampler_type]

        self.modality = modality
        require_pointmap = _require_pointmap_for_modality(modality, require_pointmap)

        super().__init__(
            config=config,
            is_evaluation=True,
            filter_quality_labels=filter_quality_labels,
        )

        sampler_options = dict(sampler_kwargs or {})
        if "require_pointmap" in sampler_options:
            raise ValueError(
                "Pass require_pointmap to PreciseCustomEvalDataset, not inside sampler_kwargs."
            )
        self.sampler = sampler_cls(
            config=config,
            dataset=self.dataset,
            combined_indices=self._combined_indices,
            dataset_success_cutoff_map=self.dataset_success_cutoff_map,
            verbose=verbose,
            require_pointmap=require_pointmap,
            **sampler_options,
        )

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx):
        return self.sampler[idx]


__all__ = ["PreciseRBMDataset", "PreciseCustomEvalDataset"]
