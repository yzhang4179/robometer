#!/usr/bin/env python3
"""Hydra schemas for the Step 4 inference entry points.

Two stock schemas are closed against Precise and are subclassed here rather than
edited:

* ``EvalServerConfig`` has no knob for the prefix clip length or for a runtime
  pointmap-bounds override, and both are things Step 4 must set per split.
* ``BaselineEvalConfig.__post_init__`` dispatches ``model_config`` on a closed
  list of ``reward_model`` names and raises ``ValueError`` for anything else, so
  ``reward_model=precise`` cannot reach it.

Both subclasses are registered under **new** ConfigStore names and are driven by
**new** yaml files, so nothing reparents the stock ``baseline_eval_config`` /
``eval_config_server`` compositions that ``run_baseline_eval.py`` and
``eval_server.py`` still use.
"""

# No `from __future__ import annotations` -- Hydra/OmegaConf resolves these
# dataclass field types at runtime, and stringised annotations break that.
# Same constraint as precise_experiment_configs.py.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hydra.core.config_store import ConfigStore

from robometer.configs.eval_configs import BaselineEvalConfig, EvalServerConfig
from robometer.configs.experiment_configs import CustomEvaluationConfig

# Wan's encoder consumes frame 0 and then whole groups of four, silently dropping
# the remainder, so a clip length must be ``4k + 1``.  The stock server's
# ``NUM_SUBSAMPLED_FRAMES = 4`` is therefore not a valid Precise clip at all: it
# would encode frame 0 and throw frames 1-3 away.
#
# The two defaults differ on purpose.  Reward alignment reproduces the in-training
# protocol, which thinned every prefix to nine frames, so its numbers stay directly
# comparable with the `save_best` pearson.  The qualitative server defaults to five,
# the shortest legal clip, which is ~1.8x cheaper per prefix.
#
# Careful: five is *not* a truncation of nine here.  Decision 2's bit-exactness
# argument holds when both clips start from the same frames, but the growing-prefix
# protocol thins each prefix with `linspace_subsample_frames`, so a 5-frame clip
# spans the same window as the 9-frame clip at roughly half the temporal density --
# a coarser view, and one frame-spacing the model never saw in training.  Run a
# split at both lengths (`--num-subsampled-frames 5 9`) to see how much it costs.
SERVER_SUBSAMPLED_FRAMES = 5
ALIGNMENT_SUBSAMPLED_FRAMES = 9


@dataclass
class PreciseEvalServerConfig(EvalServerConfig):
    """``EvalServerConfig`` plus the two Precise-only runtime knobs."""

    num_subsampled_frames: int = field(
        default=SERVER_SUBSAMPLED_FRAMES,
        metadata={"help": "Frames per growing-prefix clip. Must be 4k+1 (5, 9, 13, ... ), and <= model.precise.max_len"},
    )
    pointmap_norm_bounds: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={
            "help": "Override the checkpoint's pointmap clip bounds, e.g. "
            "'{x:[-1.0,1.0],y:[-1.2,0.5],z:[0.9,2.2]}'. None keeps the trained bounds."
        },
    )


@dataclass
class PreciseEvalModelConfig:
    """``model_config`` block for ``reward_model=precise``."""

    batch_size: int = field(default=32, metadata={"help": "Clips per forward pass"})
    # The clip length is the top-level `max_frames`, not a field here -- one knob
    # for one thing, and it matches the command shape used for the stock baselines.
    pointmap_norm_bounds: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={"help": "Override the checkpoint's pointmap clip bounds. None keeps the trained bounds"},
    )
    device: str = field(default="cuda", metadata={"help": "Device for the single model replica"})


@dataclass
class PreciseBaselineEvalConfig(BaselineEvalConfig):
    """``BaselineEvalConfig`` whose ``__post_init__`` knows ``reward_model=precise``.

    The eval datasets, the progress target convention, the discrete bin count and
    the success cutoff table are **not** fields here: they are read back out of the
    checkpoint's own ``config.yaml`` so a run cannot silently score the model under
    a different data contract than it trained on.
    """

    reward_model: str = field(default="precise", metadata={"help": "Only 'precise' is accepted by this entry point"})
    checkpoint_labels: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Optional display names, one per model_paths entry. Defaults to the checkpoint's modality"},
    )
    model_paths: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Evaluate several checkpoints in one process. Overrides model_path when set"},
    )

    def __post_init__(self):
        if isinstance(self.custom_eval, dict):
            self.custom_eval = CustomEvaluationConfig(**self.custom_eval)
        if self.reward_model != "precise":
            raise ValueError(
                f"precise_baseline_eval.py only handles reward_model=precise, got {self.reward_model!r}. "
                "Use robometer/evals/run_baseline_eval.py for the stock reward models."
            )
        if self.model_config is None or isinstance(self.model_config, dict):
            self.model_config = PreciseEvalModelConfig(**(self.model_config or {}))


cs = ConfigStore.instance()
cs.store(name="precise_eval_server_config", node=PreciseEvalServerConfig)
cs.store(name="precise_baseline_eval_schema", node=PreciseBaselineEvalConfig)


__all__ = [
    "ALIGNMENT_SUBSAMPLED_FRAMES",
    "SERVER_SUBSAMPLED_FRAMES",
    "PreciseBaselineEvalConfig",
    "PreciseEvalModelConfig",
    "PreciseEvalServerConfig",
]
