#!/usr/bin/env python3
"""Config subclasses for the precise (Wan-VAE) progress model.

Add-only: the stock `ModelConfig` / `ExperimentConfig` are untouched, and anything
that does not know about `precise` keeps working exactly as before.

`precise` is deliberately kept as a **plain dict** rather than being converted to a
`PreciseTransformerConfig` in `__post_init__`, the way `ModelConfig` converts
`rewind`. That conversion is why `load_model_from_hf` needs a custom YAML loader to
survive a `!!python/object:...ReWINDTransformerConfig` tag in a saved config
(`robometer/utils/save.py:897-901`). Our config stays plain data, so
`yaml.safe_load` reads it back with no hack. `PreciseTransformer.__init__` does the
conversion instead, at the one moment it is actually needed.

Two functions in `save.py` hardcode the base classes and therefore cannot see the
`precise` field:

| path | where | what happens to `model.precise` |
|---|---|---|
| resume training | `save.py:217` -- `{f.name for f in fields(ModelConfig)}` | silently not restored |
| eval / inference | `save.py:906` -- `ExperimentConfig(**filtered_config)` | `TypeError: unexpected keyword argument 'precise'` |

`robometer/utils/precise_setup_utils.py` provides the add-only replacements.
"""

# No `from __future__ import annotations`: pyrallis resolves config dataclasses from
# runtime annotations, and PEP 563 would hand it strings instead of classes.

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from robometer.configs.experiment_configs import (
    CustomEvaluationConfig,
    DataConfig,
    ExperimentConfig,
    LoggingConfig,
    LossConfig,
    ModelConfig,
    PEFTConfig,
    TrainingConfig,
)
from robometer.models.precise.wan_tokenizer import DEFAULT_POINTMAP_BOUNDS


def default_precise() -> Dict[str, Any]:
    """Defaults for the `model.precise` block; the yaml overrides what it cares about."""
    return {
        "modality": "rgb_pointmap",
        "hidden_dim": 512,
        "num_layers": 6,
        "num_attention_heads": 8,
        "mlp_ratio": 4,
        "dropout": 0.1,
        "max_len": 9,
        "latent_patch_size": 2,
        "prog_block_causal": False,
        "vae_model_id": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "vae_dtype": "bfloat16",
        "vae_keep_decoder": False,
        "frame_size": 256,
        # Exp 3 (lift to N cm) needs the target height in the input; false keeps the
        # model bit-identical to exp 1. See models/precise/lang_precise_transformer.py.
        "use_lang_token": False,
        "lang_vector_dim": 384,
        "pointmap_norm_bounds": {c: list(v) for c, v in DEFAULT_POINTMAP_BOUNDS.items()},
    }


@dataclass
class PreciseModelConfig(ModelConfig):
    """`ModelConfig` plus one field. Everything else is inherited unchanged."""

    precise: Optional[Dict[str, Any]] = field(
        default_factory=default_precise,
        metadata={"help": "PreciseTransformerConfig fields; kept a plain dict so yaml.safe_load can read it back"},
    )

    def __post_init__(self):
        super().__post_init__()  # handles the `rewind` sub-config, which we leave at None
        if self.precise is None:
            return
        if not isinstance(self.precise, dict):
            # A PreciseTransformerConfig can arrive here from a checkpoint round-trip.
            self.precise = dict(self.precise.to_dict() if hasattr(self.precise, "to_dict") else vars(self.precise))
        # The heads read these off the parent config, so keep the two in step.
        self.precise.setdefault("progress_loss_type", self.progress_loss_type)
        self.precise.setdefault("progress_discrete_bins", self.progress_discrete_bins or 10)


@dataclass
class PreciseExperimentConfig(ExperimentConfig):
    """`ExperimentConfig` with the model section widened to `PreciseModelConfig`."""

    model: PreciseModelConfig = field(default_factory=PreciseModelConfig)

    def __post_init__(self):
        # Convert first, so the parent's `isinstance(self.model, dict)` branch -- which
        # would build a plain ModelConfig and drop `precise` -- never fires.
        if isinstance(self.model, dict):
            self.model = PreciseModelConfig(**self.model)
        super().__post_init__()


__all__ = [
    "PreciseModelConfig",
    "PreciseExperimentConfig",
    "default_precise",
    # re-exported so callers need only one import
    "DataConfig",
    "TrainingConfig",
    "LossConfig",
    "LoggingConfig",
    "PEFTConfig",
    "CustomEvaluationConfig",
]
