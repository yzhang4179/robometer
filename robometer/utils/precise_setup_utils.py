#!/usr/bin/env python3
"""Building and loading the precise model, without touching `setup_utils.py` or `save.py`.

`setup_model_and_processor` dispatches on `cfg.base_model_id` and knows nothing
about `precise`; `load_model_from_hf` filters the saved config against
`fields(ExperimentConfig)`, which drops `model.precise` and then crashes when the
dict is fed to `ModelConfig`. Both are replaced here rather than edited there.
"""

# No `from __future__ import annotations` -- see precise_experiment_configs.py.

import os
from dataclasses import fields
from typing import Any, Dict, Optional, Tuple

import torch
import yaml

from robometer.configs.precise_experiment_configs import PreciseExperimentConfig, PreciseModelConfig
from robometer.data.collators.precise import PreciseBatchCollator
from robometer.data.datasets.precise_data import PreciseCustomEvalDataset, PreciseRBMDataset
from robometer.data.datasets.repeated_dataset import RepeatedDataset
from robometer.models.precise.lang_precise_transformer import (
    LangPreciseTransformer,
    LangPreciseTransformerConfig,
)
from robometer.models.precise.precise_transformer import PreciseTransformer, PreciseTransformerConfig
from robometer.models.precise.wan_tokenizer import WanLatentTokenizer
from robometer.utils.logger import get_logger
from robometer.utils.save import resolve_checkpoint_path

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
logger = get_logger()


def resolve_precise_config(model_config) -> PreciseTransformerConfig:
    """`model.precise` (a plain dict) -> the typed sub-config the model wants.

    `use_lang_token` picks the class: without it nothing about exp 1 changes.
    """
    precise = getattr(model_config, "precise", None)
    if precise is None:
        raise ValueError("model.precise is not set; use PreciseModelConfig or the precise_transformer.yaml preset")
    if isinstance(precise, PreciseTransformerConfig):
        return precise
    config_cls = LangPreciseTransformerConfig if dict(precise).get("use_lang_token") else PreciseTransformerConfig
    return config_cls(**precise)


def precise_model_class(precise: PreciseTransformerConfig):
    """The transformer class a resolved sub-config asks for."""
    return LangPreciseTransformer if getattr(precise, "use_lang_token", False) else PreciseTransformer


def build_latent_tokenizer(
    precise: PreciseTransformerConfig, device: torch.device | str = "cpu"
) -> WanLatentTokenizer:
    """Load the frozen Wan VAE and check it is the model the config was written for."""
    tokenizer = WanLatentTokenizer(
        model_id=precise.vae_model_id,
        dtype=DTYPES.get(precise.vae_dtype, torch.bfloat16),
        pointmap_norm_bounds=precise.pointmap_norm_bounds,
        scale_latents=precise.scale_latents,
        keep_decoder=precise.vae_keep_decoder,
        device=device,
    )
    # A different VAE would silently change every latent shape, so compare rather than trust.
    mismatches = [
        f"{name}: config says {expected}, VAE says {actual}"
        for name, expected, actual in (
            ("vae_z_dim", precise.vae_z_dim, tokenizer.z_dim),
            ("vae_spatial_ratio", precise.vae_spatial_ratio, tokenizer.spatial_ratio),
            ("vae_temporal_ratio", precise.vae_temporal_ratio, tokenizer.temporal_ratio),
        )
        if expected != actual
    ]
    if mismatches:
        raise ValueError(f"{precise.vae_model_id} does not match model.precise -- " + "; ".join(mismatches))
    return tokenizer


def build_precise_model(
    model_config,
    device: torch.device | str = "cpu",
    checkpoint_path: Optional[str] = None,
    load_vae: bool = True,
) -> PreciseTransformer:
    """Construct `PreciseTransformer` with its frozen tokenizer attached and moved."""
    precise = resolve_precise_config(model_config)
    model_cls = precise_model_class(precise)

    if checkpoint_path:
        model = model_cls.from_pretrained(checkpoint_path, config=precise)
    else:
        # Still the *outer* config: `PredictionHeadsMixin` reads `train_*_head` off it.
        # The `precise` dict inside is re-typed by `model_cls.config_class`.
        model = model_cls(model_config)

    if load_vae:
        model.attach_tokenizer(build_latent_tokenizer(precise, device=device))
    return model.to(device)


def setup_precise_model_and_processor(
    model_config,
    hf_model_id: str = "",
    peft_config=None,
):
    """Training-entry replacement for the stock model dispatch.

    Precise has neither a language tokenizer nor an image processor: its collator
    stacks cached arrays and ``WanLatentTokenizer`` normalises them inside the
    model.  Returning the same three-tuple as the stock helper keeps trainer setup
    straightforward.
    """
    del peft_config  # PEFT is intentionally unsupported for this small transformer.
    if getattr(model_config, "use_peft", False):
        raise ValueError("Precise trains its 19.5M-parameter transformer in full; set model.use_peft=false")
    if getattr(model_config, "quantization", False):
        raise ValueError("Precise does not use bitsandbytes; set model.quantization=false")
    if getattr(model_config, "use_unsloth", False):
        raise ValueError("Unsloth is a language-model path; set model.use_unsloth=false for Precise")

    checkpoint_path = None
    if hf_model_id:
        checkpoint_path = resolve_checkpoint_path(hf_model_id, hub_token=os.environ.get("HF_TOKEN"))
        if not checkpoint_path:
            raise ValueError(f"Could not resolve Precise checkpoint {hf_model_id!r}")

    # Let Trainer move the trainable module.  The unregistered VAE starts on CPU
    # and moves itself to the transformer's device on the first forward pass.
    model = build_precise_model(
        model_config,
        device="cpu",
        checkpoint_path=checkpoint_path,
        load_vae=True,
    )
    model.processor = None
    model.tokenizer = None

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    logger.info(
        f"Precise model ready: {trainable:,} trainable / {total:,} registered parameters; "
        "the frozen Wan VAE is excluded from the optimizer and checkpoints"
    )
    return None, None, model


def setup_precise_batch_collator(processor, tokenizer, config, is_eval: bool = False):
    """Build the collator while transferring only the configured visual stream(s)."""
    del processor, tokenizer
    precise = resolve_precise_config(config.model)
    return PreciseBatchCollator(
        processor=None,
        tokenizer=None,
        modality=precise.modality,
        resized_height=config.data.resized_height,
        resized_width=config.data.resized_width,
        base_model_id=config.model.base_model_id,
        load_embeddings=config.data.load_embeddings,
        use_multi_image=False,
        prog_pref=False,
        use_per_frame_progress_token=False,
        shuffle_progress_frames=False,
        inference=is_eval,
    )


def setup_precise_dataset(
    data_config,
    modality: str,
    is_eval: bool = False,
    sampler_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,
):
    """Build the training/default-eval dataset through the Precise sampler path."""
    if data_config.dataset_type != "rbm":
        raise ValueError(f"Precise requires data.dataset_type=rbm, got {data_config.dataset_type!r}")
    sampler_options = dict(sampler_kwargs or {})
    sampler_options.setdefault("random_seed", data_config.seed)
    dataset = PreciseRBMDataset(
        config=data_config,
        is_evaluation=is_eval,
        sampler_kwargs=sampler_options,
        modality=modality,
        **kwargs,
    )
    return dataset if is_eval else RepeatedDataset(dataset)


def setup_precise_custom_eval_dataset(
    data_config,
    sampler_type: str,
    modality: str,
    verbose: bool = True,
    sampler_kwargs: Optional[Dict[str, Any]] = None,
):
    """Build the reward-alignment-only validation dataset."""
    return PreciseCustomEvalDataset(
        sampler_type=sampler_type,
        config=data_config,
        verbose=verbose,
        sampler_kwargs=dict(sampler_kwargs or {}),
        modality=modality,
    )


def load_precise_experiment_config(config_path: str) -> PreciseExperimentConfig:
    """Read a saved `config.yaml` back into our subclass.

    The stock `load_model_from_hf` filters against `fields(ExperimentConfig)`, which
    silently drops `model.precise` and then raises `TypeError: ModelConfig.__init__()
    got an unexpected keyword argument 'precise'`. Filtering against our subclass and
    constructing our subclass is the whole fix.
    """
    with open(config_path) as handle:
        saved = yaml.safe_load(handle)  # plain data; no custom loader needed
    if not isinstance(saved, dict):
        raise ValueError(f"{config_path} did not parse to a mapping")

    valid = {f.name for f in fields(PreciseExperimentConfig)}
    dropped = sorted(set(saved) - valid)
    config = PreciseExperimentConfig(**{k: v for k, v in saved.items() if k in valid})
    if dropped:
        print(f"  (ignored unknown top-level config keys: {dropped})")
    if getattr(config.model, "precise", None) is None:
        raise ValueError(f"{config_path} has no model.precise block -- is this a precise checkpoint?")
    return config


def load_precise_model(
    model_path: str, device: torch.device | str = "cuda", load_vae: bool = True
) -> Tuple[PreciseExperimentConfig, PreciseTransformer]:
    """Checkpoint dir -> (experiment config, model in eval mode). Used by Step 4's eval server."""
    config_path = os.path.join(model_path, "config.yaml")
    if not os.path.exists(config_path):
        parent = os.path.join(os.path.dirname(os.path.normpath(model_path)), "config.yaml")
        if not os.path.exists(parent):
            raise FileNotFoundError(f"No config.yaml in {model_path} or its parent")
        config_path = parent

    config = load_precise_experiment_config(config_path)
    model = build_precise_model(config.model, device=device, checkpoint_path=model_path, load_vae=load_vae)
    model.eval()
    return config, model


def precise_config_from_overrides(overrides: Optional[Dict[str, Any]] = None) -> PreciseExperimentConfig:
    """A default `PreciseExperimentConfig` with `model.precise` keys overridden.

    Convenience for the dev scripts, so they can take `--set hidden_dim=256` style
    flags without a yaml file.
    """
    config = PreciseExperimentConfig(model=PreciseModelConfig())
    for key, value in (overrides or {}).items():
        if key not in config.model.precise:
            raise KeyError(f"Unknown model.precise key {key!r}; known keys: {sorted(config.model.precise)}")
        config.model.precise[key] = value
    return config
