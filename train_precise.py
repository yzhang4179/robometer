#!/usr/bin/env python3
"""Step 3 training entry point for Precise Robometer.

This is intentionally a fork of the small amount of orchestration in ``train.py``.
The stock entry point has closed model/dataset/collator/trainer dispatch tables and
registers a Hydra schema that cannot represent ``model.precise``.  Keeping this
entry point separate leaves every original Robometer path unchanged.
"""

# Precise uses neither Unsloth nor quantization. The stock setup module imports
# Unsloth eagerly, however, and this venv's optional torchao build expects a newer
# torch. Disable that unused optional backend before model imports, and provide
# the one stock setup symbol needed merely to import RBMHeadsTrainer.
import sys
import types

from transformers.utils import import_utils as transformers_import_utils

transformers_import_utils._torchao_available = False


class _UnusedFastVisionModel:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("Precise does not support the stock Unsloth model path")


if "unsloth" not in sys.modules:
    unsloth_stub = types.ModuleType("unsloth")
    unsloth_stub.FastVisionModel = _UnusedFastVisionModel
    sys.modules["unsloth"] = unsloth_stub

import copy
import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path

import datasets as hf_datasets
import torch
import torch.distributed as dist
import yaml
from hydra import main as hydra_main
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from rich import print as rprint
from rich.panel import Panel

from robometer.configs.experiment_configs import SaveBestConfig
from robometer.configs.precise_experiment_configs import (
    CustomEvaluationConfig,
    DataConfig,
    LoggingConfig,
    LossConfig,
    PEFTConfig,
    PreciseExperimentConfig,
    PreciseModelConfig,
    TrainingConfig,
)
from robometer.data.datasets.base import resolve_dataset_keys
from robometer.data.datasets.helpers import show_available_datasets
from robometer.models.precise.layout import MODALITIES, check_frame_count
from robometer.trainers.precise_trainer import PreciseTrainer
from robometer.utils.config_utils import convert_hydra_to_dataclass, display_config
from robometer.utils.distributed import banner, is_rank_0
from robometer.utils.logger import Logger, rank_0_info
from robometer.utils.precise_setup_utils import (
    resolve_precise_config,
    setup_precise_batch_collator,
    setup_precise_dataset,
    setup_precise_model_and_processor,
)
from robometer.utils.save import (
    SaveBestCallback,
    resolve_checkpoint_path,
    save_final_checkpoint,
    update_cfg_with_pretrained_ckpt,
)
from robometer.utils.setup_utils import create_training_arguments
from robometer.utils.timer import _timer

hf_datasets.logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Register every group used by config.yaml, then overwrite base_config last. Do
# not import train.py here: its later base_config registration would close the
# schema around the stock ExperimentConfig and reject model.precise.
cs = ConfigStore.instance()
cs.store(group="model", name="model_config", node=PreciseModelConfig)
cs.store(group="peft", name="peft_config", node=PEFTConfig)
cs.store(group="data", name="data_config", node=DataConfig)
cs.store(group="training", name="training_config", node=TrainingConfig)
cs.store(group="loss", name="loss_config", node=LossConfig)
cs.store(group="logging", name="logging_config", node=LoggingConfig)
cs.store(group="logging/save_best", name="save_best_config", node=SaveBestConfig)
cs.store(group="custom_eval", name="custom_eval_config", node=CustomEvaluationConfig)
cs.store(name="base_config", node=PreciseExperimentConfig)


def _checkpoint_config_path(checkpoint: str) -> tuple[str, str]:
    """Resolve a local/Hub checkpoint and find its saved training config."""
    resolved = resolve_checkpoint_path(checkpoint, hub_token=os.environ.get("HF_TOKEN"))
    if not resolved:
        raise ValueError(f"Could not resolve checkpoint {checkpoint!r}")
    candidates = (Path(resolved) / "config.yaml", Path(resolved).parent / "config.yaml")
    for candidate in candidates:
        if candidate.is_file():
            return resolved, str(candidate)
    raise FileNotFoundError(
        f"Precise checkpoint {resolved!r} has no config.yaml in the checkpoint or its parent"
    )


def update_precise_cfg_from_ckpt(cfg: PreciseExperimentConfig, checkpoint: str | None) -> None:
    """Restore the Precise input contract that the stock resume helper cannot see.

    Bounds, modality, mask semantics, and architecture belong to the checkpoint.
    A mismatching launch value is rejected rather than silently changing what the
    saved weights mean.
    """
    if not checkpoint:
        return
    _, config_path = _checkpoint_config_path(checkpoint)
    with open(config_path) as handle:
        saved_config = yaml.safe_load(handle)
    saved_precise = (saved_config or {}).get("model", {}).get("precise")
    if not isinstance(saved_precise, dict):
        raise ValueError(f"{config_path} has no model.precise mapping; it is not a Precise checkpoint")

    launch_precise = dict(cfg.model.precise or {})
    if launch_precise != saved_precise:
        launch_text = yaml.safe_dump(launch_precise, sort_keys=True).rstrip()
        saved_text = yaml.safe_dump(saved_precise, sort_keys=True).rstrip()
        raise ValueError(
            "Launch model.precise does not match the checkpoint. Resume/load must preserve the exact "
            "architecture and input contract.\n\n"
            f"launch:\n{launch_text}\n\ncheckpoint ({config_path}):\n{saved_text}\n\n"
            "Apply the checkpoint's model.precise values on the CLI, then retry."
        )
    cfg.model.precise = copy.deepcopy(saved_precise)
    rank_0_info(f"Verified and restored model.precise from {config_path}")


def validate_step3_config(cfg: PreciseExperimentConfig) -> None:
    """Fail before loading the 1.4 GB VAE when the study contract is violated."""
    precise = resolve_precise_config(cfg.model)
    # Exp 1 pinned these to the square caches. Exps 2 and 3 train on push-T and lift,
    # so the checks below are now *relational*: whatever the eval split is, in-training
    # reward alignment must score exactly that split and nothing else.
    expected_train = list(cfg.data.train_datasets)
    expected_val = list(cfg.data.eval_datasets)
    if not expected_train or not expected_val:
        raise ValueError("Step 3 needs non-empty data.train_datasets and data.eval_datasets")
    if precise.modality not in MODALITIES:
        raise ValueError(f"model.precise.modality must be one of {MODALITIES}, got {precise.modality!r}")
    if cfg.training.num_gpus != 1:
        raise ValueError(f"Step 3 is a one-GPU experiment; set training.num_gpus=1 (got {cfg.training.num_gpus})")
    if cfg.data.max_frames != 9 or precise.max_len != 9:
        raise ValueError(
            f"Step 3 requires data.max_frames=model.precise.max_len=9; got {cfg.data.max_frames} and {precise.max_len}"
        )
    check_frame_count(cfg.data.max_frames, precise.vae_temporal_ratio)
    if list(cfg.data.sample_type_ratio) != [0, 1, 0]:
        raise ValueError(
            "Step 3 is progress-only; expected data.sample_type_ratio=[0,1,0], "
            f"got {cfg.data.sample_type_ratio}"
        )
    if list(cfg.data.progress_strategy_ratio) != [0, 1, 1, 1]:
        raise ValueError(
            "Step 3 is single-task; expected data.progress_strategy_ratio=[0,1,1,1], "
            f"got {cfg.data.progress_strategy_ratio}"
        )
    if cfg.data.progress_pred_type != "absolute_first_frame":
        raise ValueError("Step 3 requires data.progress_pred_type=absolute_first_frame")
    loss_types = {
        "loss": cfg.loss.progress_loss_type,
        "data": cfg.data.progress_loss_type,
        "model": cfg.model.progress_loss_type,
        "model.precise": precise.progress_loss_type,
    }
    if any(str(value).lower() != "discrete" for value in loss_types.values()):
        raise ValueError(f"Step 3 requires every progress_loss_type to be discrete; got {loss_types}")
    bin_counts = {
        "loss": cfg.loss.progress_discrete_bins,
        "data": cfg.data.progress_discrete_bins,
        "model": cfg.model.progress_discrete_bins,
        "model.precise": precise.progress_discrete_bins,
    }
    if any(value != 10 for value in bin_counts.values()):
        raise ValueError(f"Step 3 requires ten discrete progress bins everywhere; got {bin_counts}")
    if cfg.loss.predict_last_frame_progress:
        raise ValueError("Step 3 supervises all nine tokens; set loss.predict_last_frame_progress=false")
    if set(expected_train) & set(expected_val):
        raise ValueError(f"train and eval splits overlap: {set(expected_train) & set(expected_val)}")
    if cfg.custom_eval.eval_types != ["reward_alignment"]:
        raise ValueError("Step 3 only runs custom_eval.eval_types=[reward_alignment]")
    if list(cfg.custom_eval.reward_alignment) != expected_val:
        raise ValueError(
            f"Step 3 reward alignment must use only {expected_val}; got {cfg.custom_eval.reward_alignment}"
        )
    if not cfg.custom_eval.use_frame_steps:
        raise ValueError(
            "Precise reward alignment requires custom_eval.use_frame_steps=true so each "
            "prediction sees only its causal prefix"
        )
    if not cfg.debug and cfg.custom_eval.subsample_n_frames != 9:
        raise ValueError("Full Step 3 runs require custom_eval.subsample_n_frames=9")
    if not cfg.custom_eval.pad_frames:
        raise ValueError("Precise reward-alignment prefixes require custom_eval.pad_frames=true")
    if cfg.model.use_peft or cfg.model.quantization or cfg.model.train_preference_head:
        raise ValueError("Precise requires use_peft=false, quantization=false, and train_preference_head=false")
    if not (cfg.model.train_progress_head and cfg.model.train_success_head):
        raise ValueError("Step 3 trains both progress and success heads")


def _prepare_output_dir(output_dir: str, cfg: PreciseExperimentConfig) -> None:
    """Create a per-run directory without deleting a resume checkpoint."""
    overwrite = cfg.training.overwrite_output_dir
    resuming = bool(cfg.training.resume_from_checkpoint)
    distributed = dist.is_available() and dist.is_initialized()

    if overwrite and resuming:
        raise ValueError("Do not combine overwrite_output_dir=true with resume_from_checkpoint")
    if is_rank_0() and os.path.exists(output_dir):
        if overwrite:
            rank_0_info(f"Removing existing output directory {output_dir}")
            shutil.rmtree(output_dir)
        elif resuming:
            rank_0_info(f"Reusing existing output directory for full resume: {output_dir}")
        else:
            raise ValueError(
                f"Output directory {output_dir} already exists. Choose another training.exp_name or set "
                "training.overwrite_output_dir=true."
            )
    if distributed:
        dist.barrier()
    os.makedirs(output_dir, exist_ok=True)
    if distributed:
        dist.barrier()


def _restore_dataset_random_state(train_dataset, resume_path: str | None) -> None:
    if not resume_path or not os.path.isdir(resume_path):
        return
    state_path = os.path.join(resume_path, "dataset_random_state.json")
    if not os.path.exists(state_path):
        rank_0_info("No dataset_random_state.json in checkpoint; sampler starts from its configured seed")
        return
    try:
        with open(state_path) as handle:
            state = json.load(handle)
        base_dataset = train_dataset.dataset if hasattr(train_dataset, "dataset") else train_dataset
        if hasattr(base_dataset, "set_random_state"):
            base_dataset.set_random_state(state)
            rank_0_info(f"Restored dataset random state from {state_path}")
    except Exception as error:
        rank_0_info(f"Could not restore dataset random state: {error}")


def train(cfg: PreciseExperimentConfig) -> None:
    timing_raw = {}
    run_name = cfg.training.exp_name
    if cfg.debug:
        run_name += "_debug"
        cfg.training.logging_steps = 1
        cfg.training.eval_steps = 5
        cfg.training.custom_eval_steps = 5
        cfg.training.save_steps = 5
        cfg.data.dataloader_num_workers = 0
        cfg.data.dataloader_persistent_workers = False
        cfg.training.dataloader_num_workers = 0
        cfg.training.dataloader_persistent_workers = False
        cfg.custom_eval.reward_alignment_max_trajectories = 2
        cfg.custom_eval.subsample_n_frames = 5

    torch.autograd.set_detect_anomaly(cfg.debug)
    torch.backends.cudnn.benchmark = True
    if not torch.cuda.is_available():
        raise RuntimeError("Precise Step 3 requires the RTX 4090; CUDA is not available")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Expected exactly one visible GPU, found {torch.cuda.device_count()}. Set CUDA_VISIBLE_DEVICES=0."
        )
    torch.cuda.empty_cache()

    checkpoint = cfg.training.load_from_checkpoint or cfg.training.resume_from_checkpoint
    if checkpoint:
        rank_0_info(f"Loading Precise checkpoint: {checkpoint}")
    # The stock helper restores shared head/loss fields; our add-only helper then
    # verifies/restores the model.precise block omitted by its base-class filter.
    update_cfg_with_pretrained_ckpt(cfg, checkpoint)
    update_precise_cfg_from_ckpt(cfg, checkpoint)
    validate_step3_config(cfg)
    modality = resolve_precise_config(cfg.model).modality

    banner("Setting up Precise model", f"modality={modality}")
    with _timer("time/setup_model_and_processor", timing_raw=timing_raw):
        tokenizer, processor, model = setup_precise_model_and_processor(
            cfg.model,
            hf_model_id=checkpoint or "",
            peft_config=cfg.peft,
        )

    output_dir = os.path.join(cfg.training.output_dir, run_name)
    training_args = create_training_arguments(cfg.training, output_dir)
    _prepare_output_dir(output_dir, cfg)

    logger = Logger(
        log_to=cfg.logging.log_to,
        output_dir=output_dir,
        is_main_process=is_rank_0(),
        log_level=cfg.logging.log_level,
    )
    config_path = os.path.join(output_dir, "config.yaml")
    with open(config_path, "w") as handle:
        yaml.safe_dump(asdict(cfg), handle, default_flow_style=False, sort_keys=False)
    rank_0_info(f"Saved resolved training config to {config_path}")

    wandb_info_path = os.path.join(output_dir, "wandb_info.json")
    resume_id = None
    if os.path.exists(wandb_info_path):
        try:
            with open(wandb_info_path) as handle:
                resume_id = json.load(handle).get("wandb_id")
        except Exception as error:
            rank_0_info(f"Could not load existing wandb run metadata: {error}")
    if "wandb" in (cfg.logging.log_to or []) and is_rank_0():
        logger.init_wandb(
            project=cfg.logging.wandb_project,
            entity=cfg.logging.wandb_entity,
            name=run_name,
            config=asdict(cfg),
            notes=cfg.logging.wandb_notes,
            mode=cfg.logging.wandb_mode,
            resume_id=resume_id,
        )
    logger.write_wandb_info(output_dir, run_name)

    if is_rank_0():
        show_available_datasets()
    banner("Resolving Precise dataset keys")
    cfg.data.train_datasets = resolve_dataset_keys(cfg.data.train_datasets, split="train")
    cfg.data.eval_datasets = resolve_dataset_keys(cfg.data.eval_datasets, split="eval")
    for eval_type in cfg.custom_eval.eval_types:
        values = getattr(cfg.custom_eval, eval_type)
        setattr(cfg.custom_eval, eval_type, resolve_dataset_keys(values, split="eval"))
    rank_0_info(f"Train datasets: {cfg.data.train_datasets}")
    rank_0_info(f"Reward-alignment datasets: {cfg.custom_eval.reward_alignment}")

    banner("Setting up Precise datasets and collator")
    with _timer("time/setup_data", timing_raw=timing_raw):
        collator = setup_precise_batch_collator(processor, tokenizer, cfg, is_eval=False)
        train_dataset = setup_precise_dataset(cfg.data, modality=modality, is_eval=False)
        eval_kwargs = {"max_samples": cfg.data.eval_subset_size} if cfg.data.eval_subset_size else {}
        eval_dataset = (
            setup_precise_dataset(cfg.data, modality=modality, is_eval=True, **eval_kwargs)
            if cfg.training.do_eval
            else None
        )
    rank_0_info(f"Training samples before repeat wrapper: {len(train_dataset.dataset)}")
    if eval_dataset is not None:
        rank_0_info(
            f"Default validation samples: {len(eval_dataset)} "
            "(custom reward alignment is evaluated separately)"
        )

    save_cfg = cfg.logging.save_best
    save_callback = SaveBestCallback(**asdict(save_cfg), base_model=cfg.model.base_model_id)
    trainer = PreciseTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        config=cfg,
        logger=logger,
        callbacks=[save_callback],
    )
    save_callback.setup_trainer_reference(trainer)
    logger.log_scalars(timing_raw)

    resume_path = None
    if cfg.training.resume_from_checkpoint:
        resume_path = resolve_checkpoint_path(
            cfg.training.resume_from_checkpoint,
            hub_token=save_cfg.hub_token or os.environ.get("HF_TOKEN"),
        )
    _restore_dataset_random_state(train_dataset, resume_path)
    rank_0_info(f"Starting at {'checkpoint ' + resume_path if resume_path else 'step 0'}")

    trainer.train(resume_from_checkpoint=resume_path)

    # Keep the inference-ready final model inside this run, unlike stock
    # train.py's shared-root save. Standard checkpoint-N directories above carry
    # optimizer/scheduler/RNG state for exact training resume.
    final_dir = os.path.join(output_dir, "final")
    save_final_checkpoint(trainer, final_dir, step=trainer.state.global_step)
    shutil.copy(config_path, os.path.join(final_dir, "config.yaml"))
    rank_0_info(f"Training complete. Final model: {final_dir}")


@hydra_main(version_base=None, config_path="robometer/configs", config_name="precise_transformer")
def main(cfg: DictConfig) -> None:
    banner("Starting Precise Robometer Step 3")
    experiment = convert_hydra_to_dataclass(cfg, PreciseExperimentConfig)
    display_config(experiment)
    if experiment.mode != "train":
        raise ValueError(f"train_precise.py only supports mode=train, got {experiment.mode!r}")
    if is_rank_0():
        rprint(Panel.fit("Starting Precise training + reward alignment", style="bold green"))
    train(experiment)


if __name__ == "__main__":
    main()
