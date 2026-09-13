#!/usr/bin/env python3
"""Step 4 part 2: reward-alignment and policy-ranking metrics for the three Precise checkpoints.

The stock ``run_baseline_eval.py`` cannot reach Precise at three points, all of
which are closed dispatches rather than missing arguments:

* ``BaselineEvalConfig.__post_init__`` raises for any ``reward_model`` outside its
  literal list, so ``reward_model=precise`` never gets a ``model_config``.
* ``RBMModel`` loads through ``load_model_from_hf`` + ``setup_batch_collator``,
  both of which drop or reject ``model.precise``.
* ``setup_custom_eval_dataset`` builds ``CustomEvalDataset``, whose sampler table
  returns RGB only, so the pointmap modalities would silently get no pointmap.

Two evaluations run here, on the same checkpoints and the same clips:

* ``reward_alignment`` -- ``pearson`` (the VOC number), ``loss`` and the success-head
  metrics, over the **successful** trajectories of a split.
* ``policy_ranking`` -- ``kendall``, ``ranking_acc`` and ``succ_fail_diff`` over the
  **successes and failures together**. Robometer computes succ-fail-diff and
  Kendall only here, never in reward alignment.

Everything downstream of the forward pass is the **stock** code:
``run_reward_alignment_eval_per_trajectory`` computes the metrics, so these
numbers are directly comparable with the ones the trainer logged.

Two things are deliberately taken from the checkpoint rather than from this
config: the whole ``data`` block (progress target convention, discrete bins,
success cutoff table, min/max success) and ``model.precise``. A run therefore
cannot score a model under a different data contract than it trained under.

    uv run python robometer/evals/precise_baseline_eval.py \\
        reward_model=precise \\
        model_paths=[./logs/precise/precise_rgb/final] \\
        custom_eval.eval_types=[reward_alignment,policy_ranking] \\
        custom_eval.reward_alignment=[precise_square_rbm_square_d1_val] \\
        "custom_eval.policy_ranking=[[precise_square_rbm_square_d1_val,precise_square_rbm_square_d1_val_fail]]" \\
        custom_eval.use_frame_steps=true \\
        custom_eval.subsample_n_frames=9 \\
        custom_eval.reward_alignment_max_trajectories=null \\
        max_frames=9 \\
        model_config.batch_size=32
"""

# Same import-order preamble as train_precise.py; see precise_eval_server.py.
import sys
import types

from transformers.utils import import_utils as transformers_import_utils

transformers_import_utils._torchao_available = False

if "unsloth" not in sys.modules:

    class _UnusedFastVisionModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            del args, kwargs
            raise RuntimeError("Precise does not support the stock Unsloth model path")

    _stub = types.ModuleType("unsloth")
    _stub.FastVisionModel = _UnusedFastVisionModel
    sys.modules["unsloth"] = _stub

import copy
import json
import os
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import main as hydra_main
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from robometer.configs.precise_eval_configs import PreciseBaselineEvalConfig
from robometer.data.datasets.base import resolve_dataset_keys
from robometer.evals.compile_results import run_policy_ranking_eval, run_reward_alignment_eval_per_trajectory
from robometer.evals.precise_eval_server import PreciseInferenceEngine
from robometer.models.precise.layout import POINTMAP, check_frame_count, visual_streams
from robometer.utils.config_utils import convert_hydra_to_dataclass, display_config
from robometer.utils.logger import get_logger
from robometer.utils.precise_setup_utils import setup_precise_batch_collator, setup_precise_custom_eval_dataset

# The prefix-collapse rule (group by trajectory id, order by frame_step, keep the
# last token of each prefix) is imported rather than reimplemented so the offline
# numbers use byte-identical aggregation to the ones the trainer logged.
from robometer.trainers.precise_trainer import PreciseTrainer

logger = get_logger()

MAX_PLOTS = 10


def _to_numpy(value):
    """Tensor -> numpy, demoting the half formats numpy cannot represent.

    The model runs under `torch.autocast(bfloat16)` so that these numbers match the
    ones training logged, which makes `progress_logits` and `success_logits` bf16.
    numpy has no bfloat16 dtype, so `.numpy()` on them raises
    `TypeError: Got unsupported ScalarType BFloat16`. Only floating types are cast,
    so an integer tensor keeps its dtype.
    """
    if not torch.is_tensor(value):
        return np.asarray(value)
    value = value.detach()
    if value.dtype in (torch.bfloat16, torch.float16):
        value = value.float()
    return value.cpu().numpy()


def _json_safe(obj: Any) -> Any:
    """np/torch -> plain Python, recursively. Local copy so this module has no
    import-time dependency on run_baseline_eval, which sets the global loguru level."""
    if isinstance(obj, dict):
        return {key: _json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(item) for item in obj]
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _short_name(name: str, limit: int = 60) -> str:
    return name if len(name) <= limit else f"{name[:limit - 12]}__{name[-10:]}"


def build_eval_data_config(engine: PreciseInferenceEngine, dataset_names: List[str], max_frames: int):
    """The checkpoint's own data contract, pointed at one eval dataset."""
    data_cfg = copy.deepcopy(engine.config.data)
    data_cfg.eval_datasets = list(dataset_names)
    data_cfg.dataset_type = "rbm"
    if data_cfg.max_frames != max_frames:
        logger.warning(
            f"clip length {max_frames} differs from the checkpoint's data.max_frames={data_cfg.max_frames}; "
            "prefixes will be thinned to a temporal density the model was not trained on"
        )
    data_cfg.max_frames = max_frames
    return data_cfg


def build_sampler_kwargs(cfg: PreciseBaselineEvalConfig, sampler_type: str) -> Dict[str, Any]:
    """The prefix contract, which is deliberately identical for both evaluations.

    ``subsample_n_frames`` is what makes them identical. The stock policy-ranking
    sampler has no such knob and strides by ``frame_step`` instead, which on a
    32-frame cache emits 32 prefixes per trajectory against reward alignment's 9.
    ``kendall_avg`` / ``kendall_sum`` average over whatever they are handed, so the
    two evaluations would then be reducing different clip sets to their scalars.
    ``PrecisePolicyRankingSampler`` adds the knob; both get 9 endpoints x 9 frames.
    """
    shared = {
        "random_seed": cfg.custom_eval.custom_eval_random_seed,
        # Score every prefix the sampler picks. Do NOT inherit RBMHeadsTrainer's
        # frame_step=2 shortcut, which exists only for the video VLM baselines.
        "frame_step": 1,
        "use_frame_steps": cfg.custom_eval.use_frame_steps,
        "subsample_n_frames": cfg.custom_eval.subsample_n_frames,
        "pad_frames": cfg.custom_eval.pad_frames,
    }
    if sampler_type == "reward_alignment":
        return {**shared, "max_trajectories": cfg.custom_eval.reward_alignment_max_trajectories}
    return {
        **shared,
        # null -> every trajectory of every quality tier, which is what we want:
        # the stock default of 5 per tier exists to keep a 1M-row eval affordable.
        "num_examples_per_quality_pr": cfg.custom_eval.num_examples_per_quality_pr,
        "num_partial_successes": cfg.custom_eval.num_partial_successes,
        "max_tasks": cfg.custom_eval.policy_ranking_max_tasks,
    }


def score_dataset(
    engine: PreciseInferenceEngine,
    cfg: PreciseBaselineEvalConfig,
    dataset_names: List[str],
    sampler_type: str = "reward_alignment",
) -> List[Dict[str, Any]]:
    """Run the sampler + collator + model over one eval dataset.

    Mirrors ``RBMHeadsTrainer._process_batch_progress_eval`` field for field, so the
    rows fed to the stock compiler are the same rows the trainer produced.
    """
    modality = engine.modality
    data_cfg = build_eval_data_config(engine, dataset_names, cfg.max_frames)

    sampler_kwargs = build_sampler_kwargs(cfg, sampler_type)
    dataset = setup_precise_custom_eval_dataset(
        data_config=data_cfg,
        sampler_type=sampler_type,
        modality=modality,
        verbose=True,
        sampler_kwargs=sampler_kwargs,
    )

    eval_config = copy.deepcopy(engine.config)
    eval_config.data = data_cfg
    collator = setup_precise_batch_collator(None, None, eval_config, is_eval=True)
    loader = DataLoader(
        dataset,
        batch_size=cfg.model_config.batch_size,
        collate_fn=collator,
        num_workers=data_cfg.dataloader_num_workers,
        pin_memory=False,
        drop_last=False,
        persistent_workers=False,
    )

    logger.info(f"  {len(dataset)} prefix samples -> {len(loader)} batches of {cfg.model_config.batch_size}")

    rows: List[Dict[str, Any]] = []
    for batch in tqdm(loader, desc=f"{engine.label} / {'+'.join(dataset_names)}"):
        inputs = {
            key: (value.to(engine.device) if torch.is_tensor(value) else value)
            for key, value in batch["progress_inputs"].items()
        }
        outputs, _ = engine.forward_precise_model(inputs)

        progress_pred = outputs.progress_logits["A"]
        target_progress = inputs["target_progress"]
        success_probs = success_binary = success_labels = None
        if engine.has_success_head:
            success_probs = torch.sigmoid(outputs.success_logits["A"])
            success_binary = (success_probs > 0.5).float()
            success_labels = inputs.get("success_labels")

        for index in range(progress_pred.shape[0]):
            metadata = inputs["metadata"][index] or {}
            row: Dict[str, Any] = {
                "task": inputs["task"][index],
                "target_progress": _to_numpy(target_progress[index]),
                "progress_pred": _to_numpy(progress_pred[index]),
                "data_source": inputs["data_source"][index],
                "data_gen_strategy": inputs["data_gen_strategy"][index],
                "quality_label": inputs["quality_labels"][index],
                "metadata": metadata,
                "id": metadata.get("id"),
                "video_path": metadata.get("video_path"),
                "partial_success": inputs["partial_success"][index],
            }
            if success_binary is not None:
                row["success_pred"] = _to_numpy(success_binary[index])
                row["success_probs"] = _to_numpy(success_probs[index])
                if success_labels is not None:
                    row["success_labels"] = _to_numpy(success_labels[index])
            rows.append(row)

    del loader, dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def compile_reward_alignment(engine: PreciseInferenceEngine, cfg: PreciseBaselineEvalConfig, rows: List[Dict[str, Any]]):
    """Collapse growing prefixes, then hand the stock compiler one row per trajectory."""
    if not rows:
        raise ValueError("no samples were scored -- did the quality filter remove every trajectory?")

    results = rows
    use_frame_steps = cfg.custom_eval.use_frame_steps
    if use_frame_steps:
        # Each prefix contributes only its last (leak-free) token; that stacked
        # sequence is exactly the [num_prefixes, ...] shape the compiler's
        # whole-trajectory path expects, so it is then told use_frame_steps=False.
        results = PreciseTrainer._collapse_prefix_eval_results(rows)
        use_frame_steps = False

    # The stock compiler only assigns `positive_success_acc` inside `if num_positives > 0`
    # but reads it unconditionally two lines later (compile_results.py:209-229), so a split
    # with no positive success frame dies there with a bare `UnboundLocalError`. A
    # failures-only split is exactly that case, and is a natural thing to point this at.
    # Degrade to progress-only metrics with a loud warning instead of crashing in code we
    # do not own.
    train_success_head = engine.has_success_head
    if train_success_head:
        labels = np.concatenate([np.asarray(r["success_labels"]).ravel() for r in results if "success_labels" in r])
        if labels.size == 0 or not (labels == 1).any():
            logger.warning(
                "no positive success frames in this split, so the stock metric compiler cannot "
                "produce success_auprc / positive_success_acc. Reporting progress metrics only. "
                "For a failures-only split use part 1 (dev_scripts/eval/run_precise_eval.py), "
                "which scores the success head with negative_success_acc instead."
            )
            train_success_head = False

    metrics, plots, videos, extra = run_reward_alignment_eval_per_trajectory(
        results,
        engine.config.data.progress_pred_type,
        engine.is_discrete_mode,
        engine.num_bins if engine.is_discrete_mode else None,
        results[0]["data_source"],
        use_frame_steps=use_frame_steps,
        train_success_head=train_success_head,
        last_frame_only=False,
    )
    if engine.has_success_head and not train_success_head:
        metrics["success_metrics_skipped_no_positives"] = 1.0
    return (metrics, plots, videos, extra), results


def compile_policy_ranking(engine: PreciseInferenceEngine, rows: List[Dict[str, Any]]):
    """Hand the stock policy-ranking compiler the **uncollapsed** prefix rows.

    Unlike reward alignment, no collapse happens here: ``run_policy_ranking_eval``
    groups by trajectory id itself, sorts by ``metadata["frame_step"]`` and takes
    the last token of each prefix on its own (compile_results.py:1145-1160). Passing
    it pre-collapsed rows would leave one row per trajectory and reduce every
    aggregation to the same number.
    """
    if not rows:
        raise ValueError("no samples were scored -- did the task filter remove every trajectory?")

    metrics, task_groups, task_details = run_policy_ranking_eval(
        rows,
        engine.config.data.progress_pred_type,
        engine.is_discrete_mode,
        engine.num_bins if engine.is_discrete_mode else None,
        rows[0]["data_source"],
        correlation_method="kendall",
    )
    if "error" in metrics:
        raise ValueError(
            f"policy ranking produced no metrics: {metrics['error']}. The usual cause is that "
            "successes and failures do not share a `task` string, so no task has two quality "
            "tiers and ProgressPolicyRankingSampler drops them all."
        )

    labels = {row.get("quality_label") for row in rows}
    if len(labels) < 2:
        raise ValueError(f"policy ranking needs two quality tiers; this split has only {labels}")

    # With two tiers every Kendall tau is computed over n=2 points, so it is
    # exactly `2 * ranking_acc - 1` and `kendall_rewind` collapses to +/-1. Both
    # are still reported (they are the numbers the paper quotes) but the margin
    # and the raw accuracy are what actually carry information here.
    metrics["num_quality_tiers"] = float(len(labels))
    return metrics, task_groups, task_details


def evaluate_checkpoint(
    model_path: str,
    label: Optional[str],
    cfg: PreciseBaselineEvalConfig,
    bounds: Optional[Dict[str, Any]],
) -> tuple[str, Dict[str, Dict[str, float]]]:
    """One checkpoint over every configured dataset of every configured eval type.

    Returns the engine's own label alongside the metrics. The label defaults to the
    checkpoint's modality, which is what the summary table needs -- the directory
    basenames are all `ckpt-pearson_precise_square_rbm...` and truncate to the same
    string for all three models.
    """
    engine = PreciseInferenceEngine(
        model_path=model_path,
        device=cfg.model_config.device if torch.cuda.is_available() else "cpu",
        pointmap_norm_bounds=bounds,
        num_subsampled_frames=cfg.max_frames,
        batch_size=cfg.model_config.batch_size,
        label=label,
    )
    if bounds is not None and POINTMAP not in visual_streams(engine.modality):
        logger.warning(f"pointmap_norm_bounds given but modality={engine.modality} ignores pointmaps")

    metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    for eval_type in cfg.custom_eval.eval_types:
        entries = getattr(cfg.custom_eval, eval_type)
        if not entries:
            logger.warning(f"[{engine.label}] custom_eval.{eval_type} is empty, skipping")
            continue

        run_dir = os.path.join(cfg.output_dir, engine.label, eval_type)
        os.makedirs(run_dir, exist_ok=True)
        per_dataset: Dict[str, Dict[str, float]] = {}

        for entry in entries:
            # A list entry is an explicit dataset group. Policy ranking needs one:
            # the successes and the failures of a split are separate caches that
            # must be loaded together for their shared task to have two tiers.
            dataset_names = list(entry) if isinstance(entry, (list, ListConfig)) else resolve_dataset_keys(
                [entry], split="eval"
            )
            short = _short_name("+".join(dataset_names))
            logger.info(f"[{engine.label}] {eval_type} on {dataset_names}")

            started = time.time()
            rows = score_dataset(engine, cfg, dataset_names, sampler_type=eval_type)

            plots: List[Any] = []
            if eval_type == "reward_alignment":
                (dataset_metrics, plots, video_frames_list, _), results = compile_reward_alignment(engine, cfg, rows)
                del video_frames_list
                num_trajectories = len(results)
            else:
                dataset_metrics, task_groups, task_details = compile_policy_ranking(engine, rows)
                results = {"task_groups": task_groups, "task_details": task_details}
                num_trajectories = sum(len(group) for group in task_groups.values())
            elapsed = time.time() - started

            dataset_metrics = {
                key: float(value)
                for key, value in dataset_metrics.items()
                if isinstance(value, (int, float, np.number))
            }
            dataset_metrics["num_trajectories"] = float(num_trajectories)
            dataset_metrics["seconds"] = round(elapsed, 1)
            per_dataset[short] = dataset_metrics

            with open(os.path.join(run_dir, f"{short}_results.json"), "w") as stream:
                json.dump(_json_safe(results), stream)
            with open(os.path.join(run_dir, "metrics.json"), "w") as stream:
                json.dump(_json_safe(per_dataset), stream, indent=2)

            # The stock driver renders plot+video GIFs here. Step 4 part 1 already
            # produces far better per-demo videos, so these stay as static plots.
            if plots:
                plots_dir = os.path.join(run_dir, f"{short}_plots")
                os.makedirs(plots_dir, exist_ok=True)
                for index, figure in enumerate(plots[:MAX_PLOTS]):
                    figure.savefig(os.path.join(plots_dir, f"trajectory_{index:04d}.png"), dpi=140, bbox_inches="tight")
                    plt.close(figure)
                logger.info(f"  wrote {min(len(plots), MAX_PLOTS)} plots to {plots_dir}")

            logger.info(f"  {short}: " + "  ".join(f"{k}={v:.4f}" for k, v in dataset_metrics.items()))

        metrics[eval_type] = per_dataset

    engine_label = engine.label
    del engine
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return engine_label, metrics


@hydra_main(version_base=None, config_path="../configs", config_name="precise_baseline_eval")
def main(cfg: DictConfig):
    eval_cfg = convert_hydra_to_dataclass(cfg, PreciseBaselineEvalConfig)
    display_config(eval_cfg)

    supported = {"reward_alignment", "policy_ranking"}
    unsupported = [t for t in eval_cfg.custom_eval.eval_types if t not in supported]
    if unsupported:
        raise ValueError(
            f"Precise supports {sorted(supported)} custom evaluation; got {unsupported}. "
            "confusion_matrix and quality_preference need a language branch and a preference "
            "head, neither of which this model has."
        )
    check_frame_count(eval_cfg.max_frames)

    model_paths = list(eval_cfg.model_paths or ([eval_cfg.model_path] if eval_cfg.model_path else []))
    if not model_paths:
        raise ValueError("Set model_paths=[...] (or model_path=...) to at least one Precise checkpoint")
    labels = list(eval_cfg.checkpoint_labels or [])
    if labels and len(labels) != len(model_paths):
        raise ValueError(f"checkpoint_labels has {len(labels)} entries but model_paths has {len(model_paths)}")

    bounds = eval_cfg.model_config.pointmap_norm_bounds
    if bounds is not None and not isinstance(bounds, dict):
        bounds = OmegaConf.to_container(bounds, resolve=True)

    if eval_cfg.output_dir is None:
        eval_cfg.output_dir = "./baseline_eval_output/precise"
    os.makedirs(eval_cfg.output_dir, exist_ok=True)
    logger.info(f"Output directory: {eval_cfg.output_dir}")

    all_metrics: Dict[str, Any] = {}
    for index, model_path in enumerate(model_paths):
        label = labels[index] if labels else None
        engine_label, checkpoint_metrics = evaluate_checkpoint(model_path, label, eval_cfg, bounds)
        all_metrics[engine_label] = checkpoint_metrics

    summary_path = os.path.join(eval_cfg.output_dir, "all_metrics.json")
    with open(summary_path, "w") as stream:
        json.dump(
            {
                "config": {
                    "max_frames": eval_cfg.max_frames,
                    "custom_eval": asdict(eval_cfg.custom_eval),
                    "pointmap_norm_bounds": bounds,
                    "model_paths": model_paths,
                },
                "metrics": _json_safe(all_metrics),
            },
            stream,
            indent=2,
        )
    logger.info(f"Saved summary to {summary_path}")

    def _cell(values, key):
        value = values.get(key)
        return f"{value:>10.4f}" if isinstance(value, (int, float)) else f"{'--':>10}"

    # `pearson` is the VOC number. `kendall_last` / `ranking_acc` / `succ_fail_diff`
    # only exist for policy ranking, which is the only eval that sees failures.
    columns = {
        "reward_alignment": [("pearson (VOC)", "pearson"), ("loss", "loss"), ("auprc", "success_auprc")],
        "policy_ranking": [
            ("kendall_last", "kendall_last"),
            ("rank_acc", "ranking_acc_last"),
            ("succ-fail", "avg_succ_fail_diff_last"),
        ],
    }
    for eval_type, headers in columns.items():
        rows_for_type = [
            (model_label, dataset_name, values)
            for model_label, per_type in all_metrics.items()
            for dataset_name, values in per_type.get(eval_type, {}).items()
        ]
        if not rows_for_type:
            continue
        width = 52 + 10 * len(headers)
        print("\n" + "=" * width)
        print(f"{eval_type:<22}{'dataset':<30}" + "".join(f"{name:>10}" for name, _ in headers))
        print("-" * width)
        for model_label, dataset_name, values in rows_for_type:
            print(
                f"{model_label[:21]:<22}{dataset_name[:29]:<30}"
                + "".join(_cell(values, key) for _, key in headers)
            )
        print("=" * width)
    print()
    return all_metrics


if __name__ == "__main__":
    main()
