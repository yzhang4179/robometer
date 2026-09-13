#!/usr/bin/env python3
"""Trainer integration for :class:`PreciseTransformer`.

The stock RBM trainer owns the useful training loop, metric aggregation, custom
evaluation loop, and checkpoint plumbing.  Its three runtime dispatches are
closed over the stock model/collator/dataset types, however, so Precise replaces
those seams and keeps the rest.

There are two small numerical corrections here as well:

* progress CE is reduced over valid *tokens* (the stock implementation first
  averages over time and then divides by the number of valid tokens again);
* success BCE keeps its inverse-frequency weights instead of dividing them back
  out immediately.

Reward-alignment prefixes are evaluated using the deployment convention: take
the last progress token from every growing prefix.  They are collapsed into one
sequence per trajectory before entering the stock result compiler.  That makes
its discrete CE input ``[num_prefixes, num_bins]`` and aligns Pearson with the
same last-token sequence.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from robometer.data.collators.precise import PreciseBatchCollator
from robometer.data.datasets.precise_data import PreciseCustomEvalDataset
from robometer.data.samplers.precise_progress import POINTMAP_STREAM, RGB_STREAM, precise_streams
from robometer.models.utils import convert_discrete_target_to_continuous
from robometer.trainers.rbm_heads_trainer import RBMHeadsTrainer, seed_worker
from robometer.utils.logger import get_logger
from robometer.utils.timer import _timer

logger = get_logger()


class PreciseTrainer(RBMHeadsTrainer):
    """Progress-only RBM trainer for RGB, pointmap, and fused Precise models."""

    # ------------------------------------------------------------------ dispatch

    def _precise_modality(self) -> str:
        """Read and validate the modality from the outer experiment config."""
        precise = getattr(self.config.model, "precise", None)
        if precise is None:
            raise ValueError("PreciseTrainer requires config.model.precise")
        modality = precise.get("modality") if isinstance(precise, dict) else precise.modality
        # ``precise_streams`` is the shared source of truth for valid names.
        precise_streams(modality)
        return modality

    def _use_lang_token(self) -> bool:
        precise = getattr(self.config.model, "precise", None) or {}
        if isinstance(precise, dict):
            return bool(precise.get("use_lang_token", False))
        return bool(getattr(precise, "use_lang_token", False))

    def forward_model(self, model, inputs, sample_type="progress"):
        """Call Precise with only the visual streams selected by ``modality``."""
        if sample_type != "progress":
            raise ValueError(
                "PreciseTrainer is progress-only; set data.sample_type_ratio=[0, 1, 0]. "
                f"Got sample_type={sample_type!r}."
            )

        modality = self._precise_modality()
        streams = precise_streams(modality)
        model_kwargs = {
            "sample_type": sample_type,
            "timing_raw": self.timing_raw,
        }
        if RGB_STREAM in streams:
            rgb = inputs.get("pixel_values_videos")
            if rgb is None:
                raise KeyError(f"modality={modality!r} requires 'pixel_values_videos'")
            model_kwargs["pixel_values_videos"] = rgb
        if POINTMAP_STREAM in streams:
            pointmap = inputs.get("pointmap_values")
            if pointmap is None:
                raise KeyError(f"modality={modality!r} requires 'pointmap_values'")
            model_kwargs["pointmap_values"] = pointmap
        if self._use_lang_token():
            lang_vector = inputs.get("lang_vector")
            if lang_vector is None:
                raise KeyError("model.precise.use_lang_token=true requires 'lang_vector' in the batch")
            model_kwargs["lang_vector"] = lang_vector

        with _timer("time/forward", timing_raw=self.timing_raw):
            model_output, model_timing_raw = model(**model_kwargs)
        if model_timing_raw is not None:
            self.timing_raw.update(model_timing_raw)
        return model_output, model_timing_raw

    def _make_eval_dataloader(self, dataset):
        """Build an eval loader without the stock processor/tokenizer dispatch."""
        modality = self._precise_modality()
        collator = PreciseBatchCollator(
            processor=None,
            modality=modality,
            require_pointmap=POINTMAP_STREAM in precise_streams(modality),
            resized_height=self.config.data.resized_height,
            resized_width=self.config.data.resized_width,
            base_model_id=self.config.model.base_model_id,
            load_embeddings=self.config.data.load_embeddings,
            use_multi_image=self.config.data.use_multi_image,
            prog_pref=self.config.training.predict_pref_progress,
            use_per_frame_progress_token=getattr(self.config.data, "use_per_frame_progress_token", False),
            shuffle_progress_frames=self.config.data.shuffle_progress_frames,
            inference=True,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.training.per_device_eval_batch_size,
            collate_fn=collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=False,
            # A fresh custom-eval loader is made for every dataset/eval pass.  Do
            # not retain workers (and their mmap/npz buffers) after it is deleted.
            persistent_workers=False,
            worker_init_fn=seed_worker,
        )
        return self.accelerator.prepare(dataloader)

    def _setup_eval_dataset(self, eval_type, eval_dataset):
        """Create the pointmap-aware, reward-alignment-only eval dataset."""
        if eval_type != "reward_alignment":
            raise ValueError(
                "PreciseTrainer only supports reward_alignment custom evaluation; "
                f"got {eval_type!r}. Set custom_eval.eval_types=[reward_alignment]."
            )

        eval_cfg = copy.deepcopy(self.config.data)
        eval_cfg.eval_datasets = eval_dataset if isinstance(eval_dataset, list) else [eval_dataset]
        modality = self._precise_modality()
        sampler_kwargs = {
            "random_seed": self.config.custom_eval.custom_eval_random_seed,
            "max_trajectories": self.config.custom_eval.reward_alignment_max_trajectories,
            # Precise scores every original prefix.  In particular, do not inherit
            # RBMHeadsTrainer's frame_step=2 special case for video VLMs.
            "frame_step": 1,
            "use_frame_steps": self.config.custom_eval.use_frame_steps,
            "subsample_n_frames": self.config.custom_eval.subsample_n_frames,
            "pad_frames": self.config.custom_eval.pad_frames,
        }
        dataset = PreciseCustomEvalDataset(
            sampler_type="reward_alignment",
            config=eval_cfg,
            verbose=False,
            sampler_kwargs=sampler_kwargs,
            modality=modality,
            require_pointmap=POINTMAP_STREAM in precise_streams(modality),
        )
        del eval_cfg

        logger.info(f"  Dataset size: {len(dataset)}")
        dataloader = self._make_eval_dataloader(dataset)
        logger.info(f"  Dataloader created with {len(dataloader)} batches")

        self.model.eval()
        if getattr(self, "optimizer", None) is not None:
            self.optimizer.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        return dataset, dataloader

    # -------------------------------------------------------------- loss fixes

    def _effective_progress_mask(
        self,
        mask: torch.Tensor,
        predict_last_frame_mask: torch.Tensor | None,
        sequence_length: int,
    ) -> torch.Tensor:
        """Reproduce the parent's mask rules, expanded to ``[B, T]``."""
        effective = mask
        if effective.ndim == 1:
            effective = effective.unsqueeze(-1)
        if effective.shape[1] == 1:
            effective = effective.expand(-1, sequence_length)
        elif effective.shape[1] != sequence_length:
            raise ValueError(
                f"progress mask length {effective.shape[1]} does not match prediction length {sequence_length}"
            )

        if predict_last_frame_mask is not None:
            last_mask = predict_last_frame_mask
            if last_mask.ndim == 1:
                last_mask = last_mask.unsqueeze(-1)
            if last_mask.shape[1] == 1:
                last_mask = last_mask.expand(-1, sequence_length)
            elif last_mask.shape[1] != sequence_length:
                raise ValueError(
                    "predict_last_frame_mask length "
                    f"{last_mask.shape[1]} does not match prediction length {sequence_length}"
                )
            effective = effective * last_mask.to(device=effective.device, dtype=effective.dtype)
        elif self.config.loss.predict_last_frame_progress:
            last_only = torch.zeros_like(effective)
            last_only[:, -1] = 1
            effective = effective * last_only
        return effective

    def _compute_progress_loss_helper(
        self,
        progress_pred: torch.Tensor,
        target_progress: torch.Tensor,
        mask: torch.Tensor,
        predict_last_frame_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Use the stock per-token metrics but reduce them over valid tokens once."""
        effective_last_frame_mask = predict_last_frame_mask
        if effective_last_frame_mask is None and self.config.loss.predict_last_frame_progress:
            # Supplying this explicitly also avoids the parent's C51 fallback
            # creating a three-dimensional mask from ``target_progress``.
            effective_last_frame_mask = torch.zeros(
                progress_pred.shape[:2], device=mask.device, dtype=mask.dtype
            )
            effective_last_frame_mask[:, -1] = 1
        _, spearman_corr, metrics = super()._compute_progress_loss_helper(
            progress_pred,
            target_progress,
            mask,
            predict_last_frame_mask=effective_last_frame_mask,
        )
        masked_loss = metrics["masked_loss"]
        effective_mask = self._effective_progress_mask(
            mask, effective_last_frame_mask, sequence_length=masked_loss.shape[1]
        ).to(device=masked_loss.device, dtype=masked_loss.dtype)

        denominator = effective_mask.sum().clamp_min(1e-8)
        progress_loss = masked_loss.sum() / denominator

        # These are computed from the same logits/forward as the primary all-frame
        # objective.  ``-1`` is PreciseModelOutput.last_token_index for every clip.
        last_mask = effective_mask[:, -1]
        last_denominator = last_mask.sum().clamp_min(1e-8)
        metrics["last_progress_loss"] = masked_loss[:, -1].sum() / last_denominator
        self._precise_last_progress_metrics = {
            "loss": metrics["last_progress_loss"].detach().item(),
        }
        if "masked_progress_accuracy" in metrics:
            metrics["last_progress_accuracy"] = (
                metrics["masked_progress_accuracy"][:, -1].sum() / last_denominator
            )
            all_accuracy = metrics["masked_progress_accuracy"].sum() / denominator
            self._precise_last_progress_metrics.update({
                "accuracy": metrics["last_progress_accuracy"].detach().item(),
                "all_accuracy": all_accuracy.detach().item(),
            })
        return progress_loss, spearman_corr, metrics

    def _compute_progress_loss(self, model, inputs, return_outputs=False, training=True, stratify_by_strategy=True):
        """Add last-token CE/accuracy beside the inherited all-frame metrics."""
        self._precise_last_progress_metrics = None
        result = super()._compute_progress_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            training=training,
            stratify_by_strategy=stratify_by_strategy,
        )
        if not return_outputs:
            return result

        loss, outputs = result
        # The helper ran exactly once in this progress-only path.  Recompute only
        # the two scalar reductions from the already-produced output metrics by
        # caching them during that call; never issue a second model forward.
        last_metrics = getattr(self, "_precise_last_progress_metrics", None)
        if last_metrics is not None:
            prefix = "train" if training else "eval"
            outputs[f"{prefix}/last_prog_loss"] = last_metrics["loss"]
            if "accuracy" in last_metrics:
                outputs[f"{prefix}/last_prog_accuracy"] = last_metrics["accuracy"]
                # Replace the parent's denominator, which does not include a
                # per-token predict_last_frame_mask when one is configured.
                outputs[f"{prefix}/prog_accuracy"] = last_metrics["all_accuracy"]
        return loss, outputs

    def _compute_success_loss_helper(
        self,
        success_logits,
        target_progress,
        success_labels,
        progress_loss_mask=None,
        quality_labels=None,
    ):
        """Return a genuinely class-balanced BCE while preserving stock metrics."""
        _, success_acc, batch_auprc, metrics = super()._compute_success_loss_helper(
            success_logits,
            target_progress,
            success_labels,
            progress_loss_mask=progress_loss_mask,
            quality_labels=quality_labels,
        )

        target_continuous = target_progress
        if self.config.loss.progress_loss_type.lower() == "discrete":
            target_continuous = convert_discrete_target_to_continuous(
                target_progress, num_bins=self.config.loss.progress_discrete_bins
            )
        combined_mask = ((target_continuous < self.config.data.min_success) | (success_labels > 0.5)).to(
            dtype=success_logits.dtype
        )

        if quality_labels is not None:
            quality_mask = torch.zeros_like(combined_mask)
            for index, quality_label in enumerate(quality_labels):
                if quality_label is not None and quality_label.lower() in ("suboptimal", "failure", "failed"):
                    quality_mask[index] = 1
            combined_mask = torch.maximum(combined_mask, quality_mask)

        labels = success_labels.to(dtype=success_logits.dtype)
        num_positives = (labels * combined_mask).sum()
        num_negatives = ((1 - labels) * combined_mask).sum()
        weights = combined_mask
        if num_positives > 0 and num_negatives > 0:
            if num_positives < num_negatives:
                balance = (num_negatives / num_positives).detach()
                weights = torch.where(labels > 0.5, balance * combined_mask, combined_mask)
            else:
                balance = (num_positives / num_negatives).detach()
                weights = torch.where(labels > 0.5, combined_mask, balance * combined_mask)

        plain_bce = F.binary_cross_entropy_with_logits(
            torch.clamp(success_logits, min=-50.0, max=50.0),
            labels,
            reduction="none",
        )
        success_loss = (plain_bce * weights).sum() / weights.sum().clamp_min(1e-8)
        metrics["masked_loss"] = plain_bce * weights
        return success_loss, success_acc, batch_auprc, metrics

    # -------------------------------------------------------- eval result repair

    @staticmethod
    def _as_numpy(value):
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @classmethod
    def _collapse_prefix_eval_results(cls, eval_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """One row per trajectory, containing the last token from every prefix."""
        grouped: OrderedDict[str, list[tuple[int, dict[str, Any]]]] = OrderedDict()
        for insertion_index, result in enumerate(eval_results):
            trajectory_id = result.get("id")
            if trajectory_id is None:
                raise KeyError("reward-alignment result is missing trajectory id")
            grouped.setdefault(trajectory_id, []).append((insertion_index, result))

        collapsed = []
        sequence_fields = ("progress_pred", "target_progress", "success_pred", "success_probs", "success_labels")
        for trajectory_id, indexed_results in grouped.items():
            def prefix_order(item):
                insertion_index, result = item
                frame_step = result.get("metadata", {}).get("frame_step")
                return insertion_index if frame_step is None else frame_step

            indexed_results.sort(
                key=lambda item: (prefix_order(item), item[0])
            )
            prefix_results = [item[1] for item in indexed_results]
            aggregate = dict(prefix_results[-1])
            aggregate["metadata"] = dict(aggregate.get("metadata") or {})
            aggregate["metadata"]["prefix_frame_steps"] = [
                result.get("metadata", {}).get("frame_step") for result in prefix_results
            ]
            aggregate["id"] = trajectory_id

            for field in sequence_fields:
                present = [result for result in prefix_results if field in result and result[field] is not None]
                if not present:
                    aggregate.pop(field, None)
                    continue
                if len(present) != len(prefix_results):
                    raise ValueError(f"trajectory {trajectory_id!r} has {field!r} on only some prefixes")
                last_tokens = []
                for result in prefix_results:
                    values = cls._as_numpy(result[field])
                    if values.ndim == 0 or values.shape[0] == 0:
                        raise ValueError(
                            f"trajectory {trajectory_id!r} prefix has invalid {field} shape {values.shape}"
                        )
                    last_tokens.append(values[-1])
                aggregate[field] = np.stack(last_tokens, axis=0)
            collapsed.append(aggregate)
        return collapsed

    def _compute_and_log_eval_metrics(self, eval_type, eval_results, ds_name, eval_step):
        """Feed the stock compiler a correctly shaped last-token prefix sequence."""
        if eval_type != "reward_alignment":
            raise ValueError(f"PreciseTrainer only supports reward_alignment, got {eval_type!r}")
        if not self.config.custom_eval.use_frame_steps:
            return super()._compute_and_log_eval_metrics(eval_type, eval_results, ds_name, eval_step)

        collapsed = self._collapse_prefix_eval_results(eval_results)
        # The compiler's non-frame-step path consumes one full [T, C] sequence per
        # result, which is exactly the representation above.  Restore the config
        # even if plotting or metric compilation raises.
        original_use_frame_steps = self.config.custom_eval.use_frame_steps
        self.config.custom_eval.use_frame_steps = False
        try:
            return super()._compute_and_log_eval_metrics(eval_type, collapsed, ds_name, eval_step)
        finally:
            self.config.custom_eval.use_frame_steps = original_use_frame_steps


__all__ = ["PreciseTrainer"]
