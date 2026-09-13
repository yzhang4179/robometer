#!/usr/bin/env python3
"""Inference for :class:`PreciseTransformer` -- the shared core, plus a FastAPI server.

This is the Step 4 fork of the two functions in ``robometer/evals/eval_server.py``
that are closed against Precise:

* ``forward_model`` dispatches on ``"rewind" in model.__class__.__name__.lower()``
  and otherwise calls the model with ``input_ids`` / ``attention_mask`` /
  ``image_grid_thw``.  Precise takes none of those; it takes
  ``pixel_values_videos`` and/or ``pointmap_values`` and nothing else.
* ``process_batch_helper`` builds its growing prefixes with
  ``NUM_SUBSAMPLED_FRAMES = 4``.  **Four is not a legal Wan clip length.**  Wan's
  encoder consumes frame 0 and then whole groups of four and silently drops the
  remainder, so a 4-frame clip would reach the model as frame 0 alone with frames
  1-3 discarded and no warning.  Clip lengths here are ``4k + 1`` and are checked
  by ``check_frame_count``.

The module is split so that the offline video driver and the HTTP server run
**exactly** the same arithmetic:

    PreciseInferenceEngine   one checkpoint on one device; owns the forward pass
      .forward_precise_model(...)   <- fork of eval_server.forward_model
      .compute_batch_outputs(...)   <- fork of eval_server.compute_batch_outputs
      .predict_clips(...)           batched clips  -> per-clip [T] curves
      .predict_trajectory(...)      one trajectory -> per-frame curves
      .process_batch(...)           <- fork of eval_server.process_batch_helper

``dev_scripts/eval/run_precise_eval.py`` imports the engine directly -- no HTTP,
so the three modality checkpoints share one process and one hdf5 read -- while
the FastAPI app below wraps the same engine for online use during policy
rollouts, which is what the stock ``eval_server.py`` exists for.

Deployment convention, unchanged from training: feed the growing prefix ``0:i``
and read the **last** progress token.  That token is the only leak-free one:
Wan's encoder builds latent frame ``k`` from input frames ``1+4(k-1) .. 4k``, so
every earlier progress token in a clip has already seen frames that come after
it.
"""

# Precise uses neither Unsloth nor quantization, but importing anything under
# robometer.utils pulls the stock setup module, which imports both eagerly. Same
# preamble as train_precise.py; see its comment for the torchao/unsloth details.
# (No `from __future__ import annotations` -- it would have to precede this, and
# Python 3.10 already evaluates `X | None` and `list[int]` natively.)
import sys
import types

from transformers.utils import import_utils as transformers_import_utils

transformers_import_utils._torchao_available = False

if "unsloth" not in sys.modules:  # pragma: no cover - import-order guard

    class _UnusedFastVisionModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            del args, kwargs
            raise RuntimeError("Precise does not support the stock Unsloth model path")

    _unsloth_stub = types.ModuleType("unsloth")
    _unsloth_stub.FastVisionModel = _UnusedFastVisionModel
    sys.modules["unsloth"] = _unsloth_stub

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from robometer.configs.precise_eval_configs import SERVER_SUBSAMPLED_FRAMES
from robometer.data.datasets.helpers import linspace_subsample_frames
from robometer.models.precise.layout import POINTMAP, RGB, check_frame_count, visual_streams
from robometer.models.precise.wan_tokenizer import validate_bounds
from robometer.models.utils import convert_bins_to_continuous
from robometer.utils.logger import get_logger
from robometer.utils.precise_setup_utils import build_precise_model, load_precise_experiment_config

logger = get_logger()

# [B, T, H, W, C] -> [B, T, C, H, W]: the batch axis in front of the training
# collator's (T, H, W, C) -> (T, C, H, W) permutation.
BATCH_CHANNELS_LAST_TO_FIRST = (0, 1, 4, 2, 3)

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


# --------------------------------------------------------------- prefix indices


def build_prefix_indices(end_index: int, num_subsampled_frames: int) -> np.ndarray:
    """Frame indices for the prefix ``0:end_index`` thinned to a fixed clip length.

    This delegates to robometer's own :func:`linspace_subsample_frames` rather than
    re-deriving the arithmetic, so an eval clip contains exactly the frames a
    training clip over the same prefix would have contained -- ``np.rint`` rounding,
    first and last forced, monotonic -- and then right-pads by repeating the last
    index, which is what ``pad_trajectory_to_max_frames_np`` does for a short prefix.

    The stock server instead uses ``np.linspace(..., dtype=int)``, which truncates
    rather than rounds and so drifts from training by up to one frame per position.
    """
    check_frame_count(num_subsampled_frames)
    if end_index < 0:
        raise ValueError(f"end_index must be >= 0, got {end_index}")

    carrier = np.arange(end_index + 1, dtype=np.int64)
    _, indices = linspace_subsample_frames(carrier, num_frames=num_subsampled_frames, end_idx=end_index)
    indices = list(indices)
    if not indices:
        raise ValueError(f"no indices produced for prefix 0:{end_index}")
    # Short prefixes come back shorter than the clip; pad on the right by repeating
    # the newest frame, so the last progress token still corresponds to frame end_index.
    while len(indices) < num_subsampled_frames:
        indices.append(indices[-1])
    return np.asarray(indices, dtype=np.int64)


def frame_step_end_indices(num_frames: int, frame_stride: int = 1) -> List[int]:
    """The prefix endpoints to score: ``0, stride, 2*stride, ..., num_frames-1``.

    The last frame is always included, so the curve always reaches the end of the
    trajectory regardless of stride.
    """
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")
    if frame_stride < 1:
        raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")
    ends = list(range(0, num_frames, frame_stride))
    if ends[-1] != num_frames - 1:
        ends.append(num_frames - 1)
    return ends


# --------------------------------------------------------------------- engine


class PreciseInferenceEngine:
    """One Precise checkpoint on one device, with the Step 4 inference protocol."""

    def __init__(
        self,
        model_path: str,
        device: torch.device | str = "cuda",
        pointmap_norm_bounds: Optional[Dict[str, Sequence[float]]] = None,
        num_subsampled_frames: int = SERVER_SUBSAMPLED_FRAMES,
        batch_size: int = 32,
        autocast: bool = True,
        label: Optional[str] = None,
    ):
        self.model_path = model_path
        self.device = torch.device(device)
        self.batch_size = int(batch_size)

        self._require_checkpoint_dir(model_path)
        config_path = self._resolve_config_path(model_path)
        config = load_precise_experiment_config(config_path)

        # Bounds have to be patched between reading the config and building the
        # model: `build_precise_model` hands the same dict to both the transformer
        # (which validates it) and the WanLatentTokenizer (which does the clamping),
        # so patching here is the single point that changes normalisation.
        self.trained_bounds = dict(config.model.precise.get("pointmap_norm_bounds") or {})
        self.bounds_overridden = pointmap_norm_bounds is not None
        if self.bounds_overridden:
            config.model.precise["pointmap_norm_bounds"] = {
                axis: [float(lo), float(hi)] for axis, (lo, hi) in validate_bounds(pointmap_norm_bounds).items()
            }

        self.config = config
        self.modality = config.model.precise["modality"]
        self.streams = visual_streams(self.modality)
        self.label = label or self.modality
        self.max_len = int(config.model.precise["max_len"])

        if num_subsampled_frames > self.max_len:
            raise ValueError(
                f"num_subsampled_frames={num_subsampled_frames} exceeds the checkpoint's "
                f"model.precise.max_len={self.max_len}; the per-frame progress-token table has no row for it."
            )
        check_frame_count(num_subsampled_frames, int(config.model.precise["vae_temporal_ratio"]))
        self.num_subsampled_frames = int(num_subsampled_frames)

        self.model = build_precise_model(config.model, device=self.device, checkpoint_path=model_path, load_vae=True)
        self.model.eval()

        # Exp 3's `LangPreciseTransformer` needs the instruction embedding with every
        # clip. Dataset-driven evals get it from the collator; the trajectory API
        # (`predict_trajectory`, used for qualitative videos) takes it from
        # `set_instruction`, since an hdf5 carries frames, not language.
        self.needs_lang = bool(getattr(self.model, "use_lang_token", False))
        self.lang_vector: Optional[torch.Tensor] = None

        self.is_discrete_mode = str(config.loss.progress_loss_type).lower() == "discrete"
        self.num_bins = int(config.loss.progress_discrete_bins)
        self.has_success_head = bool(config.model.train_success_head)

        # Training ran the transformer under bf16 autocast (training.bf16=true) while
        # keeping fp32 master weights, which is what the checkpoint holds. Reproducing
        # that here is what makes these numbers comparable to the in-training pearson.
        self.autocast_dtype = DTYPES.get(str(config.model.torch_dtype), torch.bfloat16)
        self.use_autocast = bool(autocast) and self.device.type == "cuda"

        logger.info(
            f"[precise] {self.label}: modality={self.modality} clip={self.num_subsampled_frames} "
            f"bins={self.num_bins if self.is_discrete_mode else 'continuous'} "
            f"success_head={self.has_success_head} device={self.device} "
            f"bounds={'OVERRIDDEN ' + str(config.model.precise['pointmap_norm_bounds']) if self.bounds_overridden else 'as trained'}"
        )

    # ------------------------------------------------------------------ loading

    @staticmethod
    def _require_checkpoint_dir(model_path: str) -> None:
        """Fail here, with the available checkpoints, rather than inside `from_pretrained`.

        A missing local directory is not detected by `PreTrainedModel.from_pretrained`:
        it falls through to the Hub and reports
        `HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name'`,
        which says nothing about the real problem. Best-metric checkpoint directories are
        rotated by `save_best` (only the top `keep_top_k` survive), so a path that worked
        yesterday can genuinely be gone today.
        """
        if os.path.isdir(model_path):
            return
        parent = os.path.dirname(os.path.normpath(model_path)) or "."
        available = sorted(
            entry for entry in os.listdir(parent) if os.path.isdir(os.path.join(parent, entry))
        ) if os.path.isdir(parent) else []
        hint = ("\n  available in %s:\n    %s" % (parent, "\n    ".join(available))) if available else ""
        raise FileNotFoundError(
            f"Precise checkpoint directory does not exist: {model_path}\n"
            "  Best-metric checkpoints are rotated by save_best, so this name may have been "
            "deleted as training improved. `final/` is stable." + hint
        )

    @staticmethod
    def _resolve_config_path(model_path: str) -> str:
        """The experiment ``config.yaml`` lives in the run dir, not in every ckpt dir."""
        candidates = (
            os.path.join(model_path, "config.yaml"),
            os.path.join(os.path.dirname(os.path.normpath(model_path)), "config.yaml"),
        )
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError(f"No config.yaml in {model_path} or its parent -- is this a Precise checkpoint?")

    @property
    def needs_rgb(self) -> bool:
        return RGB in self.streams

    @property
    def needs_pointmap(self) -> bool:
        return POINTMAP in self.streams

    def set_instruction(self, lang_vector: Optional[np.ndarray]) -> None:
        """Fix the instruction embedding used by `predict_clips` / `predict_trajectory`."""
        if lang_vector is None:
            self.lang_vector = None
            return
        vector = torch.as_tensor(np.asarray(lang_vector, dtype=np.float32)).reshape(-1)
        self.lang_vector = vector.to(self.device)

    # ------------------------------------------------------------ forward passes

    def forward_precise_model(self, batch_inputs: Dict[str, torch.Tensor], sample_type: str = "progress"):
        """Fork of ``eval_server.forward_model`` for the Precise call signature."""
        if sample_type != "progress":
            raise ValueError(f"PreciseTransformer is progress-only, got sample_type={sample_type!r}")

        model_kwargs: Dict[str, Any] = {"sample_type": "progress", "timing_raw": None}
        if self.needs_rgb:
            rgb = batch_inputs.get("pixel_values_videos")
            if rgb is None:
                raise KeyError(f"modality={self.modality!r} requires 'pixel_values_videos'")
            model_kwargs["pixel_values_videos"] = rgb
        if self.needs_pointmap:
            pointmap = batch_inputs.get("pointmap_values")
            if pointmap is None:
                raise KeyError(f"modality={self.modality!r} requires 'pointmap_values'")
            model_kwargs["pointmap_values"] = pointmap
        if self.needs_lang:
            lang_vector = batch_inputs.get("lang_vector")
            if lang_vector is None:
                raise KeyError(
                    "this checkpoint has use_lang_token=true and needs 'lang_vector' in the batch; "
                    "for the trajectory API call engine.set_instruction(embedding) first"
                )
            model_kwargs["lang_vector"] = lang_vector

        with torch.inference_mode():
            if self.use_autocast:
                with torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype):
                    model_output, extra = self.model(**model_kwargs)
            else:
                model_output, extra = self.model(**model_kwargs)
        return model_output, extra

    def compute_batch_outputs(self, batch_inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Fork of ``eval_server.compute_batch_outputs``: logits -> lists of floats.

        Returns full ``[T]`` sequences per clip.  Callers that follow the growing-prefix
        protocol keep only ``[-1]``; ``predict_clips`` does that for them.
        """
        model_output, _ = self.forward_precise_model(batch_inputs)

        progress_logits = model_output.progress_logits["A"]
        if self.is_discrete_mode:
            progress = convert_bins_to_continuous(progress_logits.detach().float().cpu())
        else:
            progress = progress_logits.detach().float().cpu()
        results: Dict[str, Any] = {"progress_pred": [row.tolist() for row in progress.numpy()]}

        if self.has_success_head and model_output.success_logits is not None:
            success = torch.sigmoid(model_output.success_logits["A"].detach().float().cpu())
            results["outputs_success"] = {"success_probs": [row.tolist() for row in success.numpy()]}
        return results

    # ------------------------------------------------------------- clip batching

    def _stack_clips(
        self,
        frames: Optional[np.ndarray],
        pointmap: Optional[np.ndarray],
        clip_indices: np.ndarray,
    ) -> Dict[str, torch.Tensor]:
        """``clip_indices`` ``[B, T]`` of frame indices -> the model's input tensors.

        Both streams are gathered with the *same* index matrix, so the RGB clip and
        the pointmap clip cannot describe different frames -- the same guarantee the
        training sampler gets from its index carrier.
        """
        batch_inputs: Dict[str, torch.Tensor] = {}
        if self.needs_rgb:
            if frames is None:
                raise ValueError(f"modality={self.modality!r} needs RGB frames but none were supplied")
            clips = frames[clip_indices]  # [B, T, H, W, 3] uint8
            clips = np.ascontiguousarray(clips.transpose(BATCH_CHANNELS_LAST_TO_FIRST))
            batch_inputs["pixel_values_videos"] = torch.from_numpy(clips).to(self.device, non_blocking=True)
        if self.needs_pointmap:
            if pointmap is None:
                raise ValueError(f"modality={self.modality!r} needs a pointmap but none was supplied")
            clips = pointmap[clip_indices]  # [B, T, H, W, 3] raw camera-frame metres
            clips = np.ascontiguousarray(clips.transpose(BATCH_CHANNELS_LAST_TO_FIRST))
            batch_inputs["pointmap_values"] = torch.from_numpy(clips).to(self.device, non_blocking=True)
        if self.needs_lang and self.lang_vector is not None:
            batch_inputs["lang_vector"] = self.lang_vector[None].expand(clip_indices.shape[0], -1)
        return batch_inputs

    def predict_clips(
        self,
        frames: Optional[np.ndarray],
        pointmap: Optional[np.ndarray],
        clip_indices: np.ndarray,
        batch_size: Optional[int] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Score ``[B, T]`` frame indices -> ``(progress [B, T], success [B, T] | None)``."""
        clip_indices = np.asarray(clip_indices, dtype=np.int64)
        if clip_indices.ndim != 2:
            raise ValueError(f"clip_indices must be [B, T], got shape {clip_indices.shape}")
        batch_size = int(batch_size or self.batch_size)

        progress_rows: List[List[float]] = []
        success_rows: List[List[float]] = []
        for start in range(0, clip_indices.shape[0], batch_size):
            chunk = clip_indices[start : start + batch_size]
            outputs = self.compute_batch_outputs(self._stack_clips(frames, pointmap, chunk))
            progress_rows.extend(outputs["progress_pred"])
            if "outputs_success" in outputs:
                success_rows.extend(outputs["outputs_success"]["success_probs"])

        progress = np.asarray(progress_rows, dtype=np.float32)
        success = np.asarray(success_rows, dtype=np.float32) if success_rows else None
        return progress, success

    def predict_trajectory(
        self,
        frames: Optional[np.ndarray],
        pointmap: Optional[np.ndarray],
        frame_stride: int = 1,
        batch_size: Optional[int] = None,
        num_subsampled_frames: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """The growing-prefix protocol over one whole trajectory.

        For every endpoint ``i`` this builds the clip for prefix ``0:i`` and keeps
        the **last** progress/success token -- the leak-free one, and the only one
        available at rollout time.

        ``num_subsampled_frames`` overrides the clip length for this call only, so
        one loaded checkpoint can produce both the 5-frame and 9-frame curves.
        Note these are *not* the same clip: over the prefix ``0:i`` the 5-frame clip
        spans the same window at roughly half the temporal density, so it is a
        coarser view rather than a truncation of the 9-frame clip.

        Returns ``end_indices``, ``progress`` and (if the checkpoint has a success
        head) ``success``, all indexed by endpoint.
        """
        clip_len = self.num_subsampled_frames if num_subsampled_frames is None else int(num_subsampled_frames)
        if clip_len > self.max_len:
            raise ValueError(f"num_subsampled_frames={clip_len} exceeds the checkpoint's max_len={self.max_len}")
        num_frames = int((frames if frames is not None else pointmap).shape[0])
        ends = frame_step_end_indices(num_frames, frame_stride)
        clip_indices = np.stack([build_prefix_indices(end, clip_len) for end in ends])

        progress, success = self.predict_clips(frames, pointmap, clip_indices, batch_size=batch_size)
        result = {
            "end_indices": np.asarray(ends, dtype=np.int64),
            "progress": progress[:, -1].astype(np.float32),
            "clip_indices": clip_indices,
        }
        if success is not None:
            result["success"] = success[:, -1].astype(np.float32)
        return result

    # ------------------------------------------------------- server-facing batch

    def process_batch(
        self,
        batch_data: List[Dict[str, Any]],
        use_frame_steps: bool = True,
        frame_stride: int = 1,
    ) -> Dict[str, Any]:
        """Fork of ``eval_server.process_batch_helper`` for JSON/npy payloads.

        Each sample is ``{"sample_type": "progress", "trajectory": {"frames": ...,
        "pointmap": ...}}``.  ``pointmap`` is the one addition to the stock payload
        and is required only by the pointmap-consuming modalities.
        """
        if not batch_data:
            raise ValueError("No samples found in batch data")

        progress_pred: List[List[float]] = []
        success_probs: List[List[float]] = []
        metadata: List[Dict[str, Any]] = []

        for sample in batch_data:
            trajectory = sample.get("trajectory") if isinstance(sample, dict) else None
            if trajectory is None:
                raise ValueError(f"sample has no 'trajectory' key: {sorted(sample) if isinstance(sample, dict) else type(sample)}")
            sample_type = sample.get("sample_type", "progress")
            if sample_type != "progress":
                raise ValueError(f"PreciseTransformer is progress-only, got sample_type={sample_type!r}")

            frames = _as_frame_array(trajectory.get("frames"), np.uint8) if self.needs_rgb else None
            pointmap = _as_frame_array(trajectory.get("pointmap"), np.float16) if self.needs_pointmap else None
            if self.needs_lang:
                self.set_instruction(trajectory.get("lang_vector"))
            if self.needs_pointmap and pointmap is None:
                raise ValueError(
                    f"modality={self.modality!r} needs trajectory['pointmap'] (T, H, W, 3) raw camera-frame metres; "
                    "the payload had none."
                )

            if use_frame_steps:
                out = self.predict_trajectory(frames, pointmap, frame_stride=frame_stride)
                progress_pred.append(out["progress"].tolist())
                if "success" in out:
                    success_probs.append(out["success"].tolist())
                metadata.append({"id": trajectory.get("id"), "end_indices": out["end_indices"].tolist()})
            else:
                # Whole-clip mode: one clip covering the trajectory, all T tokens returned.
                num_frames = int((frames if frames is not None else pointmap).shape[0])
                clip = build_prefix_indices(num_frames - 1, self.num_subsampled_frames)[None]
                progress, success = self.predict_clips(frames, pointmap, clip)
                progress_pred.append(progress[0].tolist())
                if success is not None:
                    success_probs.append(success[0].tolist())
                metadata.append({"id": trajectory.get("id"), "end_indices": clip[0].tolist()})

        return {
            "outputs_preference": None,
            "outputs_progress": {"progress_pred": progress_pred, "metadata": metadata},
            "outputs_success": {"success_probs": success_probs} if success_probs else None,
        }


def _as_frame_array(value: Any, dtype) -> Optional[np.ndarray]:
    """Payload frames/pointmap -> a contiguous ``[T, H, W, 3]`` array of ``dtype``."""
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(f"expected [T, H, W, 3], got shape {array.shape}")
    return np.ascontiguousarray(array.astype(dtype, copy=False))


def load_engines(
    model_paths: Dict[str, str],
    device: torch.device | str = "cuda",
    **kwargs,
) -> Dict[str, PreciseInferenceEngine]:
    """``{label: checkpoint}`` -> ``{label: engine}``, in the given order."""
    return {
        label: PreciseInferenceEngine(path, device=device, label=label, **kwargs)
        for label, path in model_paths.items()
    }


# ---------------------------------------------------------------- FastAPI app


def create_app(engine: "PreciseInferenceEngine", frame_stride: int = 1):
    """Wrap one engine in the same endpoints the stock RBM eval server exposes.

    The stock server keeps a pool of deep-copied replicas; Precise loads one
    replica per process instead, because a deep copy would also duplicate the
    frozen ~150M-parameter Wan encoder that the replicas could share.
    """
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware

    from robometer.evals.eval_utils import parse_npy_form_data, reconstruct_payload_from_npy

    app = FastAPI(title="Precise Robometer Evaluation Server")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _run(samples: List[Dict[str, Any]], use_frame_steps: bool, stride: int) -> Dict[str, Any]:
        return engine.process_batch(samples, use_frame_steps=use_frame_steps, frame_stride=stride)

    @app.post("/evaluate_batch")
    async def evaluate_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
        samples = batch.get("samples", batch if isinstance(batch, list) else [])
        use_frame_steps = bool(batch.get("use_frame_steps", True))
        return _run(samples, use_frame_steps, int(batch.get("frame_stride", frame_stride)))

    @app.post("/evaluate_batch_npy")
    async def evaluate_batch_npy(request: Request) -> Dict[str, Any]:
        numpy_arrays, other_data = await parse_npy_form_data(await request.form())
        # Both flags must come out before reconstruct_payload_from_npy, which walks
        # `range(len(other_data))` looking for sample_0, sample_1, ... keys.
        raw_flag = other_data.pop("use_frame_steps", True)
        use_frame_steps = raw_flag if isinstance(raw_flag, bool) else str(raw_flag).lower() == "true"
        stride = int(other_data.pop("frame_stride", frame_stride))
        samples = reconstruct_payload_from_npy(
            numpy_arrays, other_data, trajectory_keys=["trajectory"], convert_embeddings_to_torch=False
        )
        return _run(samples, use_frame_steps, stride)

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {"status": "healthy", "modality": engine.modality, "device": str(engine.device)}

    @app.get("/model_info")
    def model_info() -> Dict[str, Any]:
        precise = engine.config.model.precise
        return {
            "model_path": engine.model_path,
            "modality": engine.modality,
            "num_subsampled_frames": engine.num_subsampled_frames,
            "default_frame_stride": frame_stride,
            "max_len": engine.max_len,
            "progress_loss_type": engine.config.loss.progress_loss_type,
            "progress_discrete_bins": engine.num_bins,
            "train_success_head": engine.has_success_head,
            "pointmap_norm_bounds": precise.get("pointmap_norm_bounds"),
            "pointmap_norm_bounds_overridden": engine.bounds_overridden,
            "trainable_parameters": sum(p.numel() for p in engine.model.parameters() if p.requires_grad),
        }

    return app


def main():
    """Hydra entry point: ``uv run python robometer/evals/precise_eval_server.py model_path=...``"""
    import uvicorn
    from hydra import main as hydra_main
    from omegaconf import DictConfig, OmegaConf

    from robometer.configs.precise_eval_configs import PreciseEvalServerConfig
    from robometer.utils.config_utils import convert_hydra_to_dataclass, display_config

    @hydra_main(version_base=None, config_path="../configs", config_name="precise_eval_server")
    def _main(cfg: DictConfig):
        server_cfg = convert_hydra_to_dataclass(cfg, PreciseEvalServerConfig)
        display_config(server_cfg)
        if not server_cfg.model_path:
            raise ValueError("Set model_path to a Precise checkpoint directory")

        bounds = server_cfg.pointmap_norm_bounds
        if bounds is not None and not isinstance(bounds, dict):
            bounds = OmegaConf.to_container(bounds, resolve=True)

        engine = PreciseInferenceEngine(
            model_path=server_cfg.model_path,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            pointmap_norm_bounds=bounds,
            num_subsampled_frames=server_cfg.num_subsampled_frames,
            batch_size=server_cfg.batch_size,
        )
        app = create_app(engine)
        print(f"Precise eval server ({engine.modality}) on {server_cfg.server_url}:{server_cfg.server_port}")
        uvicorn.run(app, host=server_cfg.server_url, port=server_cfg.server_port)

    _main()


__all__ = [
    "PreciseInferenceEngine",
    "build_prefix_indices",
    "create_app",
    "frame_step_end_indices",
    "load_engines",
]


if __name__ == "__main__":
    main()
