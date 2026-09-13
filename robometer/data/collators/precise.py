#!/usr/bin/env python3
"""Modality-aware batch collator for the precise model.

No image processor and no text tokenizer are involved. The precise model has no
text token in its sequence (single task), and both visual streams are normalised on
GPU inside `WanLatentTokenizer` -- RGB by `x / 127.5 - 1`, pointmaps by the clamp to
`pointmap_norm_bounds`. Doing either here would burn dataloader-worker CPU and push
float32 through `pin_memory` instead of uint8 / float16.

The tensors for the requested modality leave this collator exactly as they came
off disk (unused streams are not stacked or transferred):

    pixel_values_videos  [B, T, 3, H, W]  uint8    0-255
    pointmap_values      [B, T, 3, H, W]  float16  RAW CAMERA-FRAME METRES

All the progress/success/mask bookkeeping is inherited from `RBMBatchCollator`.
"""

from __future__ import annotations

import numpy as np
import torch

from robometer.data.collators.rbm_heads import RBMBatchCollator
from robometer.data.dataset_types import PreferenceSample, ProgressSample
from robometer.data.samplers.precise_progress import (
    POINTMAP_KEY,
    POINTMAP_STREAM,
    RGB_STREAM,
    precise_streams,
)

CHANNELS_LAST_TO_FIRST = (0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)


class PreciseBatchCollator(RBMBatchCollator):
    """Stack only the configured visual streams; inherit target bookkeeping."""

    def __init__(
        self,
        processor=None,
        modality: str = "rgb_pointmap",
        require_pointmap: bool | None = None,
        **kwargs,
    ):
        super().__init__(processor=processor, **kwargs)
        self.modality = modality
        self.streams = precise_streams(modality)
        self.include_rgb = RGB_STREAM in self.streams
        self.include_pointmap = POINTMAP_STREAM in self.streams
        self.require_pointmap = self.include_pointmap if require_pointmap is None else bool(require_pointmap)
        if self.include_pointmap and not self.require_pointmap:
            raise ValueError(
                f"modality={modality!r} requires pointmaps; require_pointmap cannot be false."
            )
        if self.load_embeddings:
            raise ValueError("PreciseBatchCollator needs raw frames; set data.load_embeddings=false")

    def _process_preference_batch(self, preference_samples: list[PreferenceSample]):
        raise NotImplementedError(
            "The precise model is progress-only; set data.sample_type_ratio=[0, 1, 0]."
        )

    def _process_progress_batch(self, progress_samples: list[ProgressSample]) -> dict[str, torch.Tensor]:
        batch_inputs = {}

        if self.include_rgb:
            frames = np.stack(
                [np.asarray(s.trajectory.frames).transpose(CHANNELS_LAST_TO_FIRST) for s in progress_samples]
            )
            batch_inputs["pixel_values_videos"] = torch.from_numpy(frames).contiguous()

        if self.include_pointmap:
            pointmaps = [(s.trajectory.metadata or {}).get(POINTMAP_KEY) for s in progress_samples]
            missing = [s.trajectory.id for s, pointmap in zip(progress_samples, pointmaps) if pointmap is None]
            if missing:
                raise ValueError(
                    f"{len(missing)} sample(s) in this batch have no pointmap (e.g. {missing[:3]}). "
                    "They must come from PreciseProgressSampler on a cache built by preprocess_precise.py."
                )
            stacked = np.stack([np.asarray(p).transpose(CHANNELS_LAST_TO_FIRST) for p in pointmaps])
            batch_inputs["pointmap_values"] = torch.from_numpy(stacked).contiguous()

        # `_add_progress_meta` does not forward lang_vector, and `LangPreciseTransformer`
        # needs it. Stack it only when every sample has one, so an older cache whose
        # column is absent simply produces no key.
        lang_vectors = [s.trajectory.lang_vector for s in progress_samples]
        if all(vector is not None for vector in lang_vectors):
            batch_inputs["lang_vector"] = torch.from_numpy(
                np.stack([np.asarray(vector, dtype=np.float32) for vector in lang_vectors])
            )

        if progress_samples[0].trajectory.target_progress is not None:
            batch_inputs = self._add_progress_meta(batch_inputs, progress_samples)

        # `_add_progress_meta` copies `trajectory.metadata` into the batch, which would
        # ship a redundant pointmap to the GPU (including for RGB-only runs). Keep only
        # the small metadata keys; the requested pointmap stream was stacked above.
        batch_inputs["metadata"] = [
            {k: v for k, v in (s.trajectory.metadata or {}).items() if k != POINTMAP_KEY} for s in progress_samples
        ]
        batch_inputs["resample_attempts"] = [s.resample_attempts for s in progress_samples]
        return batch_inputs
