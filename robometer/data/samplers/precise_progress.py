#!/usr/bin/env python3
"""Progress sampler that carries pointmaps alongside RGB, without duplicating the sampling logic.

The stock `ProgressSampler` picks frames (forward / reverse / rewind subsampling,
uniform thinning, padding) inside `RBMBaseSampler._get_traj_from_data`, and that
method returns only the sliced RGB. Re-implementing it here to slice a second array
would mean maintaining a copy of ~120 lines of index arithmetic that decides the
training signal -- and any drift between the copies is a silent RGB/pointmap
misalignment, which every shape check in the codebase would pass.

So instead we hand the parent an **index carrier**: an `(T, 1, 1, 1)` int32 array
whose value at frame `f` is `f`. Every operation the parent performs on "frames" --
fancy indexing, `linspace_subsample_frames`, `pad_trajectory_to_max_frames_np`
repeating the last frame -- is positional, so the carrier comes back holding
exactly the frame indices the parent chose, padding included. We read them out of
`out.frames[:, 0, 0, 0]` and slice the real RGB *and* the pointmap with that one
vector.

Misalignment is therefore not merely tested for, it is unrepresentable: there is
one index vector, and both modalities are sliced with it.

The pointmap rides in `Trajectory.metadata` because `Trajectory` is a pydantic
`BaseModel` with a fixed field list -- a field added on a subclass is silently
dropped, whereas `metadata` is an open dict that `create_trajectory_from_dict`
already forwards.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from robometer.data.dataset_types import Trajectory
from robometer.data.samplers.progress import ProgressSampler
from robometer.utils.logger import get_logger

logger = get_logger()

POINTMAP_KEY = "pointmap"
FRAME_INDEX_KEY = "frame_indices"
RGB_STREAM = "rgb"
POINTMAP_STREAM = "pointmap"
PRECISE_MODALITY_STREAMS = {
    "rgb": (RGB_STREAM,),
    "pointmap": (POINTMAP_STREAM,),
    "rgb_pointmap": (RGB_STREAM, POINTMAP_STREAM),
}


def precise_streams(modality: str) -> Tuple[str, ...]:
    """Return requested streams without importing the heavyweight model package."""
    try:
        return PRECISE_MODALITY_STREAMS[modality]
    except KeyError as error:
        raise ValueError(
            f"Unknown precise modality {modality!r}; expected one of {tuple(PRECISE_MODALITY_STREAMS)}"
        ) from error


def load_precise_npz(
    npz_path: str,
    load_pointmap: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Read RGB and pointmap out of one cache file in a single open.

    `load_frames_from_npz` reads only `frames`, which is exactly why the stock RGB
    path runs on this cache unchanged -- but it also means we cannot reuse it here.
    """
    with np.load(npz_path) as cached:
        frames = cached["frames"]
        # Accessing an npz member decompresses it. RGB-only training never needs
        # this large array, so do not pay that CPU/memory cost for an unused stream.
        pointmap = cached[POINTMAP_KEY] if load_pointmap and POINTMAP_KEY in cached else None
    return frames, pointmap


class PreciseTrajectoryMixin:
    """Cooperative mixin that keeps RGB and pointmap frame selection identical.

    Put this class before an ``RBMBaseSampler`` subclass in the inheritance list.
    Its ``super()`` call then delegates all sampling, thinning, and padding to that
    stock sampler before replacing the integer carrier with the real modalities.
    This lets training and custom evaluation share exactly one implementation.
    """

    def __init__(self, *args, require_pointmap: bool = True, **kwargs):
        self.require_pointmap = require_pointmap
        super().__init__(*args, **kwargs)
        if getattr(self.config, "load_embeddings", False):
            raise ValueError(
                f"{type(self).__name__} needs raw frames; set data.load_embeddings=false. "
                "Precomputed DINO embeddings have no pointmap to go with them."
            )

    def _get_traj_from_data(
        self,
        traj: dict | Trajectory,
        subsample_strategy: str | None = None,
        frame_indices=None,
        metadata: Optional[Dict[str, Any]] = None,
        pad_frames: bool = True,
    ):
        if isinstance(traj, Trajectory):
            return traj  # already materialised; the parent does the same

        frames, pointmap = self._load(traj)
        if pointmap is None and self.require_pointmap:
            raise ValueError(
                f"{traj.get('id')}: cache file has no 'pointmap' key. Re-run preprocess_precise.py, "
                "or pass require_pointmap=False to train on RGB alone."
            )

        # The carrier: value at frame f is f, in a 4-D shape so it survives every
        # `data[indices]` and padding call the parent makes on "frames".
        carrier = np.arange(frames.shape[0], dtype=np.int32).reshape(-1, 1, 1, 1)

        trajectory = super()._get_traj_from_data(
            {**traj, "frames": carrier},
            subsample_strategy=subsample_strategy,
            frame_indices=frame_indices,
            metadata=metadata,
            pad_frames=pad_frames,
        )
        if trajectory is None:
            return None

        chosen = np.asarray(trajectory.frames).reshape(-1).astype(np.int64)
        if chosen.size == 0 or chosen.max() >= frames.shape[0] or chosen.min() < 0:
            raise ValueError(f"{traj.get('id')}: index carrier came back out of range: {chosen}")

        trajectory.frames = frames[chosen]
        trajectory.frames_shape = trajectory.frames.shape

        merged = dict(trajectory.metadata or {})
        merged[FRAME_INDEX_KEY] = chosen
        merged[POINTMAP_KEY] = None if pointmap is None else pointmap[chosen]
        trajectory.metadata = merged
        return trajectory

    def _load(self, traj: dict) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Frames + pointmap, whether the trajectory points at a cache file or holds arrays."""
        source = traj["frames"]
        if isinstance(source, str):
            return load_precise_npz(source, load_pointmap=self.require_pointmap)
        # In-memory trajectories (custom eval, tests) may carry the pointmap alongside.
        return np.asarray(source), traj.get(POINTMAP_KEY)


class PreciseProgressSampler(PreciseTrajectoryMixin, ProgressSampler):
    """`ProgressSampler` that also returns the pointmap for the frames it chose."""

    def __init__(self, is_evaluation: bool = False, require_pointmap: bool = True, **kwargs):
        super().__init__(
            is_evaluation=is_evaluation,
            require_pointmap=require_pointmap,
            **kwargs,
        )
