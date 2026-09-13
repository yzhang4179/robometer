#!/usr/bin/env python3
"""Step 1, stage B: published RBM dataset -> processed robometer cache, with pointmaps.

Subclasses the stock `DatasetPreprocessor` and changes exactly two things:

1. `_load_dataset_from_path` reads the locally-saved arrow dataset produced by
   `hdf5_to_rbm.py` (the stock loader only knows how to pull from the Hub) and
   resolves both the mp4 and the pointmap sidecar to absolute paths.
2. `_process_dataset_videos_threaded` runs the stock implementation untouched --
   so all the npz writing and index-mapping logic is inherited, not forked -- and
   then folds each trajectory's pointmap into the npz the parent just wrote.

The resulting cache file is the stock one with one extra key:

    frames/trajectory_<id>.npz
      frames       (32, 256, 256, 3)  uint8    <- what robometer already reads
      pointmap     (32, 256, 256, 3)  float16  <- camera-frame XYZ, RAW METRES
      shape, num_frames                        <- unchanged

`load_frames_from_npz()` reads only `frames`, so the plain RGB path in the original
codebase runs on this cache with zero changes.

The pointmap stays in raw metres here. The clamp to
`model.precise.pointmap_norm_bounds` happens once, on GPU, in
`WanLatentTokenizer.normalize_pointmap` -- baking it into the cache would make the
bounds a re-conversion instead of a CLI flag.

Example:
    export ROBOMETER_DATASET_PATH=/path/to/datasets
    export ROBOMETER_PROCESSED_DATASETS_PATH=/path/to/processed_datasets

    uv run python -m robometer.data.scripts.preprocess_precise \
        --train_datasets '["precise_square_rbm"]' \
        --train_subsets '[["square_d1_train"]]' \
        --eval_datasets '["precise_square_rbm"]' \
        --eval_subsets '[["square_d1_val"]]' \
        --max_frames_for_preprocessing 32 \
        --cache_dir $ROBOMETER_PROCESSED_DATASETS_PATH
"""

# NOTE: no `from __future__ import annotations` here. pyrallis reads the annotation on
# `main`'s parameter at runtime to find the config class, and PEP 563 would hand it the
# string "PrecisePreprocessConfig" instead, failing with "must be called with a dataclass".

# The stock preprocessor imports sentence_transformers -> transformers -> torchao, and
# this venv's torchao is broken (`cannot import name 'ScalingType'`). Importing unsloth
# first gets transformers loaded before that path is reached. Needs a visible GPU.
try:  # noqa: SIM105
    import unsloth  # noqa: F401
except Exception:
    pass

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import numpy as np
from pyrallis import wrap
from tqdm import tqdm

from datasets import Dataset
from robometer.data.scripts.preprocess_datasets import DataPreprocessConfig, DatasetPreprocessor
from robometer.utils.distributed import rank_0_print


@dataclass
class PrecisePreprocessConfig(DataPreprocessConfig):
    """Stock preprocessing config plus the two knobs the pointmap path needs."""

    dataset_root: str = field(
        default="",
        metadata={"help": "Dir holding the published RBM repos (default: $ROBOMETER_DATASET_PATH)"},
    )
    require_pointmap: bool = field(
        default=True,
        metadata={"help": "Fail if a trajectory has no pointmap sidecar, instead of writing an RGB-only cache entry"},
    )


class PrecisePreprocessor(DatasetPreprocessor):
    """`DatasetPreprocessor` that also carries the pointmap into the npz cache."""

    config: PrecisePreprocessConfig

    def _dataset_root(self) -> str:
        root = self.config.dataset_root or os.environ.get("ROBOMETER_DATASET_PATH", "")
        if not root:
            raise ValueError(
                "Neither --dataset_root nor ROBOMETER_DATASET_PATH is set. Point it at the directory "
                "holding the published dataset repo written by hdf5_to_rbm.py."
            )
        return root

    def _load_dataset_from_path(self, dataset_path: str, subset: str):
        """Load the locally-saved published dataset and absolutise its two file columns.

        `hdf5_to_rbm.py` writes `<root>/<repo>/<subset>` as an arrow dataset whose
        `frames` and `pointmap_path` are relative to `<root>/<repo>`, matching the
        convention the stock Hub loader patches with.
        """
        repo_dir = os.path.join(self._dataset_root(), dataset_path.split("/")[-1])
        dataset_dir = os.path.join(repo_dir, subset)
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"No published dataset at {dataset_dir} -- run hdf5_to_rbm.py first")

        rank_0_print(f"    📂 Loading published dataset from {dataset_dir}")
        dataset = Dataset.load_from_disk(dataset_dir)

        if "pointmap_path" not in dataset.column_names and self.config.require_pointmap:
            raise ValueError(
                f"{dataset_dir} has no 'pointmap_path' column -- it was not produced by hdf5_to_rbm.py. "
                "Pass --require_pointmap=false to preprocess it as RGB-only."
            )

        def absolutise(example: Dict[str, Any]) -> Dict[str, Any]:
            patched = {
                "frames_video": os.path.join(repo_dir, example["frames"]),
                "frames_path": os.path.join(repo_dir, example["frames"]),
            }
            if "pointmap_path" in example:
                patched["pointmap_abs"] = os.path.join(repo_dir, example["pointmap_path"])
            return patched

        return dataset.map(absolutise, desc="Resolving paths")

    def _process_individual_dataset(self, dataset: Dataset, cache_dir: str, cache_key: str):
        """Always take the threaded path.

        The parent picks between threaded and `.map()` on `num_threads`, but the
        `.map()` branch raises `NotImplementedError` in the stock code, so a
        `num_threads=1` run would die there.
        """
        return self._process_dataset_videos_threaded(dataset, cache_dir, cache_key)

    def _process_dataset_videos_threaded(
        self, dataset: Dataset, cache_dir: str, cache_key: str
    ) -> Tuple[Dataset, Dict[str, Any]]:
        processed_dataset, indices = super()._process_dataset_videos_threaded(dataset, cache_dir, cache_key)

        if "pointmap_abs" not in processed_dataset.column_names:
            if self.config.require_pointmap:
                raise ValueError(f"{cache_key}: no pointmap column survived preprocessing")
            rank_0_print(f"    ⚠️  {cache_key}: no pointmaps, cache holds RGB only")
            return processed_dataset, indices

        self._merge_pointmaps(processed_dataset, cache_key)
        # The absolute sidecar path has done its job; the merged npz is now the single
        # file the sampler reads. Keep the relative `pointmap_path` for provenance.
        return processed_dataset.remove_columns("pointmap_abs"), indices

    def _merge_pointmaps(self, processed_dataset: Dataset, cache_key: str) -> None:
        """Fold each pointmap sidecar into the npz the parent wrote, in place."""
        rows = [
            {"frames": row["frames"], "pointmap_abs": row["pointmap_abs"], "id": row["id"]}
            for row in processed_dataset.select_columns(["frames", "pointmap_abs", "id"])
        ]

        def merge_one(row: Dict[str, str]) -> None:
            with np.load(row["frames"]) as cached:
                frames = cached["frames"]
            with np.load(row["pointmap_abs"]) as sidecar:
                pointmap = sidecar["pointmap"]

            # The mp4 holds exactly the frames hdf5_to_rbm.py selected, and the stock
            # reader keeps all of them whenever max_frames_for_preprocessing >= that
            # count, so the two arrays are the same frames in the same order. If they
            # are not, RGB and pointmap would be silently misaligned -- so fail loudly.
            if frames.shape[0] != pointmap.shape[0]:
                raise ValueError(
                    f"{row['id']}: {frames.shape[0]} rgb frames but {pointmap.shape[0]} pointmap frames. "
                    f"Set --max_frames_for_preprocessing to at least the --max-frames used in stage A."
                )
            if frames.shape[1:3] != pointmap.shape[1:3]:
                raise ValueError(f"{row['id']}: rgb {frames.shape} and pointmap {pointmap.shape} differ spatially")

            tmp_path = row["frames"] + ".tmp.npz"
            np.savez_compressed(
                tmp_path,
                frames=frames,
                pointmap=pointmap.astype(np.float16),
                shape=np.asarray(frames.shape),
                num_frames=frames.shape[0],
            )
            os.replace(tmp_path, row["frames"])

        with ThreadPoolExecutor(max_workers=self.config.num_threads) as executor:
            futures = {executor.submit(merge_one, row): row for row in rows}
            for future in tqdm(
                as_completed(futures), total=len(futures), desc=f"Merging pointmaps {cache_key}", unit="traj"
            ):
                future.result()  # re-raise inside the main thread

        rank_0_print(f"    ✅ Merged pointmaps into {len(rows)} cache files for {cache_key}")


@wrap()
def main(config: PrecisePreprocessConfig):
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target_soft = min(hard, 65535)
        if soft < target_soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target_soft, hard))
    except Exception:
        pass

    preprocessor = PrecisePreprocessor(config)
    preprocessor.preprocess_datasets()

    print("\n=== Cache summary ===")
    for cache_key, dataset in preprocessor.datasets.items():
        cache_name = cache_key.replace("/", "_").replace(":", "_")
        print(f"  ✅ {cache_key}: {len(dataset)} trajectories  ->  data.train_datasets=[{cache_name}]")

    print(f"\n✅ Done. export ROBOMETER_PROCESSED_DATASETS_PATH={config.cache_dir}")
    print("Then inspect a batch with:")
    print("  uv run python dev_scripts/test_dataloader.py --dataset <cache_name>")


if __name__ == "__main__":
    main()
