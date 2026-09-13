#!/usr/bin/env bash
# Run the three Step-3 modality experiments sequentially on one visible GPU.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export ROBOMETER_PROCESSED_DATASETS_PATH="${ROBOMETER_PROCESSED_DATASETS_PATH:-${REPO_ROOT}/../processed_datasets}"

PRECISE_BATCH_SIZE="${PRECISE_BATCH_SIZE:-20}"
PRECISE_MAX_STEPS="${PRECISE_MAX_STEPS:-20000}"
PRECISE_OUTPUT_DIR="${PRECISE_OUTPUT_DIR:-./logs/precise}"

if [[ "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
    echo "CUDA_VISIBLE_DEVICES must expose exactly one GPU; got '${CUDA_VISIBLE_DEVICES}'." >&2
    exit 2
fi

if [[ ! -d "${ROBOMETER_PROCESSED_DATASETS_PATH}" ]]; then
    echo "Processed dataset directory not found: ${ROBOMETER_PROCESSED_DATASETS_PATH}" >&2
    echo "Set ROBOMETER_PROCESSED_DATASETS_PATH to the Step-1 cache directory." >&2
    exit 2
fi

common_overrides=(
    "training.num_gpus=1"
    "training.max_steps=${PRECISE_MAX_STEPS}"
    "training.per_device_train_batch_size=${PRECISE_BATCH_SIZE}"
    "training.per_device_eval_batch_size=${PRECISE_BATCH_SIZE}"
    "training.output_dir=${PRECISE_OUTPUT_DIR}"
)

run_precise() {
    local modality="$1"
    local exp_name="$2"
    shift 2

    echo
    echo "Starting ${exp_name}: modality=${modality}, batch=${PRECISE_BATCH_SIZE}, steps=${PRECISE_MAX_STEPS}"
    uv run python train_precise.py \
        "model.precise.modality=${modality}" \
        "training.exp_name=${exp_name}" \
        "${common_overrides[@]}" \
        "$@"
}

# Any arguments passed to this script are appended to all three Hydra commands.
# Example: dev_scripts/train_all_precise.sh logging.wandb_mode=offline
extra_overrides=("$@")

run_precise rgb precise_rgb "${extra_overrides[@]}"
run_precise pointmap precise_pointmap "${extra_overrides[@]}"
run_precise rgb_pointmap precise_rgb_pointmap "${extra_overrides[@]}"

echo
echo "All three Precise training runs completed. Outputs are under ${PRECISE_OUTPUT_DIR}."
