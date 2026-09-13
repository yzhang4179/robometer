#!/usr/bin/env bash
# Step 4 part 2: reward-alignment metrics for all three Precise models.
#
# Scores the 35 successful validation trajectories with the growing-prefix
# protocol (nine endpoints per trajectory, each prefix thinned to a nine-frame
# clip, last token read) -- exactly the protocol the trainer used, so these
# numbers are directly comparable with the logged
# eval_rew_align/pearson_precise_square_rbm_square_d1_val.
#
# Writes:
#   $OUTPUT_DIR/all_metrics.json                                  <- the summary table
#   $OUTPUT_DIR/<modality>/reward_alignment/metrics.json          <- pearson, loss, success_auprc,
#                                                                    positive/negative_success_acc
#   $OUTPUT_DIR/<modality>/reward_alignment/<dataset>_results.json <- per-trajectory curves
#   $OUTPUT_DIR/<modality>/reward_alignment/<dataset>_plots/*.png  <- first 10 trajectories
#
# All three models run in ONE process, sequentially, so the table at the end
# compares them side by side.
#
# Caveat worth keeping in the writeup: precise_square_rbm_square_d1_val is the
# split save_best selected on, so a best-pearson checkpoint's number here is
# selection-biased. Point CKPTS at the three `final/` dirs for an unbiased read.
#
# Usage:
#   bash dev_scripts/eval/run_reward_alignment.sh
#   MAX_TRAJ=3 bash dev_scripts/eval/run_reward_alignment.sh    # quick smoke
#   CLIP_LEN=5 bash dev_scripts/eval/run_reward_alignment.sh    # 5-frame clips
#
# Extra arguments are appended as Hydra overrides.

set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root; data.dataset_success_cutoff_file is repo-relative

: "${ROBOMETER_PROCESSED_DATASETS_PATH:=/home/reward/Desktop/precise_robometer/processed_datasets}"
export ROBOMETER_PROCESSED_DATASETS_PATH
: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES


: "${EVAL_DATASET:=precise_square_rbm_square_d1_val}"
: "${OUTPUT_DIR:=./baseline_eval_output/precise}"

# CLIP_LEN     = frames per prefix clip after thinning       (data.max_frames at train time)
# NUM_PREFIXES = how many prefix endpoints per trajectory     (custom_eval.subsample_n_frames)
# Both were 9 in training; they are different knobs.
: "${CLIP_LEN:=9}"
: "${NUM_PREFIXES:=9}"
: "${BATCH_SIZE:=32}"
: "${MAX_TRAJ:=null}"


# Best-metric checkpoint directories are rotated by save_best (only the top
# keep_top_k survive), so hardcoding one goes stale as training improves. Pick the
# highest-pearson directory that exists right now, and fall back to the stable
# final/ if a run has none.
pick_best_ckpt() {
    # Rank by the metric value in the directory name, then by step as a tiebreak.
    # Deliberately no `sort -t=`: this machine ships uutils coreutils, whose sort
    # rejects the attached form (`separator must be exactly one character long`)
    # that GNU sort accepts. Pulling the numbers out with sed works on both.
    local run_dir="$1"
    local best
    best=$(ls -1d "$run_dir"/ckpt-*_step=*/ 2>/dev/null \
        | sed -E 's#/+$##' \
        | sed -E 's#^(.*=([0-9]+\.[0-9]+)_step=([0-9]+))$#\2 \3 \1#' \
        | grep -E '^[0-9]' \
        | sort -k1,1g -k2,2n \
        | tail -1 \
        | cut -d' ' -f3-)
    if [[ -n "$best" && -d "$best" ]]; then
        echo "$best"
    else
        echo "$run_dir/final"
    fi
}

# Override any of these to pin a specific checkpoint, e.g.
#   RGB_CKPT=./logs/precise/precise_rgb/final bash dev_scripts/eval/run_reward_alignment.sh
: "${RGB_CKPT:=$(pick_best_ckpt ./logs/precise/precise_rgb)}"
: "${PM_CKPT:=$(pick_best_ckpt ./logs/precise/precise_pointmap)}"
: "${BOTH_CKPT:=$(pick_best_ckpt ./logs/precise/precise_rgb_pointmap)}"

echo "Checkpoints:"
echo "  rgb          : $RGB_CKPT"
echo "  pointmap     : $PM_CKPT"
echo "  rgb_pointmap : $BOTH_CKPT"

for ckpt in "$RGB_CKPT" "$PM_CKPT" "$BOTH_CKPT"; do
    if [[ ! -d "$ckpt" ]]; then
        echo "ERROR: checkpoint not found: $ckpt" >&2
        echo "       Set RGB_CKPT / PM_CKPT / BOTH_CKPT, or run: ls logs/precise/*/" >&2
        exit 1
    fi
done

# Each path is single-quoted inside the override on purpose: the best-pearson
# checkpoint directories contain '=' (ckpt-pearson_...=0.9958_step=18250), and
# Hydra's override parser splits on the first '=' and then fails to parse the
# rest ("no viable alternative at input"). Quoting makes it an opaque string.
uv run python robometer/evals/precise_baseline_eval.py \
    reward_model=precise \
    "model_paths=['$RGB_CKPT','$PM_CKPT','$BOTH_CKPT']" \
    custom_eval.eval_types=[reward_alignment] \
    "custom_eval.reward_alignment=[$EVAL_DATASET]" \
    custom_eval.use_frame_steps=true \
    custom_eval.subsample_n_frames="$NUM_PREFIXES" \
    custom_eval.reward_alignment_max_trajectories="$MAX_TRAJ" \
    custom_eval.pad_frames=true \
    max_frames="$CLIP_LEN" \
    model_config.batch_size="$BATCH_SIZE" \
    "output_dir=$OUTPUT_DIR" \
    "$@"

echo ""
echo "Summary: $OUTPUT_DIR/all_metrics.json"
