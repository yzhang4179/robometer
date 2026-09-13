#!/usr/bin/env bash
# Step 4 part 3: policy-ranking metrics (Kendall tau, ranking accuracy,
# succ-fail diff) for all three Precise models, on every shift axis.
#
# This is the ONLY evaluation that sees failures. Reward alignment drops them --
# a failure has no honest ground-truth progress curve, so pearson/loss are
# defined on successes alone. Ranking a success against a failure needs both,
# and robometer computes succ_fail_diff and Kendall only here
# (compile_results.py:_compute_policy_ranking_metrics_quality_label).
#
# Each SPLIT below loads two caches together -- the successes and the failures of
# one validation directory. They share a task string, which is what gives that
# task two quality tiers; ProgressPolicyRankingSampler silently drops any task
# that has fewer than two (progress_policy_ranking.py:76).
#
# Prefix contract is identical to run_reward_alignment.sh: NUM_PREFIXES endpoints
# per trajectory, each prefix thinned to a CLIP_LEN-frame clip, last token read.
# That is deliberate -- the two evaluations then score literally the same clips,
# so their numbers can be quoted side by side.
#
# ONE CAVEAT WORTH REPORTING. With only two quality tiers, Kendall tau-a is
# computed over n=2 points, so it reduces exactly to `2 * ranking_acc - 1`, and
# kendall_rewind collapses to +/-1 regardless of how many pairs are misordered.
# Robometer's three-tier design (failure/suboptimal/successful) is what makes tau
# informative; mimicgen gives us no middle tier. Report ranking_acc and
# succ_fail_diff beside it.
#
# Writes:
#   $OUTPUT_DIR/all_metrics.json                                 <- the summary tables
#   $OUTPUT_DIR/<modality>/policy_ranking/metrics.json            <- kendall/ranking_acc/succ_fail_diff
#   $OUTPUT_DIR/<modality>/policy_ranking/<split>_results.json    <- per-trajectory scalar rewards
#
# Usage:
#   bash dev_scripts/eval/run_policy_ranking.sh
#   SPLITS=d1 bash dev_scripts/eval/run_policy_ranking.sh       # in-distribution only
#   MAX_PER_TIER=3 bash dev_scripts/eval/run_policy_ranking.sh  # quick smoke
#
# Extra arguments are appended as Hydra overrides.

set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root; data.dataset_success_cutoff_file is repo-relative

: "${ROBOMETER_PROCESSED_DATASETS_PATH:=/home/reward/Desktop/precise_robometer/processed_datasets}"
export ROBOMETER_PROCESSED_DATASETS_PATH
: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

: "${OUTPUT_DIR:=./baseline_eval_output/precise_ranking}"

# CLIP_LEN     = frames per prefix clip after thinning    (data.max_frames at train time)
# NUM_PREFIXES = prefix endpoints per trajectory          (custom_eval.subsample_n_frames)
: "${CLIP_LEN:=9}"
: "${NUM_PREFIXES:=9}"
: "${BATCH_SIZE:=32}"
# null = every trajectory of every tier. The stock default of 5 exists to keep a
# 1M-row eval affordable; our splits are ~60 trajectories.
: "${MAX_PER_TIER:=null}"

# Which shift axes to score. Each is one (successes, failures) pair.
: "${SPLITS:=d1 d2 sideview robot}"

P=precise_square_rbm
declare -A PAIRS=(
    [d1]="${P}_square_d1_val,${P}_square_d1_val_fail"                   # in distribution
    [d2]="${P}_square_d2_val,${P}_square_d2_val_fail"                   # layout shift
    [sideview]="${P}_square_sideview_val,${P}_square_sideview_val_fail" # viewpoint shift
    [robot]="${P}_square_robot_val,${P}_square_robot_val_fail"          # embodiment shift (UR5e)
)

groups=()
for split in $SPLITS; do
    pair="${PAIRS[$split]:-}"
    if [[ -z "$pair" ]]; then
        echo "ERROR: unknown split '$split'. Known: ${!PAIRS[*]}" >&2
        exit 1
    fi
    for cache in ${pair//,/ }; do
        if [[ ! -d "$ROBOMETER_PROCESSED_DATASETS_PATH/$cache" ]]; then
            echo "ERROR: cache not found: $ROBOMETER_PROCESSED_DATASETS_PATH/$cache" >&2
            echo "       Convert it first; see the Step 4 part 3 commands in the research doc." >&2
            exit 1
        fi
    done
    groups+=("[$pair]")
done
policy_ranking="[$(IFS=,; echo "${groups[*]}")]"

# Same rotation-proof checkpoint discovery as run_reward_alignment.sh: save_best
# keeps only the top keep_top_k, so a hardcoded best-metric name goes stale.
pick_best_ckpt() {
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

: "${RGB_CKPT:=$(pick_best_ckpt ./logs/precise/precise_rgb)}"
: "${PM_CKPT:=$(pick_best_ckpt ./logs/precise/precise_pointmap)}"
: "${BOTH_CKPT:=$(pick_best_ckpt ./logs/precise/precise_rgb_pointmap)}"

echo "Checkpoints:"
echo "  rgb          : $RGB_CKPT"
echo "  pointmap     : $PM_CKPT"
echo "  rgb_pointmap : $BOTH_CKPT"
echo "Splits: $policy_ranking"

for ckpt in "$RGB_CKPT" "$PM_CKPT" "$BOTH_CKPT"; do
    if [[ ! -d "$ckpt" ]]; then
        echo "ERROR: checkpoint not found: $ckpt" >&2
        echo "       Set RGB_CKPT / PM_CKPT / BOTH_CKPT, or run: ls logs/precise/*/" >&2
        exit 1
    fi
done

# Each checkpoint path is single-quoted: the best-metric directory names contain
# '=' and Hydra's override parser splits on the first one.
uv run python robometer/evals/precise_baseline_eval.py \
    reward_model=precise \
    "model_paths=['$RGB_CKPT','$PM_CKPT','$BOTH_CKPT']" \
    custom_eval.eval_types=[policy_ranking] \
    "custom_eval.policy_ranking=$policy_ranking" \
    custom_eval.use_frame_steps=true \
    custom_eval.subsample_n_frames="$NUM_PREFIXES" \
    custom_eval.num_examples_per_quality_pr="$MAX_PER_TIER" \
    custom_eval.policy_ranking_max_tasks=null \
    custom_eval.pad_frames=true \
    max_frames="$CLIP_LEN" \
    model_config.batch_size="$BATCH_SIZE" \
    "output_dir=$OUTPUT_DIR" \
    "$@"

echo ""
echo "Summary: $OUTPUT_DIR/all_metrics.json"
