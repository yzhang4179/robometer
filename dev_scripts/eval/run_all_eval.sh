#!/usr/bin/env bash
# Step 4 part 1: qualitative progress/success videos for all three Precise models
# over the three failed test splits.
#
# Produces the tree from the Step 4 spec:
#
#   $SAVE_DIR/<task_name>/<demo key>/rgb_only.mp4
#                                    pointmap_only.mp4
#                                    rgb_pointmap.mp4
#                                    predictions.json
#
# The three modality checkpoints are loaded into one process per split, so each
# demo's hdf5 is read once and scored by all three -- the mp4s in a demo dir are
# guaranteed to be the same frames.
#
# The side-view split is rendered TWICE, into two sibling task dirs:
#   square_sideview_failed            frozen training bounds  (z: 0.2 - 1.3)
#   square_sideview_failed_rebounded  side-view bounds        (z: 0.9 - 2.2)
# Step 1 measured only 28% of side-view depth inside the frozen bounds, with the
# 1.3 m ceiling cutting through its workspace, so under frozen bounds its
# pointmaps arrive as near-uniform +1 images. Rendering both separates "pointmaps
# are a worse representation" from "the normaliser was calibrated for a different
# camera". The RGB model is unaffected and its two renders should be identical.
#
# Usage:
#   bash dev_scripts/eval/run_all_eval.sh                 # everything
#   MAX_DEMOS=2 bash dev_scripts/eval/run_all_eval.sh     # quick smoke
#   CLIP_LENGTH=9 bash dev_scripts/eval/run_all_eval.sh   # a different clip length
#
# All three test splits are failures-only, so each task dir also gets a metrics.json:
# the success head's correct-rejection rate (robometer's negative_success_acc), its
# false-positive rate, and how many demos ever cross the threshold.
#
# Extra arguments are appended to every run_precise_eval.py invocation.

set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root; Hydra and relative paths expect it

: "${SQUARE:=/home/reward/Desktop/precise_robometer/environments/mimicgen/datasets/robometer_core_datasets/square}"
: "${SAVE_DIR:=./dev_scripts/out/eval}"
: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES


# 5 = the shortest legal Wan clip (4k+1), and the length this study uses.
: "${CLIP_LENGTH:=5}"
: "${SUCCESS_THRESHOLD:=0.5}"
: "${FRAME_STRIDE:=1}"
: "${BATCH_SIZE:=32}"
: "${MAX_DEMOS:=-1}"
: "${FPS:=15}"

# Side-view specific bounds: roughly its own workspace-vs-background valley,
# measured in Step 1. Only the pointmap-consuming models are affected.
: "${SIDEVIEW_BOUNDS:=x=-1.0,1.0 y=-1.2,0.5 z=0.9,2.2}"

EXTRA_ARGS=("$@")


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
#   RGB_CKPT=./logs/precise/precise_rgb/final bash dev_scripts/eval/run_all_eval.sh
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

run_split() {
    local hdf5="$1"
    local task_name="$2"
    shift 2

    if [[ ! -f "$hdf5" ]]; then
        echo "ERROR: hdf5 not found: $hdf5" >&2
        exit 1
    fi

    echo ""
    echo "################################################################"
    echo "# $task_name"
    echo "#   $hdf5"
    echo "################################################################"

    uv run python dev_scripts/eval/run_precise_eval.py \
        --hdf5 "$hdf5" \
        --task-name "$task_name" \
        --model "$RGB_CKPT" \
        --model "$PM_CKPT" \
        --model "$BOTH_CKPT" \
        --save-dir "$SAVE_DIR" \
        --num-subsampled-frames "$CLIP_LENGTH" \
        --success-threshold "$SUCCESS_THRESHOLD" \
        --frame-stride "$FRAME_STRIDE" \
        --batch-size "$BATCH_SIZE" \
        --max-demos "$MAX_DEMOS" \
        --fps "$FPS" \
        "$@" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
}

# ---- agentview, in distribution (25 failed demos) -------------------------
run_split "$SQUARE/demo_src_square_task_D1_validation/demo_failed.hdf5" \
          "square_d1_failed"

# ---- agentview, different task layout (25 failed demos) -------------------
run_split "$SQUARE/demo_src_square_task_D2_validation/demo_failed.hdf5" \
          "square_d2_failed"

# ---- side view, frozen training bounds (23 failed demos) ------------------
run_split "$SQUARE/demo_src_square_task_D1_sideview_validation/demo_failed.hdf5" \
          "square_sideview_failed"

# ---- side view, side-view-specific bounds (same 23 demos) -----------------
run_split "$SQUARE/demo_src_square_task_D1_sideview_validation/demo_failed.hdf5" \
          "square_sideview_failed_rebounded" \
          --pointmap-bounds $SIDEVIEW_BOUNDS

echo ""
echo "Done. Videos and metrics.json under:"
for t in square_d1_failed square_d2_failed square_sideview_failed square_sideview_failed_rebounded; do
    echo "  $SAVE_DIR/$t/"
done
