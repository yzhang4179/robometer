#!/usr/bin/env bash
# Qualitative videos for exps 2 and 3, at FULL episode length.
#
# Reads the collected hdf5s directly, not the 64/32-frame caches: one video frame
# per stored source frame, each frame's prediction from the growing prefix ending
# there (thinned to the 9-frame clip). Successes and failures land in separate
# folders:
#
#   dev_scripts/out/eval_pusht/val/{successful,failure}/demo_N/rgb_only.mp4
#   dev_scripts/out/eval_lift/h{5,10,15,20}/{successful,failure}/demo_N/{rgb_only,pointmap_only,rgb_pointmap}.mp4
#
# Lift videos show the goal-marker render (agentview_goal) in the RGB panel while
# the models are fed the plain agentview frames -- the marker is never in the
# model's input, in any modality.
#
#   bash dev_scripts/eval/plot_all_qualitative.sh
#   MAX_DEMOS=2 bash dev_scripts/eval/plot_all_qualitative.sh     # smoke
#   ONLY=lift HEIGHTS="5 20" bash dev_scripts/eval/plot_all_qualitative.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PRECISE_OUTPUT_DIR="${PRECISE_OUTPUT_DIR:-./logs/precise}"
OUT="${OUT:-dev_scripts/out}"
MAX_DEMOS="${MAX_DEMOS:--1}"
CLIP_LEN="${CLIP_LEN:-9}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
HEIGHTS="${HEIGHTS:-5 10 15 20}"
ONLY="${ONLY:-pusht lift}"

pick_best_ckpt() {
    local run_dir="$1" best
    best=$(ls -1d "$run_dir"/ckpt-*_step=*/ 2>/dev/null | sed -E 's#/+$##' \
        | sed -E 's#^(.*=([0-9]+\.[0-9]+)_step=([0-9]+))$#\2 \3 \1#' | grep -E '^[0-9]' \
        | sort -k1,1g -k2,2n | tail -1 | cut -d' ' -f3-)
    if [[ -n "$best" && -d "$best" ]]; then echo "$best"; else echo "$run_dir/final"; fi
}

render() {  # render <hdf5> <save_dir> <task_name> <quality> <extra args...>
    local hdf5="$1" save_dir="$2" task_name="$3" quality="$4"; shift 4
    if [[ ! -f "$hdf5" ]]; then echo "  [skip] no $hdf5"; return; fi
    uv run python dev_scripts/eval/run_precise_eval.py \
        --hdf5 "$hdf5" --save-dir "$save_dir" --task-name "$task_name" --quality-label "$quality" \
        --num-subsampled-frames "$CLIP_LEN" --frame-stride "$FRAME_STRIDE" --max-demos "$MAX_DEMOS" "$@"
}

if [[ " $ONLY " == *" pusht "* ]]; then
    ckpt=$(pick_best_ckpt "$PRECISE_OUTPUT_DIR/precise_pusht_rgb")
    echo "push-T checkpoint: $ckpt"
    for quality in successful failure; do
        [[ $quality == successful ]] && f=demo.hdf5 || f=demo_failed.hdf5
        render "$OUT/pusht_collect/val/$f" "$OUT/eval_pusht" val "$quality" --model "rgb=$ckpt"
    done
fi

if [[ " $ONLY " == *" lift "* ]]; then
    rgb=$(pick_best_ckpt "$PRECISE_OUTPUT_DIR/precise_lift_rgb")
    pm=$(pick_best_ckpt "$PRECISE_OUTPUT_DIR/precise_lift_pointmap")
    both=$(pick_best_ckpt "$PRECISE_OUTPUT_DIR/precise_lift_rgb_pointmap")
    echo "lift checkpoints: $rgb | $pm | $both"
    for h in $HEIGHTS; do
        for quality in successful failure; do
            [[ $quality == successful ]] && f=demo.hdf5 || f=demo_failed.hdf5
            render "$OUT/lift_collect/val/h$h/$f" "$OUT/eval_lift" "h$h" "$quality" \
                --camera agentview --display-camera agentview_goal --task "lift the cube to $h cm" \
                --model "rgb=$rgb" --model "pointmap=$pm" --model "rgb_pointmap=$both"
        done
    done
fi

echo "videos under $OUT/eval_pusht and $OUT/eval_lift"
