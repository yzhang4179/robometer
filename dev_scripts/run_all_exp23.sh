#!/usr/bin/env bash
# Experiments 2 (push-T, rgb) and 3 (lift to N cm, three modalities), end to end.
#
# Stages, in order. Each is skipped if its output already exists, so the script
# is re-runnable after a crash; ONLY="stage stage" runs just those.
#
#   collect_pusht  scripted push-T rollouts       -> dev_scripts/out/pusht_collect/{train,val}/demo*.hdf5
#   convert_pusht  hdf5 -> RBM -> cache (rgb only) -> $ROBOMETER_PROCESSED_DATASETS_PATH/precise_pusht_rbm_*
#   train_pusht    precise_pusht_rgb
#   eval_pusht     reward_alignment + policy_ranking
#   collect_lift   scripted lift rollouts, 4 heights -> dev_scripts/out/lift_collect/{train,val}/h{5,10,15,20}/
#   convert_lift   hdf5 -> RBM -> cache (rgb + pointmap, lang embedding per height)
#   train_lift     precise_lift_rgb, precise_lift_pointmap, precise_lift_rgb_pointmap  (use_lang_token=true)
#   eval_lift      both evals, per height AND pooled
#
# Extra arguments are forwarded to every train_precise.py call as Hydra overrides,
# the same convention as train_all_precise.sh.
#
# Smoke test the whole thing in ~20 minutes (no debug=true: that renames the run dir):
#   MAX_DEMOS=2 PRECISE_MAX_STEPS=10 PRECISE_BATCH_SIZE=2 EXTRA_TRAIN="logging.wandb_mode=offline" \
#       bash dev_scripts/run_all_exp23.sh
#
# Rendering backends: plain-MuJoCo push-T renders through EGL; robosuite's own EGL
# context fails on this machine, so lift renders through GLX.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export ROBOMETER_DATASET_PATH="${ROBOMETER_DATASET_PATH:-${REPO_ROOT}/../datasets}"
export ROBOMETER_PROCESSED_DATASETS_PATH="${ROBOMETER_PROCESSED_DATASETS_PATH:-${REPO_ROOT}/../processed_datasets}"

MIMICGEN_PY="${MIMICGEN_PY:-$HOME/miniconda3/envs/mimicgen/bin/python}"
PRECISE_BATCH_SIZE="${PRECISE_BATCH_SIZE:-20}"
PRECISE_MAX_STEPS="${PRECISE_MAX_STEPS:-20000}"
PRECISE_OUTPUT_DIR="${PRECISE_OUTPUT_DIR:-./logs/precise}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-./baseline_eval_output}"
OUT="${OUT:-dev_scripts/out}"

# Dataset sizes. MAX_DEMOS caps every one of them, for smoke tests.
PUSHT_TRAIN="${PUSHT_TRAIN:-500}"
PUSHT_VAL="${PUSHT_VAL:-80}"
PUSHT_VAL_FAIL="${PUSHT_VAL_FAIL:-40}"
LIFT_TRAIN_PER_HEIGHT="${LIFT_TRAIN_PER_HEIGHT:-75}"
LIFT_VAL_PER_HEIGHT="${LIFT_VAL_PER_HEIGHT:-15}"
LIFT_VAL_FAIL_PER_HEIGHT="${LIFT_VAL_FAIL_PER_HEIGHT:-15}"
if [[ -n "${MAX_DEMOS:-}" ]]; then
    PUSHT_TRAIN=$MAX_DEMOS; PUSHT_VAL=$MAX_DEMOS; PUSHT_VAL_FAIL=$MAX_DEMOS
    LIFT_TRAIN_PER_HEIGHT=$MAX_DEMOS; LIFT_VAL_PER_HEIGHT=$MAX_DEMOS; LIFT_VAL_FAIL_PER_HEIGHT=$MAX_DEMOS
fi

# Cache sizes from exp2.md: 64 frames for push-T, 32 for lift; the training clip is 9.
PUSHT_CACHE_FRAMES="${PUSHT_CACHE_FRAMES:-64}"
LIFT_CACHE_FRAMES="${LIFT_CACHE_FRAMES:-32}"
FRAME_SIZE="${FRAME_SIZE:-224}"
HEIGHTS="${HEIGHTS:-5 10 15 20}"
ONLY="${ONLY:-}"
EXTRA_TRAIN="${EXTRA_TRAIN:-}"   # e.g. "debug=true logging.wandb_mode=offline"

PUSHT_TASK="push the T block onto the target"

if [[ "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
    echo "CUDA_VISIBLE_DEVICES must expose exactly one GPU; got '${CUDA_VISIBLE_DEVICES}'." >&2
    exit 2
fi

want() { [[ -z "$ONLY" || " $ONLY " == *" $1 "* ]]; }
banner() { echo; echo "=================================================================="; echo "  $*"; echo "=================================================================="; }

# ---------------------------------------------------------------- helpers

convert() {  # convert <hdf5> <repo> <subset> <task> <max_frames> <quality> [extra converter args...]
    local hdf5="$1" repo="$2" subset="$3" task="$4" max_frames="$5" quality="$6"; shift 6
    if [[ -d "$ROBOMETER_DATASET_PATH/$repo/$subset" ]]; then
        echo "  [skip] $repo/$subset already converted"; return
    fi
    uv run python -m dataset_upload.data_scripts.mimicgen_precise.hdf5_to_rbm \
        --hdf5 "$hdf5" --output-dir "$ROBOMETER_DATASET_PATH/$repo" --subset "$subset" \
        --task "$task" --lang-embedding --max-frames "$max_frames" --quality-label "$quality" "$@"
}

preprocess() {  # preprocess <repo> <max_frames> <require_pointmap> <subset>...
    local repo="$1" max_frames="$2" require_pointmap="$3"; shift 3
    local subsets=() missing=0
    for subset in "$@"; do
        subsets+=("\"$subset\"")
        [[ -d "$ROBOMETER_PROCESSED_DATASETS_PATH/${repo}_${subset}" ]] || missing=1
    done
    if [[ $missing == 0 ]]; then echo "  [skip] $repo caches already built"; return; fi
    local list; list=$(IFS=,; echo "${subsets[*]}")
    uv run python -m robometer.data.scripts.preprocess_precise \
        --train_datasets "[\"$repo\"]" --train_subsets "[[$list]]" \
        --max_frames_for_preprocessing "$max_frames" --require_pointmap="$require_pointmap" \
        --cache_dir "$ROBOMETER_PROCESSED_DATASETS_PATH"
}

train() {  # train <exp_name> <modality> <cache_frames> <overrides...>
    local exp_name="$1" modality="$2" cache_frames="$3"; shift 3
    if [[ -d "$PRECISE_OUTPUT_DIR/$exp_name/final" ]]; then
        echo "  [skip] $exp_name already trained ($PRECISE_OUTPUT_DIR/$exp_name/final exists)"; return
    fi
    # shellcheck disable=SC2086
    uv run python train_precise.py \
        "training.exp_name=$exp_name" \
        "model.precise.modality=$modality" \
        "model.precise.frame_size=$FRAME_SIZE" \
        "data.max_frames_after_preprocessing=$cache_frames" \
        "training.num_gpus=1" \
        "training.max_steps=$PRECISE_MAX_STEPS" \
        "training.per_device_train_batch_size=$PRECISE_BATCH_SIZE" \
        "training.per_device_eval_batch_size=$PRECISE_BATCH_SIZE" \
        "training.output_dir=$PRECISE_OUTPUT_DIR" \
        "$@" $EXTRA_TRAIN "${extra_overrides[@]}"
}

# Same rotation-proof checkpoint discovery as run_reward_alignment.sh.
pick_best_ckpt() {
    local run_dir="$1" best
    best=$(ls -1d "$run_dir"/ckpt-*_step=*/ 2>/dev/null | sed -E 's#/+$##' \
        | sed -E 's#^(.*=([0-9]+\.[0-9]+)_step=([0-9]+))$#\2 \3 \1#' | grep -E '^[0-9]' \
        | sort -k1,1g -k2,2n | tail -1 | cut -d' ' -f3-)
    if [[ -n "$best" && -d "$best" ]]; then echo "$best"; else echo "$run_dir/final"; fi
}

evaluate() {  # evaluate <out_dir> <reward_alignment list> <policy_ranking list> <ckpt>...
    local out_dir="$1" reward_alignment="$2" policy_ranking="$3"; shift 3
    local paths=() p
    for p in "$@"; do paths+=("'$p'"); done
    local joined; joined=$(IFS=,; echo "${paths[*]}")
    uv run python robometer/evals/precise_baseline_eval.py \
        reward_model=precise \
        "model_paths=[$joined]" \
        custom_eval.eval_types=[reward_alignment,policy_ranking] \
        "custom_eval.reward_alignment=$reward_alignment" \
        "custom_eval.policy_ranking=$policy_ranking" \
        custom_eval.use_frame_steps=true custom_eval.subsample_n_frames=9 custom_eval.pad_frames=true \
        custom_eval.reward_alignment_max_trajectories=null \
        custom_eval.num_examples_per_quality_pr=null custom_eval.policy_ranking_max_tasks=null \
        max_frames=9 \
        "output_dir=$out_dir"
    echo "  summary: $out_dir/all_metrics.json"
}

extra_overrides=("$@")

# ================================================================ Experiment 2

PUSHT_REPO=precise_pusht_rbm
PUSHT_DIR="$OUT/pusht_collect"

if want collect_pusht; then
    banner "collect_pusht  ($PUSHT_TRAIN train successes; $PUSHT_VAL val successes + $PUSHT_VAL_FAIL val failures)"
    if [[ -f "$PUSHT_DIR/train/demo.hdf5" ]]; then echo "  [skip] train exists"; else
        MUJOCO_GL="${PUSHT_MUJOCO_GL:-egl}" uv run python dev_scripts/collect/collect_pusht.py \
            --output-dir "$PUSHT_DIR/train" --num-success "$PUSHT_TRAIN" --num-failure 0 --seed 0
    fi
    if [[ -f "$PUSHT_DIR/val/demo.hdf5" ]]; then echo "  [skip] val exists"; else
        MUJOCO_GL="${PUSHT_MUJOCO_GL:-egl}" uv run python dev_scripts/collect/collect_pusht.py \
            --output-dir "$PUSHT_DIR/val" --num-success "$PUSHT_VAL" --num-failure "$PUSHT_VAL_FAIL" --seed 1000
    fi
fi

if want convert_pusht; then
    banner "convert_pusht  ($PUSHT_CACHE_FRAMES frames/trajectory, rgb only)"
    convert "$PUSHT_DIR/train/demo.hdf5"       $PUSHT_REPO pusht_train    "$PUSHT_TASK" $PUSHT_CACHE_FRAMES successful --no-pointmap
    convert "$PUSHT_DIR/val/demo.hdf5"         $PUSHT_REPO pusht_val      "$PUSHT_TASK" $PUSHT_CACHE_FRAMES successful --no-pointmap
    convert "$PUSHT_DIR/val/demo_failed.hdf5"  $PUSHT_REPO pusht_val_fail "$PUSHT_TASK" $PUSHT_CACHE_FRAMES failure    --no-pointmap
    preprocess $PUSHT_REPO $PUSHT_CACHE_FRAMES false pusht_train pusht_val pusht_val_fail
fi

if want train_pusht; then
    banner "train_pusht  (rgb, ${FRAME_SIZE}px, $PRECISE_MAX_STEPS steps)"
    train precise_pusht_rgb rgb $PUSHT_CACHE_FRAMES \
        "data.train_datasets=[${PUSHT_REPO}_pusht_train]" \
        "data.eval_datasets=[${PUSHT_REPO}_pusht_val]" \
        "custom_eval.reward_alignment=[${PUSHT_REPO}_pusht_val]" \
        "logging.save_best.metric_names=[eval_rew_align/pearson_${PUSHT_REPO}_pusht_val]"
fi

if want eval_pusht; then
    banner "eval_pusht"
    evaluate "$EVAL_OUTPUT_DIR/precise_pusht" \
        "[${PUSHT_REPO}_pusht_val]" \
        "[[${PUSHT_REPO}_pusht_val,${PUSHT_REPO}_pusht_val_fail]]" \
        "$(pick_best_ckpt $PRECISE_OUTPUT_DIR/precise_pusht_rgb)"
fi

# ================================================================ Experiment 3

LIFT_REPO=precise_lift_rbm
LIFT_DIR="$OUT/lift_collect"

if want collect_lift; then
    banner "collect_lift  (per height: $LIFT_TRAIN_PER_HEIGHT train; $LIFT_VAL_PER_HEIGHT + $LIFT_VAL_FAIL_PER_HEIGHT val)"
    for h in $HEIGHTS; do
        if [[ -f "$LIFT_DIR/train/h$h/demo.hdf5" ]]; then echo "  [skip] train h$h exists"; else
            MUJOCO_GL="${LIFT_MUJOCO_GL:-glx}" "$MIMICGEN_PY" dev_scripts/collect/collect_lift.py \
                --height-cm "$h" --output-dir "$LIFT_DIR/train/h$h" --camera-size "$FRAME_SIZE" \
                --num-success "$LIFT_TRAIN_PER_HEIGHT" --num-failure 0 --seed "$h"
        fi
        if [[ -f "$LIFT_DIR/val/h$h/demo.hdf5" ]]; then echo "  [skip] val h$h exists"; else
            MUJOCO_GL="${LIFT_MUJOCO_GL:-glx}" "$MIMICGEN_PY" dev_scripts/collect/collect_lift.py \
                --height-cm "$h" --output-dir "$LIFT_DIR/val/h$h" --camera-size "$FRAME_SIZE" \
                --num-success "$LIFT_VAL_PER_HEIGHT" --num-failure "$LIFT_VAL_FAIL_PER_HEIGHT" --seed "$((1000 + h))"
        fi
    done
fi

lift_subsets=()
for h in $HEIGHTS; do lift_subsets+=("lift_h${h}_train" "lift_h${h}_val" "lift_h${h}_val_fail"); done

if want convert_lift; then
    banner "convert_lift  ($LIFT_CACHE_FRAMES frames/trajectory, rgb + pointmap, one instruction per height)"
    for h in $HEIGHTS; do
        task="lift the cube to $h cm"
        convert "$LIFT_DIR/train/h$h/demo.hdf5"      $LIFT_REPO "lift_h${h}_train"    "$task" $LIFT_CACHE_FRAMES successful --camera agentview
        convert "$LIFT_DIR/val/h$h/demo.hdf5"        $LIFT_REPO "lift_h${h}_val"      "$task" $LIFT_CACHE_FRAMES successful --camera agentview
        convert "$LIFT_DIR/val/h$h/demo_failed.hdf5" $LIFT_REPO "lift_h${h}_val_fail" "$task" $LIFT_CACHE_FRAMES failure    --camera agentview
    done
    preprocess $LIFT_REPO $LIFT_CACHE_FRAMES true "${lift_subsets[@]}"
fi

join() { local IFS=,; echo "$*"; }
lift_train=(); lift_val=(); lift_pairs=(); lift_val_fail=()
for h in $HEIGHTS; do
    lift_train+=("${LIFT_REPO}_lift_h${h}_train")
    lift_val+=("${LIFT_REPO}_lift_h${h}_val")
    lift_val_fail+=("${LIFT_REPO}_lift_h${h}_val_fail")
    lift_pairs+=("[${LIFT_REPO}_lift_h${h}_val,${LIFT_REPO}_lift_h${h}_val_fail]")
done
first_height=${HEIGHTS%% *}

if want train_lift; then
    banner "train_lift  (rgb / pointmap / rgb_pointmap, ${FRAME_SIZE}px, use_lang_token=true, mixed heights)"
    for modality in rgb pointmap rgb_pointmap; do
        train "precise_lift_${modality}" "$modality" $LIFT_CACHE_FRAMES \
            "model.precise.use_lang_token=true" \
            "data.train_datasets=[$(join "${lift_train[@]}")]" \
            "data.eval_datasets=[$(join "${lift_val[@]}")]" \
            "custom_eval.reward_alignment=[$(join "${lift_val[@]}")]" \
            "logging.save_best.metric_names=[eval_rew_align/pearson_${LIFT_REPO}_lift_h${first_height}_val]"
    done
fi

if want eval_lift; then
    banner "eval_lift  (per height, then pooled)"
    # Per height, then one pooled group. Policy ranking pools by *task string*, so the
    # pooled entry ranks within each height and averages -- the honest pooled number.
    reward_alignment="[$(join "${lift_val[@]}"),[$(join "${lift_val[@]}")]]"
    policy_ranking="[$(join "${lift_pairs[@]}"),[$(join "${lift_val[@]}"),$(join "${lift_val_fail[@]}")]]"
    evaluate "$EVAL_OUTPUT_DIR/precise_lift" "$reward_alignment" "$policy_ranking" \
        "$(pick_best_ckpt $PRECISE_OUTPUT_DIR/precise_lift_rgb)" \
        "$(pick_best_ckpt $PRECISE_OUTPUT_DIR/precise_lift_pointmap)" \
        "$(pick_best_ckpt $PRECISE_OUTPUT_DIR/precise_lift_rgb_pointmap)"
fi

banner "done"
echo "  push-T eval : $EVAL_OUTPUT_DIR/precise_pusht/all_metrics.json"
echo "  lift eval   : $EVAL_OUTPUT_DIR/precise_lift/all_metrics.json"
echo "  videos      : bash dev_scripts/eval/plot_all_qualitative.sh"
