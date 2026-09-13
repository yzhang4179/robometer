# Precise Robometer — Experiments 2 & 3 (as built)

Same question as exp 1 (**rgb vs pointmap vs rgb+pointmap**) on two tasks that had no data.
Model, trainer and both evaluations are reused from exp 1. **Status: built and verified end to end**
(collect → convert → train → both evals → videos) at `MAX_DEMOS=2`; 50-episode collections done for
both tasks. Nothing in the original robometer codebase was edited.

| | exp 2 — push-T | exp 3 — lift to N cm |
|---|---|---|
| env | `pushT.xml` from model-based-diffusion, stepped in **plain MuJoCo** | robosuite `Lift` subclasses `Lift_H{5,10,15,20}` |
| policy | **MPPI** over `mujoco.rollout` | mimicgen `WaypointTrajectory` reach → grasp → lift |
| modalities | rgb | rgb, pointmap, rgb+pointmap (+ language token) |
| resolution | 224×224 (square, required) | 224×224 |
| cache frames / clip | 64 / 9 | 32 / 9 |
| data targets (runner defaults) | 500 train, 80 val, 40 val-fail | 75 train + 15 val + 15 val-fail **per height** |
| training runs | 1 | 3 |

---

## Important updates

1. **Push-T needs no Brax/jax/clone.** The upstream env is `pushT.xml` + `n_frames=5` + three small
   functions (reset, reward, `done = reward > 0.95`), all reproduced in `pusht_env.py`. Same XML,
   same task; the data is generated from scratch, so the integrator's rounding is irrelevant.
2. **Push-T policy is MPPI** (64 samples × 20-step horizon, `mujoco.rollout`, multithreaded).
   A hand-written pusher and a greedy stroke search took ~600–1000 steps and were dropped.
   Success **30/30** (strong planner); a deliberately weak planner (8 × 6, execution noise 0.5)
   supplies near-miss failures. **Kept as is — no more tuning.**
3. **Language token** (`LangPreciseTransformer`): one token from the 384-d `lang_vector` as block
   **B0**, ahead of every visual block, so the block-causal mask needs no change.
   `use_lang_token=false` (default) builds the plain model with zero extra parameters — exp-1
   checkpoints load and score unchanged. Needed because "lift to 5 cm" and "lift to 20 cm" are
   pixel-identical inputs with different labels.
4. **Goal marker never reaches depth.** It lives in geom group 2, which `MjvOption.geomgroup`
   switches off before the scene is built; `goal_marker_visible()` turns it on for one extra RGB
   render of the same state. Measured: RGB changes by 3.7e5, **depth by 0.0**. Each lift episode
   stores `agentview_image` + `agentview_pointmap` (model input) and `agentview_goal_image`
   (videos only) from one rollout.
5. **224 px is a Hydra override** (`model.precise.frame_size=224` in the runner); the yaml stays
   256 so exp-1 reruns are untouched. Grid 14×14 → 7×7 = 49 tokens/frame/stream, 303-token
   sequence vs 393. `forward` now raises if frames ≠ `frame_size`.
6. **Lift needs its own pointmap bounds.** Its `z` spans 0.45–2.72 m; exp-1's frozen bounds clip
   11–15 % of it. Fit them before the real training (command below).
7. **Eval fixes / flags.** rgb-only eval no longer crashes (`find_camera` required a pointmap);
   `run_precise_eval.py` gained `--quality-label`, `--camera`, `--display-camera`, `--task`.
   `train_precise.py` no longer pins the square caches (train ≠ eval, reward_alignment == eval).
8. **Episode length.** Push-T frames are stored every 4 physics steps (25 fps) so a success
   averages ~50–65 frames for the 64-frame cache; shorter episodes are kept whole (no padding,
   ≥ 9 frames needed). Lift episodes are 106 steps (112 for `drop`) by construction, so length
   carries no success information.

## Measured (50 episodes each, `dev_scripts/out/{pusht,lift}/`)

| | success | episode length |
|---|---|---|
| push-T, strong MPPI (30) | **100 %** | 38 control steps mean (8–176), 1.9 s, 48 frames |
| push-T, weak planner (20) | 5 % | 120 steps (timeout) |
| lift, nominal plan (35) | **100 %**, lifts 1.3–1.45 cm above target | 106 steps, 5.3 s |
| lift, injected `under`/`drop`/`miss` (15) | 0 % | 106 / 112 / 106 steps |

Per-episode videos: `dev_scripts/out/{pusht,lift}/demo_{id}_{success,fail}.mp4`; records in
`pusht/success_rate.json` and `lift/summary.json` (per height, per mode, per episode).

---

## Files

| new | what |
|---|---|
| `robometer/models/precise/lang_precise_transformer.py` | `LangPreciseTransformer` + config; `prepend_lang_block` |
| `dev_scripts/collect/pusht_env.py`, `assets/pushT.xml` | plain-MuJoCo push-T + top-down camera |
| `dev_scripts/collect/pusht_policy.py` | MPPI planner (`STRONG` / `WEAK`) |
| `dev_scripts/collect/collect_pusht.py` | episodes → `demo.hdf5` / `demo_failed.hdf5` / json / mp4s |
| `dev_scripts/collect/lift_hnn.py` | `Lift_H{5,10,15,20}`, goal marker, `MG_Lift` |
| `dev_scripts/collect/collect_lift.py` | waypoint planner, failure modes, 3 streams per episode |
| `dev_scripts/run_all_exp23.sh` | collect → convert → train → eval, both experiments |
| `dev_scripts/eval/plot_all_qualitative.sh` | full-length videos, `{successful,failure}` folders |

| changed (ours) | what |
|---|---|
| `precise_transformer.py` | `_prefix_tokens` hook, pixel-size check, `self.config_class` |
| `precise_setup_utils.py`, `precise_trainer.py`, `collators/precise.py` | pick the lang class, forward `lang_vector` |
| `precise_eval_server.py` | `needs_lang`, `set_instruction()` |
| `hdf5_to_rbm.py` | `--no-pointmap`, `--lang-embedding` |
| `run_precise_eval.py` | rgb-only fix, `--quality-label/--camera/--display-camera/--task` |
| `train_precise.py`, `dataset_success_cutoff_precise.txt` | relational dataset guard; push-T / lift entries |

---

## Commands

Paths default to `../datasets` and `../processed_datasets`; override with
`ROBOMETER_DATASET_PATH` / `ROBOMETER_PROCESSED_DATASETS_PATH`. Push-T renders with
`MUJOCO_GL=egl`, robosuite with `MUJOCO_GL=glx` (its EGL context fails here); the runner sets both.

### Everything

```bash
bash dev_scripts/run_all_exp23.sh                     # ~10 min push-T + ~2 h lift collection, 4 × 20k-step runs, evals
ONLY="collect_pusht convert_pusht" bash dev_scripts/run_all_exp23.sh   # stages: collect_pusht convert_pusht train_pusht eval_pusht
ONLY="train_lift eval_lift" bash dev_scripts/run_all_exp23.sh          #         collect_lift  convert_lift  train_lift  eval_lift
# stages skip themselves when their output exists; extra args are Hydra overrides for train_precise.py

# 20-minute smoke of the whole pipeline (do NOT use debug=true: it renames the run dir)
MAX_DEMOS=2 PRECISE_MAX_STEPS=10 PRECISE_BATCH_SIZE=2 EXTRA_TRAIN="logging.wandb_mode=offline" \
    bash dev_scripts/run_all_exp23.sh
```

### Before the real lift training — fit the pointmap bounds

```bash
uv run python dev_scripts/inspect_pointmap_bounds.py \
    --hdf5 dev_scripts/out/lift_collect/train/h5/demo.hdf5 --hdf5 dev_scripts/out/lift_collect/train/h20/demo.hdf5 \
    --bounds x=-0.6,0.6 y=-1.0,0.25 z=0.2,1.3 --out-dir dev_scripts/out/pointmap_bounds
ONLY=train_lift bash dev_scripts/run_all_exp23.sh 'model.precise.pointmap_norm_bounds={x:[..,..],y:[..,..],z:[..,..]}'
```

### Collection by hand

```bash
# push-T: 50 episodes, every one kept, one mp4 each
MUJOCO_GL=egl uv run python dev_scripts/collect/collect_pusht.py \
    --output-dir dev_scripts/out/pusht --video-dir dev_scripts/out/pusht \
    --num-success 50 --num-failure 50 --max-attempts 50 --iterations 1 --seed 7

# lift, one height (~13 s per episode); --video-start-index lets the four heights share one folder
MUJOCO_GL=glx ~/miniconda3/envs/mimicgen/bin/python dev_scripts/collect/collect_lift.py \
    --height-cm 5 --output-dir dev_scripts/out/lift/h5 --video-dir dev_scripts/out/lift --video-start-index 0 \
    --num-success 13 --num-failure 13 --max-attempts 13 --iterations 1
```

### Convert + cache by hand

```bash
# push-T (rgb only, 64 frames)
uv run python -m dataset_upload.data_scripts.mimicgen_precise.hdf5_to_rbm \
    --hdf5 dev_scripts/out/pusht_collect/train/demo.hdf5 --output-dir $ROBOMETER_DATASET_PATH/precise_pusht_rbm \
    --subset pusht_train --task "push the T block onto the target" --lang-embedding --max-frames 64 --no-pointmap
uv run python -m robometer.data.scripts.preprocess_precise \
    --train_datasets '["precise_pusht_rbm"]' --train_subsets '[["pusht_train","pusht_val","pusht_val_fail"]]' \
    --max_frames_for_preprocessing 64 --require_pointmap=false --cache_dir $ROBOMETER_PROCESSED_DATASETS_PATH

# lift (rgb + pointmap, 32 frames, one instruction per height; failures get --quality-label failure)
uv run python -m dataset_upload.data_scripts.mimicgen_precise.hdf5_to_rbm \
    --hdf5 dev_scripts/out/lift_collect/train/h5/demo.hdf5 --output-dir $ROBOMETER_DATASET_PATH/precise_lift_rbm \
    --subset lift_h5_train --camera agentview --task "lift the cube to 5 cm" --lang-embedding --max-frames 32
uv run python dev_scripts/test_dataloader.py --dataset precise_lift_rbm_lift_h5_train   # alignment / metres / coverage
```

### Train by hand (what the runner does)

```bash
uv run python train_precise.py training.exp_name=precise_lift_rgb model.precise.modality=rgb \
    model.precise.frame_size=224 model.precise.use_lang_token=true \
    'data.train_datasets=[precise_lift_rbm_lift_h5_train,precise_lift_rbm_lift_h10_train,precise_lift_rbm_lift_h15_train,precise_lift_rbm_lift_h20_train]' \
    'data.eval_datasets=[precise_lift_rbm_lift_h5_val,precise_lift_rbm_lift_h10_val,precise_lift_rbm_lift_h15_val,precise_lift_rbm_lift_h20_val]' \
    'custom_eval.reward_alignment=[precise_lift_rbm_lift_h5_val,precise_lift_rbm_lift_h10_val,precise_lift_rbm_lift_h15_val,precise_lift_rbm_lift_h20_val]' \
    'logging.save_best.metric_names=[eval_rew_align/pearson_precise_lift_rbm_lift_h5_val]' \
    training.max_steps=20000 training.per_device_train_batch_size=20
# push-T: exp_name=precise_pusht_rgb, modality=rgb, no use_lang_token, the pusht_train / pusht_val caches
```

### Evaluate (VOC + ranking; lift per height and pooled)

```bash
uv run python robometer/evals/precise_baseline_eval.py reward_model=precise \
    "model_paths=['logs/precise/precise_lift_rgb/final','logs/precise/precise_lift_pointmap/final','logs/precise/precise_lift_rgb_pointmap/final']" \
    custom_eval.eval_types=[reward_alignment,policy_ranking] \
    'custom_eval.reward_alignment=[precise_lift_rbm_lift_h5_val,[precise_lift_rbm_lift_h5_val,precise_lift_rbm_lift_h20_val]]' \
    'custom_eval.policy_ranking=[[precise_lift_rbm_lift_h5_val,precise_lift_rbm_lift_h5_val_fail]]' \
    custom_eval.use_frame_steps=true custom_eval.subsample_n_frames=9 custom_eval.pad_frames=true max_frames=9 \
    output_dir=baseline_eval_output/precise_lift
# a nested list = one pooled group. Policy ranking pools by task string, so a group with all four
# heights ranks within each height and averages -- that is the honest pooled number.
```

### Qualitative videos (full episode length, successes and failures separate)

```bash
bash dev_scripts/eval/plot_all_qualitative.sh                  # MAX_DEMOS=2 HEIGHTS="5 20" ONLY=lift for a subset
uv run python dev_scripts/eval/run_precise_eval.py --hdf5 dev_scripts/out/lift_collect/val/h5/demo.hdf5 \
    --model rgb=logs/precise/precise_lift_rgb/final --model pointmap=logs/precise/precise_lift_pointmap/final \
    --save-dir dev_scripts/out/eval_lift --task-name h5 --quality-label successful \
    --camera agentview --display-camera agentview_goal --task "lift the cube to 5 cm" --num-subsampled-frames 9
```

### Outputs

```text
baseline_eval_output/precise_pusht/all_metrics.json
baseline_eval_output/precise_lift/all_metrics.json           # h5 / h10 / h15 / h20 and pooled
dev_scripts/out/eval_pusht/val/{successful,failure}/demo_N/rgb_only.mp4
dev_scripts/out/eval_lift/h{5,10,15,20}/{successful,failure}/demo_N/{rgb_only,pointmap_only,rgb_pointmap}.mp4
dev_scripts/out/{pusht,lift}/demo_{id}_{success,fail}.mp4    # raw collection videos (lift: goal-marker view)
```

---

## Gotchas

- `MUJOCO_GL`: `egl` for plain MuJoCo (2.6 ms/frame), `glx` for robosuite (its EGL context fails on
  this machine). `nvidia-smi` shows an NVML mismatch; CUDA works.
- `debug=true` appends `_debug` to the run dir, which the runners' checkpoint lookup does not
  follow. Smoke with `PRECISE_MAX_STEPS=10` instead.
- Clip length must be `4k+1` (9); 64 and 32 are cache sizes, never the clip.
- mimicgen `waypoint.py:211` (`skip_interpolation=True`) reads an undefined `gripper`;
  `collect_lift.py` builds fixed segments with `WaypointSequence.from_poses` instead.
- Robosuite's stock lift success (`table + 0.04`) is only ~2 cm of real lift because the cube's
  origin already sits 2.1 cm above the table; `Lift_HNN` measures `cube_z − cube_z_at_reset`.
- Lift failures are all injected (`--failure-rate 0.25`: `miss` / `under` / `drop`); the nominal
  plan never misses. Push-T failures come from the weak planner (mostly 120-step timeouts).

## Next

1. `bash dev_scripts/run_all_exp23.sh` (collection + push-T training run unattended).
2. After `collect_lift`, fit the lift pointmap bounds and pass them to `train_lift`.
3. Report `all_metrics.json` for both, lift per height and pooled, and the videos.
