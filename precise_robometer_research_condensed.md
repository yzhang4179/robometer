# Precise Robometer — Condensed Reference

Study: progress/success prediction from **rgb only**, **pointmap only**, **rgb+pointmap**.
Everything is new code inside `robometer/`; **no original robometer file was edited.**

---

## 1. Data

**Every split has both `demo.hdf5` and `demo_failed.hdf5`** (measured, not assumed), so each
validation dir is a self-contained success-vs-failure ranking set.

| split | robot / camera | #succ | #fail | shift axis |
|---|---|---:|---:|---|
| `demo_src_square_task_D1` | Panda / agentview | 577 | 25 | **train** |
| `..._D1_validation` | Panda / agentview | 35 | 25 | **in-distribution** (also the training eval) |
| `..._D2_validation` | Panda / agentview | 25 | 25 | layout shift |
| `..._D1_sideview_validation` | Panda / **sideview** | 41 | 23 | viewpoint shift |
| `..._D1_robot_validation` | **UR5e** / agentview | 28 | 25 | embodiment shift |

All 8 validation halves are converted; cache names are `precise_square_rbm_square_{d1,d2,sideview,robot}_val[_fail]`.

Root: `/home/reward/Desktop/precise_robometer/environments/mimicgen/datasets/robometer_core_datasets/square`

Measured facts:
- `agentview_pointmap` **already exists** in the hdf5: `(T,256,256,3) float32`, camera-frame XYZ (`[...,2] == depth`). No pointmap generation needed.
- Sideview split uses `sideview_image` / `sideview_pointmap` keys — the converter auto-detects the camera prefix.
- **No vertical flip.** These frames render upright (unlike LIBERO); flipping would mirror RGB against its pointmap.
- `datagen_info/robometer_progress` is **ignored entirely**; progress comes from robometer's `compute_progress_from_segment`.
- Trajectories are 134–171 frames, downsampled to 32 in the cache.

### How the pointmap is stored (the schema question)

Published RBM schema keeps its 8 columns + **one** new column `pointmap_path` → a sidecar
`.npz` (`pointmap` float16 raw metres, `shape`, `num_frames`, `frame_indices`).
Processed cache adds the `pointmap` key to the **same** `trajectory_<id>.npz`:

```text
trajectory_<id>.npz
  frames    (32,256,256,3) uint8     <- the key stock robometer reads (RGB path unchanged)
  pointmap  (32,256,256,3) float16   <- camera-frame XYZ, RAW METRES
```

- float16 is 2–8× finer than what the bf16 VAE consumes; ~6.5 MB/traj compressed, ~4.0 GB for train+val.
- RGB and pointmap are sliced with the **same index vector**, so they cannot drift.
- Rejected: a second MP4 (H.264 is 8-bit lossy, destroys metric XYZ) and uint16 quantisation (bakes bounds into the cache).

---

## 2. Architecture

**Tokenizer:** frozen **Wan2.2 TI2V-5B VAE** (`diffusers.AutoencoderKLWan`).
`z_dim=48`, temporal ×4, spatial ×16 → `T_lat = 1 + (T-1)//4`. At 256×256, one latent frame is `48×16×16`.
Decoder is **dropped** (555 M of 705 M params); VAE is a plain attribute, **not in `state_dict`**.

**Transformer:** decoder-only, ~19.5 M trainable (`hidden_dim=512`, `num_layers=6`, `latent_patch_size=2`).

### Token layout (9 frames → 3 latent frames), rgb+pointmap

```text
[   0:128 ] B1 = [t1_rgb(64), t1_pm(64)]         latent frame 1
[ 128:129 ] B2 = [prog_1]
[ 129:257 ] B3 = [t2_rgb(64), t2_pm(64)]         latent frame 2
[ 257:261 ] B4 = [prog_2..prog_5]
[ 261:389 ] B5 = [t3_rgb(64), t3_pm(64)]         latent frame 3
[ 389:393 ] B6 = [prog_6..prog_9]
seq = 393 (patch 2) / 1545 (patch 1)
```

**Mask: purely block-causal.** Every block sees itself in full + everything before it.
`True` = blocked (ReWiND convention). Progress blocks are **bidirectional inside**
(`prog_block_causal: false`) — `prog_2..prog_5` read the same latent frame, so ordering them
removes context for no gain.

### Position encoding (the 2-modality question)

Three separate signals — the mistake is conflating them:
1. **One shared 3-D RoPE** over `(t,h,w)` latent-grid coords. RGB and pointmap at the same `(t,h,w)` get the **same** position — that is what binds them. Do *not* flatten into one long 1-D sequence.
2. **Additive learned modality vectors** `emb_rgb` / `emb_pointmap` / `emb_prog`. Since positions are shared, this is the *only* thing distinguishing modalities — required, not optional.
3. **Progress tokens**: RoPE slot `(t = block's latent index, h = H_lat, w = W_lat)` just outside the visual grid, plus a learned `prog_token[k]` from a `(1, max_frames, D)` table — the only signal telling `prog_2..prog_5` which frame they predict.

Separate input projections `rgb_proj` / `pm_proj` (`Linear(48*p*p → hidden)`), not shared — different statistics.

### Pointmap normalisation — frozen bounds, clamp in exactly one place

```yaml
model.precise.pointmap_norm_bounds:   # metres, camera frame
  x: [-0.6, 0.6]
  y: [-1.0, 0.25]
  z: [0.2, 1.3]
```

Bounds are **hand-set to the workspace, not derived from data** (data min/max would put `z_max`
at the back wall ~2.7 m and squeeze the table into the bottom quarter). ~6–9% saturates, all far background.

The clamp lives **only** in `WanLatentTokenizer.normalize_pointmap`, on GPU, right before the encode:

```text
hdf5 → sidecar npz → cache npz → sampler → collator      ALL RAW METRES, NO CLAMP
                                              ↓
   WanLatentTokenizer.normalize_pointmap   <<< THE CLAMP >>>  clamp → 2*(x-lo)/(hi-lo)-1 → bf16
                                              ↓ frozen Wan VAE → [B,48,3,16,16]
```

Cache stays raw so bounds remain a CLI knob (no re-conversion for a sweep).
RGB path: `x_uint8/127.5 - 1`, no clamp. Resizing is **NEAREST, never bilinear**.

---

## 3. Key corrections & gotchas (all measured)

1. **The frozen VAE is 82–97% of every training step.** `latent_patch_size` is a *modelling*
   choice, not a throughput one — patch 1 costs only 12–18% more wall clock, not 4×.
   rgb+pointmap costs ~2× rgb because it runs the VAE **twice**, not because the sequence is longer.
   Fusing both streams into one VAE call: no faster (972 vs 973 ms), ~2× peak memory. Left as two calls.

   | modality | patch | seq | VAE ms | rest ms | total ms | peak GB |
   |---|---|---|---:|---:|---:|---:|
   | rgb | 2 | 201 | 486 | 17 | 503 | 5.13 |
   | rgb+pm | 2 | 393 | 972 | 35 | 1007 | 5.20 |
   | rgb+pm | 1 | 1545 | 973 | 216 | 1189 | 6.98 |

2. **`max_frames` must be `4k+1`.** Wan encodes `1 + (T-1)//4` chunks; 16 frames silently drops
   frames 14–16 — no error. `check_frame_count` now rejects anything else. 9 and 5 are valid; **4 is not**.

3. **Temporal leakage is real and unfixable by masking.** Latent frame 2 is built from input
   frames 2–5, so `prog_2..prog_4` see their own future. Measured: perturbing frame 5 moves
   `prog_2/3/4` but `|Δprog_1| = 0.00000` (the mask genuinely works).
   **Decided: supervise all 9 tokens, and log a last-token-only metric** — that is the number matching Step 4.

4. **Two `save.py` traps, both reproduced and fixed.** `ModelConfig` is hardcoded there, so a
   `precise` field is invisible: **resume silently drops it** (`save.py:217`) and **eval hard-crashes**
   (`save.py:906`). Fixed with add-only `update_precise_cfg_from_ckpt` + `load_precise_model`,
   which refuse to start on a mismatch. Keeping `precise` a plain dict also means `yaml.safe_load` suffices.

5. **Sideview does not fit the frozen bounds.** Only **28%** of its depth falls inside `z:[0.2,1.3]`
   (agentview: 88.8%), and the 1.3 ceiling cuts through its *workspace*. Bounds were **not changed** —
   they are right for train/val. Step 4 renders sideview **twice**, the second with
   `z:[0.9,2.2]`, and reports both.

6. **Environment:** no system `ffmpeg` (scripts symlink the one inside `imageio-ffmpeg`); this venv's
   `torchao` breaks on any `transformers`/`diffusers` import, so every entry script starts with
   `try: import unsloth / except: pass` — meaning even the data scripts need a visible GPU.

7. **Kendall τ degenerates with two quality tiers — it equals `2·ranking_acc − 1`.**
   Robometer's `policy_ranking` is built for three tiers (`failure:1, suboptimal:2, successful:3`).
   `_compute_policy_ranking_metrics_quality_label` draws one trajectory per tier via `product(...)`
   and computes τ-a over **k points, k = number of tiers**. With k=2 every τ is ±1, so the average
   is exactly `2·ranking_acc − 1`; `kendall_rewind` (τ over per-tier *mean* reward) collapses to a
   single bit and reads 1.0 even when half the pairs are misordered. Verified on real data:
   rgb/d1 gave `kendall_last=0.9817`, `ranking_acc_last=0.9909` → `2(0.9909)−1 = 0.9818`. ✓
   **Report `ranking_acc_failure_vs_successful` and `avg_succ_fail_diff` beside τ; skip `kendall_rewind`.**
   Note also `combinations(present_labels, k)` at [:774] is a no-op loop — choosing k from k yields
   exactly one tuple. Mentally delete it when reading that function.

8. **Where each metric lives.** `succ_fail_diff` and Kendall exist **only** in `policy_ranking`,
   never in `reward_alignment` — which drops failures at the dataset level
   (`filter_quality_labels=["successful"]`, `custom_eval.py:30`) and skips non-successful rows in
   its accumulator anyway. **VOC = `pearson`** from `reward_alignment`.

9. **The two evals sample a different number of clips per episode by default.**
   `RewardAlignmentSampler` honours `subsample_n_frames` (9 endpoints). `ProgressPolicyRankingSampler`
   does **not** — it strides by `frame_step`, giving 32 prefixes on our 32-frame cache. Since
   `kendall_avg`/`kendall_sum` average over however many prefixes they get, the two would not be
   comparable. `PrecisePolicyRankingSampler` adds the knob, so **both use 9 endpoints × 9 frames**,
   endpoints `[0,3,7,11,15,19,23,27,31]` — literally the same clips.

Smaller ones: preference head deleted (would never get a gradient — FSDP hazard);
`partial_success` types as `null` (fine unless concatenated with a stock RBM dataset);
eval thinning uses robometer's `linspace_subsample_frames` (rounds) not `np.linspace(dtype=int)` (truncates).

---

## 4. New files (nothing in robometer was edited)

**Model / config**
| file | what |
|---|---|
| `robometer/models/precise/wan_tokenizer.py` | frozen Wan VAE, `normalize_rgb`, `normalize_pointmap` (**the only clamp**), `encode` |
| `robometer/models/precise/precise_transformer.py` | 3-D RoPE, modality embeddings, prog tokens, block-causal attention, heads |
| `robometer/models/precise/layout.py` | single source of truth for token positions + who attends to whom |
| `robometer/configs/precise_experiment_configs.py` | `PreciseModelConfig` / `PreciseExperimentConfig` |
| `robometer/configs/precise_transformer.yaml` | the preset (frozen bounds live here) |
| `robometer/utils/precise_setup_utils.py` | `build_precise_model` / `load_precise_model` (the `save.py` fix) |

**Data**
| file | what |
|---|---|
| `dataset_upload/data_scripts/mimicgen_precise/hdf5_to_rbm.py` | stage A: hdf5 → published RBM (mp4 + pointmap sidecar) |
| `robometer/data/scripts/preprocess_precise.py` | stage B: published → processed cache, pointmap folded in |
| `robometer/data/samplers/precise_progress.py` | `PreciseProgressSampler` — index-carrier trick |
| `robometer/data/samplers/eval/precise_reward_alignment.py` | aligned RGB/pointmap prefix sampler |
| `robometer/data/samplers/eval/precise_policy_ranking.py` | same, for policy ranking; **adds the `subsample_n_frames` the stock sampler lacks** |
| `robometer/data/collators/precise.py`, `datasets/precise_data.py` | collator + dataset |
| `robometer/data/dataset_success_cutoff_precise.txt` | all `1.0` (sim has exact trajectory ends) |

**Train / eval**
| file | what |
|---|---|
| `train_precise.py` | forked entry point (repo root) |
| `robometer/trainers/precise_trainer.py` | forward/loss/eval integration |
| `robometer/evals/precise_eval_server.py` | `PreciseInferenceEngine` + FastAPI server |
| `robometer/evals/precise_baseline_eval.py` + `configs/reward_model/precise.yaml` | `reward_model=precise` |
| `dev_scripts/` | `test_dataloader.py`, `test_precise_model.py`, `bench_memory.py`, `train_all_precise.sh`, `eval/run_precise_eval.py`, `eval/run_all_eval.sh`, `eval/run_reward_alignment.sh`, `eval/run_policy_ranking.sh` |

---

## 5. Commands

### Setup (every session)

```bash
cd /home/reward/Desktop/precise_robometer/robometer
export SQUARE=/home/reward/Desktop/precise_robometer/environments/mimicgen/datasets/robometer_core_datasets/square
export ROBOMETER_DATASET_PATH=/home/reward/Desktop/precise_robometer/datasets
export ROBOMETER_PROCESSED_DATASETS_PATH=/home/reward/Desktop/precise_robometer/processed_datasets
export CUDA_VISIBLE_DEVICES=0
```

### Step 1 — data (~5 min total)

```bash
# A. hdf5 -> published RBM, one per split
uv run python -m dataset_upload.data_scripts.mimicgen_precise.hdf5_to_rbm \
    --hdf5 $SQUARE/demo_src_square_task_D1/demo.hdf5 \
    --output-dir $ROBOMETER_DATASET_PATH/precise_square_rbm \
    --subset square_d1_train --data-source precise_square_d1_train --max-frames 32

uv run python -m dataset_upload.data_scripts.mimicgen_precise.hdf5_to_rbm \
    --hdf5 $SQUARE/demo_src_square_task_D1_validation/demo.hdf5 \
    --output-dir $ROBOMETER_DATASET_PATH/precise_square_rbm \
    --subset square_d1_val --data-source precise_square_d1_val --max-frames 32

# B. published -> processed cache (--max_frames_for_preprocessing must be >= --max-frames above)
uv run python -m robometer.data.scripts.preprocess_precise \
    --train_datasets '["precise_square_rbm"]' --train_subsets '[["square_d1_train"]]' \
    --eval_datasets  '["precise_square_rbm"]' --eval_subsets  '[["square_d1_val"]]' \
    --max_frames_for_preprocessing 32 --cache_dir $ROBOMETER_PROCESSED_DATASETS_PATH

# C. verify (alignment / raw-metres / clip-coverage checks + preview png)
uv run python dev_scripts/test_dataloader.py --dataset precise_square_rbm_square_d1_train --with-sampler
uv run python dev_scripts/test_dataloader.py --dataset precise_square_rbm_square_d1_train --precise-sampler
```

Optional, to re-pick bounds before freezing them:

```bash
uv run python dev_scripts/inspect_pointmap_bounds.py \
    --hdf5 $SQUARE/demo_src_square_task_D1/demo.hdf5 \
    --hdf5 $SQUARE/demo_src_square_task_D1_sideview_validation/demo.hdf5 \
    --bounds x=-0.6,0.6 y=-1.0,0.25 z=0.2,1.3 --out-dir dev_scripts/out/pointmap_bounds
```

### Step 2 — model checks (no dataset needed with `--synthetic`)

```bash
uv run python dev_scripts/test_precise_model.py --synthetic --stage tokenizer   # downloads Wan VAE, 2.8 GB
uv run python dev_scripts/test_precise_model.py --synthetic --stage all --check-leakage
uv run python dev_scripts/test_precise_model.py --synthetic --modality rgb
uv run python dev_scripts/test_precise_model.py --synthetic --latent-patch-size 1
```

### Step 3 — training (~12–14 h for all three)

Contract: batch 16, `data.max_frames=9`, 10 discrete bins, progress-only samples `[0,1,0]`,
target `absolute_first_frame`, strategies `[0,1,1,1]`, all 9 tokens supervised, reward-alignment eval only.

```bash
uv run python dev_scripts/bench_memory.py --batch-sizes 8 16 24 32        # optional

# 10-step smoke -> logs/precise/precise_smoke_debug/final/
uv run python train_precise.py debug=true training.max_steps=10 \
    training.exp_name=precise_smoke training.overwrite_output_dir=true logging.log_to=[]

uv run python train_precise.py model.precise.modality=rgb           training.exp_name=precise_rgb
uv run python train_precise.py model.precise.modality=pointmap      training.exp_name=precise_pointmap
uv run python train_precise.py model.precise.modality=rgb_pointmap  training.exp_name=precise_rgb_pointmap

bash dev_scripts/train_all_precise.sh                       # all three, sequential
bash dev_scripts/train_all_precise.sh logging.wandb_mode=offline

# resume (modality must match the checkpoint; path must start with ./)
uv run python train_precise.py model.precise.modality=rgb training.exp_name=precise_rgb \
    training.resume_from_checkpoint=./logs/precise/precise_rgb/checkpoint-1000
```

Estimates at 20k steps: rgb ≈ 3.0–3.5 h, pointmap ≈ 3.0–3.5 h, rgb+pointmap ≈ 6.0–6.7 h.

### Step 4 — inference (~15–20 min total)

Both scripts **auto-discover the best-pearson checkpoint** at launch; override with `RGB_CKPT=` / `PM_CKPT=` / `BOTH_CKPT=`.

**Part 1 — failure videos, clip length 5** (3 panels: RGB | progress | success, 1280×384):

```bash
MAX_DEMOS=2 SAVE_DIR=./dev_scripts/out/eval_debug bash dev_scripts/eval/run_all_eval.sh   # smoke
bash dev_scripts/eval/run_all_eval.sh                                                     # full sweep

CLIP_LENGTH=9 bash dev_scripts/eval/run_all_eval.sh
SUCCESS_THRESHOLD=0.3 bash dev_scripts/eval/run_all_eval.sh
RGB_CKPT=./logs/precise/precise_rgb/final PM_CKPT=./logs/precise/precise_pointmap/final \
  BOTH_CKPT=./logs/precise/precise_rgb_pointmap/final bash dev_scripts/eval/run_all_eval.sh
```

Output tree (4 task folders — sideview rendered twice, frozen vs rebounded `z:[0.9,2.2]`; the
rebounded one is a *control*, not a correction — see the sideview box in part 3):

```text
dev_scripts/out/eval/<task>/metrics.json
dev_scripts/out/eval/<task>/demo_N/{rgb_only,pointmap_only,rgb_pointmap}.mp4 + predictions.json
```

**Part 2 — reward alignment (pearson = VOC) on the successful demos, clip length 9:**

```bash
MAX_TRAJ=3 OUTPUT_DIR=./baseline_eval_output/precise_debug bash dev_scripts/eval/run_reward_alignment.sh
bash dev_scripts/eval/run_reward_alignment.sh                      # d1 only

# all four shift axes in one process
EVAL_DATASET="precise_square_rbm_square_d1_val,precise_square_rbm_square_d2_val,precise_square_rbm_square_sideview_val,precise_square_rbm_square_robot_val" \
OUTPUT_DIR=./baseline_eval_output/precise_voc bash dev_scripts/eval/run_reward_alignment.sh
```

Expanded form (**single-quote each path** — best-checkpoint dirs contain `=` and Hydra splits on it):

```bash
uv run python robometer/evals/precise_baseline_eval.py \
    reward_model=precise \
    "model_paths=['./logs/precise/precise_rgb/final','./logs/precise/precise_pointmap/final','./logs/precise/precise_rgb_pointmap/final']" \
    custom_eval.eval_types=[reward_alignment] \
    custom_eval.reward_alignment=[precise_square_rbm_square_d1_val] \
    custom_eval.use_frame_steps=true custom_eval.subsample_n_frames=9 \
    custom_eval.reward_alignment_max_trajectories=null \
    max_frames=9 model_config.batch_size=32
```

**Measured, 2026-09-12** — pearson is VOC. Successes only (35 / 25 / 41 / 28 trajectories).

| model | d1 (in-dist) | d2 (layout) | sideview (view) | robot (UR5e) |
|---|---:|---:|---:|---:|
| rgb | 0.9958 | **0.9809** | **0.7062** | **0.9884** |
| pointmap | 0.9963 | 0.7768 | 0.3790 | 0.9726 |
| rgb+pointmap | **0.9966** | 0.9288 | 0.6160 | 0.9839 |

All three are near-ceiling in distribution (0.996) and separate only under shift. Pointmap alone
degrades hardest. The sideview column was rerun with sideview-fitted bounds to test whether that
number is just a clipping artifact — **it is not**; see the box below the ranking table.

### Step 4 part 3 — policy ranking: Kendall τ, ranking accuracy, succ-fail diff (~5 min)

The **only** evaluation that sees failures. Each entry loads a success cache and a failure
cache together; they share a `task` string, which is what gives that task two quality tiers.

```bash
bash dev_scripts/eval/run_policy_ranking.sh                       # 3 models x 4 shift axes
SPLITS=d1 bash dev_scripts/eval/run_policy_ranking.sh             # in-distribution only
MAX_PER_TIER=3 bash dev_scripts/eval/run_policy_ranking.sh        # quick smoke
```

Expanded form:

```bash
uv run python robometer/evals/precise_baseline_eval.py \
    reward_model=precise \
    "model_paths=['./logs/precise/precise_rgb/final','./logs/precise/precise_pointmap/final','./logs/precise/precise_rgb_pointmap/final']" \
    custom_eval.eval_types=[policy_ranking] \
    "custom_eval.policy_ranking=[[precise_square_rbm_square_d1_val,precise_square_rbm_square_d1_val_fail]]" \
    custom_eval.use_frame_steps=true custom_eval.subsample_n_frames=9 \
    custom_eval.num_examples_per_quality_pr=null custom_eval.policy_ranking_max_tasks=null \
    max_frames=9 model_config.batch_size=32 \
    output_dir=./baseline_eval_output/precise_ranking
```

**Measured, 2026-09-12** — best-pearson checkpoints, all trajectories, 9 clips x 9 frames.
`kendall_last` is `2*ranking_acc_last - 1` by construction (two tiers), so read `rank_acc` and
`succ-fail` as the informative columns.

| model | d1 (in-dist) | d2 (layout) | sideview (view) | robot (UR5e) |
|---|---|---|---|---|
| | rank_acc / succ-fail | rank_acc / succ-fail | rank_acc / succ-fail | rank_acc / succ-fail |
| rgb | 0.991 / 0.517 | 0.818 / 0.319 | 0.764 / 0.207 | 0.956 / 0.509 |
| pointmap | 0.937 / 0.365 | 0.632 / 0.091 | 0.694 / 0.124 | **0.991** / 0.258 |
| rgb+pointmap | **0.998** / 0.519 | **0.896** / 0.405 | **0.805** / 0.131 | 0.981 / **0.523** |

Trajectory counts: d1 60, d2 50, sideview 64, robot 53.

Three things this says. **rgb+pointmap wins on three of four axes**, and by the widest margin
under layout shift (0.896 vs 0.818 rgb). **Pointmap alone is the weakest everywhere except the
UR5e split**, where it is the best of the three at separating success from failure ordering --
consistent with geometry being robot-agnostic in a way RGB appearance is not. **Sideview is the
hardest axis for every model.**

#### Sideview is not a clipping artifact -- refitting the bounds makes it *worse*

The frozen agentview bounds saturate ~72% of sideview depth, so the obvious suspicion was that the
pointmap models look bad there only because their input arrives as a near-uniform +1 image. Rerun
with sideview-fitted bounds `x:[-1.0,1.0] y:[-1.2,0.5] z:[0.9,2.2]`:

| model | rank_acc frozen -> refit | VOC frozen -> refit |
|---|---|---|
| rgb (ignores bounds) | 0.7635 -> 0.7635 | 0.7062 -> 0.7062 |
| pointmap | 0.6935 -> **0.6013** | 0.3790 -> **0.3394** |
| rgb+pointmap | 0.8049 -> **0.6681** | 0.6160 -> **0.4162** |

rgb being bit-identical both ways is the free consistency check that the override reached only the
pointmap streams. Both pointmap models got **worse**, so the explanation is the opposite of the
suspicion: **the bounds are part of the input contract the model trained under.** Changing them at
test time is itself a distribution shift, and a larger one than the saturation it removes. The
model learned to read pointmaps with the workspace sitting in a particular part of `[-1,1]`;
refitting moves it somewhere it has never seen. Saturated-but-familiar beats unsaturated-but-novel.

(One counter-current worth noting rather than over-reading: the *success head* improved under refit
-- `success_auprc` 0.158 -> 0.390 for pointmap, 0.182 -> 0.316 for rgb+pointmap -- while progress
correlation fell. Refit helps detecting "done", hurts estimating "how far along".)

Consequence for Step 4 part 1: rendering sideview twice was the right call, but the second folder
is evidence that per-camera bounds are **not** a free fix, not a better view of the same data.

Converting a new failure split (the `--quality-label` flag already existed):

```bash
uv run python -m dataset_upload.data_scripts.mimicgen_precise.hdf5_to_rbm \
    --hdf5 $SQUARE/demo_src_square_task_D1_validation/demo_failed.hdf5 \
    --output-dir $ROBOMETER_DATASET_PATH/precise_square_rbm \
    --subset square_d1_val_fail --data-source precise_square_d1_val_fail \
    --quality-label failure --max-frames 32

uv run python -m robometer.data.scripts.preprocess_precise \
    --eval_datasets '["precise_square_rbm"]' \
    --eval_subsets '[["square_d1_val_fail"]]' \
    --max_frames_for_preprocessing 32 --cache_dir $ROBOMETER_PROCESSED_DATASETS_PATH
```

Optional HTTP server (writes nothing itself; for online rollout):

```bash
uv run python robometer/evals/precise_eval_server.py \
    model_path=./logs/precise/precise_rgb_pointmap/final num_subsampled_frames=5 server_port=8000
uv run python dev_scripts/eval/precise_server_client.py \
    --hdf5 $SQUARE/demo_src_square_task_D1_validation/demo_failed.hdf5 --num-demos 2 --frame-stride 4
```

---

<!-- ## 6. Eval semantics worth remembering

- **`use_frame_steps=true`** = growing prefixes `0:1 … 0:T`, keep only the **last** progress token.
  That is the deployment protocol *and* the only leak-free token.
- **`subsample_n_frames`** = number of prefix endpoints per trajectory (9).
  **`max_frames`** = frames inside each clip (9). Different knobs that share a value.
- **`reward_alignment_max_trajectories`** = per-dataset cap; `null` = all. We use `null` (35 trajectories).
- **Clip 5 is not clip 9 truncated.** Same window, half the temporal density — a frame spacing
  the model never trained on. Do not quote part-1 and part-2 curves as the same thing.
- **Part-1 metrics are failures-only**, so `success_auprc` / `positive_success_acc` are reported
  as `null` (not `0.0`). What is reported: `negative_success_acc`, `false_positive_rate`,
  `false_alarm_demo_rate`, `mean_peak_success_prob`, `mean_final_progress` (context only).
  Per-frame and per-demo can disagree sharply — 5% of frames was 20% of demos in testing.
- **Selection bias:** best-pearson checkpoints were selected on `precise_square_rbm_square_d1_val`,
  the same split part 2 scores. Fair *between* models, not a clean held-out number.
  Re-run with the three `final/` dirs (~2 min) for the unbiased read; report both. -->

<!-- ## 7. Disk

| artifact | per traj | 577 train + 35 val |
|---|---:|---:|
| published mp4 (RGB) | ~46 KB | ~28 MB |
| pointmap sidecar (deletable after stage B) | ~5.3 MB | ~3.2 GB |
| **processed cache npz** (RGB + pointmap) | **~6.5 MB** | **~4.0 GB** |

Step 4 videos: 288 mp4s → ~100–200 MB. -->

---

## 6. Step 5 — Experiments 2 & 3 (push-T, lift-to-N-cm)

Same question as exp 1 on two tasks with no existing data. Model, trainer and both
evaluations are reused; the new work is collection, one model variant, and two runners.
Verified end to end at `MAX_DEMOS=2` (collect → convert → 10-step train → both evals →
qualitative videos) for both experiments.

### What was built (nothing in robometer edited)

| file | what |
|---|---|
| `robometer/models/precise/lang_precise_transformer.py` | `LangPreciseTransformer`: one language token as block **B0**, ahead of every visual block. `use_lang_token=false` (default) → plain `PreciseTransformer`, zero extra params, exp-1 checkpoints unchanged. |
| `dev_scripts/collect/pusht_env.py` | push-T in **plain MuJoCo** (same `pushT.xml`, same reset/reward/`done`, `n_frames=5`) + a top-down 224×224 camera. No jax/brax/clone needed. |
| `dev_scripts/collect/pusht_policy.py` | **MPPI** over `mujoco.rollout` (64 samples × 20 steps). 40/40 success, ~55 control steps mean. Weak variant (8×6, noise 0.5) produces near-miss failures. |
| `dev_scripts/collect/collect_pusht.py` | episodes → `demo.hdf5` / `demo_failed.hdf5` + `success_rate.json`; frames every 4 physics steps (25 fps) → ~65 frames per success. |
| `dev_scripts/collect/lift_hnn.py` | `Lift_H{5,10,15,20}` (success = `cube_z − cube_z_at_reset > N`), goal-marker disc in its **own geom group**, `MG_Lift` interface. |
| `dev_scripts/collect/collect_lift.py` | `WaypointTrajectory` reach→grasp→lift; failure modes `miss` / `under` / `drop`; writes `agentview_image`, `agentview_pointmap`, `agentview_goal_image` from one rollout. |
| `dev_scripts/run_all_exp23.sh` | collect → convert → train → eval, both experiments, skip-if-exists per stage, `ONLY=` to pick stages. |
| `dev_scripts/eval/plot_all_qualitative.sh` | full-length videos, `{successful,failure}` folders, goal-marker RGB panel for lift. |

Small changes to our own files: `hdf5_to_rbm.py` `--no-pointmap` / `--lang-embedding`;
`run_precise_eval.py` `--quality-label` / `--camera` / `--display-camera` / `--task` and the
rgb-only `find_camera` fix; `precise_eval_server.py` `set_instruction`; `PreciseBatchCollator`
forwards `lang_vector`; `train_precise.py` dataset guard is now relational (train ≠ eval,
reward_alignment == eval) instead of pinned to the square caches.

### Decisions that are not in exp2.md

1. **Push-T needs no Brax.** The Brax env is `pushT.xml` + `n_frames=5` + three small functions;
   stepping the same XML with `mj_step` gives the same task *and* renders. Only the task
   definition has to match, not an integrator's rounding — the data is ours from scratch.
2. **Push-T policy is MPPI, not a hand-written pusher.** Greedy stroke search converged in
   ~1000 steps; MPPI in ~55 (median 34; heavy tail, min 9). Kept as is.
3. **Goal marker cannot leak into depth.** It lives in geom group 2, which
   `MjvOption.geomgroup` switches off *before* the scene is built. Measured: RGB differs by
   3.7e5, depth by exactly 0.0, plain render bit-identical before/after.
4. **224 is a Hydra override, not a yaml edit.** `precise_transformer.yaml` still says 256,
   so exp-1 reruns are untouched; the runner passes `model.precise.frame_size=224`.
5. **Lift needs its own pointmap bounds.** The lift scene's `z` runs 0.45–2.72 m; exp-1's
   frozen bounds clip 11–15 % of `z`. Run `inspect_pointmap_bounds.py` on
   `lift_collect/train/h5/demo.hdf5` and pass `model.precise.pointmap_norm_bounds=…` to the
   two pointmap runs before the real training.
6. **Upstream bug avoided:** mimicgen `waypoint.py:211` (`skip_interpolation=True`) reads an
   undefined `gripper`; `collect_lift.py` builds fixed segments with
   `WaypointSequence.from_poses` instead.

### Gotchas

- robosuite's EGL context fails on this machine (`EGLGLContext has no _context`); use
  `MUJOCO_GL=glx` for robosuite and `MUJOCO_GL=egl` for plain MuJoCo (2.6 ms/frame). The
  runner sets both.
- `debug=true` renames the run dir to `<exp>_debug`, which the runners' checkpoint lookup
  does not follow — smoke with `PRECISE_MAX_STEPS=10` instead.
- Two pixel checks now exist: the model raises if frames ≠ `frame_size`, and the push-T
  render is square because `grid_h == grid_w` is hardcoded.

### Commands

```bash
# everything, both experiments (≈ 1 h collection + 4 × 20k-step trainings + evals)
bash dev_scripts/run_all_exp23.sh

# smoke (≈ 20 min)
MAX_DEMOS=2 PRECISE_MAX_STEPS=10 PRECISE_BATCH_SIZE=2 EXTRA_TRAIN="logging.wandb_mode=offline" \
    bash dev_scripts/run_all_exp23.sh

# one stage at a time
ONLY="collect_pusht convert_pusht" bash dev_scripts/run_all_exp23.sh
ONLY="train_lift" bash dev_scripts/run_all_exp23.sh model.precise.pointmap_norm_bounds='{x:[..],y:[..],z:[..]}'

# qualitative videos (full episode length, successes/failures in separate folders)
bash dev_scripts/eval/plot_all_qualitative.sh
MAX_DEMOS=2 HEIGHTS="5 20" bash dev_scripts/eval/plot_all_qualitative.sh

# outputs
baseline_eval_output/precise_pusht/all_metrics.json     # VOC + ranking, push-T
baseline_eval_output/precise_lift/all_metrics.json      # per height h5/h10/h15/h20 and pooled
dev_scripts/out/eval_pusht/val/{successful,failure}/demo_N/rgb_only.mp4
dev_scripts/out/eval_lift/h{5,10,15,20}/{successful,failure}/demo_N/{rgb_only,pointmap_only,rgb_pointmap}.mp4
```
