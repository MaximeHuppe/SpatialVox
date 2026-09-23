# The null head decides emptiness; the carver always paints

**Date:** 2026-09-22 · **Branch:** `dev-SpatialVox-V1` (uncommitted on top of `f2067a6`) · **Status:** implemented and unit-tested. The first test run, **B12** (`runs/mask-valid-seed1`), ran on 2026-09-23 from 00:09 to 04:36. Held-out Dice is 0.727 / 0.774 at `best.pt`, against 0.368 / 0.460 for the baseline; one seed. See §6.

## In one paragraph

The carver no longer learns to paint nothing. Until now, the ~16% of training prompts that name no structure trained the mask towards an empty volume. That turned the carver into a second, image-based "names nothing" detector, and the detector also fired on real targets it had never been supervised on: that is the shy painting. A new setting, `train.stage_b.mask_on: valid` (now shipped in all three configs), trains the mask only on prompts that name a structure. Deciding "names nothing" is left to the null head alone, through a new gate `null_gated`, and every Dice is now reported both with and without that gate. Separately, the carver's output head was rewritten as an exact algebraic equivalent that uses 40% less peak memory in the carver. The function, parameters and checkpoints are unchanged.

---

## 1. Rationale: why this change

### 1.1 The bottleneck is the carver's decision to paint, not the region

The evidence comes from `runs/relational-seed1/eval_val_{train,val,test}` (real MRI, one seed):

| | trained classes | caudate + putamen (held-out) | hippocampus (held-out) |
|---|---|---|---|
| Dice | 0.794 | 0.005 | 0.001 |
| prompt-blind floor | 0.307 | 0.110 | 0.063 |
| masks that come out **empty** | 0% | **75%** | **73%** |
| predicted / true volume | 0.98 | 0.04 | 0.17 |
| region contains the target's centre (`where_raw > 0.5`) | 83% | 86% | 90% |
| null head says "names nothing" | 0.5% | 0.5% | 0% |

For held-out targets the region is right, and the null head says the prompt is valid. Yet the carver paints nothing three times in four. Held-out Dice sits *below* the prompt-blind floor.

### 1.2 Why it is silent: the carver learned its own rejection

- **The carver rejects by itself.** On 382 impossible prompts the carver emits a mask only **5.5%** of the time, while the null head flags just **63%** of them. The carver therefore holds a prompt-level "nothing here" decision that the null head, with only four inputs, cannot make.
- **Only one training signal can teach that rejection.** Only prompts whose target is an empty mask reward a completely empty answer: the flips that name nothing, about 0.25 × 65.5% ≈ **16%** of training prompts on `data/mri`, and about 18% on `synthetic-mri`.
- **On those prompts, any painting is very costly.** The soft Dice term (`src/engine.py::segmentation_loss`) on an empty target is `1 − 1/(Σp + 1)`, so one voxel's worth of probability already costs half of it. When unsure, silence or a tiny mask is the cheapest answer, and an unfamiliar structure is exactly where the carver is unsure.
- **The rejection is driven by the image.** Swapping in another subject's MRI *raises* held-out Dice (0.005 → 0.029). The no-image run (`prompt-only-seed1`), trained on the same empty prompts, does not go silent on held-out classes; it scores 0.117, at the floor.
- **The MRI-like synthetic corpus shows the same two symptoms.** CPU evaluation of `runs/mri-stage-b/best.pt`: 25% of held-out-val masks are empty, and the non-empty ones cover 0.54 of the true volume.

### 1.3 Why this is the single change

- **It targets the only training signal that rewards silence.** With every supervised mask non-empty, an empty or timid answer always costs the full Dice, so when unsure the carver's best move becomes painting its best candidate inside the region.
- **It breaks no invariant (CLAUDE.md §1–§4).** Nothing new reaches the model. The null head keeps training on every prompt, empty ones included. The flip keeps its re-scoring and its `keep` weight.
- **The other options did not move held-out Dice** (see the runs analysis):
  - leave-one-out: +0.04 against 0.09 of noise;
  - pretrained `B`: no effect;
  - oracle anchors: no gain.
- **Rival fixes aim at a different failure.** Removing the anchor masks from the carver targets the *recall* route, but on real MRI the dominant failure is silence, not painting the wrong class. Many more training shapes is the deeper fix for the carver copying trained shapes, but it is a corpus project. It becomes the next step if this change produces masks that land on the wrong structure.

### 1.4 The price, stated up front

Impossible prompts will now leak through whenever the null head misses them. Today the carver keeps 94.5% of them silent. Afterwards only the null head can, and its four inputs cap it at an AUC of about 0.85 on `data/mri` (CLAUDE.md §7). That is why every Dice and every empty-prompt leak is now also reported **gated**.

---

## 2. Changes applied to the code

### 2.1 `train.stage_b.mask_on`: the mask term supervises only valid prompts

| file | change |
|---|---|
| `src/engine.py` | New constant `MASK_ON = ("all", "valid")`, documented with the measurements above. New `StageBTask` field `mask_on: str = "all"`, validated in `__post_init__`; an unknown value raises. In `StageBTask.loss`, `mask_weight = keep * valid if self.mask_on == "valid" else keep` feeds the `mask` term (Dice + BCE). **No other term changes**: `null_bce`, `centroid`, `field_centroid` and `L_far` keep their weights exactly. |
| `scripts/train.py` | Passes `mask_on=str(stage_cfg.mask_on)` to `StageBTask`. It is a required key, like `field_centroid_on`. |
| `scripts/evaluate.py` | Passes the same setting. It only affects the reported loss. |
| `configs/config.yaml`, `configs/synthetic.yaml`, `configs/synthetic-hard.yaml` | `mask_on: valid`, with the rationale in a comment. **`mask_on: all` reproduces every run before this date.** The code default stays `all`, so a `StageBTask` built without the argument behaves exactly as before. |

It is a training-schedule value, so it is recorded in the checkpoint's `meta["config"]["stage"]` and the `.json` sidecar (CLAUDE.md §8). Old checkpoints are unaffected.

### 2.2 `null_gated`: the null head's decision, applied and reported

| file | change |
|---|---|
| `src/engine.py` | New `null_gated(probability, valid)`, which zeroes the mask of every sample whose null logit is ≤ 0. That is the same threshold `null_summary` scores. |
| `src/engine.py` (`Trainer.evaluate`, every epoch, Stage B only) | Two new metrics for every validation population (`val`, `val:targets.val`, `val:targets.test`). **`dice_null_gated`** is the Dice after the gate. **`empty_rate`** is the fraction of prompts that name a structure but get an empty mask: the shy-painting trend, logged every epoch. The epoch line prints `empty` for val and each held-out population. |
| `scripts/evaluate.py` | `score` adds `dice_null_gated` and `empty_prediction_rate_null_gated`. `empty_prompt_report` adds `false_positive_rate_null_gated` and `false_positive_voxels_null_gated`. Both are printed beside the ungated numbers. The existing keys are unchanged. `--save-masks` still writes the ungated carver output. |
| naming | The new keys say `null_gated`, not just `gated`, so they never share a name with `gate_fraction_*`: "the gate" of CLAUDE.md §6, `where_raw > 0.5` at the target centroid. `evaluate.py`'s own docstring warns against two numbers sharing a name. They were renamed before any run logged them. |
| `src/engine.py` and `SpatialVox.md` §14.2 | The comment above the `L_far` term said that on an impossible prompt "the empty-mask loss is already the penalty everywhere". That is now true only under `mask_on: all`, so it was reworded to cover both settings, in the code and in its verbatim copy in the document. |

### 2.3 Efficiency: the mask head in two exact halves

| file | change |
|---|---|
| `src/models.py` (`Carver.forward`) | Before: upsample the 16 feature channels to full resolution, concatenate `B(I)` into a `[B,32,D,H,W]` tensor, then apply the 1×1 `head`. After: apply the head's feature half (`head.weight[:, :width]` plus its bias) on the 64³ working grid, upsample **one** channel, and add the `B(I)` half (`head.weight[:, width:]`) at full resolution. |

**Why it is exact.** A 1×1 convolution and a trilinear upsample are both linear, and the upsample's weights sum to one (so the bias passes through):
`head(cat[up(f), B]) = W_f·up(f) + W_B·B + b = up(W_f·f + b) + W_B·B`.
The same `self.head` parameter is used, so the state dict is identical and every existing checkpoint loads and behaves the same.

**Measured** (a throwaway benchmark: the carver alone, forward and backward, bf16 autocast, NVIDIA GeForce RTX 5090 Laptop GPU, shipped widths, 10 timed repetitions after 3 warm-up):

| input | old | new | change |
|---|---|---|---|
| batch 4, 128³ (`data/mri`) | 219.1 ms, 3602 MiB | 204.1 ms, 2146 MiB | time −7%, **peak memory −40%** |
| batch 16, 64³ (`synthetic-mri`) | 105.5 ms, 1801 MiB | 97.2 ms, 1073 MiB | time −8%, **peak memory −40%** |

The difference between old and new logits is 1.9e-6 in fp32 at 128³ (logits up to 9.5). In bf16 it is 0.064, one bf16 rounding step at that magnitude. A float64 unit test holds it to 1e-10.

### 2.4 Tests

| test | holds |
|---|---|
| `tests/test_models.py::test_the_split_mask_head_is_exactly_the_1x1_on_the_upsampled_concatenation` (with and without the skip) | new head = literal concat + 1×1, float64, non-zero head |
| `tests/test_engine.py::test_the_mask_term_can_be_restricted_to_prompts_that_name_a_structure` | painting on an impossible prompt costs `all` and not `valid`; every other term is identical; a bad value raises |
| `tests/test_engine.py::test_the_null_gate_empties_exactly_the_prompts_the_null_head_rejects` | `valid ≤ 0` empties the mask, `valid > 0` leaves it untouched |
| `tests/test_engine.py::test_the_loop_trains_stage_b_and_writes_a_reloadable_checkpoint` (extended) | `dice_null_gated` and `empty_rate` are logged each epoch; gated ≤ ungated |
| `tests/test_evaluate.py::test_the_report_runs_end_to_end` (extended) | the gated evaluation and empty-prompt keys exist, and gating only removes masks |
| `tests/test_config.py::test_every_block_the_code_reads_is_present` (extended) | `mask_on` is in the shipped config |

Result: **152 passed** (69 in models, engine, evaluate and config, plus 83 in data, geometry, mapper, mri and synthetic), all on CPU.

### 2.5 Documentation kept in sync

- `documentation/SpatialVox.md`:
  - §12.2: the carver code, its shape table and the identity;
  - §14.2: the loss code, the mask row of the terms table and a callout on `mask_on`;
  - §16.2: gated reporting;
  - §17: the tensor table;
  - §20.4: two new deviations;
  - §21: the config table;
  - the "Open" list.
- `documentation/Flowchart.md` and `Flowchart.drawio`: three carver labels. The layout was not touched; a backup of the `.drawio` is in the session scratchpad.
- `documentation/Model Info/modules/Carver.md`, `NullHead.md` and `Phase B/MODEL PHASE B.md`.

---

## 3. Deliberately **not** changed, so the effect can be attributed

- **`field_centroid_on`** stays `always`. `empty-only` is a separate, still-unrun experiment (B06).
- **`carver_sees_anchors`** stays on. The no-anchor arm has never actually run with the flag off.
- **`flip_probability`** stays 0.25. The null head still needs the empty prompts.
- **The null head's inputs** are unchanged. CLAUDE.md §1 fixes them at four scalars, and showing it the MRI is forbidden.
- **Checkpoint selection** still uses the ungated trained-class Dice on held-out subjects (CLAUDE.md §5). Validation has no impossible prompts, so the gate barely changes it.

## 4. How to test it, and what would refute it

Nothing has been launched. When runs are allowed again, start on `synthetic-mri`, which is fast and has per-class held-out populations. Use two seeds. Keep `epochs: 30` so the learning-rate schedule matches `mri-stage-b` epoch for epoch, and stop after about 12 epochs. Two runs in parallel take about 3.6 h.

```bash
.venv/bin/python scripts/train.py b --config configs/synthetic-hard.yaml --out runs/mask-valid-seed1 \
  --set logging.wandb.name=stageb-mri16-mask-valid-s1
.venv/bin/python scripts/train.py b --config configs/synthetic-hard.yaml --out runs/mask-valid-seed2 \
  --set train.seed=20260916 --set logging.wandb.name=stageb-mri16-mask-valid-s2
```

`mask_on` is read from the config (`valid`), so no boolean override is needed. Booleans passed with `--set` must be written `True`/`False`, because lowercase `false` is a truthy string. The parent is `mri-stage-b` (B09), and the matched-epoch noise comes from the B09/B11 replicate pair: 0.091 on held-out val and 0.065 on held-out test.

**It worked if:**
- the held-out `empty_rate` falls towards 0; it was about 25% on `synthetic-mri` and 75% on `data/mri`;
- held-out val Dice clears its floor of 0.247 by more than 0.09, in both seeds;
- trained-class Dice holds at about 0.94;
- image replacement (`scripts/evaluate.py`) now *lowers* held-out Dice, meaning the image is used to paint rather than to reject;
- the gated empty-prompt leak is reported and understood; it should approach the null head's miss rate.

**It failed, or needs the next step, if:**
- held-out masks stay empty. Then silence does not come from the empty-target supervision but from class-specific features, and the fix is many more training shapes.
- masks appear but land on another structure. Today hollow_cylinder already puts 62% of its painted voxels on another structure, usually a trained one. The shyness is then fixed, but the painting still copies trained shapes, and the next change is again more training shapes.

**A 10-minute preview that needs no training.** Evaluate the existing checkpoint at thresholds below 0.5, for example `scripts/evaluate.py runs/relational-seed1/best.pt --split val --classes val --set train.threshold=0.1`. If correctly placed held-out masks appear, the carver can already find the target and is holding back, which is what this change releases.

## 5. Risks

- **The carver will now paint on impossible prompts.** If the null head is poorly calibrated, the gated false-positive rate rises. It is measured and reported; the null head's four-scalar ceiling is a known limit.
- **Without the rejection habit, the carver may over-paint valid prompts.** Dice penalises that, and `L_far` still bounds painting far from the field. Watch the predicted/true volume ratio in `scripts/evaluate.py`.
- **Every number will again be single-seed.** Hence two seeds from the start.

---

## 6. Log

### 2026-09-23 00:09 — first test run launched (B12)

- **Why:** at your request, a Stage B training with the Stage A of `runs/mri-stage-a` (A09), the segmenter trained on `data/synthetic-mri`. That is the Stage A the parent B09 and both arms used, and the one `configs/synthetic-hard.yaml` points to. It was passed explicitly with `--segmenter`.
- **Run:** `runs/mask-valid-seed1`, main PID 139968, detached with `setsid nohup`, console log in `runs/mask-valid-seed1/console.log`, wandb `stageb-mri16-mask-valid-s1` (https://wandb.ai/imag2/spatial-vox/runs/em9h065h).
- **Command:**
  ```bash
  setsid nohup .venv/bin/python scripts/train.py b --config configs/synthetic-hard.yaml \
    --segmenter runs/mri-stage-a/best.pt --out runs/mask-valid-seed1 \
    --set logging.wandb.name=stageb-mri16-mask-valid-s1 \
    --set "logging.wandb.tags=['stage-b','synthetic-mri','16-classes','transfer','arm:mask-valid']"
  ```
- **Only difference from B09:** `mask_on: valid`, read from the config. Seed 20260915 (B09's), 30 epochs, cosine schedule with warm-up 2, batch 16, and Stage A run live because `data/synthetic-mri` has no anchor cache, exactly as B09. One seed for now; a second seed can run beside it.
- **Tracker:** new note `documentation/Result_tracker/Experiments/B12 mask-valid-seed1.md`, written before launch with its prediction. The hub `Result_tracker.md` was also corrected: its "Needs attention" callout still said B10/B11 were running, which stopped being true on 2026-09-22, and B12 was added to the lineage graph. The B10/B11 notes themselves are unchanged.
- **How to stop it:** `kill -TERM -139968`, which signals the whole process group, workers included.

### 2026-09-23 01:46 — interim result, B12 epochs 0–10 (one seed, not yet a result)

At matched epochs against B09 (identical config and seed, only `mask_on` differs):

| | B12 `valid` | B09 `all` | Δ | replicate noise |
|---|---|---|---|---|
| held-out val, mean over epochs 0–10 | 0.459 | 0.306 | **+0.154** | 0.091 |
| held-out test, mean over epochs 0–10 | 0.572 | 0.396 | **+0.176** | 0.065 |
| held-out val at epoch 10 | 0.560 | 0.368 | +0.192 | |
| supervised at epoch 10 | 0.955 | 0.937 | +0.018 | 0.022 |
| held-out val empty rate | 0.0–0.2% at every epoch | ~25% at `best.pt` (CPU evaluation) | | |

Held-out val is higher in 9 of 11 epochs, and the gap widens (epochs 7–10: +0.19 to +0.37). This is consistent with the prediction in §4: the shyness disappears and held-out Dice rises. It is still one seed, and the price is not measured yet: the gated leak on impossible prompts needs `scripts/evaluate.py` on the final `best.pt`. *Train* Dice and `loss_mask` are no longer comparable with B09 (see the B12 tracker note).

### 2026-09-23 04:36 — B12 finished (30 epochs, one seed)

| | B12 `valid` | B09 `all` (best, epoch 10) | B11 replicate (best, epoch 12) |
|---|---|---|---|
| trained classes at `best.pt` | **0.962** (epoch 22) | 0.937 | 0.944 |
| held-out val at `best.pt` | **0.727** (null-gated 0.714) | 0.368 | 0.467 |
| held-out test at `best.pt` | **0.774** (null-gated 0.761) | 0.460 | 0.513 |
| held-out val, mean over the last 5 epochs | 0.732 | — (stopped at 11) | 0.321 (epochs 12–16) |
| held-out val empty rate | 0.0% at every epoch | ~25% (CPU evaluation) | not logged |
| held-out val centroid error | 5.0 mm | 6.5 mm | — |

**Reading.**
- The prediction of §4 held on every per-epoch criterion. The empty rate went to 0, and held-out val clears its floor (0.247) by far more than the noise: +0.15 over matched epochs 0–10, and +0.21 over the 14 matched epochs against the replicate.
- Trained classes held and improved.
- The transfer curve reversed. It used to be best at epoch 0 and then decay; now it rises through the whole schedule.

**Not yet established.**
- A second seed.
- The price: the impossible-prompt leak with and without the null gate.
- Whether the held-out gain is broad, or carried by the prism twin; this needs the per-class evaluation.
- The image-replacement test.

B09 never trained past epoch 11, so the only defensible comparison is at matched epochs.

### 2026-09-23 — evaluation of B12 `best.pt` (epoch 22, `--split val`, `configs/synthetic-hard.yaml`)

Reports are in `runs/mask-valid-seed1/eval_val_{train,val,test}/`.

| | trained classes | held-out val | held-out test |
|---|---|---|---|
| Dice / null-gated Dice | 0.962 / 0.950 | 0.724 / 0.714 | 0.775 / 0.763 |
| empty masks (ungated / gated) | 0% / 1.3% | 0% / 1.4% | 0% / 2.0% |
| predicted / true volume | 1.00 | 0.82 | 0.84 |
| centroid error | 1.4 mm | 5.0 mm | 4.8 mm |
| image replacement: Dice | 0.962 → 0.174 | 0.723 → 0.161 | 0.777 → 0.156 |
| impossible prompts that get a mask, ungated / gated | 100% / 30.5% | 100% / 32.3% | 100% / 29.0% |

Per class, against B09 `best.pt` (earlier CPU evaluation):
- banana 0.68 → **0.82**
- cross 0.36 → **0.80**
- hollow_cylinder 0.02 → **0.53**
- crescent 0.33 → **0.77**
- hourglass 0.21 → **0.68**
- triangular_prism 0.89 → 0.89

**Reading.**
- The gain is broad; the prism twin is not carrying it. The prism was already at 0.89 and stayed there, while the five other held-out shapes rose by 0.14 to 0.51.
- The image is now used to paint: swapping in another scene drops held-out Dice by 78%, where it used to raise it.
- The counterfactuals still hold, with `permute_both` unchanged.
- The price is as predicted. The carver paints on every impossible prompt, and the null head removes only about 69% of those (a gated leak of 29–32%, against 5.5% before, on `data/mri`).
- The centroid does not "hold" under image replacement (5 → 10 mm): the heatmap also reads the image.

