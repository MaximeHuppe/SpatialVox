---
id: "B12"
kind: "stage-b"
status: "done"
date: "2026-09-23"
run: "runs/mask-valid-seed1"
git: "f2067a6 + the uncommitted change set of _update_ideas/2026-09-22-null-head-decides-emptiness.md"
corpus: "data/synthetic-mri"
stage_a: "[[A09 mri-stage-a]]"
parent: "[[B09 mri-stage-b]]"
change: "train.stage_b.mask_on: valid — the mask term supervises only prompts that name a structure; the null head alone decides emptiness"
assumption: "If the empty-target supervision is what makes the carver shy, the held-out empty rate falls towards 0 and held-out Dice rises above B09's at matched epochs by more than the replicate noise"
epochs: "30 / 30"
seeds: 1
metric: "val Dice, targets.train (10)"
score: 0.9616
floor: 0.1383
heldout_val: 0.7268
heldout_test: 0.7740
delta: 0.154
delta_on: "held-out val (targets.val), mean per epoch over matched epochs 0–10 against B09; +0.208 over 14 matched epochs against the B11 replicate"
benefit: "yes"
verdict: "The shy painting is gone: held-out empty rate 0% at every epoch. Held-out Dice RISES through training, to 0.727 / 0.774 at best.pt (epoch 22), where B09 reached 0.368 / 0.460. +0.15 / +0.18 at matched epochs 0–10 against 0.09 / 0.07 of noise. Trained classes 0.962. One seed; the impossible-prompt leak and the per-class held-out Dice are not measured yet"
tags:
  - experiment
  - stage-b
  - synthetic
  - transfer
---
# B12 mask-valid-seed1

> [!abstract] Verdict
> **`mask_on: valid` removed the shy painting and reversed the transfer curve.**
> - Held-out masks are never empty; B09's `best.pt` had about 25% empty.
> - Held-out Dice now rises through training instead of peaking at epoch 0: 0.727 on val and 0.774 on test at `best.pt` (epoch 22), where B09 reached 0.368 and 0.460, and 0.738 / 0.796 at the end.
> - At matched epochs 0–10 the gain is +0.15 on val and +0.18 on test, against 0.09 / 0.07 of replicate noise.
> - Trained classes are also higher: 0.962 against 0.937.
> - **One seed.** The price, how many impossible prompts slip past the null head, is not measured yet.

## Question
Does the carver stop being shy when it is never rewarded for painting nothing? Prediction, written before the run:
- **If empty-target supervision causes the shy painting:**
  - `val:targets.val.empty_rate` falls well below B09's ~25% (CPU evaluation of B09's `best.pt`);
  - held-out val Dice beats B09's at matched epochs by more than 0.091, the mean per-epoch |Δ| of the B09/B11 replicate pair, and stays above its 0.247 floor;
  - supervised Dice holds at about 0.94.
- **If the held-out masks stay empty,** the silence comes from class-specific features, not from the empty targets, and the next change is target diversity.
- **If masks appear on the wrong structure,** the shyness is fixed but the painting still copies trained shapes.

## Setup
- **Command:**
  ```
  setsid nohup .venv/bin/python scripts/train.py b --config configs/synthetic-hard.yaml \
    --segmenter runs/mri-stage-a/best.pt --out runs/mask-valid-seed1 \
    --set logging.wandb.name=stageb-mri16-mask-valid-s1 \
    --set "logging.wandb.tags=['stage-b','synthetic-mri','16-classes','transfer','arm:mask-valid']"
  ```
  Launched from `dev-SpatialVox-V1` on 2026-09-23 at 00:09, main PID 139968, detached. Console log: `runs/mask-valid-seed1/console.log`.
  wandb: [stageb-mri16-mask-valid-s1](https://wandb.ai/imag2/spatial-vox/runs/em9h065h).
- **The one change against the parent:** `train.stage_b.mask_on: valid`, from the config. B09 ran the equivalent of `all`.
- **Everything else matches B09:**
  - same config file, seed 20260915, 30 epochs, cosine schedule with warm-up 2, batch 16, `flip_probability` 0.25, same loss weights;
  - `field_centroid_on: always` and `carver_sees_anchors: true`;
  - Stage A [[A09 mri-stage-a]] run live, since there is no anchor cache, exactly as B09.
  - The carver's mask head is now computed in two exact halves, the same function to 1.9e-6.
- **Check that it applied:** `runs/mask-valid-seed1/best.json` → `meta.config.stage.mask_on == "valid"`.
- **New per-epoch columns to read:**
  - `empty_rate`: prompts that name a structure but get no mask, for val and each held-out population;
  - `dice_null_gated`: Dice after the null head's gate.

## Result
All 30 epochs; the run finished 2026-09-23 at 04:36. `best.pt` is epoch 22, selected on trained classes as CLAUDE.md §5 requires. Floors: held-out val 0.2467, held-out test 0.1617.

| epoch | supervised | floor | held-out val | held-out test | held-out val empty | centroid | `flip_direction` drop | `permute_both` drop |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.7894 | 0.1383 | 0.4478 | 0.4981 | 0.2% | 2.02 | 0.645 | -0.0009 |
| 1 | 0.8694 | 0.1383 | 0.4285 | 0.5245 | 0.0% | 1.39 | 0.733 | +0.0005 |
| 2 | 0.9021 | 0.1383 | 0.3093 | 0.4753 | 0.0% | 1.10 | 0.748 | -0.0006 |
| 3 | 0.9066 | 0.1383 | 0.3899 | 0.4721 | 0.0% | 1.37 | 0.781 | -0.0001 |
| 4 | 0.9297 | 0.1383 | 0.3495 | 0.5278 | 0.2% | 0.87 | 0.776 | +0.0016 |
| 5 | 0.9344 | 0.1383 | 0.5161 | 0.6162 | 0.0% | 1.38 | 0.771 | +0.0000 |
| 6 | 0.9365 | 0.1383 | 0.3721 | 0.4683 | 0.0% | 1.07 | 0.776 | +0.0011 |
| 7 | 0.9315 | 0.1383 | 0.5111 | 0.6517 | 0.0% | 1.24 | 0.800 | +0.0010 |
| 8 | 0.9486 | 0.1383 | 0.5662 | 0.6611 | 0.0% | 1.65 | 0.789 | +0.0007 |
| 9 | 0.9504 | 0.1383 | 0.6020 | 0.7089 | 0.0% | 1.40 | 0.775 | +0.0029 |
| 10 | 0.9551 | 0.1383 | 0.5595 | 0.6869 | 0.0% | 1.16 | 0.819 | +0.0024 |
| 11 | 0.9541 | 0.1383 | 0.5863 | 0.7032 | 0.0% | 1.39 | 0.804 | +0.0012 |
| 12 | 0.9549 | 0.1383 | 0.6491 | 0.7177 | 0.0% | 1.46 | 0.779 | +0.0002 |
| 13 | 0.9576 | 0.1383 | 0.5514 | 0.6649 | 0.0% | 1.16 | 0.832 | +0.0002 |
| 14 | 0.9520 | 0.1383 | 0.6027 | 0.6671 | 0.2% | 0.89 | 0.784 | -0.0011 |
| 15 | 0.9599 | 0.1383 | 0.6375 | 0.7308 | 0.0% | 1.42 | 0.823 | +0.0001 |
| 16 | 0.9600 | 0.1383 | 0.6594 | 0.7322 | 0.0% | 1.46 | 0.814 | -0.0002 |
| 17 | 0.9585 | 0.1383 | 0.6892 | 0.7507 | 0.0% | 1.48 | 0.808 | +0.0002 |
| 18 | 0.9600 | 0.1383 | 0.7054 | 0.7628 | 0.0% | 1.30 | 0.821 | +0.0000 |
| 19 | 0.9594 | 0.1383 | 0.7214 | 0.7830 | 0.0% | 1.47 | 0.823 | +0.0017 |
| 20 | 0.9558 | 0.1383 | 0.6839 | 0.7612 | 0.0% | 1.42 | 0.845 | +0.0000 |
| 21 | 0.9600 | 0.1383 | 0.7335 | 0.7839 | 0.0% | 1.45 | 0.834 | -0.0004 |
| 22 (best) | 0.9616 | 0.1383 | 0.7268 | 0.7740 | 0.0% | 1.44 | 0.823 | +0.0006 |
| 23 | 0.9608 | 0.1383 | 0.7274 | 0.7854 | 0.0% | 1.40 | 0.829 | +0.0002 |
| 24 | 0.9609 | 0.1383 | 0.7165 | 0.7917 | 0.0% | 1.43 | 0.837 | +0.0002 |
| 25 | 0.9607 | 0.1383 | 0.7172 | 0.7818 | 0.0% | 1.48 | 0.840 | +0.0003 |
| 26 | 0.9602 | 0.1383 | 0.7302 | 0.7909 | 0.0% | 1.45 | 0.841 | +0.0001 |
| 27 | 0.9602 | 0.1383 | 0.7365 | 0.7951 | 0.0% | 1.42 | 0.840 | +0.0000 |
| 28 | 0.9602 | 0.1383 | 0.7384 | 0.7969 | 0.0% | 1.47 | 0.841 | +0.0001 |
| 29 | 0.9603 | 0.1383 | 0.7378 | 0.7963 | 0.0% | 1.45 | 0.840 | +0.0001 |

## Against the parent
B09 at matched epochs (0–10), and the B11 replicate for 0–16.

**Interim (epochs 0–10, one seed).**
- Held-out val: **+0.154** mean per epoch against B09 (0.459 vs 0.306), and +0.177 against B11. It is higher at 9 of 11 epochs; the exceptions are epoch 0 (tied) and epoch 2.
- Held-out test: **+0.176** against B09 (0.572 vs 0.396).
- The gap widens with training: +0.19 to +0.37 over epochs 7–10, while B09 stays between 0.24 and 0.37.
- Held-out empty rate: **0.0–0.2% at every epoch**.
- Supervised: 0.955 against B09's 0.937 at epoch 10, higher at every epoch from 1 on; within 0.022 of noise per epoch, but consistent.
- Noise for comparison: 0.091 held-out val, 0.065 held-out test. The effect is about twice the replicate's per-epoch spread, but it is still one seed.

**Two numbers that are not comparable to B09.**
- *Train* Dice (0.78 at epoch 10, against 0.94) still averages over the ~18% of training prompts that name nothing, where the ungated carver now paints and so scores 0.
- `loss_mask` (0.041 against 0.113) no longer contains those prompts.
- The flip drop is also not a like-for-like comparison. Under `all`, a flip that names nothing produced silence and so the maximal drop. Under `valid` it produces a best guess. It still removes 86% of the Dice (91% for B09). Noise: mean |Δ| per epoch 0.022 supervised, 0.091 held-out val, 0.065 held-out test ([[Result_tracker#Noise]]). One seed: a Δ below about 0.1 is not evidence.

## Reading
- **The predicted signature appeared in full.**
  - The held-out empty rate stays between 0 and 0.2% for 30 epochs.
  - Held-out val climbs from 0.45 to 0.74, with a mean of 0.732 over the last 5 epochs. The B09/B10/B11 family never trended up, and its best held-out val was 0.47.
  - The old pattern, transfer best at epoch 0 and then decaying, is reversed.
- **The baseline never ran this long.** B09 stopped at epoch 11 and B11 at 17, so past epoch 16 there is nothing to compare against. The matched-epoch gain (epochs 0–16) is the defensible number, and the late climb is part of what the change bought.
- **The gate costs little on valid prompts.** The null head rejects 1.7% of held-out val prompts, so the null-gated Dice is 0.714 against 0.727 ungated.
- **Not measured yet:**
  - the per-class held-out Dice (does the prism twin carry the test score?);
  - the image-replacement test;
  - the leak on impossible prompts, with and without the null gate.
  All three come from `scripts/evaluate.py` on `best.pt`.
- **Next:** that evaluation, and a second seed.

Afterwards, Afterwards, run `scripts/evaluate.py` on `best.pt` for both held-out populations. Read the gated and ungated numbers, the empty-prompt leak with and without the null gate, and the image-replacement test: held-out Dice should now **fall** when another scene's image is swapped in.

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

