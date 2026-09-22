---
id: "B03"
kind: "stage-b"
status: "done"
date: "2026-09-22"
run: "runs/relational-seed1"
git: "008da30"
corpus: "data/mri"
stage_a: "[[A01 phase-a-current]]"
parent: "[[L01 legacy attention Stage B]]"
change: "The relational architecture replaces the attention Stage B: frozen Stage A, detached soft masks, parameter-free mapper, then B(I), carver and null head. B from scratch, predicted anchors from the cache"
assumption: "The model does the relational task from the image, and transfers to target classes it was never supervised on"
epochs: "20 / 20"
seeds: 1
metric: "val Dice, targets.train (8 classes, 600 val-split prompts)"
score: 0.7812
floor: 0.3067
heldout_val: 0.0053
heldout_test: 0.0006
delta:
delta_on: "not comparable with L01 (leaky occupancy input, different architecture). Read against the floor: +0.4745"
benefit: "mixed"
verdict: "Reads the prompt and uses the image on supervised classes (0.781 vs a 0.307 floor; permute probes go to exactly 0.0000). Does NOT transfer: 0.005 vs a 0.110 floor, with 75% empty masks"
tags:
  - experiment
  - stage-b
  - mri
  - baseline
---
# B03 relational-seed1

> [!abstract] Verdict
> The **MRI baseline** of the relational architecture. On the classes it is supervised on, it reads the prompt and draws from the image. On held-out target classes it **falls silent**: Dice 0.0052 against a 0.1097 floor, with 75% of masks empty.

## Setup
`scripts/train.py b --out runs/relational-seed1 --set train.stage_b.epochs=20 --set train.stage_b.scheduler.warmup_epochs=2`. There are 5,855 supervised training prompts over 160 subjects, `B` starts from scratch, and predicted anchors come from the cache. 268,275 trainable parameters sit beside the frozen 17.0M. About 12 minutes per epoch. Its `metrics.jsonl` names the held-out curves `val_ood` (= `val:targets.val`) and `test_ood` (= `val:targets.test`). **Both are val-split subjects.**

## The curve
| epoch | supervised 8 | centroid | held-out 4 | held-out 2 | `flip_direction` drop | `permute_both` drop |
|---|---|---|---|---|---|---|
| 0 | 0.4368 | 5.16 mm | 0.1375 | 0.0198 | +0.384 | +0.0003 |
| 1 | 0.5508 | 3.45 mm | 0.0813 | 0.0100 | +0.456 | −0.0000 |
| 2 | 0.5718 | 2.71 mm | 0.0499 | 0.0022 | +0.500 | −0.0000 |
| 3 | 0.6002 | 2.58 mm | 0.0099 | 0.0006 | +0.545 | +0.0034 |
| 7 | 0.7206 | 2.05 mm | 0.0050 | 0.0003 | +0.656 | +0.0021 |
| 9 | 0.7412 | 2.10 mm | 0.0092 | 0.0006 | +0.655 | +0.0000 |
| 13 | 0.7753 | 1.86 mm | 0.0055 | 0.0007 | +0.696 | +0.0008 |
| 15 | 0.7794 | 1.76 mm | 0.0068 | 0.0008 | +0.692 | +0.0007 |
| 19 (best) | **0.7812** | 1.94 mm | 0.0053 | 0.0006 | +0.702 | +0.0011 |

Training Dice ends at 0.8385 (a gap of about 0.06), and the selection curve has been flat since epoch 15. The counterfactuals are decisive **from epoch 0**. On the selection curve then (base 0.4368), `permute_channels` and `permute_clauses` give 0.0000 (drop 0.4368), `flip_direction` gives 0.0528 (drop 0.3840), and `permute_both` gives 0.4365 (drop 0.0003). Losses at epoch 0 total 1.31: mask 0.79, `field_centroid` 0.30, `centroid` 0.14, `null_bce` 0.07, `far` 0.0001.

## The full report (`scripts/evaluate.py`, final checkpoint, val split)
| | supervised 8 | held-out 4 | held-out 2 |
|---|---|---|---|
| examples | 400 | 392 | 127 |
| Dice | **0.7943** | **0.0052** | 0.0006 |
| prompt-blind floor | 0.3067 | 0.1097 | 0.0630 |
| anchor-set ceiling, this population | 71.0% | 83.9% | 99.2% (uninformative) |
| centroid error | 1.72 mm | 24.86 mm | 28.32 mm |
| emitted an **empty** mask | 0.0% | **75.0%** | 73.2% |
| predicted / true voxels | 1597 / 1631 | **93** / 2219 | 389 / 2322 |
| anchor Dice | 0.8127 | 0.8046 | 0.8138 |
| gate, predicted anchor centroids | 0.8275 | 0.8571 | 0.8976 |
| HD95 | 2.61 mm | 27.62 mm | 26.67 mm |

| counterfactual (supervised 8) | resulting Dice | drop |
|---|---|---|
| `permute_channels` | **0.0000** | 0.7943 |
| `permute_clauses` | **0.0000** | 0.7943 |
| `flip_direction` | 0.0631 | 0.7312 |
| `permute_both` (control) | 0.7934 | 0.0009 |

| prompts that name nothing (382) | |
|---|---|
| null head says invalid | 0.6257 |
| **emitted any mask at all** | **0.0550** |
| mean false-positive voxels | 164.9 |

| image replacement (360 different-subject pairs, 40 same-subject pairs skipped) | own MRI | another subject's | expected |
|---|---|---|---|
| Dice | 0.7922 | **0.4747** | falls ✓ |
| centroid error | 1.72 mm | 5.47 mm | holds ✓ (the field's own centre is 19.2 mm off) |

**Intermediate checkpoint** (a best-so-far checkpoint mid-run, 200 val prompts per population). The failure was already "says nothing" rather than "says the wrong thing":

| population | Dice | empty masks | predicted / true voxels | centroid |
|---|---|---|---|---|
| `targets.train` (8) | 0.6876 | 0.0% | 1413 / 1609 | 2.1 mm |
| `targets.val` (4) | 0.0096 | 55.0% | 135 / 2241 | 14.4 mm |
| `targets.test` (2) | 0.0021 | 51.2% | 397 / 2322 | 21.0 mm |

**The report path, verified on the epoch-0 checkpoint** (`--classes val --limit 80`): anchor Dice 0.8121, centroid 12.39 mm, gate 0.9125, ceiling 85.0%. Counterfactuals 0.0000 / 0.0000 / 0.0126 / 0.1633 against a base of 0.1631. 381 empty prompts: 0.5433 called invalid, 0.3123 emitted any mask, 164.7 false-positive voxels. Image replacement over 63 pairs (17 same-subject pairs skipped): Dice 0.1609 → 0.1567, centroid 12.62 → 13.17 mm. At epoch 0 the carver barely uses `B(I)`, so this near-null result was **uninformative**. Image replacement needs a trained model.

## Reading
- **The prompt is read.** Moving the masks against the words, or the words against the masks, gives exactly zero overlap, while moving both moves it by 0.0009. That control is weak by construction ([[SpatialVox#16.1 The four counterfactuals]]).
- **The image is used.** Swapping the MRI costs 40% of the Dice while the centroid stays close, which is the signature that the words placed the structure and the image drew it. The prompt-only ablation is [[B04 prompt-only-seed1]].
- **Transfer fails by silence, not by error.** On held-out classes the anchors (0.8046) and the gate (0.857) are as good as on supervised ones, so nothing upstream failed. The carver has learned a class-conditional size and shape prior and defaults to empty outside it. 16% of its training examples *are* empty (flips that name nothing), so "when in doubt, say nothing" costs nothing on the training distribution.
- It falls from 0.1375 at epoch 0 to 0.0099 by epoch 3: early on the carver puts a generic blob near the field, and it stops once it specialises.
- **The localiser is not what fails.** On the held-out four the heatmap's centroid error is 14.6–25.9 mm across epochs (24.9 mm at the end), against **26.8 mm** for the field's own centre of mass on that population ([[D02 mapper gate and tau sweep]]). The model is no worse than the geometry it is given, and that geometry is the ceiling.
- **Single seed.**
