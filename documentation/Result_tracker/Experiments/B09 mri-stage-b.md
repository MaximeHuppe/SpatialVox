---
id: "B09"
kind: "stage-b"
status: "stopped"
date: "2026-09-22"
run: "runs/mri-stage-b"
git: "b8de31b"
corpus: "data/synthetic-mri"
stage_a: "[[A09 mri-stage-a]]"
parent: "[[B08 hard-stage-b]]"
change: "Corpus to synthetic-mri: 16 classes in 12 families, no family split across supervision, MRI-measured appearance (threshold IoU 0.2065), 400 scenes; Stage A from A09"
assumption: "With geometric twins removed and an appearance measured from the MRI, does transfer survive?"
epochs: "11 / 30 (log ends 17:25; the arms were launched at 17:30)"
seeds: 1
metric: "val Dice, targets.train (10)"
score: 0.9369
floor: 0.1383
heldout_val: 0.3682
heldout_test: 0.4604
delta:
delta_on: "not comparable (different corpus)"
benefit: "n/a"
verdict: "Supervised fine (0.937 vs 0.138). Transfer is above floor on both held-out populations (0.37 vs 0.25, 0.46 vs 0.16) but swings 0.006 to 0.448 between epochs. Single seed, stopped at 11/30"
tags:
  - experiment
  - stage-b
  - synthetic
  - baseline
---
# B09 mri-stage-b

> [!abstract] Verdict
> The **current synthetic baseline**, and the parent of the two arms. Transfer to held-out *families* is above floor but unstable. The run stopped at epoch 10 of 30.

## Setup
`scripts/train.py b --config configs/synthetic-hard.yaml --out runs/mri-stage-b`, with the corpus `data/synthetic-mri` (the config name is historical). The target split is 10 / 3 / 3 by family (see [[SpatialVox#5.5 Synthetic corpora]]). Stage A is [[A09 mri-stage-a]], **run live** because there is no anchor cache for this corpus. `far.dilation 4`, warm-up 2, batch 16, about 8.8 minutes per epoch. Uploaded to wandb afterwards as `stageb-mri16-baseline`.

## Result
| epoch | supervised | held-out val | held-out test | centroid | `flip_direction` drop | `permute_both` |
|---|---|---|---|---|---|---|
| 0 | 0.7761 | 0.4480 | 0.5083 | 2.09 | 0.654 | −0.000 |
| 3 | 0.8780 | 0.3881 | 0.3937 | 1.28 | 0.787 | −0.001 |
| 4 | 0.7711 | **0.0055** | 0.1464 | 1.88 | 0.699 | 0.003 |
| 5 | 0.9166 | 0.3949 | 0.4678 | 1.32 | 0.829 | −0.000 |
| 8 | 0.9365 | 0.3187 | 0.3609 | 1.42 | 0.861 | 0.001 |
| 10 (best) | **0.9369** | **0.3682** | **0.4604** | 0.98 | 0.857 | 0.000 |

| population | floor | ceiling |
|---|---|---|
| `targets.train` (10) | 0.1383 | 30.6% |
| `targets.val`: hollow_cylinder, cross, banana | 0.2467 | 68.3% |
| `targets.test`: crescent, hourglass, triangular_prism | 0.1617 | 66.1% |

## Reading
- Against MRI's collapse ([[B03 relational-seed1]]: 0.005 vs 0.110), held-out Dice here stays **above floor** on both populations at most epochs. This corpus has threshold IoU 0.2065 against the MRI's 0.0796.
- The held-out curves move by 0.14 on average between consecutive epochs (range 0.006–0.448). A same-config replicate ([[B11 arm-noanchor]]) differs from this run by a mean of 0.091 per epoch on held-out-val. **Read held-out numbers only across epochs and seeds.**
- Single seed, 11 of 30 epochs.
