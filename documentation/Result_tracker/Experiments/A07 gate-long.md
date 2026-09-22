---
id: "A07"
kind: "stage-a"
status: "done"
date: "2026-09-22"
run: "runs/gate-long"
git: "c143832"
corpus: "data/synthetic-gate (first version; inferred from timestamps)"
parent: "[[A05 gate-stage-a]]"
change: "14 to 50 epochs, same corpus"
assumption: "Stage A sits in the predict-empty optimum for about 20 epochs before it climbs"
epochs: "50 / 50"
seeds: 1
metric: "val Dice, all 16 classes (320 masks)"
score: 0.2428
delta: 0.1777
delta_on: "val Dice, best vs best on the same corpus and val population (A05's best is at epoch 4)"
benefit: "yes"
verdict: "Still rising at epoch 49 (0.013, 0.057, 0.086, 0.144, 0.216, 0.243 at epochs 0/9/19/29/39/49) with a train/val gap of about 0.02: the corpus is learnable, and the short gate was under-training"
tags:
  - experiment
  - stage-a
  - synthetic
  - gate
---
# A07 gate-long

> [!abstract] Verdict
> **Benefit shown: +0.1777** over [[A05 gate-stage-a]] from the schedule alone. The corpus is learnable, and the 14-epoch gate was simply too short.

## Setup
The same corpus as A05 (inferred: finished at 15:18, before the corpus was rewritten at 15:26), 50 epochs, everything else unchanged. The cosine schedule stretches over 50 epochs instead of 14, so early epochs are not at the same learning rate as A05's.

## Result
| epoch | 0 | 9 | 19 | 29 | 39 | 49 |
|---|---|---|---|---|---|---|
| val Dice | 0.0127 | 0.0570 | 0.0861 | 0.1444 | 0.2162 | **0.2428** |
| train Dice | 0.0089 | 0.0530 | 0.0729 | 0.1499 | 0.2314 | 0.2600 |

Per class at the end, highest: hollow_cylinder 0.515, tetrahedron 0.380, cross 0.320. Lowest: cone 0.145, torus 0.179, pyramid 0.176.

## Reading
With 60 training scenes the curve has not flattened. The next steps are the MRI-measured appearance ([[A08 gate-mri-like]]) and then more scenes ([[A09 mri-stage-a]]).
