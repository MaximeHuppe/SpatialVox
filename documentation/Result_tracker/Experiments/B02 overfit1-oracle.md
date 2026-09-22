---
id: "B02"
kind: "stage-b"
status: "stopped"
date: "2026-09-22"
run: "runs/overfit1-oracle"
git: "008da30"
corpus: "data/mri (1 training scene, 41 prompts)"
stage_a: "[[A01 phase-a-current]]"
parent: "[[B01 overfit1]]"
change: "anchor_source predicted to oracle (ground-truth anchor masks replace Stage A's)"
assumption: "If Stage A's masks, through the A_i > 0.5 exclusion, eat part of the target, oracle anchors raise the overfit ceiling"
epochs: "33 / 120 (stopped once the comparison was unambiguous)"
seeds: 1
metric: "Dice on the memorised scene"
score: 0.5584
delta: -0.0072
delta_on: "Dice, mean over 33 matched epochs (sd 0.0542)"
benefit: "no"
verdict: "Indistinguishable from predicted anchors: the carver, not Stage A, limits the overfit, and predicted ≈ oracle on MRI"
tags:
  - experiment
  - stage-b
  - mri
  - diagnostic
---
# B02 overfit1-oracle

> [!abstract] Verdict
> **No benefit.** Oracle anchors track predicted ones within noise: mean (oracle − predicted) **−0.0072 ± 0.0542** over 33 matched epochs.

## Setup
`scripts/train.py b --overfit 1 --set train.stage_b.anchor_source=oracle`, otherwise identical to [[B01 overfit1]]. It was stopped at epoch 33 to give the GPU back to [[B03 relational-seed1]].

## Result
| epoch | predicted (B01) | oracle (B02) | centroid, predicted / oracle |
|---|---|---|---|
| 0 | 0.0000 | 0.0000 | 17.32 / 17.37 mm |
| 12 | 0.3211 | 0.3545 | 4.68 / 4.60 mm |
| 18 | 0.4464 | 0.4698 | 3.26 / 3.33 mm |
| 24 | 0.5257 | 0.4399 | 1.57 / 3.17 mm |
| 30 | 0.5698 | **0.5584** (best) | 1.70 / 1.92 mm |

## Reading
The exclusion `logits = background where A_i > 0.5` does not remove enough of the target to matter, which fits Stage A's sub-voxel centroid error ([[A01 phase-a-current]]). `oracle` is kept as a labelled diagnostic for corpora where segmentation is hard.
