---
id: "B01"
kind: "stage-b"
status: "done"
date: "2026-09-22"
run: "runs/overfit1"
git: "008da30"
corpus: "data/mri (1 training scene, 41 prompts)"
stage_a: "[[A01 phase-a-current]]"
parent: ""
change: "The relational architecture, trained and validated on the same single scene with no flips (--overfit 1)"
assumption: "A model that cannot memorise one scene has its channel order, world coordinates or prompt indices wrong"
epochs: "120 / 120"
seeds: 1
metric: "Dice on the memorised scene (41 prompts over 8 classes)"
score: 0.7879
delta:
delta_on: "root (wiring check)"
benefit: "n/a"
verdict: "Wiring correct: centroid error 17.32 mm to 1.20 mm, below one voxel. This is NOT a capacity ceiling: B03 passes it on training Dice by epoch 11"
tags:
  - experiment
  - stage-b
  - mri
  - wiring
---
# B01 overfit1

> [!abstract] Verdict
> The wiring check passes. Dice 0.0000 → **0.7879** and centroid error 17.32 → **1.20 mm** (sub-voxel), which shows the channel order, the world coordinates and the prompt indices are right.

## Setup
`scripts/train.py b --overfit 1 --set train.stage_b.epochs=120`. Train and val are the same first training scene, the flip is off, `anchor_source: predicted`, `B` starts from scratch, warm-up 5, batch 4, about 6.4 s per epoch.

## Result
| epoch | 0 | 9 | 29 | 59 | 89 | 116 (best) | 119 |
|---|---|---|---|---|---|---|---|
| Dice | 0.0000 | 0.3215 | 0.5801 | 0.6204 | 0.7697 | **0.7879** | 0.7877 |
| centroid (mm) | 17.32 | 4.31 | 1.70 | 1.36 | 1.23 | 1.20 | 1.22 |

The Dice is exactly 0 for the first 8 epochs: the prior-initialised head starts at the base rate and has to climb out of the empty mask. `field_centroid` plateaus around 0.32 for the whole run while `centroid` falls to about 0.008. That is the two-heatmap-target conflict ([[SpatialVox#20.3 Where the specification was ambiguous]]).

## Reading
A one-scene memorisation score is a **wiring check and nothing more**. The same carver exceeds it on *training* Dice by epoch 11 of [[B03 relational-seed1]] (0.8048 with 160 subjects).
