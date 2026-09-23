---
id: "A09"
kind: "stage-a"
status: "done"
date: "2026-09-22"
run: "runs/mri-stage-a"
git: "c143832"
corpus: "data/synthetic-mri"
parent: "`A08 gate-mri-like` (archived)"
change: "60 to 400 training scenes (and 20 to 50 val scenes), same appearance; the first 60 / 20 scenes are identical to A08's"
assumption: "The gate corpus was limited by data, not by its appearance"
epochs: "50 / 50"
seeds: 1
metric: "val Dice, all 16 classes (50 val scenes, 800 masks)"
score: 0.9233
delta: 0.6402
delta_on: "val Dice, best vs best (epoch 49 both); the val set grew from 20 to 50 scenes and is a superset of A08's"
benefit: "yes"
verdict: "Data, not appearance, limited A08: 0.283 to 0.923 with 6.7× the scenes. This is the frozen Stage A of B09–B11"
tags:
  - experiment
  - stage-a
  - synthetic
---
# A09 mri-stage-a

> [!abstract] Verdict
> **Benefit shown: +0.6402** over `A08 gate-mri-like` (archived) from more scenes alone (same appearance, same first 60 / 20 scenes). Stage A of the current synthetic series: `B09 mri-stage-b` (archived), `B10 arm-loo` (archived), `B11 arm-noanchor` (archived).

## Setup
`scripts/train.py a --config configs/synthetic-hard.yaml`, which now points at `data/synthetic-mri`: 400 / 50 / 50 scenes, 16 classes, threshold IoU 0.2065, 64³. 8,023,299 parameters. 50 epochs, batch 16, about 10 s per epoch.

## Result
| epoch | 0 | 49 (best) |
|---|---|---|
| val Dice | 0.0625 | **0.9233** |
| train Dice | 0.0418 | 0.9283 |

Per class: cube 0.963, cuboid 0.961, sphere 0.958, ellipsoid 0.941, cylinder 0.940, torus 0.939, cross 0.938, capsule 0.936, banana 0.932, tetrahedron 0.926, triangular_prism 0.915, cone 0.908, pyramid 0.901, crescent 0.882, hourglass 0.882, **hollow_cylinder 0.851**.

## Reading
Stage A handles the held-out *target* classes as anchors just as well (they are in its vocabulary). There is **no anchor cache** for `data/synthetic-mri`, so B09–B11 run this Stage A live under bf16 autocast.
