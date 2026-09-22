---
id: "A08"
kind: "stage-a"
status: "done"
date: "2026-09-22"
run: "runs/gate-mri-like"
git: "c143832"
corpus: "data/synthetic-gate (rewritten 15:26 with the MRI-measured appearance, threshold IoU 0.2065)"
parent: "[[A07 gate-long]]"
change: "Appearance replaced by the model measured from data/mri: tissue classes, octave texture, a shared structure band [0.10, 0.85], a head envelope, bias field 0.30"
assumption: "An appearance measured from the MRI, rather than invented, is still learnable at gate scale"
epochs: "50 / 50"
seeds: 1
metric: "val Dice, all 16 classes (320 masks)"
score: 0.2831
delta:
delta_on: "+0.0403 vs A07, but not strictly comparable: the images changed, and the earlier version is no longer on disk to check that the labels match"
benefit: "n/a"
verdict: "Learnable at gate scale (0.283, still rising), which justified building the full 500-scene corpus (A09)"
tags:
  - experiment
  - stage-a
  - synthetic
  - gate
---
# A08 gate-mri-like

> [!abstract] Verdict
> The MRI-measured appearance passes the full-schedule gate (0.2831 at epoch 49, still rising), so the 400-scene corpus was built on it ([[A09 mri-stage-a]]).

## Setup
`data/synthetic-gate` after its 15:26 rewrite: 60 / 20 / 10 scenes, 16 classes, threshold IoU **0.2065**. The appearance is quoted in units of the grey–white gap `separation 0.28`: tissue fractions 0.112 / 0.496 / 0.392, `texture_sd 0.39`, `tissue_length 3.5`, `structure [0.10, 0.85]`, `class_spread: shared`, `noise 0.01`, `blur 0.6`, head envelope, `bias_field 0.30`. Its scenes are **identical** to the first 60 / 20 scenes of `data/synthetic-mri` (checked on `train_00000` and `val_00000`).

## Result
| epoch | 0 | 49 (best) |
|---|---|---|
| val Dice | 0.0002 | **0.2831** |
| train Dice | 0.0000 | 0.2996 |

Highest classes: torus 0.471, cross 0.424, banana 0.321. Lowest: cone 0.131.

## Reading
The +0.04 over A07 cannot be credited to the appearance with certainty, because the images changed and the old ones were overwritten. What matters is that the curve still rises with the same data-limited shape.
