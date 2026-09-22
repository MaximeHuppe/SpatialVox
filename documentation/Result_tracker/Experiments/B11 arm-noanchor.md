---
id: "B11"
kind: "stage-b"
status: "invalid"
date: "2026-09-22"
run: "runs/arm-noanchor"
git: "f11b02f"
corpus: "data/synthetic-mri"
stage_a: "[[A09 mri-stage-a]]"
parent: "[[B09 mri-stage-b]]"
change: "INTENDED: carver_sees_anchors false (22 carver channels). ACTUAL: the flag stayed true, because the CLI value false is the string 'false', which is truthy"
assumption: "The anchor masks' shapes let the carver name the target from the anchor identities (ceiling 30.6% here, 67.8% on MRI) instead of solving the conjunction"
epochs: "13 / 30 at snapshot (still running)"
seeds: 1
metric: "val Dice, targets.train (10)"
score: 0.9406
floor: 0.1383
heldout_val: 0.3482
heldout_test: 0.4489
delta:
delta_on: "not a test of its change; used instead as a same-config replicate of B09 (the noise estimate)"
benefit: "invalid"
verdict: "INVALID for its hypothesis: the checkpoint records carver_sees_anchors true. It is the only noise estimate: vs B09, mean |Δ| per epoch 0.022 supervised, 0.091 held-out-val, 0.065 held-out-test (epochs 0–10)"
snapshot: "2026-09-22 21:55, metrics.jsonl through epoch 12; score and held-out values are epoch 10, matched to B09"
tags:
  - experiment
  - stage-b
  - synthetic
  - arm
  - invalid
  - running
---
# B11 arm-noanchor

> [!danger] Invalid: the change never applied
> Launched with `--set model.stage_b.carver_sees_anchors=false` (recovered from wandb, `stageb-mri16-noanchor`). `parse_overrides` keeps `false` as the **string** `"false"`, `scripts/train.py` passes `bool("false")`, which is **True**, and `runs/arm-noanchor/best.json` records `"carver_sees_anchors": true`. The carver still received all 25 channels. The run is **still training** (17 epochs of about 18 minutes, roughly 5 GPU-hours, left at 21:55) on a configuration identical to its parent. At epoch 12 `best.json` reads 0.9444, with held-out 0.4669 / 0.5129.

## What it is instead: the noise estimate
Same config, same seed (20260915) and same corpus as [[B09 mri-stage-b]]. The only differences are the code revision (a no-op for this config) and the non-determinism of the kernels, since nothing sets `cudnn.deterministic`. Per epoch, epochs 0–10:

| curve | mean \|Δ\| | max \|Δ\| | mean Δ (sd) |
|---|---|---|---|
| supervised val | **0.022** | 0.134 | +0.009 (0.041) |
| held-out val | **0.091** | 0.223 | −0.023 (0.113) |
| held-out test | **0.065** | 0.200 | +0.018 (0.082) |
| centroid (voxels) | 0.24 | 0.69 | +0.01 |
| `flip_direction` drop | 0.015 | 0.096 | +0.006 |
| train Dice | 0.007 | 0.040 | −0.007 |

| epoch | supervised, B09 / B11 | held-out val | held-out test |
|---|---|---|---|
| 0 | 0.7761 / 0.7860 | 0.4480 / 0.4680 | 0.5083 / 0.5613 |
| 3 | 0.8780 / 0.8696 | 0.3881 / 0.1710 | 0.3937 / 0.2936 |
| 4 | 0.7711 / 0.9053 | 0.0055 / 0.2284 | 0.1464 / 0.3464 |
| 10 | 0.9369 / **0.9406** | 0.3682 / **0.3482** | 0.4604 / **0.4489** |

## To actually test the hypothesis
`scripts/train.py b --config configs/synthetic-hard.yaml --set model.stage_b.carver_sees_anchors=False --out runs/arm-noanchor-v2`, then check `best.json → model.carver_sees_anchors` is `false`. The carver's stem should report 22 input channels, and Stage B 266,979 trainable parameters.
