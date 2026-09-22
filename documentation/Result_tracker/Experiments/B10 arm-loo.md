---
id: "B10"
kind: "stage-b"
status: "running"
date: "2026-09-22"
run: "runs/arm-loo"
git: "f11b02f"
corpus: "data/synthetic-mri"
stage_a: "[[A09 mri-stage-a]]"
parent: "[[B09 mri-stage-b]]"
change: "train.stage_b.leave_one_out true: one supervised class withheld from the loss each epoch, cycling through the 10"
assumption: "The model recognises which supervised class is asked for and paints its remembered shape; withholding a class during training makes that route fail and forces relational drawing"
epochs: "13 / 30 at snapshot"
seeds: 1
metric: "val Dice, targets.train (10)"
score: 0.9379
floor: 0.1383
heldout_val: 0.3787
heldout_test: 0.566
delta: 0.0021
delta_on: "held-out-val Dice, mean over matched epochs 0–10 vs B09, whose log ends at epoch 10 (sd 0.121). Supervised +0.0061, held-out-test +0.0223. Replicate noise: 0.091 / 0.065 mean |Δ|"
benefit: "not yet"
verdict: "Inside replicate noise so far. The epoch-10 held-out-test +0.106 is one epoch of a curve that swings by about ±0.2. Wait for 30 epochs, and seeds"
snapshot: "2026-09-22 21:55, metrics.jsonl through epoch 12; score and held-out values are epoch 10, matched to the parent"
tags:
  - experiment
  - stage-b
  - synthetic
  - arm
  - running
---
# B10 arm-loo

> [!info] Running (snapshot 2026-09-22 21:55)
> 13 of 30 epochs are logged in `metrics.jsonl` (epochs 0–12), and `best.json` is still epoch 10. The parent's log ends at epoch 10, so the matched comparison stops there. Update this note when the run ends.

> [!abstract] Verdict so far
> **Not yet distinguishable** from its parent [[B09 mri-stage-b]]. Every matched-epoch difference is inside the same-config replicate noise.

## Setup
The launch arguments, recovered from wandb (`stageb-mri16-loo`):
`scripts/train.py b --config configs/synthetic-hard.yaml --set train.stage_b.leave_one_out=true --set logging.wandb.name=stageb-mri16-loo --out runs/arm-loo`.

> [!warning] It worked by accident
> `leave_one_out=true` reached the code as the **string** `"true"` (the checkpoint records `"leave_one_out": "true"`), and `bool("true")` is True. The same mistake with `false` did not work in [[B11 arm-noanchor]]. Write `True`/`False`.

It shares the GPU with B11, so each epoch takes about 18 minutes.

## Result, matched to the parent
| epoch | supervised, B09 / B10 | held-out val, B09 / B10 | held-out test, B09 / B10 |
|---|---|---|---|
| 0 | 0.7761 / 0.7854 | 0.4480 / 0.4711 | 0.5083 / 0.5286 |
| 2 | 0.8841 / 0.8691 | 0.3540 / 0.1703 | 0.4358 / 0.3450 |
| 4 | 0.7711 / 0.8779 | 0.0055 / 0.3106 | 0.1464 / 0.3952 |
| 6 | 0.9284 / 0.9234 | 0.2939 / 0.3442 | 0.4190 / 0.4254 |
| 8 | 0.9365 / 0.9229 | 0.3187 / 0.2087 | 0.3609 / 0.3879 |
| 10 | 0.9369 / **0.9379** | 0.3682 / **0.3787** | 0.4604 / **0.5660** |

| Δ over epochs 0–10 | vs B09: mean (sd) | replicate noise, mean \|Δ\| |
|---|---|---|
| supervised | +0.0061 (0.036) | 0.022 |
| held-out val | +0.0021 (0.121) | 0.091 |
| held-out test | +0.0223 (0.090) | 0.065 |

Past the parent's last epoch: epoch 11 gives 0.9324 / 0.4162 / 0.4896 and epoch 12 gives 0.9374 / 0.3371 / 0.4729. Against the replicate [[B11 arm-noanchor]] over epochs 0–12, the mean Δ is −0.003 supervised, +0.026 held-out val (sd 0.110) and +0.012 held-out test (sd 0.090), also inside the noise.

## Reading
Leave-one-out should show in the held-out curves, not the supervised one, and those curves are the noisiest. A benefit needs held-out Δ well above 0.1 **sustained across epochs**, or ≥ 3 seeds per arm.
