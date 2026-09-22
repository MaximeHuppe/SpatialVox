---
id: "A06"
kind: "stage-a"
status: "done"
date: "2026-09-22"
run: "runs/control-stage-a"
git: "c143832"
corpus: "data/synthetic-control"
parent: "[[A05 gate-stage-a]]"
change: "The same 14-epoch gate on a control corpus with a much easier appearance (threshold IoU 0.5784)"
assumption: "If an appearance known to be learnable also fails the 14-epoch gate, the gate is at fault, not the corpus"
epochs: "14 / 14"
seeds: 1
metric: "val Dice, all 16 classes (320 masks)"
score: 0.065
delta:
delta_on: "not comparable (different corpus); the result is qualitative: the control fails the same gate"
benefit: "n/a"
verdict: "The control failed too and even collapsed (val 0.0650 at epoch 2 to 0.0012 at epoch 13), so the 14-epoch gate reads under-training as unlearnability"
tags:
  - experiment
  - stage-a
  - synthetic
  - gate
  - control
---
# A06 control-stage-a

> [!abstract] Verdict
> The control that exonerated the corpus. An appearance with threshold IoU 0.5784 (far easier than the MRI) **also** fails a 14-epoch gate, and even collapses towards predicting nothing.

## Setup
`data/synthetic-control`: 60 / 20 / 10 scenes, 16 classes, `structure [0.50, 0.78]`, `background [0.45, 0.03]`, `texture 0.9`, `bias_field 0.3`, no tissue model and no head envelope. Threshold IoU **0.5784**. 14 epochs.

## Result
| epoch | train Dice | val Dice |
|---|---|---|
| 0 | 0.0236 | 0.0016 |
| 2 (best) | 0.0443 | **0.0650** |
| 13 | 0.0025 | 0.0012 |

## Reading
Escaping the empty optimum takes optimisation steps, not easier images. From here on, the gate for a synthetic appearance is the full 50-epoch schedule ([[A07 gate-long]]).
