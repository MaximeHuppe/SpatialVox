---
id: "B07"
kind: "stage-b"
status: "stopped"
date: "2026-09-22"
run: "runs/synthetic-stage-b"
git: "008da30"
corpus: "data/synthetic"
stage_a: "[[A03 synthetic-stage-a]]"
parent: "[[B03 relational-seed1]]"
change: "The same architecture on the easy synthetic corpus (far dilation 4 for 64³)"
assumption: "If transfer fails here too the architecture is at fault; if it succeeds, the MRI's appearance becomes the suspect"
epochs: "4 / 30 (stopped)"
seeds: 1
metric: "val Dice, targets.train (7)"
score: 0.9826
floor: 0.25
heldout_val: 0.9829
heldout_test: 0.9749
delta:
delta_on: "not comparable (different corpus)"
benefit: "n/a"
verdict: "Transfer perfect and meaningless: one global threshold separates everything (IoU ≈ 0.994), so B(I) gets every boundary for free. Already 0.949 supervised and 0.950 held out after one epoch"
tags:
  - experiment
  - stage-b
  - synthetic
---
# B07 synthetic-stage-b

> [!abstract] Verdict
> Transfer is **perfect and uninformative**. With threshold IoU ≈ 0.994 the carver never has to learn a class, because `B(I)` hands it every boundary. This result is what motivated the appearance knobs (commit `c143832`).

## Setup
`scripts/train.py b --config configs/synthetic.yaml`: 7 / 2 / 1 target split (`cube, sphere, cylinder, cone, pyramid, torus, capsule` / `cuboid, ellipsoid` / `triangular_prism`), batch 16, about 4 minutes per epoch. Stopped after 4 epochs.

## Result
| epoch | supervised | held-out val | held-out test | centroid | `flip_direction` drop |
|---|---|---|---|---|---|
| 0 | 0.9490 | 0.9504 | 0.9488 | 1.33 | 0.896 |
| 1 | 0.9723 | 0.9648 | 0.9733 | 1.20 | 0.909 |
| 2 (best) | **0.9826** | **0.9829** | 0.9749 | 0.85 | 0.948 |
| 3 | 0.9082 | 0.9328 | 0.8877 | 0.93 | 0.907 |

Floors: 0.2500 / 0.2167 / 0.1847. Ceilings: 29.8% / 72.5% / 100%, and the single-class test population is uninformative.

## Reading
It supports "the architecture can transfer when the image carries class-agnostic boundaries", which is what the method predicts. It cannot say whether the architecture transfers when it has to *learn* them. Hence [[B08 hard-stage-b]].
