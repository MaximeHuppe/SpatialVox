---
id: "B06"
kind: "stage-b"
status: "failed"
date: "2026-09-22"
run: "runs/field-empty-only-seed1"
git:
corpus: "data/mri"
stage_a: "[[A01 phase-a-current]]"
parent: "[[B03 relational-seed1]]"
change: "train.stage_b.field_centroid_on always to empty-only"
assumption: "The field-centre heatmap target sits 20 mm from the true centroid, a constant opposing gradient on every valid prompt; restricting it to empty prompts should help localisation"
epochs: "0 logged"
seeds: 1
metric: "val Dice, targets.train (8)"
score:
floor: 0.3067
delta:
delta_on: "no result"
benefit: "n/a"
verdict: "No result: metrics.jsonl is empty (created 2026-09-22 17:30) and no epoch was logged. Still to run"
tags:
  - experiment
  - stage-b
  - mri
  - queued
---
# B06 field-empty-only-seed1

> [!failure] No result
> `runs/field-empty-only-seed1/metrics.jsonl` exists and is empty (created 2026-09-22 17:30). No epoch was logged and there is no checkpoint. The cause is not recorded.

## Question
Under `field_centroid_on: always`, the heatmap is pulled towards `where_raw`'s centre of mass, which lies **20.2 mm** from the target's centroid, on every valid prompt. On [[B01 overfit1]] that term plateaus at 0.32 while `centroid` falls to 0.008, about 40% of the loss spent on a constant opposing gradient. Does restricting it to prompts that name nothing (where it is the only target) improve the centroid, and through it anything else?

## To run
`scripts/train.py b --out runs/field-empty-only-seed1 --set train.stage_b.field_centroid_on=empty-only` (a string value, so no boolean trap), matched to [[B03 relational-seed1]]'s 20 epochs and warm-up 2.
