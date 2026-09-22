---
id: "B08"
kind: "stage-b"
status: "done"
date: "2026-09-22"
run: "runs/hard-stage-b"
git: "c143832"
corpus: "data/synthetic-hard"
stage_a: "[[A04 hard-stage-a]]"
parent: "[[B07 synthetic-stage-b]]"
change: "Appearance easy to hard (threshold IoU 0.2704) and structures ×1.8; Stage A from A04"
assumption: "At MRI-like separability with no anatomy prior, transfer collapses as on MRI if separability is the cause"
epochs: "30 / 30"
seeds: 1
metric: "val Dice, targets.train (7)"
score: 0.8763
floor: 0.2133
heldout_val: 0.7074
heldout_test: 0.3532
delta:
delta_on: "not comparable (different corpus)"
benefit: "n/a"
verdict: "Transfer holds far above floor (0.71 vs 0.15), but the held-out cuboid and ellipsoid are geometric twins of the supervised cube and sphere, so it is inflated. This led to the family split (B09)"
tags:
  - experiment
  - stage-b
  - synthetic
---
# B08 hard-stage-b

> [!abstract] Verdict
> Transfer survives the hard appearance (held-out 0.71 against a 0.15 floor). But the held-out `cuboid` and `ellipsoid` are the supervised `cube` and `sphere` under other names (IoU 1.0 at canonical parameters), so the number is **inflated**. The fix is a family-level split ([[B09 mri-stage-b]]).

## Setup
`data/synthetic-hard`, 7 / 2 / 1 split as in B07, Stage A [[A04 hard-stage-a]] (anchor-centroid p95 25.6 voxels), predicted anchors from the cache, `far.dilation 4`, 30 epochs, about 4.9 minutes per epoch.

## Result
| epoch | supervised | held-out val | held-out test | centroid | `flip_direction` drop |
|---|---|---|---|---|---|
| 0 | 0.7591 | 0.6810 | 0.6412 | 3.94 | 0.224 |
| 4 | 0.8663 | **0.7529** | 0.6963 | 4.16 | 0.279 |
| 6 (best) | **0.8763** | 0.7074 | 0.3532 | 2.68 | 0.296 |
| 10 | 0.8706 | 0.7185 | 0.4439 | 4.36 | 0.409 |
| 11 | 0.8217 | 0.6681 | 0.4255 | 4.02 | 0.654 |
| 29 | 0.8253 | 0.6312 | 0.2320 | 3.71 | 0.696 |

Floors: 0.2133 / 0.1500 / 0.2729. Ceilings: 29.9% / 71.5% / 100%, and the single-class test population is uninformative.

## Reading
- Held-out-val stays at 0.63–0.75 throughout, against a 0.15 floor, but for the wrong reason (twin shapes). An earlier module note quoted "0.737 against a 0.1625 floor" for this run. The values here are re-read from `metrics.jsonl` (600-prompt curves) and `corpus_report.py` (600 sampled prompts), and they supersede that quote.
- **A regime change at epoch 11.** The `flip_direction` drop jumps from about 0.3 to 0.65, and train Dice rises from 0.71 to 0.88, while supervised val falls from 0.87 to 0.82–0.84. `best.pt` comes from before it. Until then the model leant on the anchors and the field more than on the direction words.
- Single seed.
