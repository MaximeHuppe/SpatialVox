---
id: "A03"
kind: "stage-a"
status: "done"
date: "2026-09-22"
run: "runs/synthetic-stage-a"
git: "008da30"
corpus: "data/synthetic"
parent: ""
change: "Stage A on the easy synthetic corpus: 10 primitives, 64³, 4 widths reaching 8³"
assumption: "A synthetic corpus gives a controlled stand-in for the MRI, with no anatomy prior and near-oracle anchors"
epochs: "50 / 50"
seeds: 1
metric: "val Dice, all 10 classes (50 val scenes, 500 masks)"
score: 0.9976
delta:
delta_on: "root of the synthetic line"
benefit: "n/a"
verdict: "Near perfect because the appearance is trivially separable (one threshold reaches IoU ≈ 0.994). Anchors are effectively oracle (centroid error median 0.10 voxel)"
tags:
  - experiment
  - stage-a
  - synthetic
---
# A03 synthetic-stage-a

> [!abstract] Verdict
> 0.9976, and meaningless as a test of Stage A: the easy appearance lets a single global intensity threshold recover every structure (threshold IoU ≈ 0.994). It is the Stage A of `B07 synthetic-stage-b` (archived).

## Setup
`scripts/train.py a --config configs/synthetic.yaml`: `data/synthetic`, 400 / 50 / 50 scenes, 10 classes (`cube, cuboid, sphere, ellipsoid, cylinder, cone, pyramid, triangular_prism, torus, capsule`), batch 16, about 9 s per epoch.

## Result
| epoch | val Dice |
|---|---|
| 0 | 0.1732 |
| 48 (best) | **0.9976** |

Every class ≥ 0.994. Anchor centroid error: median 0.10 / p95 0.28 / worst 2.67 voxels (anchor Dice 0.998).

## Reading
This is the easy end of the separability axis. Any transfer result built on it inherits the same triviality (`B07 synthetic-stage-b` (archived)).
