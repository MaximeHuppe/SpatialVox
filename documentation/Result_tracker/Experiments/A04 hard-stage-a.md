---
id: "A04"
kind: "stage-a"
status: "done"
date: "2026-09-22"
run: "runs/hard-stage-a"
git: "008da30"
corpus: "data/synthetic-hard"
parent: "[[A03 synthetic-stage-a]]"
change: "Appearance easy to hard (threshold IoU ≈0.994 to 0.2704: overlapping intensity ranges, texture 0.9 at scale 12, bias field 0.6) and every structure ×1.8 in size"
assumption: "A corpus between the easy synthetic and the MRI (0.0796) is still learnable by Stage A"
epochs: "50 / 50"
seeds: 1
metric: "val Dice, all 10 classes (500 masks)"
score: 0.8831
delta:
delta_on: "not comparable: different images and different label geometry (structures ×1.8, foreground 1.7% to 9.7% of the volume)"
benefit: "n/a"
verdict: "Learnable (0.883), but cube 0.607 and cuboid 0.642 confuse each other, and the anchor-centroid tail is long (p95 25.6 voxels)"
tags:
  - experiment
  - stage-a
  - synthetic
---
# A04 hard-stage-a

> [!abstract] Verdict
> Stage A learns the hard appearance (0.8831). The two box shapes are hard to tell apart, which puts a long tail on anchor centroid errors. Stage A of [[B08 hard-stage-b]].

## Setup
`data/synthetic-hard`, 400 / 50 / 50 scenes, the same 10 classes as A03, threshold IoU **0.2704** (`meta.json`). Appearance: `structure [0.45, 0.75]`, `background [0.45, 0.04]`, `class_spread: shared`, `noise 0.035`, `blur 0.6`, `texture 0.9` at scale 12, `bias_field 0.6`. Structures are scaled by `size: 1.8`, so label volumes differ from A03's (checked: foreground 1.7% → 9.7% on `train_00000`).

## Result
| epoch | val Dice |
|---|---|
| 0 | 0.1268 |
| 46 (best) | **0.8831** |
| 49 | 0.8829 |

Per class: sphere 0.967, cylinder 0.963, ellipsoid 0.957, pyramid 0.953, torus 0.951, cone 0.935, capsule 0.934, triangular_prism 0.923, **cuboid 0.642, cube 0.607**. Anchor centroid error: median 0.67 / p95 **25.6** / worst 36.2 voxels (anchor Dice 0.889). A mistaken box pulls the centroid across the scene.

## Reading
This is the first corpus on which `predicted` and `oracle` anchors should differ. The Δ against A03 is not a comparison, because both images and geometry changed.
