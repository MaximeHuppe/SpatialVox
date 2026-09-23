---
id: "A01"
kind: "stage-a"
status: "done"
date: "2026-09-20"
run: "runs/phase-a/current"
git: "dbc1e0c"
corpus: "data/mri"
stage_a:
parent: ""
change: "Stage A on the 23-structure MRI corpus: encoder [32,64,128,256,256] (8³ bottleneck, 512 tokens), deep supervision at 4 scales [0.05,0.1,0.25,0.6]"
assumption: "A promptable segmenter trained on every name that may be an anchor gives anchors good enough for the mapper, which reads their centroids rather than their masks"
epochs: "50 / 50"
seeds: 1
metric: "val Dice, all 23 classes (20 val subjects, 460 masks)"
score: 0.8158
floor:
heldout_val:
heldout_test:
delta:
delta_on: "root of the MRI Stage A line"
benefit: "n/a"
verdict: "Shipped Stage A. Anchor centroid error median 0.84 mm (p95 2.13, worst 8.14), sub-voxel, so predicted anchors are nearly oracle for the mapper"
tags:
  - experiment
  - stage-a
  - mri
  - shipped
---
# A01 phase-a-current

> [!abstract] Verdict
> The **shipped Stage A**, frozen inside every MRI Stage B run (B01–B06). Its anchors' centroid error is sub-voxel, and that is what the mapper consumes.

## Question
Can one promptable segmenter, trained on all 23 names, produce anchor masks whose **centroids** are accurate enough for the parameter-free mapper?

## Setup
- `scripts/train.py a` on `data/mri`: 128³ at 1.25 mm, 23 classes, 160 training subjects, every name prompted on every item, one random octahedral rotation per item.
- 17,004,292 parameters. AdamW lr 1e-3, wd 1e-5, cosine with 2 warm-up epochs, batch 4, bf16, seed 20260915. About 32–37 s per epoch.
- Anchor cache: `data/mri/anchors/d429368a0730/`, which is this checkpoint's SHA-256, `normalize: zscore-brain`, float32 then float16 crops.

## Result
| epoch | val Dice (23 classes) |
|---|---|
| 0 | 0.0519 |
| 49 (best) | **0.8158** |

Per class at the best epoch: Brain-Stem 0.918, Thalami 0.893 / 0.893, Putamen 0.864 / 0.862, Caudate 0.855 / 0.860, Hippocampus 0.838 / 0.851, Amygdala 0.826 / 0.827, VentralDC 0.821 / 0.817, Lateral-Ventricles 0.825 / 0.840, 3rd/4th ventricle 0.815 / 0.838. Weakest: Pallidum 0.712 / 0.761, Accumbens 0.733 / 0.716, Inf-Lat-Vent 0.685 / 0.712.

**Anchor quality**, from `scripts/gate_mapper.py --segmenter runs/phase-a/current/best.pt`:

| | value |
|---|---|
| anchor Dice (as consumed by Stage B) | 0.80–0.82 (0.8172 on the selection curve) |
| centroid error, median / p95 / worst | **0.84 mm** / 2.13 mm / 8.14 mm (the voxel is 1.25 mm) |
| smallest predicted anchor mass | 3.2e-6 of the volume, which is why `min_mass = 1e-6` (see `D02 mapper gate and tau sweep` (archived)) |

## Reading
The mapper reads centroids, so anchor *Dice* is the wrong summary. With a sub-voxel median error, `anchor_source: oracle` and `predicted` are currently non-discriminating on MRI (`B02 overfit1-oracle` (archived)). The error tail is still real: from predicted anchors the gate falls to 0.83 against 0.97 from ground truth. Re-measure with `gate_mapper.py --segmenter` whenever Stage A is retrained.
