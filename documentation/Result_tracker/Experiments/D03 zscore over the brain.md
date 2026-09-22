---
id: "D03"
kind: "diagnostic"
status: "adopted"
date: "2026-09-21"
run: "(commit 3714857)"
git: "3714857"
corpus: "data/mri"
parent: ""
change: "data.normalize zscore (whole volume) to zscore-brain (non-zero voxels only)"
assumption: "With the brain mask applied, whole-volume statistics are dominated by the 66.5% of exact zeros"
epochs: "n/a"
seeds:
metric: "brain-tissue mean after normalisation (8 val subjects)"
score: 0.0
delta: -1.348
delta_on: "tissue mean +1.348 (sd 0.523, per-subject spread 0.097) to 0 (sd 1)"
benefit: "yes"
verdict: "Adopted in the shipped config. Plain zscore is kept as a mode so older runs stay reproducible"
tags:
  - experiment
  - diagnostic
  - data
  - mri
---
# D03 zscore over the brain

> [!abstract] Verdict
> **Adopted.** Plain `zscore` compressed every anatomical contrast into about half a sigma, with an offset that drifted per subject. `zscore-brain` puts tissue at mean 0 and sd 1 for every subject.

## The measurement
With `mri.apply_brainmask: true`, 66.5% of an HCP volume is exact zero. Over eight val subjects, whole-volume `zscore` put brain tissue at mean **+1.348** with sd **0.523**, and the offset varied by 0.097 across subjects with head size and brain-mask tightness. That matters most on a corpus where intensity is already nearly uninformative.

## Notes
- `!= 0` stands in for "inside the brain" only because the corpus was written with `apply_brainmask: true`.
- The `data/mri` anchor cache (`d429368a0730`) was built with `normalize: zscore-brain`. A checkpoint's `meta.config` does not record `data.normalize`.
