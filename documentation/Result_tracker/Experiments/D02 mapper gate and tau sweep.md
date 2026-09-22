---
id: "D02"
kind: "diagnostic"
status: "adopted"
date: "2026-09-22"
run: "scripts/gate_mapper.py"
git: "2e7ef04"
corpus: "data/mri"
parent: ""
change: "mapper.tau 2.0 to 0.5 mm and min_mass 1e-3 to 1e-6, set from scripts/gate_mapper.py"
assumption: "The gate (target centroid inside where_raw > 0.5) is a property of tau and the corpus, and must pass before the carver is trained"
epochs: "n/a"
seeds:
metric: "gate: fraction of target centroids with where_raw > 0.5 (ground-truth centroids, train split)"
score: 0.9742
delta: 0.3234
delta_on: "gate at tau 0.5 vs the proposal's 2.0 (0.6508)"
benefit: "yes"
verdict: "Adopted. Also measured: a flip leaves 0.0000 at every tau; the null-head ceiling is AUC 0.848; the field's centre is 20.2 mm from the target (the wedge); 36% / 94% of target voxels lie inside the field / its dilation"
tags:
  - experiment
  - diagnostic
  - mapper
  - mri
---
# D02 mapper gate and tau sweep

> [!abstract] Verdict
> **Adopted** `tau = 0.5 mm` (gate 0.9742 against 0.6508 at the proposal's 2.0) and `min_mass = 1e-6` (the proposal's 1e-3 rejects almost every anchor). The sweep also produced four measurements that shape how every later result is read.

## The sweep (`scripts/gate_mapper.py --segmenter runs/phase-a/current/best.pt`)
| `tau` (mm) | gate | gate after one flip | field volume | dilated | target in field | in dilation | null AUC |
|---|---|---|---|---|---|---|---|
| 0.25 | 0.9908 | 0.0000 | 0.28% | 2.93% | 0.319 | 0.929 | 0.849 |
| **0.50** | **0.9742** | **0.0000** | **0.33%** | **3.13%** | **0.364** | **0.935** | **0.848** |
| 1.00 | 0.8917 | 0.0000 | 0.43% | 3.55% | 0.448 | 0.945 | 0.848 |
| 2.00 | 0.6508 | 0.0000 | 0.66% | 4.40% | 0.580 | 0.958 | 0.847 |
| 4.00 | 0.2417 | 0.0000 | 1.20% | 6.08% | 0.735 | 0.973 | 0.843 |

## `min_mass`: anchor masses from Stage A ([[A01 phase-a-current]])
| | fraction of 128³ |
|---|---|
| Brain-Stem, the largest | 5.0e-3 |
| Left-Thalamus | 2.0e-3 |
| Left-Accumbens | 1.5e-4 |
| Right-Inf-Lat-Vent, the smallest class | 4.6e-5 |
| smallest single anchor observed | **3.2e-6** |

## Flip outcomes (re-scored by the corpus rule)
Names two or more: **33.2%** (dropped). Names none: **65.5%** (empty mask, null target invalid). Retargets: **1.2%**.

## The field contains the target but does not point at it
With `--examples 600`: gate 0.9767 / 0.8850 / 0.6333 at tau 0.5 / 1 / 2, while the field's centre of mass is **20.2 / 20.2 / 20.3 mm** from the target's centroid. Used as a prompt-only localiser:

| population | mean | median | p90 |
|---|---|---|---|
| `targets.train` (8) | 19.20 mm | 10.93 mm | 48.14 mm |
| `targets.val` (4) | 26.79 mm | 21.50 mm | 50.98 mm |
| `targets.test` (2) | 30.15 mm | 33.84 mm | 40.11 mm |

## Reading
The null head's ceiling (AUC 0.848) is a property of its four inputs, not of its width. The 20 mm wedge makes the always-on field heatmap term a constant opposing gradient ([[B06 field-empty-only-seed1]] tests the alternative). Re-run this whenever Stage A or the corpus changes.
