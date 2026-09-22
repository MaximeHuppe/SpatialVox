---
id: "P01"
kind: "pretrain"
status: "done"
date: "2026-09-22"
run: "runs/boundary-seed1"
git: "008da30"
corpus: "data/mri"
parent: ""
change: "B(I) pretrained alone with three class-agnostic objectives: reconstruct blanked cubes, predict the label-adjacency boundary map, regress |∇I|"
assumption: "B can learn where edges are with no class id at all"
epochs: "30 / 30"
seeds: 1
metric: "boundary-map Dice (val, 20 subjects)"
score: 0.6327
delta:
delta_on: "root (pretext task)"
benefit: "n/a"
verdict: "Train and val agree (0.629 vs 0.626 at the end), so B learns edges rather than memorising subjects. Loaded into B05, where it did not restore transfer"
tags:
  - experiment
  - pretrain
  - mri
---
# P01 boundary-seed1

> [!abstract] Verdict
> `B(I)` learns a class-agnostic edge map: boundary Dice **0.6327** on val at best (epoch 28), 0.6258 at the end, with train at 0.6290. Used to initialise [[B05 pretrained-b-seed1]].

## Setup
`scripts/train.py boundary`: `BoundaryPretrainer(widths=[16, 32, 32])` with 228,579 parameters (the encoder plus three 1×1 heads of 17). `mask_fraction 0.5`, `patch 16`, loss weights boundary 1.0 / reconstruct 1.0 / edge 0.5. AdamW 5e-4, 30 epochs, batch 4, about 15–25 s per epoch, roughly 10 minutes in total. The label-adjacency target covers about 1.63% of voxels. Before this run, the path was smoke-tested at full resolution: 0.229M parameters, 1.6 GB peak memory, loss 4.70 → 4.24 over three steps.

## Result
| epoch | loss | boundary Dice, train | val |
|---|---|---|---|
| 0 | 3.514 | 0.0041 | 0.0163 |
| 6 | 2.047 | 0.5313 | 0.5516 |
| 18 | 1.278 | 0.6067 | 0.6138 |
| 28 (best) | 1.213 | 0.6297 | **0.6327** |
| 29 | 1.206 | 0.6290 | 0.6258 |

## Reading
The boundary map covers all 23 structures, the held-out ones included, so this prior tests *relational transfer*, not the lesion claim. `B` has seen those outlines, with no class channel and no target indicator. The optional contrastive term is not implemented.
