---
id: "A02"
kind: "stage-a"
status: "done"
date: "2026-09-21"
run: "runs/phase-a/new-model"
git: "008da30"
corpus: "data/mri"
parent: "[[A01 phase-a-current]]"
change: "4 encoder widths [32,64,128,256], so a 16³ bottleneck (4,096 attention tokens), and 3-scale deep supervision [0.1,0.3,0.6]"
assumption: "A finer bottleneck gives the name queries more spatial resolution to attend over"
epochs: "50 / 50"
seeds: 1
metric: "val Dice, all 23 classes (20 val subjects, 460 masks)"
score: 0.8018
delta: -0.014
delta_on: "val Dice, best vs best (epoch 44 vs 49); same corpus, same val population"
benefit: "no"
verdict: "Worse by 0.014 with about 8.0M parameters instead of 17.0M. A01 stays shipped. Used only by the legacy query heads (L01)"
tags:
  - experiment
  - stage-a
  - mri
---
# A02 phase-a-new-model

> [!abstract] Verdict
> **No benefit.** −0.0140 val Dice against [[A01 phase-a-current]], on the same corpus and the same 460 validation masks. A01 remains the shipped Stage A.

## Question
Does a 16³ bottleneck (four widths instead of five) segment the anchors better?

## Setup
- The same as A01 except `encoder_channels: [32, 64, 128, 256]`, `bottleneck: 16` and `deep_supervision: [0.1, 0.3, 0.6]`. **Two changes at once**, so the width list and the supervision schedule are confounded.
- About 8.0M parameters (the checkpoint is 32 MB against A01's 68 MB). Batch 4, 50 epochs, git 008da30.

## Result
| | A01 | A02 |
|---|---|---|
| best val Dice | 0.8158 (ep 49) | **0.8018** (ep 44) |
| final val Dice | 0.8158 | 0.8009 |
| Brain-Stem / Thalamus L | 0.918 / 0.893 | 0.911 / 0.869 |
| Accumbens L / R | 0.733 / 0.716 | 0.721 / 0.706 |

## Reading
The smaller network with the finer attention grid is uniformly slightly worse. It was used as the Stage A of the legacy `qh`, `rel` and `sel` heads ([[L01 legacy attention Stage B]]). No relational run uses it.
