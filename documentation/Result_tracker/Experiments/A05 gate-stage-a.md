---
id: "A05"
kind: "stage-a"
status: "done"
date: "2026-09-22"
run: "runs/gate-stage-a"
git: "c143832"
corpus: "data/synthetic-gate (first version; inferred from timestamps)"
parent: "[[A04 hard-stage-a]]"
change: "A 60-scene gate corpus with 16 classes in 12 families and a new appearance, trained on a 14-epoch schedule"
assumption: "A short 14-epoch Stage A run is enough to tell whether an appearance is learnable"
epochs: "14 / 14"
seeds: 1
metric: "val Dice, all 16 classes (20 val scenes, 320 masks)"
score: 0.0651
delta:
delta_on: "not comparable: corpus, class set and schedule all changed"
benefit: "n/a"
verdict: "Looked unlearnable (flat around 0.06). Superseded: A06 shows the 14-epoch gate itself was at fault, and A07 shows the same corpus climbing with 50 epochs"
tags:
  - experiment
  - stage-a
  - synthetic
  - gate
---
# A05 gate-stage-a

> [!abstract] Verdict
> A misleading gate. At 14 epochs the result reads as "unlearnable" (best 0.0651 at epoch 4, 0.0573 at epoch 13). The control [[A06 control-stage-a]] and the longer run [[A07 gate-long]] show that this was **under-training**.

## Setup
- A small corpus for fast decisions: 60 / 20 / 10 scenes, 16 classes (the family split later used by `synthetic-mri`), 64³.
- **Corpus attribution is inferred.** `data/synthetic-gate` was rewritten at 15:26 on 2026-09-22 with the MRI-measured appearance, and this run finished at 15:14. It therefore trained on the earlier version, whose appearance parameters are no longer on disk.
- 14 epochs, batch 16, about 2 s per epoch.

## Result
| epoch | train Dice | val Dice |
|---|---|---|
| 0 | 0.0000 | 0.0195 |
| 4 (best) | 0.0460 | **0.0651** |
| 13 | 0.0590 | 0.0573 |

## Reading
Stage A spends its first ~20 epochs near the predict-empty optimum that class imbalance creates (`prior_foreground` 0.0016, structures about 1% of voxels). Read a **flat loss** as unlearnable, and a falling loss with low Dice as under-trained. Always pair a short gate with a control at a known-learnable appearance.
