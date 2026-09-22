---
id: "B04"
kind: "stage-b"
status: "done"
date: "2026-09-22"
run: "runs/prompt-only-seed1"
git: "008da30"
corpus: "data/mri"
stage_a: "[[A01 phase-a-current]]"
parent: "[[B03 relational-seed1]]"
change: "use_image false: B(I) removed, anchors and fields kept (--prompt-only). 32,819 trainable parameters, 9 carver input channels"
assumption: "The mandatory test: if its Dice approaches the full model's, the mask is a shape redrawn from the spatial prior"
epochs: "20 / 20"
seeds: 1
metric: "val Dice, targets.train (8)"
score: 0.5232
floor: 0.3067
heldout_val: 0.1171
heldout_test: 0.054
delta: -0.258
delta_on: "supervised val Dice at epoch 19 vs 19 (mean over 20 matched epochs −0.2575). Held-out-4 +0.1119 at epoch 19 (mean +0.1161), held-out-2 +0.0533 (mean +0.0770)"
benefit: "mixed"
verdict: "Mandatory test passed: the image is used (−0.258 on supervised). But the image-free carver transfers 22× better (0.117 vs 0.005), just above the 0.110 floor. It still reads the prompt (permute_channels drop 0.528)"
tags:
  - experiment
  - stage-b
  - mri
  - ablation
  - mandatory
---
# B04 prompt-only-seed1

> [!abstract] Verdict
> **The mandatory test passes, and it turns up something unexpected.** Removing `B(I)` costs **0.258** on supervised classes, so the mask is not a spatial prior. On held-out classes the image-free carver scores **0.1171** against the full model's **0.0053**. That is 22× better, and the only MRI arm above the held-out-4 floor (0.1097), though only just.

## Setup
`scripts/train.py b --prompt-only`, the same 20 epochs and seed as [[B03 relational-seed1]]. The carver's input is `A` (3) + `F` (3) + `where_raw` + both logs, 9 channels in all, with no skip. About 3.7 minutes per epoch.

## Result, matched epochs
| epoch | full: sup / ho-4 / ho-2 | prompt-only: sup / ho-4 / ho-2 |
|---|---|---|
| 0 | 0.4368 / 0.1375 / 0.0198 | 0.2350 / 0.1689 / 0.1131 |
| 5 | 0.6411 / 0.0132 / 0.0003 | 0.4452 / 0.1168 / 0.1011 |
| 10 | 0.7601 / 0.0049 / 0.0008 | 0.4954 / 0.1571 / 0.0807 |
| 19 | **0.7812** / 0.0053 / 0.0006 | **0.5232** / **0.1171** / 0.0540 |

| | full (B03) | prompt-only |
|---|---|---|
| centroid (epoch 19) | 1.94 mm | 4.32 mm |
| `permute_channels` / `permute_clauses` drop | 0.781 / 0.781 | 0.528 / 0.528 |
| `permute_both` drop | +0.001 | −0.0001 |
| floors: sup / ho-4 / ho-2 | 0.3067 / 0.1097 / 0.0630 | same |

## Reading
The two rows cross. A carver that can see `B(I)` learns what each of its eight supervised classes *looks like*, and then has nothing to say about a ninth. A carver without the image can only put a blob where the field points, so it stays class-agnostic and transfers, badly but above chance. **The image is what makes the supervised number good and the transfer number bad.** Held-out-2 (0.054) is still below its floor (0.063). Single seed.
