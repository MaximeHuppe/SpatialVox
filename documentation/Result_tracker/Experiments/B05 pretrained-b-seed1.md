---
id: "B05"
kind: "stage-b"
status: "killed"
date: "2026-09-22"
run: "runs/pretrained-b-seed1"
git: "008da30"
corpus: "data/mri"
stage_a: "[[A01 phase-a-current]]"
parent: "[[B03 relational-seed1]]"
change: "B(I) initialised from P01 (boundary-seed1) and trained at 0.1× the carver's learning rate"
assumption: "A B trained from scratch has no class-agnostic prior. Pretraining should restore transfer: the empty rate falls and the predicted volume tracks the truth"
epochs: "14 / 20 (killed by an external process)"
seeds: 1
metric: "val Dice, targets.train (8)"
score: 0.7491
floor: 0.3067
heldout_val: 0.0066
heldout_test: 0.0004
delta: -0.0095
delta_on: "supervised val Dice, mean over 14 matched epochs (sd 0.0276). Held-out-4 +0.0012 (sd 0.0109)"
benefit: "no"
verdict: "Falsified: transfer collapses identically (0.112 to 0.009 by epoch 12). The cause is the carver's objective, not B's features"
tags:
  - experiment
  - stage-b
  - mri
  - ablation
---
# B05 pretrained-b-seed1

> [!abstract] Verdict
> **No benefit; the hypothesis is falsified.** A class-agnostic edge prior does not restore transfer. Held-out Δ is **+0.0012 ± 0.0109** over 14 matched epochs, and the curve collapses exactly as in [[B03 relational-seed1]].

## Setup
`--set train.stage_b.boundary_checkpoint=runs/boundary-seed1/best.pt` ([[P01 boundary-seed1]]), `boundary_lr_scale 0.1`: `B` has 228,528 parameters at 3e-5 against 39,747 at 3e-4. Otherwise B03's config. The run was killed by an external process at epoch 14 of 20: the log stops mid-epoch with no traceback, and another session had taken the GPU. By then both curves had separated and settled, so it was not restarted.

## Result, matched epochs
| epoch | supervised, scratch / pretrained | held-out 4, scratch / pretrained |
|---|---|---|
| 0 | 0.4368 / 0.4647 | 0.1375 / 0.1123 |
| 3 | 0.6002 / 0.5870 | 0.0099 / 0.0089 |
| 4 | 0.6663 / 0.6268 | 0.0091 / 0.0162 |
| 5 | 0.6411 / 0.6886 | 0.0132 / 0.0173 |
| 8 | 0.7268 / 0.7020 | 0.0072 / 0.0256 |
| 12 | 0.7724 / 0.7455 | 0.0093 / 0.0087 |
| 13 (best) | 0.7753 / **0.7491** | 0.0055 / 0.0066 |

| mean (pretrained − scratch), 14 epochs | Δ | sd |
|---|---|---|
| supervised 8 | −0.0095 | 0.0276 |
| held-out 4 | +0.0012 | 0.0109 |

## Reading
Nothing in "Dice + BCE on eight classes" rewards class-agnostic behaviour, and 16% of training examples are supervised to be empty, so silence is a locally optimal policy. The next thing to change is the **supervision**, not the network: `flip_probability`, leave-one-class-out ([[B10 arm-loo]]), or the image-free route that [[B04 prompt-only-seed1]] stumbled on. A wider carver is not indicated.
