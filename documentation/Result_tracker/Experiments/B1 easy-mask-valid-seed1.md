---
id: "B1"
kind: "stage-b"
status: "running"
date: "2026-09-23"
run: "runs/easy-mask-valid-seed1"
git: "d14f201"
corpus: "data/synthetic"
stage_a: "[[A03 synthetic-stage-a]]"
parent: "[[B0 mask-valid-seed1]]"
change: "Same method as B0 (mask_on: valid), on the easy synthetic corpus (threshold IoU 0.99)"
assumption: "A sanity check, not evidence of transfer: the corpus is trivially separable and its held-out shapes are twins of trained ones. The change must not break it; expect Dice of about 0.98 everywhere"
epochs: "0 / 30 (running)"
seeds: 1
metric: "val Dice, targets.train (7)"
score:
floor: 0.2500
heldout_val:
heldout_test:
delta:
delta_on: "not comparable: different corpus"
benefit: "n/a"
verdict:
tags:
  - experiment
  - stage-b
  - synthetic
---
# B1 easy-mask-valid-seed1

> [!abstract] Verdict
> Running.

## Setup
- **Command:** `setsid nohup .venv/bin/python scripts/train.py b --config configs/synthetic.yaml --segmenter runs/synthetic-stage-a/best.pt --out runs/easy-mask-valid-seed1 --set logging.backend=wandb --set logging.wandb.name=stageb-easy-mask-valid-s1 --set "logging.wandb.tags=['stage-b','synthetic-easy','transfer','baseline']"`
- **Code:** commit `d14f201`. **Stage A:** [[A03 synthetic-stage-a]]. `mask_on: valid`, 30 epochs, seed 20260915.
- **Replaces** `B07 synthetic-stage-b` (archived). That run stopped after 4 epochs under `mask_on: all`, at 0.983 / 0.983 / 0.975.
- **Floors:** trained 0.2500, held-out val 0.2167, held-out test 0.1847. Test is the single class `triangular_prism`, whose anchor-set ceiling is 100%.
- The evaluation of `best.pt` (`--split val`, three populations) runs automatically when training ends.

## Result
To be filled when it ends.
