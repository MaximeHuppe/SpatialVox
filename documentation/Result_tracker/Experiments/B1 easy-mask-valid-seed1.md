---
id: "B1"
kind: "stage-b"
status: "done"
date: "2026-09-23"
run: "runs/easy-mask-valid-seed1"
git: "d14f201"
corpus: "data/synthetic"
stage_a: "[[A03 synthetic-stage-a]]"
parent: "[[B0 mask-valid-seed1]]"
change: "Same method as B0 (mask_on: valid), on the easy synthetic corpus (threshold IoU 0.99)"
assumption: "A sanity check, not evidence of transfer: the corpus is trivially separable and its held-out shapes are twins of trained ones. The change must not break it; expect Dice of about 0.98 everywhere"
epochs: "30 / 30"
seeds: 1
metric: "val Dice, targets.train (7)"
score: 0.9952
floor: 0.2500
heldout_val: 0.9965
heldout_test: 0.9981
delta:
delta_on: "not comparable: different corpus"
benefit: "n/a"
verdict: "Sanity check passed: the baseline method does not break the trivial corpus. 0.994 / 0.997 / 0.998, 0% empty masks, probes and image replacement pass. Impossible-prompt leak after the null gate 26–29%, as on B0"
tags:
  - experiment
  - stage-b
  - synthetic
---
# B1 easy-mask-valid-seed1

> [!abstract] Verdict
> Passed. The baseline method holds on the trivial corpus: 0.99+ everywhere. The result says nothing about transfer, because a threshold separates every shape and the held-out shapes are twins of trained ones.

## Setup
- **Command:** `setsid nohup .venv/bin/python scripts/train.py b --config configs/synthetic.yaml --segmenter runs/synthetic-stage-a/best.pt --out runs/easy-mask-valid-seed1 --set logging.backend=wandb --set logging.wandb.name=stageb-easy-mask-valid-s1 --set "logging.wandb.tags=['stage-b','synthetic-easy','transfer','baseline']"`
- **Code:** commit `d14f201`. **Stage A:** [[A03 synthetic-stage-a]]. `mask_on: valid`, 30 epochs, seed 20260915.
- **Replaces** `B07 synthetic-stage-b` (archived). That run stopped after 4 epochs under `mask_on: all`, at 0.983 / 0.983 / 0.975.
- **Floors:** trained 0.2500, held-out val 0.2167, held-out test 0.1847. Test is the single class `triangular_prism`, whose anchor-set ceiling is 100%.
- The evaluation of `best.pt` (`--split val`, three populations) runs automatically when training ends.

## Result
`best.pt` is epoch 23. Evaluated with `--split val` on 2026-09-23, reports in `runs/easy-mask-valid-seed1/eval_val_*/`:

| | trained (7) | held-out val: cuboid, ellipsoid | held-out test: triangular prism |
|---|---|---|---|
| Dice (null-gated) | 0.994 (0.979) | 0.997 (0.987) | 0.998 (0.984) |
| empty masks | 0% | 0% | 0% |
| centroid error | 1.1 mm | 1.1 mm | 1.3 mm |
| `permute_channels` / `flip_direction` / `permute_both` | 0.009 / 0.102 / 0.994 | 0.000 / 0.076 / 0.997 | 0.002 / 0.088 / 0.998 |
| another scene's image | 0.994 → 0.024 | 0.996 → 0.029 | 0.998 → 0.022 |
| impossible prompts that get a mask, ungated / gated | 62% / 29% | 52% / 26% | 57% / 28% |
