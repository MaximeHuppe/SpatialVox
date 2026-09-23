---
id: "B2"
kind: "stage-b"
status: "running"
date: "2026-09-23"
run: "runs/mri-mask-valid-seed1"
git: "32de4e3 (code identical to d14f201)"
corpus: "data/mri"
stage_a: "[[A01 phase-a-current]]"
parent: "[[B03 relational-seed1]]"
change: "The baseline method (mask_on: valid) on real MRI. Also 30 epochs against B03's 20: compare at matched epochs 0–19"
assumption: "If the shy carver was what blocked transfer on MRI too, the held-out empty rate falls from 75% towards 0 and held-out val Dice clears its 0.110 floor, where B03 had 0.005"
epochs: "0 / 30 (running)"
seeds: 1
metric: "val Dice, targets.train (8)"
score:
floor: 0.3067
heldout_val:
heldout_test:
delta:
delta_on: "held-out val at matched epochs 0–19 against B03"
benefit: "not yet"
verdict:
tags:
  - experiment
  - stage-b
  - mri
  - transfer
---
# B2 mri-mask-valid-seed1

> [!abstract] Verdict
> Running.

## Question
Does the baseline's fix transfer to real MRI?

The prediction, written before launch:
- the held-out empty rate falls well below B03's 75%;
- held-out val (caudate, putamen) clears its 0.110 floor by more than the noise;
- held-out test (hippocampus) clears its 0.063 floor, which has a 99.2% anchor-set ceiling, so a high score there says little;
- trained classes stay near 0.78.

If the held-out masks come back but land on the wrong structure, the anchor-set shortcut is the next suspect: on MRI's fixed anatomy the anchors alone name the target 84% of the time.

## Setup
- **Command:**
  ```
  setsid nohup .venv/bin/python scripts/train.py b --config configs/config.yaml \
    --segmenter runs/phase-a/current/best.pt --out runs/mri-mask-valid-seed1 \
    --set train.stage_b.scheduler.warmup_epochs=2 --set logging.wandb.name=stageb-mri-mask-valid-s1
  ```
- **Setup:** code `32de4e3`; Stage A [[A01 phase-a-current]] (5-stage, 0.816), from the anchor cache; 30 epochs, warmup 2, batch 4, seed 20260915.
- **Against B03:** B03 used the same Stage A, model, loss weights and seed. Only `mask_on` differs, plus the length of the schedule.
- **Check that it applied:** `best.json` → `meta.config.stage.mask_on == "valid"`.
- **Floors:** trained 0.3067, held-out val 0.1097, held-out test 0.0630.

## Result
To be filled when it ends.
