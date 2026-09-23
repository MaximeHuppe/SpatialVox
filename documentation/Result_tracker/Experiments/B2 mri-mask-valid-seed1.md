---
id: "B2"
kind: "stage-b"
status: "done"
date: "2026-09-23"
run: "runs/mri-mask-valid-seed1"
git: "32de4e3 (code identical to d14f201)"
corpus: "data/mri"
stage_a: "[[A01 phase-a-current]]"
parent: "[[B03 relational-seed1]]"
change: "The baseline method (mask_on: valid) on real MRI. Also 30 epochs against B03's 20: compare at matched epochs 0–19"
assumption: "If the shy carver was what blocked transfer on MRI too, the held-out empty rate falls from 75% towards 0 and held-out val Dice clears its 0.110 floor, where B03 had 0.005"
epochs: "30 / 30"
seeds: 1
metric: "val Dice, targets.train (8)"
score: 0.7924
floor: 0.3067
heldout_val: 0.0218
heldout_test: 0.0114
delta:
delta_on: "held-out val at matched epochs 0–19 against B03"
benefit: "no"
verdict: "No transfer on real MRI: held-out 0.022 / 0.011 against floors of 0.110 / 0.063. Trained classes improve (0.792). The shyness is fixed, but it is replaced by painting a trained neighbour: the carver fine term encodes class identity"
tags:
  - experiment
  - stage-b
  - mri
  - transfer
---
# B2 mri-mask-valid-seed1

> [!abstract] Verdict
> Done (30 epochs, `best.pt` epoch 17). Final evaluation: trained 0.792; held-out 0.022 / 0.011 against floors of 0.110 / 0.063; 10% / 4% empty masks; another subject's image *raises* held-out Dice. Interim, epochs 0–9: trained 0.774, held-out 0.017 / 0.002 against floors of 0.110 / 0.063. The model now answers, but it paints a trained neighbour instead of the held-out target.

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
**Interim, epochs 0–9** (2026-09-23 14:30; `best.pt` = epoch 9):
- trained **0.774** (B03 at epoch 9: 0.741);
- held-out val **0.017** (floor 0.110; B03 0.009);
- held-out test **0.002** (floor 0.063);
- held-out empty rate 7–46% (B03 75%).

**The diagnostic of `best.pt`:**
- held-out masks are 12–22% of the target's volume;
- 59–62% of the painted voxels land on another structure, almost always a trained one (caudate → accumbens or pallidum; hippocampus → thalamus, pallidum or amygdala);
- the centroid is 22–28 mm off.

The shyness is fixed, and it has been replaced by recognise-and-recall. The full analysis, with the candidate causes, is in `_update_ideas/2026-09-23-b2-mri-no-generalisation-analysis.md`.
