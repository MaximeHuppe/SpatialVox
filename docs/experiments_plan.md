# Experiments plan — checks and tests to prepare

Written 2026-09-18. Nothing in this file has been run and no code has been
changed to produce it. Every item states, explicitly, whether it needs a
config override, a corpus regeneration, or a code change before it can run —
that's the point of writing this down before starting.

## 1. Already confirmed by reading the code (no training needed)

### 1.1 Occupancy source follows `mode`; occupancy_mode is independent

- `train.stage_b.mode: predicted` (the default) → Stage A's predictions for
  both anchors and occupancy. A pretrained `phase_a_checkpoint` (or
  `--segmenter`) is required; missing it is an error.
- `train.stage_b.mode: oracle` → ground truth for both. This is the only
  bypass of the checkpoint requirement; a leftover path in the config is
  ignored. They cannot be mixed.
- `train.stage_b.occupancy_mode: all | anchors-only | none` chooses which
  masks of that source are unioned. The shipped default is `anchors-only`.
  `all` is the ceiling (target included whenever the source can name it).
  `anchors-only` and `none` do not leak the target.
- `StageB.forward` still subtracts the anchor channels from occupancy, so
  `anchors-only` is empty at the decoder.

### 1.2 Stage A always sees all 10 structures, in every split

- `scripts/train.py:52-54` (comment + code): Stage A's `SceneDataset` is
  built identically for every split. Only the *target* vocabulary is
  restricted by `configs/config.yaml: targets`; the anchor/name vocabulary
  never is.
- So "train Stage A on 9/10 structures" as originally proposed is not
  something the current pipeline supports via config — see §3.4.

### 1.3 Target-class split is one config block, no code change needed

- `configs/config.yaml:34-37` — `targets.{train,val,test}`, currently 7/2/1,
  `test: [triangular_prism]`.
- `src/config.py: parse_overrides` uses `ast.literal_eval`, so list-valued
  CLI overrides work, e.g. `--set "targets.test=['torus']"`. Confirmed by
  reading the parser; not executed.

### 1.4 `data.n_anchors` is a corpus-generation-time parameter, not a training-time one

- Anchor selection (`select_anchors`) runs once, at `scripts/generate_data.py`
  time, and is baked into `train.jsonl` / `val.jsonl` / `test.jsonl`.
  Overriding `data.n_anchors` on a `train.py` call without regenerating the
  corpus changes nothing — the manifests already fix the anchors.
- The n_anchors sweep (§3.3) needs three separate corpora (e.g.
  `data/synthetic_na2`, `_na3`, `_na4`), one `generate_data.py` run each with
  `--set data.n_anchors=N`.

### 1.5 The synthetic vocabulary is hardcoded, not data-driven

- `src/synthetic.py:34` — `SHAPES` is a literal Python dict;
  `SHAPE_NAMES = tuple(SHAPES)` (line 92). There is no config knob to drop a
  shape from generation.
- The literal version of "exclude one structure from Stage A training
  entirely" (§3.4) needs either a code change (an allow-list in the
  generator) or a post-hoc corpus filter that also remaps label ids. Flag
  this before attempting it — it is the one item here that is not a
  config-only change.

## 2. Open checks before designing the runs further (no training)

- [ ] Read `Trainer.evaluate` in `src/engine.py` end to end, to confirm it
      threads the same `StageBTask` instance used for the training loss.
      (Inferred from `__call__` being the single occupancy source; the
      `Trainer.evaluate` body itself hasn't been read line by line.)
- [ ] If `data/synthetic` already exists, check `train.jsonl` / `val.jsonl`
      size and count distinct `(anchor_names, directions)` triplets, to see
      how much combinatorial anchor-triplet coverage there already is before
      treating "unseen anchor combos" as a separate experiment.
- [ ] Check actual wall-clock per epoch for one Stage B run (30 epochs, batch
      16, 64³) on the available hardware, to size the 10-way leave-one-out
      sweep (§3.2) and the 3-arm occupancy ablation (§3.1) before committing
      to them — no run needed if a prior `runs/*/metrics.jsonl` exists to
      read timestamps from.
- [ ] Decide logging before running more than one arm: `logging.backend` is
      `none` by default (`configs/config.yaml:80`). Turn on `wandb`, or
      commit to a naming convention for `--out` run directories — comparing
      10-30 runs by hand from stdout is the thing most likely to go wrong.

## 3. Tests to prepare, ranked by information per GPU-hour

### 3.1 Occupancy ablation — top priority

- **Question.** Is Stage B localizing the target from the relations, or
  snapping to "the blob occupancy already highlighted"? This is the
  deployment-realistic question (§1.1).
- **Arms.**
  1. `occupancy_mode: all` — union of every shape the source can name (ceiling, not the default).
  2. `occupancy_mode: anchors-only` — union of the three prompt anchors only (current default).
  3. `occupancy_mode: none` — empty occupancy channel.
- **Done.** Source and occupancy_mode are now separate. Source is not a free
  parameter: `train.stage_b.mode: predicted` (default) → segmenter for
  *both* anchors and occupancy, and a `phase_a_checkpoint` is required;
  `mode: oracle` → ground truth for both (the only way to skip the
  checkpoint). Occupancy mode is
  `train.stage_b.occupancy_mode: all | anchors-only | none`. `all` is the
  ceiling in both sources (GT includes the target's voxels; a segmenter
  trained on the target's name will paint it). `anchors-only` and `none`
  do not. `StageB.forward` still subtracts the anchor channels, so
  anchors-only input is empty at the decoder — not a bug,
  `tests/test_models.py:
  test_stage_b_subtracts_the_anchors_from_the_occupancy_it_is_given` pins
  it as the architecture's "anchors are given, not repeated" invariant.
- **Open question, not yet checked**: once Stage A is actually trained (it
  sees all 10 names in every split, `scripts/train.py:52-54`), predicted
  `all` may converge toward oracle `all` for any shape Stage A segments
  well — including the target, which is why `all` is a ceiling. Compare GT
  vs segmenter for the same mode once a real Stage A checkpoint exists.
- **Report.** Dice on test, next to the `permute_both` control for each arm
  (a real accuracy drop and a slot-dependence artifact look identical
  without it) and next to the anchor Dice already reported.

### 3.2 Leave-one-target-class-out sweep

- **Question.** Corrected version of the original "9/10 structures" idea:
  does Stage B generalize to a held-out *target* class (already the design
  in `targets.test`), and does that depend on the held-out class having a
  close relative in train?
- **Arms.** Rotate which of the 10 primitives is `targets.test`, one at a
  time (10 runs), keeping `n_anchors`/everything else fixed. Config-only,
  via `--set "targets.train=[...]" --set "targets.val=[...]" --set "targets.test=['X']"`
  — no code change (§1.3).
- Group results afterward by "held-out class has a close relative in train"
  (current default: `triangular_prism` relative to `cube`/`pyramid`) vs. "no
  close relative" (e.g. `torus`, `capsule` alone) to separate compositional
  generalization from true zero-shot.
- **Cost check first.** 10 Stage B runs at current epoch/batch settings —
  confirm this is affordable (§2) before scheduling it.

### 3.3 `data.n_anchors` sweep (2 / 3 / 4)

- **Question.** Is the 3-way conjunction load-bearing? With 2 anchors the
  target is under-determined by construction — a Dice drop there is evidence
  the intersection module matters, not a bug.
- **Needs corpus regeneration**, not just a training override (§1.4): three
  `generate_data.py` runs, three corpus directories, then `data.root` points
  at each for the corresponding `train.py` run.

### 3.4 Anchor-name generalization (the literal reading of the original idea)

- **Question.** If Stage A never trains on a class name at all, can it/Stage
  B still use that class correctly when it appears only as an *anchor* at
  test time? This tests embedding-table coverage (`docs/method/03_prompt_encoder.md`'s
  `pair` table is `6 × |vocab|`, so an unseen name means untrained rows), not
  spatial reasoning — label it as that when reporting, not as a spatial-only
  capacity result.
- **Needs a code change** to the generator (§1.5) — no config path exists to
  exclude a shape from every scene while keeping label ids consistent across
  train/val/test. Scope this properly before starting; it's the most
  invasive item on this list.

### 3.5 Anchor-corruption sweep (cheapest — eval-only, no retraining)

- **Question.** Full anchor-quality-vs-target-Dice degradation curve, instead
  of the one operating point Stage A happens to produce.
- **Arms.** Erode / dilate / drop-one-channel on ground-truth anchors, at
  eval time only, several corruption levels.
- **Needs a small code change** to `scripts/evaluate.py` (or a new script) to
  inject the corruption before `StageBTask.anchors()` — no retraining
  required, existing checkpoints suffice.

### 3.6 Stage A rotation augmentation on/off

- **Question.** `docs/method/02_data.md` already names this as an unrun
  experiment: does augmenting Stage A fix `cube`/`cuboid` and
  `sphere`/`ellipsoid` confusion (both pairs differ only by anisotropy)?
- **Config-only**: `scripts/train.py a --set train.stage_a.augment=true`. No
  code change.

### 3.7 Difficulty stratification on existing runs (no new training)

- **Question.** Turn one Dice number into a limit curve using data already
  produced by any completed run.
- Bucket `predictions.jsonl` / `report.json` rows by target–anchor distance
  and by the count of non-anchor structures that satisfy 2-of-3 relations
  (near-misses for the conjunction). Pure analysis, no training, no code
  change beyond a throwaway analysis script.

## 4. Hygiene for every arm above

- ≥3 seeds per arm (`train.seed` in `configs/config.yaml:53`) — the
  occupancy and n_anchors deltas are the actual result; without seed spread
  they are unreadable.
- Report the `permute_both` control (`scripts/evaluate.py: CONTROL`) for
  every arm, not just the baseline.
- Report per-target stratification alongside any headline number — test is
  currently one class (`triangular_prism`); a single-class Dice is a noisy
  estimate on its own (§3.2 is what fixes this).
- The counterfactual battery (`evaluation.counterfactuals` in
  `configs/config.yaml:91`) already tests reliance on mask/prompt
  *correspondence* per example, for free, on every existing checkpoint. It is
  not a substitute for §3.1 (occupancy) or §3.3 (n_anchors), which test
  capacity under training conditions the battery never perturbs.
