# Experiments plan — checks and tests to prepare

Written 2026-09-18, before anything had been run. **Updated 2026-09-19:** §3.1
has been run, and it found an architecture bug that has since been fixed — that
section now reports results and the change it caused, rather than a plan. Every
remaining item still states, explicitly, whether it needs a config override, a
corpus regeneration, or a code change before it can run.

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
  `anchors-only` is empty at the decoder — **which makes it a duplicate of
  `none`, not an arm.** See §3.1; this line was written before that was
  understood.

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

### 3.1 Occupancy ablation — **run, and it found an architecture bug**

Three arms were run (`runs/occ_oracle_{all,anchors-only,none}`, `mode: oracle`,
1 seed). Full analysis and figures:
[`notebooks/occupancy_ablation.py`](../notebooks/occupancy_ablation.py).

| arm | best val Dice | what it actually measured |
|---|---|---|
| `all` | 0.9925 | a silhouette leak, not capacity — see below |
| `anchors-only` | 0.2212 | **bit-for-bit identical input to `none`** |
| `none` | 0.2294 | the same experiment, run twice |

**What went wrong.** `StageB.forward` multiplies occupancy by
`(1 - anchors.amax(1))`, and `anchors-only` sets occupancy *to* that anchor
union. Binary masks, so `u * (1 - u) == 0`: the decoder got zeros either way and
the two arms were one experiment. §1.1 of this file noted the emptiness but
framed it as a design note rather than "this arm is a duplicate", which is why
the run was scheduled.

**Two findings that survive, both eval-only on the existing checkpoints:**

- `all` scores **exactly 0.0000** when the target's voxels are removed from
  occupancy — not degraded, zero. A model selecting among the 6 remaining
  candidate blobs would pick a wrong one and still score ~0.1-0.2. It learned
  `output ⊆ occupancy`, so its 0.9925 (Hausdorff 0.66, sub-voxel) is the
  answer's outline being handed to it as an input.
- `none` scores the same handed a full occupancy map as an empty one. It never
  built a pathway for that channel at all.

**The real defect.** With an empty occupancy channel Stage B could not see that
*any* structure existed — three blobs in an empty volume and a sentence. The
0.22 arms were not reasoning badly; they were guessing, which is all the inputs
allowed. And the occupancy that made `all` work was built from ground-truth
labels: oracle information no deployment has.

**Fixed:** Stage B now takes the intensity volume at the decoder, area-pooled to
each scale (`model.stage_b_image`, default on). The encoder still sees only the
anchors, so localisation stays provably relational. Same `none` arm: **0.2294
after 30 epochs → 0.8912 after 3.**

**Read the synthetic numbers with this caveat.** `synthetic.appearance` puts
background at `[0.12, 0.04]` and structures at `[0.45, 0.75]`, which do not
overlap: pooled over ten val scenes, background p99.9 is 0.260 and structure
p0.1 is 0.275, so a single threshold recovers `labels > 0` at IoU 0.9998. On
*this corpus* the intensity volume is therefore nearly as informative as
`occupancy_mode: all` — it is the same candidate set, just un-thresholded. What
the image changes is where that information legitimately comes from (an acquired
volume, available at inference) rather than from ground-truth labels. It does
not make the synthetic task harder, and 0.89 should be read as approaching the
`all` ceiling of 0.99, not as beating it. Per-class means span 0.488-0.586 with
standard deviations near 0.07, so intensity identifies *foreground*, never a
class. On real MRI no threshold separates structures at all — that is what Stage
A is for, and why `mode: predicted` is the setting that actually tests this.

`occupancy_mode` survives as an independent axis on top of the image, now
four-valued: `all` | `distractors-only` | `anchors-only` | `none` (default
`none`). `distractors-only` is new — every non-anchor structure except the
target — and is an oracle **diagnostic**: it reads `batch["target"]`, so
`StageBTask` refuses it under `mode: predicted`. Excluding a target you have not
found is not something inference can do; the deployable version of that question
is §3.4.

**Still open.**

- Re-run the four arms with the image on, ≥3 seeds (§4). The prior that
  `distractors-only` lands between `all` and `none` is *not* safe: `none` ignored
  occupancy entirely, so it may land at neither.
- Report `permute_both` per arm. On the old `all` checkpoint the control moved
  0.22 (0.9925 → 0.7722) while the real perturbations moved 0.70-0.72 — real
  relational sensitivity plus a slot-order artifact. On the 0.22 arms everything
  collapsed to ~0.02-0.06, which says little.
- Compare GT vs segmenter occupancy for the same mode once a real Stage A
  checkpoint exists.
- **Encoder-side image injection** is the obvious follow-up and was deliberately
  not built. It would break the invariant three tests assert directly
  (`test_stage_b_sees_the_anchors_and_an_anonymous_occupancy_and_nothing_else`
  checks `torch.equal(seen["encoder"], anchors)`), so it needs its own arm and
  an explicit decision to relax the project's central leak check — not a config
  flag that silently disables it.

**Noise floor, measured for free.** `anchors-only` and `none` shared a seed and
a bitwise-identical input, so they should have been bit-identical. Nothing pins
non-determinism (`src/engine.py` seeds and stops; no `cudnn.deterministic`), and
they diverged by 0.012 mean / 0.034 max per epoch. That is a **lower bound** on
seed-to-seed spread, which is why §4's ≥3 seeds is not optional.

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
