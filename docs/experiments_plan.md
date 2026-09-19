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
`none`). **Only `all` and `none` are arms.** The other two are controls:

- `anchors-only` is bit-for-bit `none` at the decoder (§3.1 above).
- `distractors-only` was proposed here as "the honest middle ground". **That was
  wrong once the image became a permanent input.** Occupancy is every non-anchor
  structure except the target, and the image gives the foreground, so
  `target == foreground - occupancy - anchors` exactly. Measured on
  `runs/occ_img_oracle_distractors-only`: Dice **1.0000**, Hausdorff **0.0**,
  and all four counterfactuals **1.0000** — the prompt is not used at all. It is
  a *negative* control: useful once, to confirm the leak behaves as predicted,
  and never comparable with the real arms.

### The ordering leak (found 2026-09-19, fixed)

`select_anchors` returned candidates ranked by distance and the manifest stored
that order, so the slot index was a perfect proxy for proximity: **slot 1 was
the target's nearest structure in 100% of examples**, readable without parsing a
direction word. Fixed by shuffling the returned order (`src/geometry.py`), in
`build_examples` and in `ExampleDataset._augment`; `scripts/rebuild_manifests.py`
rewrites the manifests in ~3 s without re-rendering volumes. Stored order is now
sorted-by-distance in ~17% of examples, which is chance for 3! = 6.

Priced on `runs/occ_img_oracle_none` (val), by re-scoring one checkpoint:

| scored on | Dice | `permute_both` drop |
|---|---|---|
| pre-fix corpus | 0.9860 | 0.22 |
| shuffled corpus | **0.8753** | **0.0152** |

Two things this settles. The 0.11 Dice gap is what the leak was worth. And the
`permute_both` control — which had dropped ~0.22 on *every* checkpoint and
looked like an architectural permutation-invariance defect — was measuring the
leak: once order carries no information, it behaves. **No symmetry refactor is
needed.** `flip_direction` still costs 0.574, so the relational sensitivity is
real.

### The three real modes, re-scored on the clean corpus

`mode: oracle`, val, checkpoints trained pre-fix and re-scored post-fix (so
these understate a cleanly-trained model). Prompt-blind baseline: **0.674**.

| occupancy | Dice | flip_direction | permute_channels | permute_clauses | **permute_both** (control) |
|---|---|---|---|---|---|
| `all` | 0.9206 | **-0.693** | -0.496 | -0.469 | -0.036 |
| `anchors-only` | 0.8816 | **-0.568** | -0.097 | -0.170 | -0.013 |
| `none` | 0.8753 | **-0.574** | -0.162 | -0.181 | -0.015 |

Reading: the control no longer moves, so the battery is trustworthy again, and
flipping **one** clause's direction costs 0.57-0.69 Dice. **The prompt is
load-bearing in all three modes** — the earlier "the model ignores the prompt"
result was specific to `distractors-only`, which is not one of them. `all`
buys about +0.045 over `none`, and `anchors-only` sits within noise of `none`
exactly as the subtraction predicts.

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

## 5. Making the data harder — what each knob buys (NOT applied)

Written 2026-09-19 as an explanation, not a change. Every measurement in §3.1
says the same thing: the corpus is too easy in three independent ways, and each
has its own fix. Ranked by information gained per unit of work.

### 5.1 Break the proximity shortcut — **implemented, one config line**

`data.anchor_pool` (default 3) draws the anchors from the *k* nearest feasible
candidates instead of taking the nearest ones deterministically. Already in the
code; applying it is `--set data.anchor_pool=5` plus
`scripts/rebuild_manifests.py` (~3 s, no volumes re-rendered).

| | prompt-blind baseline |
|---|---|
| `anchor_pool: 3` | 0.674 |
| `anchor_pool: 5` | **0.478** |

Same example counts, same feasibility, zero duplicate directions. The cost is
realism: a reader does name *nearby* landmarks, so a very wide pool describes a
task nobody would pose. 5 is a reasonable compromise; report the baseline
whichever value is used.

### 5.2 Make segmentation non-trivial — **the one that revives two dead axes**

`synthetic.appearance` puts background at `[0.12, 0.04]` and structures at
`[0.45, 0.75]`. Measured over ten val scenes: background p99.9 = 0.260,
structure p0.1 = 0.275. **They do not overlap**, so one threshold recovers
`labels > 0` at IoU 0.9998. Three consequences, all bad for the experiment:

- Stage A scores 0.998, so `mode: predicted` ≈ `mode: oracle` (0.8913 vs
  0.8905) and the source axis measures nothing.
- The image is as informative as `occupancy_mode: all`, so the occupancy axis
  measures nothing either.
- "Segmentation" is free, so Stage B's Dice is almost entirely a selection
  score — which flatters it relative to any real deployment.

The fix is to overlap the distributions: raise `background` mean/std toward the
structure range, widen `noise`, and increase `blur` so partial-volume edges are
genuinely ambiguous. Start by targeting a Stage A Dice around 0.90-0.95 rather
than 0.998 — that is the regime where an occupancy prior from Stage A can
actually help the decoder, and where `predicted` vs `oracle` becomes a real
comparison. **This is the highest-value change after 5.1**, because without it
neither the occupancy ablation nor the end-to-end claim can produce a number.

### 5.3 Make the conjunction load-bearing

Today a scene holds exactly one instance of all ten primitives (`pack_scene`),
so the model chooses 1 of 7 non-anchor candidates, and on average only **0.50**
of them satisfy 2 of the 3 stated relations. A prompt where no distractor is a
near-miss does not need all three clauses. Options, in increasing difficulty:

- Place structures so more candidates are near-misses — reject a scene whose
  best distractor satisfies fewer than 2 relations. Pure generator change, and
  it directly raises the floor that §3.3's `n_anchors` sweep is supposed to test.
- Allow several instances of one shape per scene. This is the realistic case
  (two kidneys, many gyri) but it breaks the prompt language: "the cube" stops
  being a unique referent, so `src/vocab.py` would need instance
  disambiguation. Scope it properly before starting.
- Raise the structure count so 1-of-7 becomes 1-of-15.

### 5.4 Truly unseen structures

Needed for the "never seen" claim in `CLAUDE.md` §4, which does not hold today —
the held-out target class is still an anchor 811 times in training and Stage A
sees all ten classes. `src/synthetic.py: SHAPES` is a literal dict with no
allow-list, and label ids must stay consistent across splits. See §3.4.

### 5.5 Lower-value, but worth listing

- **Non-axis-aligned poses.** Augmentation uses the octahedral group, and
  structures are placed axis-aligned, so `superior`/`anterior` land on voxel
  axes. Arbitrary rotations would stop the direction rule coinciding with the
  grid.
- **Anisotropic spacing.** Supported everywhere (`data.spacing`) and never
  exercised; real MRI is rarely isotropic.
- **Size and shape variation**, so the target's extent is not predictable from
  its class — currently `draw_params` samples from a narrow per-shape range.

### 5.6 What none of this replaces

Hardening the corpus does not make real MRI unnecessary — it makes the move
*interpretable*. Real anatomy is more positionally stereotyped than these
scenes, so every shortcut above gets stronger there, not weaker: a model can
memorise the atlas and never read the prompt. Before moving, build the MRI
analogue of the prompt-blind baseline ("predict the class-average location") and
carry the §4 reporting rules with it.
