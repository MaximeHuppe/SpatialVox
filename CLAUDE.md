# Project constraints

Segment a structure the prompt never names — it only locates it, by its relations
to structures that *are* named. These are the invariants that make that claim
mean something. Breaking one does not fail a test; it quietly makes the results
unpublishable. Read this before changing `src/models.py`, `src/engine.py` or
`configs/config.yaml`.

Findings and run history live in `docs/experiments_plan.md` §3.1 and
`notebooks/occupancy_ablation.py`. This file is constraints only.

---

## 1. What Stage B may see

**Exhaustive.** Four tensors reach the network. Anything else is a leak.

| input | shape | enters at | pooling |
|---|---|---|---|
| ordered anchor masks | `[B, 3, D, H, W]` | **encoder** (+3 world-coordinate channels → 6) | — |
| clause ids (`direction_ids`, `name_ids`) | `[B, 3]` | prompt encoder, structure encoder, decoder FiLM | — |
| the scene image | `[B, 1, D, H, W]` | **decoder guidance only** | `avg` (area) |
| occupancy | `[B, 1, D, H, W]` | **decoder guidance only**, after anchor subtraction | `max` |

- **The image is always sent.** `model.stage_b_image: true` is the project
  default. `false` exists only to reproduce the reference architecture
  bit-for-bit (`tests/test_reference_parity.py`) and for ablations — it is not a
  deployment setting. With it off, an empty occupancy leaves Stage B unable to
  see that any structure exists at all.
- **The encoder stays anchors-only.** The image and occupancy must not reach it.
  Grounding queries that could see the target's own voxels let the model pick "a
  blob that is not an anchor" instead of reading the prompt. Relaxing this needs
  an explicit, recorded decision — never a config flag that silently disables the
  guarantee.
- **Occupancy has the anchors subtracted** (`src/models.py`, `1 - anchors.amax`).
  The anchors are already three encoder channels; re-supplying them is
  redundant by construction. This is why `occupancy_mode: anchors-only` is
  bit-for-bit `none` at the decoder — correct behaviour, not a bug, but it means
  **`anchors-only` is not an experimental arm.** It is kept because it is the
  natural thing to reach for, and documented so nobody schedules it twice.

Pinned by `tests/test_models.py::test_stage_b_sees_the_anchors_and_an_anonymous_occupancy_and_nothing_else`
and `::test_stage_b_image_reaches_the_decoder_but_never_the_encoder`.

### 1a. Anchor order: randomised, but shared by everything

Two rules, and they are not in tension — the order is chosen once, then used
everywhere.

1. **The order carries no information.** `select_anchors` ranks candidates by
   distance to *choose* them and then shuffles before returning
   (`src/geometry.py`). Storing the ranking made the slot index a perfect proxy
   for proximity: slot 1 was the target's nearest structure in **100%** of
   examples, readable without parsing a single direction word. After the fix,
   the stored order is sorted-by-distance in ~17% of examples — chance for
   3! = 6 permutations.
2. **Channel `i` is the structure clause `i` talks about.** The randomised order
   is the order of `anchors`, of `directions`, and of the rendered prompt's
   clauses, all from one list in `build_examples`. Downstream, *everything*
   indexes by that same slot: the encoder's mask channels
   (`masks_from(labels, anchors)`), `name_ids = anchors - 1`, the relation
   tokens, the per-slot `Evidence` grounding branches, the `Intersection` maps
   and the decoder's FiLM context. **Never reorder one without the others.**

That correspondence is the thing the `permute_channels` counterfactual is built
to detect, so it must hold by construction rather than by luck. Pinned by
`tests/test_geometry.py::test_a_manifest_clause_always_describes_its_own_mask_channel`
and `::test_a_manifest_anchor_order_is_not_sorted_by_distance`.

Anchors always have **pairwise-distinct directions** — two clauses naming the
same side would not narrow the conjunction
(`src/geometry.py: _distinct_directions`).

## 2. What Stage B may never see

The label volume, the target mask, the target's name, its centroid, its size.
Pinned by `tests/test_models.py::test_stage_b_signature_admits_nothing_that_identifies_the_target`.

`occupancy_mode: distractors-only` is the one mode that reads `batch["target"]`.
`StageBTask.__post_init__` refuses it under `mode: predicted`, because excluding
a target you have not found is not something inference can do. That guard is a
constraint. Do not relax it.

**It is a negative control, never an arm.** Occupancy is every non-anchor
structure *except* the target, and the image supplies the foreground, so

    target  ==  foreground(image)  -  occupancy  -  anchors

exactly — verified, Dice 1.0 by pure set arithmetic. A model trained on it
learns that subtraction and ignores the prompt completely: `runs/occ_img_oracle_
distractors-only` scores Dice **1.0000**, Hausdorff **0.0**, and **all four
counterfactuals 1.0000**. Run it once to confirm the leak still behaves that
way; never report it as a capacity result, and never compare it with `all` or
`none`.

## 3. Occupancy source

`occupancy_mode` ∈ `all` | `distractors-only` | `anchors-only` | `none`, and the
*source* is `train.stage_b.mode`:

- **`mode: predicted` is the only deployment-relevant setting.** Occupancy is
  then Stage A's segmentation, which is what the design intends: "all the
  structures Stage A found", "only the anchors", or "nothing".
- **`mode: oracle` builds occupancy from ground-truth labels.** It is a
  diagnostic for isolating the relational architecture from Stage A's error.
  **Every run in `runs/` used it** — so none of them tested the intended
  occupancy path. Do not report an oracle number as an end-to-end result.

Caveat to state whenever this matters: on the current synthetic corpus Stage A
scores 0.998 Dice, so `predicted` ≈ `oracle` (0.8913 vs 0.8905). The distinction
is **currently non-discriminating** and only becomes real on data where
segmentation is hard.

## 4. Generalisation the project claims

**Holds today.** Stage B is supervised only on `targets.train` (7 classes) and
tested on `targets.test` — currently `triangular_prism`, never a *target* in
training. It scores 0.836 there.

**Does not hold today, and matters.** "Never seen" is stronger than "never a
target":

- `triangular_prism` appears as an **anchor 811 times** in training, so its name
  embedding row is trained.
- Stage A sees all 10 classes in every split (`scripts/train.py`, deliberate:
  the target-class split constrains what Stage B may be supervised on, not what
  anatomy exists).

Excluding a class from the corpus entirely needs a generator change —
`src/synthetic.py: SHAPES` is a literal dict with no allow-list, and label ids
must stay consistent across splits. Scoped in `docs/experiments_plan.md` §3.4.
**Do not describe the current setup as zero-shot on unseen structures.**

## 5. Reporting rules

A bare Dice number is not interpretable in this project. Any reported Dice must
carry both of these:

1. **The prompt-blind baseline.** "Pick the non-anchor structure nearest the
   anchor centroid" never reads the prompt, and it scores:

   | `data.anchor_pool` | prompt-blind baseline |
   |---|---|
   | 3 (default — anchors are the nearest feasible set) | **0.674** |
   | 5 | **0.478** |

   A perfect relational solver scores 1.000 (the conjunction is unique in 97.8%
   of examples). So at the default the readable range is **0.674 → 1.000**, not
   0 → 1. Recompute it whenever `anchor_pool` or the generator changes —
   `notebooks/occupancy_ablation.py` §9 does it in seconds, from the manifests
   alone.
2. **The `permute_both` control.** It preserves every relation and must not
   move. Measured on `runs/occ_img_oracle_none`, one checkpoint, val:

   | corpus it is scored on | Dice | `permute_both` drop |
   |---|---|---|
   | pre-fix (anchors stored nearest-first) | 0.9860 | **0.22** |
   | shuffled (current) | 0.8753 | **0.0152** |

   The control was measuring the leak, not an architectural defect: with order
   randomised there is nothing for a permutation to destroy, and it behaves.
   **So no permutation-invariance refactor is needed** — the encoder's 3 mask
   channels, `Intersection`'s ordered `cat` and the FiLM context stay
   order-dependent, which is legitimate once order carries no information.

   Two consequences that are constraints, not observations:
   - **Any checkpoint trained on a pre-fix corpus is contaminated.** Its
     headline number includes ~0.11 Dice of ordering leak. Everything in `runs/`
     written before 2026-09-19 14:07 is in that category; re-score on the
     current corpus or retrain before quoting it.
   - If the control ever moves materially again, something has reintroduced
     information into slot order. Check `select_anchors` and
     `ExampleDataset._augment` first.

Also: non-determinism alone moves val Dice by ~0.01 typical / ~0.03 worst case
(two same-seed runs with bitwise-identical inputs diverged that much; nothing
sets `cudnn.deterministic`). That is a *lower* bound on seed spread. **≥3 seeds
per arm**, or the arm is unreadable.

## 6. Known properties of the synthetic corpus

Not constraints on the code — constraints on what you may conclude from it.

- **Foreground is trivially separable.** Background `[0.12, 0.04]` and structures
  `[0.45, 0.75]` do not overlap: one threshold recovers `labels > 0` at IoU
  0.9998. So the image is nearly as informative as `occupancy_mode: all`, and
  the occupancy axis is largely **redundant on this corpus**. It will only
  discriminate where segmentation is hard.
- **Intensity does not identify a class.** Per-class means span 0.488–0.586 with
  std ≈ 0.07. Foreground, never identity.
- **Every scene holds exactly one of all 10 primitives** (`pack_scene`), so there
  are always 3 anchors, 1 target and 6 distractors — a 1-of-7 selection.

## 7. Architecture compatibility

`StageB.config` is what `load_model` rebuilds from, so every architectural
parameter must appear there or old checkpoints break. `model.stage_b_image:
false` is bit-identical to the pre-image architecture (verified: identical
logits, evidence maps and all 77 initial weight tensors).

`tests/test_reference_parity.py` **skips** when the `exp/realistic-appearance`
branch is absent, as in this clone. A green suite is therefore not a parity
guarantee — check the test *ran* before claiming one.

## 8. Conventions

Arrays are `(z, y, x)`; world coordinates are `(x, y, z)` in a RAS frame. A label
volume stores `vocabulary index + 1`, 0 for background. A relation always
describes the target relative to the anchor. Every tunable lives in
`configs/config.yaml`; nothing in `src/` hard-codes a value that lives there.
