# Project constraints

Segment a structure the prompt never names — it only locates it, by its
relations to structures that *are* named, and the mask is painted from the MRI
rather than chosen from a list of proposals. These are the invariants that make
that claim mean something. Breaking one does not fail a test; it quietly makes
the results unpublishable. Read this before changing `src/models.py`,
`src/mapper.py`, `src/engine.py` or `configs/config.yaml`.

The strategy, method and architecture are `documentation/SpatialVox.md`. Every
departure from the original proposal, with the measurement that justified it, is
in its section "Deviations from the original proposal", and every experiment is in
`documentation/Result_tracker/`. This file is constraints only.

---

## 1. What Stage B may see

**Exhaustive.** Three things reach `StageB.forward`, and everything else is
derived inside it.

| input | shape | what reads it |
|---|---|---|
| the MRI | `[B, 1, D, H, W]` | the frozen Stage A, and `B(I)` |
| `name_ids` | `[B, 3]` | **Stage A only** |
| `direction_ids` | `[B, 3]` | **the mapper only** |

Everything downstream is a function of those:

```
A_i         = stop_gradient(sigmoid(stage_a(image, name_ids)_i))     soft, detached
F_i         = sigmoid(margin_i / tau)  · [mass_i >= min_mass]        the pyramid
where_raw   = F_0 · F_1 · F_2                                        never renormalised
carver      <- B(I), A_0..2, F_0..2, where_raw, log(where_raw), where_mass
null head   <- where_mass, mass_0, mass_1, mass_2                    four scalars, no pixels
```

- **Names stop at Stage A.** There is no name, pair or slot embedding anywhere
  downstream. `tests/test_models.py::test_names_reach_stage_a_and_stop_there`
  holds every output bit-identical under an arbitrary renaming once the masks
  are fixed. A token would let the triple of anchor names stand in for the
  target.
- **The direction id is consumed by the mapper and nowhere else.** The direction
  is already the shape of `F_i`.
- **No coordinate grid reaches `B` or the carver.** World coordinates exist only
  inside the mapper, which consumes them to place a pyramid and emits fields.
- **Stage A's feature pyramid stays out.** Those features were trained to light
  up *named* structures. `B` is a separate encoder trained without class ids, so
  a structure Stage A has never seen is still a boundary in the image. That is
  the whole lesion claim.

`anchors=` and `boundary_image=` are the only other arguments `forward` admits.
`anchors` substitutes exactly the three detached soft masks Stage A would have
produced — the precomputed cache, and the ground-truth diagnostic — and cannot
identify the target. `boundary_image` is the §7 image-replacement test. Which
source a run used is recorded in its checkpoint.

## 2. What Stage B may never see

The label volume, the target mask, the target's name, its centroid, its size, an
occupancy map built from labels, any mask other than the three anchor
probabilities, and a candidate list. Pinned by
`tests/test_models.py::test_stage_b_signature_admits_nothing_that_identifies_the_target`
and `tests/test_engine.py::test_the_task_never_hands_the_model_anything_from_the_label_volume`.

The label volume is read **in the task**, to build a training target and to
score. It is never an argument to the model.

The unsigned boundary map is a pretraining target for `B` alone
(`BoundaryPretext`), and lives outside `StageB` so it is not reachable from the
relational forward.

## 3. The anchors

**Stage A is frozen and inside Stage B.** It is trained beforehand on every name
that may be an anchor. `StageB.train()` keeps it in `eval`,
`trainable_parameters()` excludes it, and the relational loss does not reach it.
Relaxing that needs an explicit, recorded decision — never a config flag.

**The mask handed downstream is the detached probability, not a threshold.** A
cut at 0.5 makes the centroid the mapper reads jump and can delete a dim but
real anchor in one step.

`mapper.min_mass` rejects an anchor Stage A failed to find. Measured on
`data/mri`, the smallest mass a real structure gets is **3.2e-6** and the
smallest class averages 4.6e-5, so the threshold is 1e-6. **The proposal's
starting value of 1e-3 sits above every structure in the vocabulary except the
brainstem and the thalami.** Re-measure with `scripts/gate_mapper.py --segmenter`
whenever Stage A is retrained.

### 3a. Anchor order: randomised, but shared by everything

Two rules, and they are not in tension — the order is chosen once, then used
everywhere.

1. **The order carries no information.** `anchor_first_examples` shuffles the
   triple before returning it. Storing a distance ranking made the slot index a
   perfect proxy for proximity, readable without parsing a single direction word.
2. **Slot `i` is the structure clause `i` names.** The same order is the order of
   `anchors`, of `directions`, of the rendered prompt's clauses, of `name_ids`,
   of the mask channels and of `F_i`. **Never reorder one without the others** —
   `roll_anchors` in `src/engine.py` exists so a counterfactual cannot.

Anchors always have **pairwise-distinct directions**: two clauses naming the same
side would not narrow the conjunction.

## 4. The corpus

**Anchor-first only.** The triple is fixed first and the targets follow, so one
anchor set serves several targets and only the direction words tell them apart.
A triple whose conjunction is not unique **on that scene** is dropped, so
per-scene well-posedness holds by construction.

**Global triple stability.** Per-scene uniqueness is not enough on real MRI: the
same unordered set of `(anchor name, direction)` pairs can uniquely mean
Left-Thalamus on one subject and Left-Caudate on another. After per-scene
generation, `stabilize_relational_manifests` (default `define_on="train"`)
decides stability from **train subjects only** and drops unstable rows from
train only. Val/test rows that collide with a train target are kept and counted
as an exposure stratum, so the recogniser shortcut stays measurable. Pass
`define_on="all"` only to reproduce the legacy global filter. Rebuild with
`scripts/rebuild_manifests.py` (or re-import); `meta.json` records
`triple_stability: train-unique-target`. Audit with
`scripts/triple_cross_patient.py`.

The target-first generator is **deleted, not configurable**. It picked the
anchors nearest the target, which on fixed anatomy made the anchor identities a
name tag: the unordered anchor set alone recovered the target 98.9% of the time
against 94.5% for solving the conjunction, so ignoring the prompt strictly beat
reading it. No pool width inverts that; the selection rule itself was the leak.

**Stage B has no rotation augmentation.** Its one augmentation is the direction
flip (§5 of the proposal). That is also what makes Stage A's output a function of
the scene alone, which is what `scripts/cache_anchors.py` relies on. Stage A
keeps its rotations.

**The flip is training-only.** A validation curve mixing retargeted and
empty-mask prompts would move `best.pt` for reasons unrelated to the model, and
an empty prediction against an empty target scores Dice 1.0. `scripts/evaluate.py`
builds the empty-prompt population separately.

**A flip is re-scored, never assumed empty.** Measured: it names two or more
structures 33.2% of the time (dropped), names none 65.5% (empty mask, null target
invalid) and retargets 1.2%. "Dropped" is the per-example `keep` weight, and it
must reach *every* loss term and `Metrics` — `weighted_mean` is that weight.

## 5. Generalisation the project claims

**Holds.** Stage B is supervised only on `targets.train` (8 classes).
`targets.val` (caudate, putamen) and `targets.test` (hippocampus) are scored
every epoch and **never used to choose a checkpoint** — `best.pt` is selected on
held-out *subjects* with *trained* classes.

**Does not hold, and matters.** "Never seen" is stronger than "never a target":

- every class, including the held-out ones, appears as an **anchor** in training,
  so its name embedding row in Stage A is trained;
- Stage A sees all 23 classes in every split, deliberately: the target-class
  split constrains what Stage B may be supervised on, not what anatomy exists.

**Do not describe this as zero-shot on unseen structures.** It is "never
supervised as a relational target". The lesion claim rests on the carver — that
the output need not be a structure Stage A knows how to draw — and the two
mandatory tests in §6 are what support it.

## 6. Reporting rules

A bare Dice is not interpretable in this project. Any reported Dice must carry:

1. **The prompt-blind floor.** "Of the non-anchor structures, take the one
   nearest the anchor centroid" never reads the prompt. Measured on `data/mri`,
   val split (`scripts/corpus_report.py`):

   | population | prompt-blind Dice | anchor-set ceiling |
   |---|---|---|
   | all target classes | **0.2017** | 42.3% |
   | `targets.train` (8, the selection curve) | **0.3067** | 67.8% |
   | `targets.val` (4, transfer) | **0.1097** | 83.9% |
   | `targets.test` (2, further held out) | **0.0630** | 99.2% |

   Read each number against **its own row**. Conditioning on fewer classes
   inflates the shortcut ceiling: on the two-class test population the anchor set
   alone identifies the target 99.2% of the time, so a high score there says
   almost nothing. Recompute whenever the generator changes.

2. **The counterfactuals.** `permute_channels`, `permute_clauses` and
   `flip_direction` must drop. `permute_both` preserves every relation and must
   not move — but note that `where_raw` is a product and therefore *exactly*
   permutation-invariant, so the only order dependence left is the carver's
   `cat`. A flat control is a much weaker statement here than it was under the
   attention architecture.

3. **Image use**, before a Dice is treated as evidence that the image was used
   at all. Two complementary tests:

   - **Image replacement** (`scripts/evaluate.py`): another subject's MRI into
     `B`, this subject's anchors and fields. The centroid should hold and the
     Dice should fall.
   - **Trained twin** (`scripts/train.py b --twin`): `refine` / q / K / V stay
     at their zero init, `B` is not computed, same seeds as the full run. The
     success bar is full model minus twin, never an absolute Dice. Switching
     `refine` off only at eval is not a twin — coarse alone is nearly flat.

4. **The gate.** `scripts/gate_mapper.py` measures the fraction of target
   centroids with `where_raw > 0.5`. §2 of the proposal makes it a precondition:
   a carver trained on top of a mapper that disagrees with the prompts is not
   worth scoring. At the shipped `tau = 0.5` it is **0.9742**; at the proposal's
   starting `tau = 2.0` it is 0.6508.

Also: non-determinism alone moved val Dice by ~0.01 typical / ~0.03 worst case on
this project's earlier runs, and nothing sets `cudnn.deterministic`. That is a
*lower* bound on seed spread. **A single-seed number is not a result** — say so
explicitly when only one seed was run.

## 7. Known properties of the corpus

Not constraints on the code — constraints on what may be concluded from it.

- **Predicted anchors are nearly oracle here.** Stage A's centroid error against
  ground truth is a median of **0.84 mm** on a 1.25 mm grid (95th percentile
  2.13 mm, worst 8.14 mm). So `anchor_source: oracle` and `predicted` are
  *currently non-discriminating* for the mapper, and the distinction only becomes
  real on data where segmentation is hard. Anchor Dice itself is the wrong
  summary — the mapper consumes the centroid, not the mask.
- **The null head's ceiling is a property of its inputs, not its width.**
  `where_mass` alone separates "names one structure" from "names none" at
  **AUC 0.848**. The mapper cannot see which regions hold tissue, so a roomy but
  empty conjunction looks exactly like a valid one — and §3 forbids showing it
  the MRI, for the right reason. A trained head at ~0.85 is working as well as
  its four numbers permit. Report Dice both gated and ungated by it.
- **Only ~36% of a target's voxels lie inside `where_raw > 0.05`**, and ~94%
  inside the 8-voxel dilation `L_far` measures mass outside. The pyramid
  describes where a *centroid* would satisfy the clause; part of the body
  legitimately lies outside it. `where_raw` is a channel and a weak bias — never
  a crop, never a mask, and `logit(where)` is not added to the logits unless the
  scalar-`alpha` ablation is on.

## 8. Architecture compatibility

`StageB.config` is what `load_model` rebuilds from, so every architectural
parameter must appear there or old checkpoints break — including `segmenter`,
the frozen Stage A's own config, which makes a Stage B checkpoint
self-contained. `flip_probability` and the loss weights are training-schedule
values and live in the checkpoint's `meta["config"]["stage"]`, mirrored into the
`.json` sidecar; see "Where the specification was ambiguous" in `documentation/SpatialVox.md`.

Checkpoints of the previous attention-based Stage B do not load here, and are
not meant to.

## 9. Conventions

Arrays are `(z, y, x)`; world coordinates are `(x, y, z)` in a RAS frame. A label
volume stores `vocabulary index + 1`, 0 for background. A relation always
describes the target relative to the anchor. **`mapper.tau` and every distance
are in world units** — the corpus is 1.25 mm/voxel, so they are millimetres, and
`corpus.spacing` (not `data.spacing`) is the authority. Every tunable lives in
`configs/config.yaml`; nothing in `src/` hard-codes a value that lives there.
