# Confrontation with the entity-selector blueprint

The blueprint in the design discussion is not a refinement of
[`relational_architecture.md`](relational_architecture.md). The carver in that
document, with names discarded after Stage A, is the method. Anonymous entity
selection is a secondary baseline for the easier case where the target was
seen as a mask and never as a relational target. The two are not combined:
a candidate list would mean the output has to be a structure Stage A already
knows how to draw.

The proposal paints a mask from the MRI inside the region the three clauses
describe, so a structure that has never had a mask — a lesion between known
anchors — can still be segmented. The blueprint selects one anonymous proposal
and only then refines it, inside a band around that proposal. A structure that
was never proposed has no path to an output.

Both can be true of a paper. They cannot be the same network.

## What already matches this repository

| blueprint | this repo |
|---|---|
| inputs are the MRI and three clauses; the target name is not an input | the contract in `CLAUDE.md` and in the proposal |
| backbone widths `[32, 64, 128, 256]` at 128³, 64³, 32³, 16³ | `model.stage_a.encoder_channels` is `[32, 64, 128, 256, 256]`, five stride-2 stages from 128 to 8³, checked against `model.stage_a.bottleneck`. Stage B has no such backbone at all: `B(I)` is three stages at 16-32 channels |
| InstanceNorm or GroupNorm, not BatchNorm | Stage A already uses `InstanceNorm3d` |
| no language model on the inference path | `Vocabulary.clause_ids` is the only text compiler |
| slot `i` aligns clause `i` with anchor channel `i`; slots are permuted as a triple | `shuffle_clauses: true`, and the tests that lock the correspondence |
| a direction flip is not assumed to be an empty prompt | the proposal already retargets a flip that still names one structure, and drops a flip that names two |
| checkpointing must not look at the held-out-as-target classes | **now what the code does.** `best.pt` is selected on val-split subjects restricted to `targets.train`; `targets.val` and `targets.test` are scored every epoch under the names `val:targets.val` and `val:targets.test` and never selected on |
| report more than Dice | `scripts/evaluate.py` reports six blocks: Dice with anchor Dice, centroid error with the empty-prediction rate, the gate on the split, the four counterfactuals, the empty-prompt population, and image replacement |

`model.stage_b_selection` and `occupancy_mode` were the blueprint’s selector and
its silhouette leak, and both were switched on when this note was written.
**Both are now gone** — not turned off, deleted, along with the attention Stage B
they lived in (`deviations.md` §6). `occupancy_mode: all` handed the decoder the
target's own outline; the selection head copied a Stage A mask. Neither belongs
in the forward the proposal describes, so neither is reachable from it.

## Where the blueprint contradicts the repository

**Directions.** The vocabulary is `anterior, posterior, superior, inferior,
medial, lateral`. Medial and lateral are distances to the mid-sagittal plane,
not the unit vectors `+x` and `−x`. A clause is the one dominant axis of the
centroid offset (`classify`), so a pair is not simultaneously superior and
anterior. Anchors are pairwise distinct, and so are directions
(`Vocabulary.validate`). The blueprint’s `{left, right, …}`, its six independent
predicates, and its permission to repeat an anchor are a different language.
Implementing them rewrites `src/geometry.py`, the manifests, and every
counterfactual.

**Canonical template coordinates.** Scans are already in a shared ACPC frame.
A further deformable registration, and a head trained to predict those
coordinates from the image, teaches the backbone where it sits in an atlas. On
this data a centroid in template space is close to a class name: the
configuration notes that, with anatomy fixed, the anchor set alone recovers the
target more often than the conjunction does. The proposal keeps world
coordinates inside the mapper, where medial and lateral need the volume centre,
and does not feed a coordinate grid to the carver.

**The selector is the path that was set aside.** Protocol A trains entity masks
on all twelve labels, including the four that are never relational targets, by
Hungarian matching. The held-out structure is therefore segmented by name-free
but class-seen mask supervision, and the output copies the matched proposal.
The refiner is allowed to edit only `dilate(selected) ∩ dilate(fields)`. That
cannot draw a lesion Stage A has never been given a mask for. Protocol B, which
withholds those four from every mask loss, also withholds them as anchors. The
requirement that all twelve structures be usable as anchors and the requirement
that the four be visually unseen cannot be the same training run. The blueprint
says so. The implementation keeps one claim: anchors may be any named
structure; the target mask is painted from the image; the four are held out of
the relational loss only.

## What is not carried across

- `K` anonymous queries, Hungarian assignment, and an entity-embedding volume.
  Matching uses the target mask as a training target for every labelled
  structure. That is the silhouette the selection head was added to copy.
- A geometry MLP that scores candidates from canonical centroids and extent.
  After registration those numbers identify the class.
- A learned residual on the relational field (`delta_g` conditioned on the
  anchor and the coordinates). The field stays the soft form of `classify`,
  with one constant `τ` and no learned gain. A learned field that can move
  itself onto a blob is the shortcut the fixed pyramid exists to close.
- A null class and a selector softmax. An empty mask is the carver’s output
  when the clauses describe nothing. A separate null score is unnecessary
  until a selector exists.
- Dense pairwise relation loss as a prerequisite. The corpus already stores one
  direction per pair. A six-predicate BCE against that table would supervise
  contradictions. Pairwise geometry can be logged as a diagnostic. It is not a
  new head in the first implementation.
- Self-supervised masked pretraining, surface Dice, SDF loss, and a
  coordinate-reconstruction head. They do not decide whether the clauses select
  the structure.

## What the implementation does take

From the blueprint, only the gates and the reports:

1. Validation subjects used to save `best.pt` are drawn from the eight
   supervised targets. The four held-out classes are scored and never used to
   choose a checkpoint. Hippocampi stay a further held-out pair.
2. A direction is flipped only in the token and the pyramid together. If the
   new clauses name exactly one structure, that structure is the mask target.
   If they name none, the mask target is empty. If they name two, the example
   is dropped. “Impossible” is measured, not assumed.
3. The report decomposes a run into: fraction of target centroids inside
   `where`, centroid error, Dice, anchor Dice, alternate-prompt switch rate,
   and false-positive volume on empty prompts. A single Dice is not the result.
4. The direct comparison the blueprint asks for is the one already specified.
   The prompt-conditioned path is now the mapper plus the carver — the three
   attention branches this line used to name are gone with the rest of the
   attention Stage B. A geometry-only selector is an ablation that has to beat
   that path on switch rate and on the four held-out targets before it replaces
   the carver. It is not part of the refactor.

The network to implement remains the one in `relational_architecture.md`.
