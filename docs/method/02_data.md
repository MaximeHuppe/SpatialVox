# 02 — Data and relations

`src/geometry.py`, `src/synthetic.py`, `src/data.py`

## A corpus is two volumes per scene

```
<root>/
  meta.json                          shape, spacing, anchors per prompt, target split
  vocab.json                         ordered structure names; label id == index + 1
  scenes/<scene_id>/image.nii.gz     float intensity volume   (RAS)
  scenes/<scene_id>/labels.nii.gz    integer label volume     (RAS)
  train.jsonl val.jsonl test.jsonl   one relational example per line
```

That is the entire interface between "where the data came from" and "everything
else". `src/synthetic.py` produces an `(image, labels)` pair; so does an MRI
archive. Nothing downstream can tell which it got.

A manifest line is deliberately small:

```json
{"scene": "train_00007", "id": "train_00007__pyramid", "target": 7,
 "anchors": [3, 1, 9], "directions": ["superior", "medial", "anterior"],
 "prompt": "segment the structure that is superior to the sphere, ..."}
```

**No masks are stored.** Every mask in this project is `labels == id`, built on
the accelerator inside the training step (`src/engine.py: masks_from`). One label
volume per scene replaces up to `len(vocab)` mask volumes, which is what keeps a
sample cheap at 128³ with a large vocabulary — and it makes it impossible for a
stored mask to drift out of step with the labels it came from.

## The direction rule

`src/geometry.py: classify`. Six tokens, no synonyms:

```
anterior  posterior  superior  inferior  medial  lateral
```

From `delta = centroid(target) - centroid(anchor)` in world units:

1. the deciding axis is `argmax(|dx|, |dy|, |dz|)`, ties broken `z > y > x`;
2. axis `z` → `superior` / `inferior`;
3. axis `y` → `anterior` / `posterior`;
4. axis `x` → whichever structure is **farther from the mid-sagittal plane** is
   `lateral`, the nearer one `medial`.

Rule 4 is the one worth reading twice. `medial`/`lateral` is not the sign of `dx`
— it is a comparison of distances to the centre plane, so two structures both
left of centre still have a well-defined medial/lateral relation. That is the
anatomical meaning, and it is the reason the model is given *world* coordinates
rather than tensor indices: without a centre, `medial` is undefined.

Two configurations are undecidable and raise `AmbiguousDirection` rather than
getting a made-up answer: coincident centroids, and an exact tie in centre-plane
distance. Callers drop that pair. A direction is never invented.

Working in world units (not voxel indices) is what makes anisotropic spacing
correct: 3 voxels of `y` beats 2 of `z` until `z` voxels are twice as long, and
then it does not.

## Nearest-feasible anchor selection

`src/geometry.py: select_anchors`. For a chosen target:

1. rank every other structure by centroid distance, label id as tie-break;
2. walk the ranking, keeping a structure only if its direction is not already
   used;
3. stop at `n_anchors` (3) distinct directions;
4. if no such set exists, **drop this target** — the scene keeps its other
   examples.

The result is the nearest *feasible* set, not the three nearest structures: if
the two closest neighbours are both `superior`, the second is skipped for a
farther one that adds a new direction. Distinct directions are what force the
intersection to be informative — three `superior` clauses would leave the target
underdetermined.

The resulting order — ascending distance — is shared by the prompt clauses, the
`anchors` list, and therefore the mask channels. Clause *i* and channel *i* are
the same anchor, everywhere, always. That correspondence is the thing the
counterfactuals in [07](07_evaluation.md) attack.

## Splitting by target class

`configs/config.yaml: targets` assigns each split a disjoint set of structures it
may supervise as the **target**. Every structure remains available as an
**anchor** in every split. So:

- the vocabulary is fully seen — no embedding is untrained;
- validation and test measure transfer to structures never supervised as targets.

The default synthetic assignment is 7 / 2 / 1, chosen so that each held-out class
has a close relative inside training (`cuboid` ← `cube`, `ellipsoid` ← `sphere`,
`triangular_prism` between `cube` and `pyramid`), making the transfer
compositional rather than a jump to an unseen family.

## The synthetic corpus

`src/synthetic.py`. Ten axis-aligned primitives — cube, cuboid, sphere,
ellipsoid, cylinder, cone, pyramid, triangular prism, torus, capsule. Each is one
entry in the `SHAPES` table: its parameter ranges, its membership predicate on
world offsets from its centre, and half its bounding box. Adding a primitive is
adding a row.

A scene is packed by rejection sampling — draw size, draw a centre that keeps the
bounding box inside the margin, rasterise, reject if empty or overlapping, retry;
bulky classes first. Sizes are written for a 64-voxel axis and rescale linearly,
so a 128³ corpus holds the same structures at twice the sampling. Everything
derives from the scene seed, so a scene is bit-for-bit reproducible from that
seed alone.

Intensities are painted afterwards over untouched labels: a noisy background,
a per-structure mean, intra-structure noise, and a light Gaussian blur so edges
are partial-volume rather than a 0/1 cut. The mean is the midpoint of one shared
range plus a small deterministic per-class offset (`class_bias`, T1-like) plus a
per-structure draw (`jitter`) of comparable size, clipped back into the range.
The overlap is the point: brightness carries a hint of identity but never
determines it, so Stage A has to use shape and position — the situation it will
face on real MRI.

Images are fed to the network as generated. `data.normalize: zscore` turns on
per-volume z-scoring, which real MRI needs and this corpus does not.

Two classes need an anti-degeneracy rule: a `cuboid` drawn isotropically is a
cube, and an `ellipsoid` drawn isotropically is a sphere. `ANISOTROPIC` rejects
those draws.

## Rotation augmentation

`src/data.py`. One element of the 24-rotation octahedral group per item, drawn
from `(epoch, index)`.

The usual hard part of augmenting relational data is rewriting the labels: a 90°
turn moves a pair of structures from the signed `superior`/`inferior` rule to the
centre-distance `medial`/`lateral` rule, and neither token carries what the other
needs, so relabelling token-by-token is wrong.

This project does not rewrite anything. It rotates the volume and runs
`select_anchors` again on the rotated geometry — the same function that built the
corpus. The clauses are *re-derived*, so the prompt describes the tensor by
construction rather than by an argument. If a pose admits no feasible anchor set,
it is skipped and the stored pose is used. `tests/test_data.py` checks the
invariant directly: for every augmented item, recompute each direction from the
rotated label volume and compare.

Augmentation is attached to the training split only — validation and test stay in
the stored pose, or the metric stops being comparable across runs — and it is set
per stage:

- **Stage B: on.** The prompt has to follow the volume, and that is exactly what
  the re-derivation guarantees.
- **Stage A: off.** It is label-consistent (the scene and every mask rotate
  together, and a rotated cube is still a cube), but the generator only ever
  emits shapes in their canonical pose, so rotating them shifts Stage A's
  training distribution away from an evaluation set that is not rotated. Turn it
  on deliberately, as an experiment: on the synthetic corpus it is what stops
  Stage A confusing `cube` with `cuboid` and `sphere` with `ellipsoid`, pairs
  that differ only by anisotropy.
