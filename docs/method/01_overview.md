# 01 — Overview

## The problem

Segment a 3D structure that the prompt never names, describes or points at. All
the prompt gives is where the target sits relative to three structures that *are*
named:

```
segment the structure that is superior to the cube, medial to the torus,
and anterior to the sphere.
```

No single relation identifies anything — "superior to the cube" is satisfied by a
whole half-volume. Only the intersection of all three picks out one region. The
model has to represent each relation separately and then combine them
conjunctively; a model that pools the sentence into one vector before grounding
it cannot express the conjunction at all.

This is a proof of concept for a clinical setting where a radiologist describes a
finding by its relations to known anatomy — "the lesion inferior to the thalamus,
lateral to the ventricle" — rather than by a name the model was trained on.

## Why two networks

| | Stage A | Stage B |
|---|---|---|
| input | intensity volume + structure names | anchor masks + relational prompt + occupancy |
| output | one mask per requested name | one target mask |
| knows the target? | it is just another structure | never — it has to infer it |
| on real MRI | an anatomy segmenter | unchanged |

The split is not decoration. Stage B consumes *masks*, not images, and measures
every geometric feature from the masks it is handed. So the same trained Stage B
runs two ways with no code change:

- **ground-truth anchors** — isolates the relational architecture. This is the
  primary measurement: if it fails here, no amount of segmentation quality helps.
- **predicted anchors** — Stage A segments the three named structures from the
  image and hands them over. The gap between the two is attributable segmentation
  error, and the anchors' own Dice is reported next to the result so you can see
  it rather than guess.

It also means the interesting half of the project survives the move to MRI. Real
anatomy segmenters exist; relational grounding is what has to be built.

## What makes a result believable

A high Dice proves nothing on its own. A model that ignores the prompt entirely
and learns "segment the nearest non-anchor blob" can score well on a corpus where
that heuristic usually works. Two things guard against that:

1. **Held-out target classes.** The structures a split may supervise as targets
   are disjoint across train, validation and test. Every structure stays
   available as an *anchor* everywhere, so the vocabulary is fully seen — but the
   validation and test scores are on targets never once supervised.
2. **Counterfactuals.** Permute the mask channels while holding the prompt fixed;
   permute the prompt while holding the channels fixed; flip one direction to its
   opposite. Dice should collapse. Permute both together, preserving the
   correspondence, and it should not move. A model that scores the same under all
   four is not using the relations. See [07 — Evaluation](07_evaluation.md).

## Conventions

Fixed everywhere, never re-derived locally:

- arrays are indexed `(z, y, x)`, shape `(D, H, W)`;
- world coordinates are ordered `(x, y, z)` in a RAS frame — `x` right/lateral,
  `y` anterior, `z` superior;
- `spacing` is world units per voxel, ordered `(x, y, z)`;
- a label volume stores `vocabulary index + 1`; 0 is background;
- a relation always describes the **target relative to the anchor**.
