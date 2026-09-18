# 08 — Scaling: real MRI, real structure names, 128³

The three changes this project was rebuilt to make cheap. Each is one edit.

## 128³ instead of 64³

```bash
scripts/generate_data.py --set data.resolution=128
scripts/train.py a
```

That is all. Nothing in `src/models.py` mentions 64.

- `depth = log2(resolution / model.bottleneck)`, so 64³ gets three stride-2
  stages and 128³ gets four. The **bottleneck stays at 8³ = 512 tokens**, so the
  cross-attention budget is identical at either resolution — which is the whole
  reason the depth is derived rather than fixed.
- Widths double from `base_channels` and stop at `max_channels`, so the extra
  stage does not force a 512-wide bottleneck. Lower `max_channels` if memory
  becomes the constraint.
- World coordinates are recomputed per scale from the world frame, so a position
  means the same thing at every resolution.
- Synthetic structure sizes are written for a 64-voxel axis and rescale linearly,
  so a 128³ corpus holds the same structures at twice the sampling rather than
  structures twice as large.

What does change is cost: eight times the voxels. Expect to lower
`train.batch_size` and raise `train.accum` to keep the effective batch.
`model.prior_foreground` does not need changing — the fraction of a volume one
structure occupies is scale-invariant.

Any power-of-two multiple of the bottleneck works; anything else is rejected with
a clear error rather than silently rounded.

## Real structure names

The vocabulary is data, not code. `<root>/vocab.json` is an ordered list of
names, and every table that depends on it — the name embedding shared by both
stages, the `(direction, name)` pair table, the Stage A mask head — is sized from
its length. Ten primitives and eighty anatomical labels are the same code path.

The prompt parser is built from the vocabulary at call time, so
`"segment the structure that is superior to the Brain-Stem, lateral to the
Right-Thalamus, and anterior to the Left-Putamen."` parses and round-trips with
no change. The only constraint on a name is that it must not contain `", "`,
which separates clauses; `Vocabulary` rejects such a name at construction.

Two things are worth checking on a large anatomical vocabulary:

- `train.stage_a.prompts_per_item` — sampling a subset of names per training item
  keeps Stage A affordable when the vocabulary is large. It is safe because a
  Stage A logit for one name does not depend on which other names were requested
  ([04](04_stage_a.md)); evaluation still prompts every name.
- structures that are absent from a subject are handled: a mask is `labels == id`,
  which is simply empty, and the metric convention scores empty-against-empty as
  a match, so "not present" is a learnable answer.

## Real MRI data

Everything downstream of the corpus is written against an `(image, labels)` pair,
so `src/synthetic.py` is the only file real data replaces. The entry point is
`src.data.import_corpus`:

```python
from src.data import import_corpus, load_nifti

scenes = {
    subject: (load_nifti(root / subject / "t1.nii.gz"),
              load_nifti(root / subject / "aseg.nii.gz", dtype="int32"))
    for subject in subjects
}

import_corpus(
    "data/mri",
    scenes,
    label_names={17: "Left-Hippocampus", 53: "Right-Hippocampus",
                 16: "Brain-Stem", 10: "Left-Thalamus", ...},
    splits={"train": train_ids, "val": val_ids, "test": test_ids},
    spacing=(1.0, 1.0, 1.0),
    targets={"train": [...], "val": [...], "test": [...]},
)
```

It remaps the source label ids (FreeSurfer's 17, 53, 16, … need not be
contiguous) to `vocabulary index + 1`, writes the volumes as RAS NIfTI, computes
every scene's relations, and writes the manifests. Ids not listed in
`label_names` are dropped. Then:

```bash
scripts/train.py a --set data.root=data/mri
scripts/train.py b --set data.root=data/mri
```

### What to check before trusting the result

**Orientation.** The direction rule assumes RAS: `x` right/lateral, `y` anterior,
`z` superior, arrays `(z, y, x)`. Reorient to RAS on the way in (`nibabel`'s
`as_closest_canonical`) or every relation will be confidently wrong in a way no
metric reveals.

**Spacing.** Pass the real voxel spacing. The direction rule works in world
units, so anisotropic voxels change which axis dominates a relation — 3 voxels of
`y` beats 2 of `z` until `z` voxels are twice as long. The network itself sees
only normalised coordinates, in which spacing cancels; spacing matters for the
relations and for Hausdorff distances in millimetres.

**The mid-sagittal plane.** `medial`/`lateral` is defined against the volume's
`x` centre. That is right for a registered, centred brain and wrong for an
arbitrarily cropped field of view. If volumes are not centred, pass the anatomical
midline as the `center` argument to `select_anchors` rather than the volume
centre.

**Intensity.** Set `data.normalize: zscore`. The default is `none`, because the
synthetic generator emits a bounded [0, 1] image and that is what the published
runs were trained on; real MRI has no such guarantee. Swap in percentile clipping
or a bias-field correction in `src.data.normalize` if a modality needs it — it is
one branch in one function.

**Registration is not required.** Stage B consumes masks and measures its own
geometry from them, so it does not care whether subjects are in a common space.
Stage A does benefit from consistent orientation, which reorienting to RAS
already provides.

**Anchor feasibility.** A target with no set of `n_anchors` pairwise-distinct
directions is dropped. On anatomy — where many structures are bilateral and
roughly coplanar — expect a higher drop rate than the ~2% seen on synthetic
scenes. `generate_data.py` prints the retained count; if it is low, lower
`data.n_anchors` to 2 or widen the direction vocabulary.

## Things that were left out

Removed deliberately in the interest of a small codebase; each is easy to add
back if a result demands it.

- **Input ablations** (anchor-masks-only, prompt-only, union-mask). The
  counterfactual battery in [07](07_evaluation.md) tests the same claim per
  example rather than per training run, and does not cost three extra trainings.
- **Grid escalation** when packing fails. `generate_scene` retries with a fresh
  seed instead, which is simpler and keeps every scene the same shape — a
  precondition for batching.
- **Non-cubic volumes.** The datasets and the decoder handle them; `depth_for`
  takes `min(shape)`, so a strongly anisotropic field of view would waste stages.
  Crop or pad to a cube, or make the depth per-axis.
