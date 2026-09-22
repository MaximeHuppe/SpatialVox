# Relational architecture

The model segments a structure the prompt never names. Its only inputs are the
MRI volume and three clauses. Each clause is a direction and the name of an
anchor. The target may be a structure that has never had a mask — a lesion
between known anchors. The mask is painted from the image. It is not chosen
from a list of proposals.

```
image   [B, 1, 128, 128, 128]     z-scored over brain
clauses [B, 3] × {direction, name}
        →  target logits [B, 1, 128, 128, 128]
        →  null logit              one number: the clauses name nothing
```

Clause `i` and anchor channel `i` are the same slot. The order is chosen once
when the example is built and then used everywhere.

Names are used only to obtain the three anchor masks. After that, localization
and carving are name-free. The anchor’s identity downstream is its mask and
its location.

Visual: [`relational_architecture.drawio`](relational_architecture.drawio).
The entity-selector blueprint is a separate baseline, not this pipeline:
[`blueprint_confrontation.md`](blueprint_confrontation.md).

---

## What is computed, and what is not

| step | sees | does not see |
|---|---|---|
| Stage A | image, the three anchor names | the target name, the directions |
| `PositionalMapper3D` | detached soft anchor masks, the three direction ids | the image, every name |
| null head | `where_mass`, the three anchor masses | the image, names, direction embeddings |
| boundary encoder `B` | the MRI | prompt, names, directions, coordinates, labels |
| carver | `B(I)`, the three soft masks, the three fields, `where_raw`, `log(where_raw)`, `where_mass` | names, direction ids, tokens, coordinates, Stage A features |

There is no name embedding, pair embedding, or slot embedding after Stage A.
The direction is already the shape of `F_i`. A token would let the triple of
anchor names stand in for the target.

Stage A’s feature pyramid stays out. Those features were trained to light up
named structures. `B` is a separate encoder, trained without class ids, so a
structure Stage A has never seen is still a boundary in the image.

---

## 1. Anchors

The three names are read off the clauses and passed to Stage A. Channel `i`
is the structure clause `i` names. Stage A is trained beforehand on every name
that may be an anchor, then frozen. The relational loss does not flow back
into it.

The mask handed downstream is the detached probability, not a hard threshold:

```
A_i = stop_gradient(sigmoid(anchor_logits_i))
mass_i = mean(A_i)
c_i = sum(A_i · x) / (sum(A_i) + ε)          # confidence-weighted centroid
```

`mapper.min_mass` (starting value `1e-3` of the volume) rejects an anchor. A
rejected channel writes `F_i = 0`, the product is zero, and the null head is
what declares the prompt empty. A threshold at 0.5 makes the centroid jump and
can delete a dim but real anchor in one step. Because the mask is detached,
using the soft values does not train Stage A through the mapper.

Stage A is queried for those three names only. No other mask is offered to
the carver.

---

## 2. PositionalMapper3D — the WHERE

The mapper is the predicate in `classify`, written as a soft field. It has no
parameters. It does not see the image. A clause is true when one axis of the
centroid offset dominates, inside a square pyramid of 45°. There is no axial
fade: a direction is a relation, not a distance. There is no learned gain.

### One clause

Centroid `c` from the soft mask, in world `(x, y, z)`. Volumes stay indexed
`(z, y, x)`. `center` is `volume_center_world`. For a voxel at world point
`p`, `d = p − c`.

| direction | margin |
|---|---|
| superior | `d_z − max(\|d_x\|, \|d_y\|)` |
| inferior | `−d_z − max(\|d_x\|, \|d_y\|)` |
| anterior | `d_y − max(\|d_x\|, \|d_z\|)` |
| posterior | `−d_y − max(\|d_x\|, \|d_z\|)` |
| lateral | `min(\|d_x\| − max(\|d_y\|, \|d_z\|), \|p_x − center_x\| − \|c_x − center_x\|)` |
| medial | `min(\|d_x\| − max(\|d_y\|, \|d_z\|), \|c_x − center_x\| − \|p_x − center_x\|)` |

```
F_i(p) = sigmoid(margin_i(p) / τ)     if mass_i ≥ min_mass, else 0
```

`τ` is `mapper.tau` in world units (starting value 2 mm). It is the only
geometric constant. Lateral and medial use the midline, as `classify` does.
The direction id is consumed here. It is not an input to the carver.

### The product is not renormalized

```
where_raw  = F_0 · F_1 · F_2                         # [B, 1, 128³]
where_mass = mean(where_raw)                        # [B, 1]
```

`where_raw` is the spatial map. Dividing by its own maximum is not done. A
sigmoid is never exactly zero, so an impossible prompt still has a tiny peak,
and dividing by that peak would turn it into 1. The null head is what reads
`where_mass`. A meaningless spike stays small.

Each `F_i` is kept. The carver sees the three fields and the product. A branch
does not re-weight another clause, because there is no learned fusion of the
clauses at all.

The pyramid describes where a **centroid** would satisfy the clause. Part of
the body lies outside it. `where_raw` is a channel and a weak bias. It is not
a crop, and it is not the initial segmentation.

### Manifest check, before the carver is trained

On the training manifests, with ground-truth anchors used only as a check: the
fraction of examples whose target centroid has `where_raw > 0.5`. The
threshold is absolute. Flipping one direction must move that centroid off the
high region. A low fraction means the mapper disagrees with the prompts, and
the carver is not trained on top of it.

---

## 3. Null head

A small MLP, with no image and no names:

```
valid = MLP(where_mass, mass_0, mass_1, mass_2)
```

It is trained to say whether the clauses name exactly one structure. On an
impossible prompt `where_mass` is tiny and the head is trained to say invalid.
At inference, invalid empties the mask. Ambiguous prompts (more than one
structure) are dropped in the data, not given a third label.

The head cannot see the MRI, so it cannot decide “empty” by looking at tissue.
Feasible-field mass is the reason an impossible prompt loses.

---

## 4. Boundary encoder and carver — the WHAT

### `B`, image only

```
B(I) → boundary features          # no prompt, name, direction, coordinate, or label
```

Two or three stages, 16–32 channels, small beside Stage A. It is pretrained,
then given a lower learning rate while the carver trains. The pretraining
objectives are class-agnostic:

- masked-volume reconstruction of `I`;
- local intensity agreement and an edge consistency term;
- a binary boundary map taken from the label volume: a voxel is positive where
  two neighbouring voxels differ in label. No class channel, no target
  indicator. This map is a pretraining target, not an inference input;
- optionally, a contrastive term that pulls voxels of one connected region
  together and separates them across a boundary, again with no class ids.

Stage A features are not a substitute. They carry named-structure semantics.

### Carver

```
x        = concat(B(I), A_0, A_1, A_2, F_0, F_1, F_2,
                  where_raw, log(where_raw + ε), broadcast(where_mass))
residual = two 16-channel blocks, stride-2 stem, then 1×1 up to 128³
logits   = residual
logits   = background where A_i > 0.5
heatmap  = a separate 1×1, soft-argmax → centroid
```

`log(where_raw)` is clamped. `where_mass` is one number, repeated across
space, so the carver can see that the conjunction has no mass without
renormalizing the map.

The additive form is an ablation, not the default:

```
logits = residual + α · logit(clamp(where_raw))
```

`α` is a single scalar, initialised at 0.35, and it has no other input. It
cannot depend on the MRI, a class, or a name. A residual added to the full
`logit(where)` makes the initial mask equal to the conjunction, and a
16-channel block then has to undo a large negative outside the pyramid, on the
side where the body actually continues. The default gives the carver the maps
and lets Dice move the boundary. The ablation measures whether a weak explicit
bias helps.

Anchor voxels above 0.5 are set to the background after the residual. The
exclusion uses the predicted soft mask, not the label volume, and it is not
dilated.

The heatmap is not trained through the mask loss. When the null head says the
prompt is empty, the mask target is empty and the heatmap is still trained on
the peak of `where_raw`, if that peak exists. Soft-argmax is the expectation
under a softmax over voxels.

### What this can resolve

An unseen structure is segmented where `B(I)` carries a boundary. A lesion
that contrasts with its surroundings, inside the conjunction of three known
anchors, is the case. Two nuclei that share an intensity are not, and the
carver must not cover that with a memorised training shape. Centroid right and
Dice wrong is that failure. It is not a reason to attach a proposal list.

---

## 5. Losses

| loss | prompt names one structure | prompt names none | prompt names two |
|---|---|---|---|
| Dice + BCE | that structure’s mask | the empty mask | example dropped |
| null BCE | valid | invalid | example dropped |
| heatmap vs structure centroid | that centroid | — | example dropped |
| heatmap vs the field | peak of `where_raw` | peak of `where_raw` | example dropped |
| `L_far` | mass outside `dilate(where_raw > ε)` | — | example dropped |

`L_far` is the only spatial penalty on a valid prompt. A coverage term on the
whole exterior fights the body that extends past the pyramid. Dice is allowed
to follow an MRI boundary a few voxels outside the intersection. `ε` and the
dilation radius live in config (starting point: `ε = 0.05`, radius 8 voxels).

On an impossible prompt the empty-mask loss is the penalty everywhere. `L_far`
is not added on top.

One direction is replaced by its opposite with probability
`train.flip_probability` (start at 0.25), using `OPPOSITE` in
`src/geometry.py`. The flip changes `F_i` and nothing else, because there is
no token left to update. The new clauses are scored with the same rule as the
corpus:

- exactly one structure → that mask, that centroid, null target valid;
- none → empty mask, null target invalid, heatmap still on the field;
- more than one → dropped.

A flip is not assumed to be empty. Same image, same anchors, a different
field. There is no deep supervision.

---

## 6. What is withheld

Not inputs, in any mode:

- the label volume, the target mask, the target name, the target centroid, at
  inference and inside the carver;
- an occupancy built from labels;
- any mask other than the three detached anchor probabilities;
- Stage A features;
- a name, pair, or slot embedding after Stage A;
- a direction vector outside the mapper;
- a coordinate grid in `B` or the carver.

The unsigned boundary map is a pretraining target for `B` only.

The 8/4 split is a test on this corpus. Caudate and putamen are held out as
relational targets and are still anchors, so Stage A has seen them. It is not
the lesion claim. The lesion claim is the carver: the output need not be a
structure Stage A knows how to draw.

---

## 7. Experiment

Eight structures are supervised as the relational target. Four are never a
relational target. All twelve may be anchors.

| role | structures |
|---|---|
| supervised as target (8) | `targets.train`: L/R thalamus, pallidum, amygdala, accumbens |
| held out as target (4) | `targets.val`: L/R caudate, putamen. Scored. Not used to save `best.pt` |
| anchors | all 12, every split |
| further held-out pair | `targets.test`: L/R hippocampus. Scored, not used to choose a checkpoint |
| landmarks, never targets | ventricles, brainstem, ventral DC. May be anchors |

Checkpoints are chosen on subjects whose relational targets are among the
eight. Today `Loop` saves `best.pt` on validation Dice, and `targets.val` is
the four. That has to change before a number from this model is quoted.

A Dice on the four is reported with the prompt-blind baseline recomputed on
this corpus, and with `permute_both`. `permute_channels`, `permute_clauses`
and `flip_direction` must drop. Three seeds minimum.

| metric | reading |
|---|---|
| fraction of target centroids with `where_raw > 0.5` | the mapper agrees with the prompts |
| centroid error, mm, on the 4 | the field located the structure |
| Dice, and anchor Dice beside it | outline, and whether Stage A failed |
| null rate and false-positive volume on prompts that name nothing | the spike was not renormalized into a mask |
| alternate-prompt switch rate | a prompt that names a different structure moves the mask |

### Two tests that are part of the result

**Prompt-only carver.** `B(I)` is removed. Anchors and `where_raw` stay. If its
Dice approaches the full model, the carver is redrawing a shape from the
spatial prior and is not using the MRI.

**Image replacement.** Anchors and fields stay those of the subject. The volume
fed to `B` is a different subject’s MRI, or the same MRI strongly corrupted.
The centroid should hold. Boundary Dice should fall. That is the signature
that the words placed the structure and the image drew it.

---

## 8. Decision recorded against the current contract

`CLAUDE.md` keeps the image out of the relational encoder so a query cannot
pick “the blob that is not an anchor”. Here the image enters only through
`B`, which never sees the prompt, and through the carver, which sees `B(I)`
beside a field computed from detached masks and direction ids. The null head
does not see the image. There is no `occupancy_mode`. `model.stage_b_selection`
is off: a candidate list is the other method.

`StageB.config` must contain `mapper.tau`, `mapper.min_mass`, `α` if the
ablation is on, the width of `B`, `flip_probability`, and the loss weights.
Checkpoints of the attention-based Stage B do not load here.

---

## 9. Plan

### Step 1 — the pyramid on soft masks

`src/mapper.py`. Soft masks `[B, 3, D, H, W]`, direction ids, spacing, volume
centre, `min_mass`. Returns `F`, `where_raw`, `where_mass`, and the three
masses. No parameters. No division by the maximum.

Tests: each of the six regions; product peak nearer the satisfying centroid
than either anchor; a flip moves it; mass below `min_mass` yields `F_i = 0`
and `where_raw = 0`; an impossible product’s maximum stays far below 1; a
voxel of the target on the near side of the pyramid may score low.

Then the manifest fraction at `where_raw > 0.5`. The carver waits on it.

### Step 2 — `B`, the null head, the carver

`StageB.forward(image, direction_ids, name_ids)` returns `logits`, `where_raw`,
`centroid`, and `valid`. Inside: frozen Stage A, detached sigmoids, the
mapper, `B(image)`, the carver, the null MLP. No embedding table in that
module. No coordinate grid. No Stage A pyramid.

Tests: with the carver at random init, an impossible prompt does not produce
a peak of 1 in `where_raw`; the null head’s input tensor does not include the
image; replacing the image changes `B` and not `where_raw`; the label volume
and the target name are not arguments.

### Step 3 — losses

Dice + BCE, null BCE, both heatmap terms, `L_far` on valid prompts only.
Tests for the three flip outcomes.

### Step 4 — the two MRI tests, then the report

Prompt-only carver and image replacement, on the four held-out targets, before
a Dice is treated as evidence that the image was used.

---

## Fixed by this document

- Names stop at Stage A. Downstream: soft detached masks, `F_i`, and `where_raw`.
- `where_raw` is the product. It is not divided by its maximum. `where_mass`
  feeds a null head that does not see the image.
- The carver reads `B(I)` and the geometric maps. `logit(where)` is not added.
  The scalar-`α` form is an ablation.
- `L_far` on valid prompts penalises mass far outside the field. Dice may
  follow the MRI past the intersection.
- A prompt-only carver and an image-replacement test are mandatory.
- Entity selection is a baseline. It is not in this forward.
