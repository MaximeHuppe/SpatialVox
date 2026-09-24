# Relational3D

Segment a structure the prompt never names — only locates, by its relations to
three structures that *are* named.

```text
segment the structure that is superior to the Left-Thalamus,
medial to the Right-Putamen, and anterior to the Brain-Stem.
```

No single relation identifies anything; only the conjunction of all three picks
out one region. The target is never an input, and the mask is **painted from the
MRI** rather than chosen from a list of proposals — so the output need not be a
structure the segmenter already knows how to draw.

```text
image   [B, 1, 128, 128, 128]     z-scored over the brain
clauses [B, 3] × {direction, name}
        →  target logits [B, 1, 128, 128, 128]
        →  null logit               one number: the clauses name nothing
        →  centroid [B, 3]          from a heatmap, not from the mask
```

**Names stop at Stage A.** They buy three soft anchor masks and are then gone:
no name, pair or slot embedding exists anywhere downstream. The direction words
are consumed by a parameter-free geometric mapper and by nothing else.

The strategy, the method and the architecture, step by step from an HCP scan to a
reported number, are in [`documentation/SpatialVox.md`](documentation/SpatialVox.md).
That covers every departure from the original proposal, with the measurement that
justified it. The model is drawn on one page in
[`documentation/Flowchart.md`](documentation/Flowchart.md), and every experiment,
its parent and what the change did to the score is in
[`documentation/Result_tracker/`](documentation/Result_tracker/Result_tracker.md).
`documentation/` is an Obsidian vault. `CLAUDE.md` is the short version: the
invariants that make the claim mean something.

## The four pieces

| | what it sees | what it is |
|---|---|---|
| **Stage A** | the image, three anchor names | a promptable segmenter, trained beforehand on every name that may be an anchor, then **frozen** |
| **`PositionalMapper3D`** | detached soft anchor masks, three direction ids | `classify` written as a soft 45° pyramid. No parameters, no image, no names. `where_raw = F₀·F₁·F₂`, never divided by its own maximum |
| **`B(I)`** | the MRI | generic boundary features, pretrained without class ids. Stage A's pyramid is not a substitute — those features were trained to light up *named* structures |
| **carver** | `B(I)`, the three masks, the three fields, `where_raw` and its mass | two 16-channel blocks. It never receives a name, a direction id, or a coordinate grid |

A separate null head reads four scalars — the field's mass and the three anchor
masses — and no pixels, so it cannot decide "empty" by looking at tissue.

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest          # ~3 min
```

Install the torch build that matches the hardware (CUDA, or MPS on Apple
Silicon).

## Run it

```bash
# 1. the corpus: HCP subjects -> scenes, manifests, vocabulary
.venv/bin/python scripts/import_mri.py
.venv/bin/python scripts/corpus_report.py            # the prompt-blind floor a Dice is read against

# 2. the gate: a precondition, not a diagnostic.
.venv/bin/python scripts/gate_mapper.py --segmenter runs/phase-a/current/best.pt

# 3. Stage A, then frozen
.venv/bin/python scripts/train.py a
.venv/bin/python scripts/cache_anchors.py --segmenter runs/phase-a/current/best.pt

# 4. the relational model
.venv/bin/python scripts/train.py b --overfit 1 --set train.stage_b.epochs=120   # wiring test
.venv/bin/python scripts/train.py boundary                                       # optional: pretrain B(I)
.venv/bin/python scripts/train.py b

# 5. the report
.venv/bin/python scripts/evaluate.py runs/stage_b/best.pt --split val --classes val
```

Every tunable lives in `configs/config.yaml`; any leaf can be overridden from
the command line:

```bash
.venv/bin/python scripts/train.py b --set train.stage_b.epochs=5 --set model.stage_b.mapper.tau=1.0
```

## Reading a result

A single Dice is not the result. `scripts/evaluate.py` reports six things, and
the Dice is the third:

1. **Dice, with anchor Dice beside it** — a drop is either a worse outline or a
   Stage A failure, and those are different problems.
2. **Centroid error in millimetres**, from the heatmap rather than the mask.
   Dice fuses "did it point at the right structure" with "did it draw it", and
   those two transfer differently.
3. **The gate on this split** — the fraction of target centroids with
   `where_raw > 0.5`.
4. **Four counterfactuals.** A model that segments "the nearest thing that is
   not an anchor" scores well without reading a word. `permute_channels`,
   `permute_clauses` and `flip_direction` must fall; `permute_both` preserves
   every relation and must not move.
5. **Prompts that name nothing** — the null rate and the false-positive volume.
   This is the check that the tiny spike in `where_raw` was not renormalised
   into a confident answer.
6. **Image replacement** — another subject's MRI into `B`, this subject's
   anchors and fields kept. The centroid should hold and the Dice should fall.
   That pattern is the signature that the words placed the structure and the
   image drew it.

## Layout

```text
configs/config.yaml   every tunable for the MRI corpus, in one file
configs/synthetic*.yaml  the synthetic corpora (synthetic-hard.yaml now points at data/synthetic-mri)
src/config.py         load it, override any leaf from the command line
src/geometry.py       centroids, the direction rule, anchor-first generation
src/vocab.py          structure names, and the prompt language over them
src/mapper.py         PositionalMapper3D - the WHERE, with no parameters
src/mri.py            HCP/FreeSurfer volumes -> this project's corpus
src/synthetic.py      packed primitives with a controllable appearance (the synthetic corpora)
src/data.py           corpus on disk, the datasets, the direction flip
src/models.py         Stage A; the boundary encoder, carver and null head
src/engine.py         the five losses, metrics, one training loop per stage
scripts/              import_mri, generate_data, rebuild_manifests, corpus_report,
                      gate_mapper, cache_anchors, train, evaluate
tests/                the mapper, the model contracts, the losses, the corpus
documentation/        Obsidian vault: SpatialVox (the document), Flowchart (+ .drawio), Result_tracker,
                      Model Info (one note per module)
```

## Conventions

Arrays are indexed `(z, y, x)`; world coordinates are ordered `(x, y, z)` in a
RAS frame (`x` right/lateral, `y` anterior, `z` superior). A label volume stores
`vocabulary index + 1`, with 0 for background. A relation always describes the
target relative to the anchor. `mapper.tau` and every distance are in **world
units** — the corpus is 1.25 mm/voxel, so they are millimetres.
