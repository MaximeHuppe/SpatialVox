# Relational3D

Segment a 3D structure that the prompt never names — only locates, by its
relations to three structures that *are* named.

```text
segment the structure that is superior to the cube, medial to the torus,
and anterior to the sphere.
```

No single relation identifies anything; only the intersection of all three picks
out one region. The target is never an input.

Two networks. **Stage A** segments named structures from an intensity volume — on
real MRI this is an anatomy segmenter. **Stage B** takes three of those masks in
the order the prompt names them, plus the prompt, plus a binary map of where
*any* structure is, and outputs the target. Because Stage B consumes masks and
measures its own geometry from them, the same trained model runs on ground-truth
anchors (which isolates the relational architecture) or on Stage A's predictions
(the end-to-end setting), and the difference is attributable segmentation error.

**[docs/method/](docs/method/) explains how and why.**

The networks are the ones from `exp/realistic-appearance`, parameter for
parameter — `tests/test_reference_parity.py` ports a checkpoint from that branch
into these classes and checks every output tensor is bit-identical, so results
stay comparable across the rewrite.

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest          # ~15 s
```

Install the torch build that matches the hardware (CUDA, or MPS on Apple
Silicon).

## Run it

```bash
.venv/bin/python scripts/generate_data.py --smoke   # 12 scenes, ~20 s
.venv/bin/python scripts/generate_data.py           # 500 scenes

.venv/bin/python scripts/train.py a                 # Stage A
.venv/bin/python scripts/train.py b --overfit 1 --set train.stage_b.epochs=200 --set train.stage_b.mode=oracle
.venv/bin/python scripts/train.py b                 # Stage B, predicted anchors (needs Stage A)
.venv/bin/python scripts/train.py b --set train.stage_b.mode=oracle  # Stage B, ground-truth anchors

.venv/bin/python scripts/evaluate.py runs/stage_b/best.pt
.venv/bin/python scripts/evaluate.py runs/stage_b/best.pt --set train.stage_b.mode=oracle
```

Every tunable lives in `configs/config.yaml`; any leaf can be overridden from the
command line:

```bash
.venv/bin/python scripts/train.py b --set train.stage_b.epochs=5 --set model.base_channels=8
```

`notebooks/demo.py` walks one example end to end — load a volume, generate its
prompt, run both stages, score it, and look at the result in an interactive 3D
view. It is a script and a notebook at once (`# %%` cells); `pip install plotly`
for the figure.

`evaluate.py` reports Dice / IoU / Hausdorff — overall and stratified by target,
anchor, direction and clause slot — and then four counterfactual probes. Those
matter more than the Dice: a model that ignores the prompt and segments "the
nearest non-anchor structure" can score well, and only the probes tell the two
apart. See [docs/method/07_evaluation.md](docs/method/07_evaluation.md).

## Layout

```text
configs/config.yaml   every tunable, in one file
src/config.py         load it, override any leaf from the command line
src/geometry.py       centroids, the direction rule, anchor selection
src/vocab.py          structure names, and the prompt language over them
src/synthetic.py      the synthetic corpus — the only file real MRI replaces
src/data.py           corpus on disk, datasets, rotation augmentation
src/models.py         shared blocks, Stage A, Stage B
src/engine.py         losses, metrics, one training loop for both stages
scripts/              generate_data.py, train.py, evaluate.py
tests/                geometry, data, model contracts, training, counterfactuals,
                      config, parity with exp/realistic-appearance
notebooks/            one example end to end, with a 3D view
docs/method/          how the model works, and why
```

## Scaling

The project is built so that the three obvious next steps are one edit each — see
[docs/method/08_scaling.md](docs/method/08_scaling.md).

| | how |
|---|---|
| 128³ instead of 64³ | `--set data.resolution=128`; the depth follows, the 512-token bottleneck does not move |
| real structure names | `vocab.json` is an ordered list; every table is sized from it, and the prompt parser is built from it |
| real MRI | `src.data.import_corpus(...)` — remaps source label ids, writes the corpus, and nothing downstream changes |

## Conventions

Arrays are indexed `(z, y, x)`; world coordinates are ordered `(x, y, z)` in a
RAS frame (`x` right/lateral, `y` anterior, `z` superior). A label volume stores
`vocabulary index + 1`, with 0 for background. A relation always describes the
target relative to the anchor.
