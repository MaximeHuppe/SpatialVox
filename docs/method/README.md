# Method

How the model works, and why each piece is there. Read in order the first time;
each page names the file it describes, so the code is never more than one hop
away.

| | | |
|---|---|---|
| [01 — Overview](01_overview.md) | the problem, and the two-stage answer | — |
| [02 — Data and relations](02_data.md) | scenes, the direction rule, anchor selection, augmentation | `src/geometry.py`, `src/synthetic.py`, `src/data.py` |
| [03 — Prompt encoder](03_prompt_encoder.md) | how a sentence becomes three tokens | `src/vocab.py`, `src/models.py` |
| [04 — Stage A](04_stage_a.md) | the promptable structure segmenter | `src/models.py` |
| [05 — Stage B](05_stage_b.md) | the relational target segmenter | `src/models.py` |
| [06 — Training](06_training.md) | losses, schedule, the four phases | `src/engine.py` |
| [07 — Evaluation](07_evaluation.md) | metrics, and the counterfactuals that matter more | `scripts/evaluate.py` |
| [08 — Scaling](08_scaling.md) | real MRI, real structure names, 128³ | — |

## The whole project on one page

A prompt names three known structures and says where the target lies relative to
each of them. The target itself is never named, never outlined, never pointed at:

```
segment the structure that is superior to the cube, medial to the torus,
and anterior to the sphere.
```

Two networks:

- **Stage A** turns an image into masks for named structures. It is the part
  that would be an anatomy segmenter on real MRI.
- **Stage B** takes three of those masks, in the order the prompt names them,
  plus the prompt, plus a binary map of where *any* structure is, and outputs
  the target. It never sees the label volume, so it has to find the one region
  that satisfies all three relations at once.

```
        image ──────────────► Stage A ──► masks for the three named anchors
                                              │
   prompt ──► three clause tokens ────────────┤
                                              ▼
   labels > 0 (anonymous occupancy) ──────► Stage B ──► target mask
```

Splitting them is what makes the claim testable. Train Stage B with ground-truth
anchors and you measure the relational architecture alone; swap in Stage A's
predictions, change nothing else, and the difference is segmentation error. And
because the supervised target classes are held out per split, a good score on
validation and test is a score on structures never once supervised as a target.

## Layout

```
configs/config.yaml   every tunable, in one file
src/config.py         load it, override any leaf from the command line
src/geometry.py       centroids, the direction rule, anchor selection
src/vocab.py          structure names, and the prompt language over them
src/synthetic.py      the synthetic corpus (swap out for real MRI)
src/data.py           corpus on disk, datasets, rotation augmentation
src/models.py         shared blocks, Stage A, Stage B
src/engine.py         losses, metrics, one training loop for both stages
scripts/              generate_data.py, train.py, evaluate.py
```
