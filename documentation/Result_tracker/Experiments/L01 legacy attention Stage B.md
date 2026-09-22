---
id: "L01"
kind: "legacy"
status: "superseded"
date: "2026-09-20 to 2026-09-21"
run: "runs/phase-b/*"
git: "dbc1e0c, 008da30, 5ec74a9"
corpus: "data/mri"
stage_a: "[[A01 phase-a-current]]"
parent: ""
change: "The attention / query-space Stage B family that preceded the relational carver"
assumption: "(historical) relational attention over a Stage A bottleneck, with occupancy or candidate selection, can segment the described structure"
epochs: "2 to 13 per run"
seeds: 1
metric: "val Dice, 8 supervised classes (as logged at the time)"
score: 0.7058
delta:
delta_on: "not comparable with any relational run: occupancy_mode all leaked the target silhouette, selection copied Stage A candidates, different architecture, possibly different val populations"
benefit: "n/a"
verdict: "Superseded by the relational carver (B03). Recorded here for lineage only"
tags:
  - experiment
  - legacy
  - stage-b
  - mri
---
# L01 legacy attention Stage B

> [!warning] Not comparable
> These runs used a **different Stage B** that was deleted with the move to the relational architecture: `RelationPrompt`, `StructureEncoder`, `Evidence`, `Intersection`, a FiLM decoder, and `SelectionHead` or query heads. Some of them fed the decoder `occupancy_mode: all`, which handed it the target's own outline. Their val curves were logged under older conventions. Only config keys and logged numbers are recorded here, with no interpretation of heads whose code is gone. The design history is in [[SpatialVox#2.2 What this is not]].

| run | git | model keys (from `best.json`) | Stage A | epochs logged | best val Dice (epoch) | other logged metrics |
|---|---|---|---|---|---|---|
| `phase-b/EXP2-seed1` | dbc1e0c | 5 widths, 8³, `intersection_hidden 256`, `image: true`, `selection: true`; stage `occupancy_mode: all`, `mode: predicted` | phase-a/current | 10 | 0.5881 (9) | selection_accuracy 0.898, HD95 9.09 |
| `phase-b/EXP2-16_bottleneck` | 008da30 | 4 widths, 16³, same flags | phase-a/current | 11 | 0.5673 (9) | selection_accuracy 0.920, HD95 10.18 |
| `phase-b/rel-seed1` | 5ec74a9 | embedded 4-width 16³ Stage A, `head: relational` | phase-a/new-model | 13 | **0.7058** (8) | query_top1 0.910, select_top1 0.903 |
| `phase-b/sel-seed1` | 5ec74a9 | `head: selection` | phase-a/new-model | 2 | 0.5066 (1) | select_top1 0.687 |
| `phase-b/qh-seed1` | 5ec74a9 | query head | phase-a/new-model (`--segmenter`) | 8 | 0.2650 (5) | query_top1 0.363 |
| `phase-b/query-w4-seed1` | 5ec74a9 | query head | 4-width 16³ | 2 | 0.0994 (1) | query_top1 0.110 |
| `phase-b/query-seed1` | — | — | — | 0 (empty `metrics.jsonl`) | — | — |
| `phase-b/rel-seed2` | — | — | — | 0 (empty `metrics.jsonl`) | — | — |

All were trained with rotation augmentation (`augment: true`), which the relational Stage B no longer has.
