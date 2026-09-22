---
id: "D01"
kind: "diagnostic"
status: "adopted"
date: "2026-09-21"
run: "scripts/corpus_report.py"
git: "2978078"
corpus: "data/mri"
parent: ""
change: "Prompt generation target-first (anchors = the 3 structures nearest the target) to anchor-first (fix a triple, keep the targets its conjunction names uniquely), with clause order shuffled"
assumption: "On fixed anatomy the anchor identities name the target unless the triple is chosen before the target"
epochs: "n/a"
seeds:
metric: "anchor-set shortcut ceiling, all target classes (val)"
score: 0.423
floor: 0.2017
delta: -0.566
delta_on: "anchor-set ceiling 98.9% to 42.3%; prompt-blind floor 0.775–1.000 to 0.2017"
benefit: "yes"
verdict: "Adopted, and target-first deleted: ignoring the prompt had strictly beaten reading it (98.9% vs 94.5%). Shuffling slots also closed a slot-order leak worth about 0.11 Dice"
tags:
  - experiment
  - diagnostic
  - corpus
  - mri
---
# D01 anchor-first prompt generation

> [!abstract] Verdict
> **Adopted.** The anchor-set ceiling fell from **98.9%** to **42.3%** and the prompt-blind floor from **0.775–1.000** to **0.2017**. Target-first generation is deleted, not configurable.

## The measurement
With target-first generation the unordered anchor set alone recovered the target **98.9%** of the time, while solving the conjunction was right **94.5%** of the time, so ignoring the prompt strictly beat reading it. No pool width inverts that, because the selection rule itself was the leak. Anchor-first fixes the triple first and keeps a target only if the conjunction is **unique**, which it now is in 100% of manifest prompts.

## Per population (val, anchor-first, `scripts/corpus_report.py --split val --scenes 600`)
| population | prompt-blind Dice | anchor-set ceiling |
|---|---|---|
| all target classes | 0.2017 | 42.3% |
| `targets.train` (8) | 0.3067 | 67.8% |
| `targets.val` (4) | 0.1097 | 83.9% |
| `targets.test` (2) | 0.0630 | 99.2% |

## Also closed
- **Slot order.** A stored distance ranking made the slot index a proxy for proximity. `shuffle_clauses: true` removed about 0.11 Dice of shortcut.
- The floors apply per population: filtering to fewer classes inflates the ceiling.
