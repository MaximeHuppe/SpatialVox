---
id: 
kind: 
status: 
date: 
run: 
git: 
corpus: 
stage_a: 
parent: 
change: 
assumption: 
epochs: 
seeds: 1
metric: 
score: 
floor: 
heldout_val: 
heldout_test: 
delta: 
delta_on: 
benefit: 
verdict: 
tags:
  - experiment
---
# ID run-name

> [!abstract] Verdict
> One sentence, written after the run: what changed, what moved, and whether it moved more than the noise.

## Question
What this run is meant to decide, written **before** it starts. State the assumption as a prediction that can fail, e.g. "if X is the cause, the held-out empty rate should fall below Y".

## Setup
- **Command:** `scripts/train.py b --config … --set …`. Use `True`/`False`: `--set x=false` is the string `"false"`, which is truthy.
- **The one change against the parent:** …
- **Check that it applied:** open `runs/<name>/best.json` and read the changed key in `model` or `meta.config.stage`.
- **Corpus / Stage A / epochs / seed / git:** …

## Result
| epoch | supervised | floor | held-out val | held-out test | centroid | `flip_direction` drop | `permute_both` drop |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

## Against the parent
Give Δ only when the parent shares the corpus **and** the population, at matched epochs. Put it beside the replicate noise (see [[Result_tracker#Noise]]). Otherwise write "not comparable" and say why.

## Reading
What the numbers mean and what they do not, and what to run next.

<!--
Fields
  id        A = Stage A, B = Stage B, P = pretraining, D = diagnostic or decision, L = legacy. Zero-pad: B12.
  kind      stage-a | stage-b | pretrain | diagnostic | legacy
  status    done | running | stopped | killed | failed | invalid | adopted | superseded
  parent    "`B09 mri-stage-b` (archived)", the run this one is compared to. Empty for a root.
  stage_a   "[[A09 mri-stage-a]]", the frozen Stage A this run uses (Stage B only).
  score     the `metric` value, a bare number, at the epoch named in delta_on or in the note.
  floor     the prompt-blind floor of that population (Stage B only).
  delta     a bare number, only when comparable. Otherwise leave it empty and explain in delta_on.
  benefit   yes | no | mixed | not yet | n/a | invalid
-->
