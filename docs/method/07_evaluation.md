# 07 — Evaluation

`scripts/evaluate.py`, `src/engine.py: Metrics`

```bash
scripts/evaluate.py runs/stage_b/best.pt
scripts/evaluate.py runs/stage_b/best.pt --segmenter runs/stage_a/best.pt
scripts/evaluate.py runs/stage_b/best.pt --split val --save-masks
```

Writes `report.json`, `predictions.jsonl`, and with `--save-masks` every
predicted mask beside the target it was scored against.

## Metrics

```yaml
evaluation:
  metrics: [dice, iou, hausdorff]
  hausdorff_percentile: 95
  stratify_by: [target, anchor, direction, slot]
  counterfactuals: [permute_channels, permute_clauses, flip_direction, permute_both]
```

Dice and IoU are always computed; listing `hausdorff` turns on the one metric
that costs a surface extraction per sample, at `hausdorff_percentile`. Results
are reported overall and stratified by whatever `stratify_by` lists:

| stratum | what it answers |
|---|---|
| `target` | does it work on the held-out target classes, or only the trained ones? |
| `anchor` | is some structure a bad anchor — too small, too symmetric? |
| `direction` | is `medial`/`lateral` weaker than `superior`/`inferior`? |
| `slot` | does clause position matter more than it should? |

`Metrics` keeps one row per scored channel rather than folding into running
means, so a single pass supports any of these breakdowns without deciding in
advance — `stratify_by` selects from what was recorded rather than changing what
is measured.

Hausdorff is measured between mask *surfaces* — a few hundred voxels rather than
a few hundred thousand — so the exact pairwise distance is affordable. Two empty
masks match perfectly; one empty mask is undefined and returned as `nan` so it is
counted rather than silently averaged in. Dice and IoU use the same convention:
two empty masks score 1.0, one empty against one non-empty scores 0.0.

With `--segmenter`, the predicted anchors' own Dice against the ground-truth
masks is reported beside the result. Without it you can see that a run got worse;
with it you can see whether the relational model or the segmenter is responsible.

## The counterfactuals are the actual result

A high Dice proves nothing on its own. A model that ignores the prompt and learns
"segment the nearest non-anchor structure" scores well on any corpus where that
heuristic usually holds — and on synthetic scenes it often does. The probes
separate the two hypotheses by perturbing one correspondence at a time and
leaving the volumes untouched.

| probe | what moves | expected |
|---|---|---|
| `permute_channels` | the mask channels; the prompt stays put | **large drop** |
| `permute_clauses` | the prompt; the mask channels stay put | **large drop** |
| `flip_direction` | one clause asks for the opposite side | **large drop** |
| `permute_both` | both, together — every relation preserved | **no change** |

The fourth is the control, and it is the one that makes the other three
meaningful. (`evaluation.counterfactuals` selects which probes run; dropping the
control is possible and unwise.) Any perturbation degrades a fragile model; only a model that is
really reading the correspondence collapses on the first three *and* holds on the
fourth.

Making this possible is the reason `anchors` (which mask sits in which channel)
and `name_ids` (which structure each clause names) are separate inputs. In a
normal batch they are locked together — `name_ids` defaults to `anchors - 1` — and
a probe is precisely the act of unlocking them.

Read the numbers this way:

- **all four move by about the same amount** — the model is not using the prompt.
  It has learned a geometric heuristic. This is the common failure, and a high
  aggregate Dice alongside it is not evidence of anything.
- **the first three drop, the fourth holds** — the model is using
  direction–anchor correspondence.
- **`permute_both` drops too** — the model is slot-dependent: it has attached
  meaning to clause position rather than to clause content. Suspect the slot
  embeddings, or too little data for the shared evidence branch to generalise
  across slots.
- **`flip_direction` does not move but `permute_clauses` does** — the model reads
  *which* anchor a clause names but not *what* the relation to it is. Direction
  embeddings are not being used.

`tests/test_evaluate.py` checks each probe changes exactly what it claims to, and
that all four reach the model. The probes themselves cannot be silently broken.

## Reading a result honestly

The headline number is Dice **on the test split**, whose target class was never
supervised — not the aggregate over everything. Report it next to:

1. the counterfactual table;
2. the oracle-versus-predicted pair, with the anchors' own Dice;
3. the per-target stratum, so a single easy class cannot carry the average.
