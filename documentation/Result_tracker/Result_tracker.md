---
tags:
  - spatialvox
  - result-tracker
---
# Result tracker

Every experiment gets one note in `Result_tracker/Experiments/`. Its properties record what was run and what was assumed, and above all its **`parent`** (the experiment it is compared against) and its **`delta`** (the change in score against that parent, on the metric named in `delta_on`). **`benefit`** records whether the change helped. The method and architecture are documented in [[SpatialVox]], and the model is drawn in [[Flowchart]].

> [!warning] Needs attention (2026-09-23 00:10)
> - [[B12 mask-valid-seed1]] **finished** (30 epochs), with `mask_on: valid`. Held-out empty rate 0%, and held-out Dice 0.727 / 0.774 at `best.pt` against B09's 0.368 / 0.460 (+0.15 / +0.18 at matched epochs). One seed: still to do are `scripts/evaluate.py` on its `best.pt` (per-class Dice, image replacement, impossible-prompt leak with and without the null gate) and a second seed.
> - [[B10 arm-loo]] and [[B11 arm-noanchor]] were **killed at 17 of 30 epochs** on 2026-09-22, and their run folders were later deleted by another session's relaunch. Their notes still show epoch-10 snapshots. Leave-one-out showed no effect beyond the replicate noise. B11 was invalid (the flag never applied) and serves as B09's replicate. `carver_sees_anchors: False` has still never been run.
> - [[B06 field-empty-only-seed1]] **produced no epoch**. Still to run.

## All experiments

![[Result_tracker.base]]

The table has four views: *All experiments*, *Stage B*, *Stage A and pretraining*, *Needs attention*. Numbers are bare floats so the columns sort. An empty `delta` means **not comparable**, and `delta_on` says why.

## Lineage

A solid arrow means *parent → child* (the comparison). A dotted arrow means *uses* (the frozen Stage A, or a pretrained `B`, that a run is built on).

```mermaid
flowchart LR
    subgraph MRI["data/mri - HCP, 23 structures, 128³ at 1.25 mm"]
        direction LR
        A01["A01 Stage A<br/>0.8158 - shipped"] --> A02["A02 16³ bottleneck<br/>0.8018, Δ -0.014"]
        L01["L01 legacy attention B<br/>best 0.706, not comparable"]
        P01["P01 B(I) pretraining<br/>boundary Dice 0.633"]
        B01["B01 overfit 1 scene<br/>0.788, 1.20 mm"] --> B02["B02 oracle anchors<br/>Δ -0.007 ± 0.054"]
        L01 --> B03["B03 relational baseline<br/>0.781 vs floor 0.307<br/>held-out 0.005 vs 0.110"]
        B03 --> B04["B04 prompt-only<br/>Δ -0.258, held-out +0.112"]
        B03 --> B05["B05 pretrained B<br/>Δ -0.010, held-out +0.001"]
        B03 --> B06["B06 field empty-only<br/>0 epochs"]
        D01["D01 anchor-first prompts"]
        D02["D02 gate, tau 0.5"]
        D03["D03 zscore-brain"]
    end
    subgraph SYN["synthetic corpora - 64³"]
        direction LR
        A03["A03 easy Stage A<br/>0.9976"] --> A04["A04 hard Stage A<br/>0.8831"]
        A04 --> A05["A05 gate, 14 epochs<br/>0.065"]
        A05 --> A06["A06 control, 14 epochs<br/>0.065 then collapse"]
        A05 --> A07["A07 gate, 50 epochs<br/>0.243, Δ +0.178"]
        A07 --> A08["A08 MRI-measured look<br/>0.283"]
        A08 --> A09["A09 400 scenes<br/>0.923, Δ +0.640"]
        B07["B07 easy Stage B<br/>0.983, held-out 0.98"] --> B08["B08 hard Stage B<br/>0.876, held-out 0.71"]
        B08 --> B09["B09 synthetic-mri<br/>0.937, held-out 0.37 / 0.46"]
        B09 --> B10["B10 leave-one-out<br/>killed at 17/30, no effect"]
        B09 --> B11["B11 no-anchor<br/>INVALID, replicate of B09"]
        B09 --> B12["B12 mask_on: valid<br/>0.962, held-out 0.73 / 0.77"]
    end
    B03 --> B07
    A01 -.-> B01
    A01 -.-> B03
    P01 -.-> B05
    A03 -.-> B07
    A04 -.-> B08
    A09 -.-> B09
    D01 -.-> B03
    D02 -.-> B03
    D03 -.-> A01

    classDef stageA fill:#e1f5fe,stroke:#0288d1,color:#1a1a1a;
    classDef stageB fill:#fbe9e7,stroke:#e64a19,color:#1a1a1a;
    classDef diag fill:#f3e5f5,stroke:#7b1fa2,color:#1a1a1a;
    classDef legacy fill:#eeeeee,stroke:#9e9e9e,color:#1a1a1a;
    classDef running fill:#fff8e1,stroke:#f9a825,stroke-width:2px,color:#1a1a1a;
    classDef bad fill:#ffebee,stroke:#c62828,stroke-width:2px,color:#1a1a1a;
    class A01,A02,A03,A04,A05,A06,A07,A08,A09 stageA;
    class B01,B02,B03,B04,B05,B07,B08,B09 stageB;
    class D01,D02,D03,P01 diag;
    class L01 legacy;
    class B06,B11 bad;
    class B10,B12 stageB;
```

Keep this graph in step with the `parent` properties when you add a note. Obsidian's graph view draws the same edges from the `parent` and `stage_a` links.

## How to read a row

| property | meaning |
|---|---|
| `parent` | the run this one is compared against, which is normally the same config minus one change. Empty for a root |
| `change` | the **one** thing that differs from the parent. If several things changed, say so, because the Δ is then confounded |
| `assumption` | the hypothesis, written *before* the run, as a prediction that can fail |
| `score` | the value of `metric` at the epoch the note states. Stage B: val Dice on `targets.train` (the selection curve) |
| `floor` | the prompt-blind Dice of that population (Stage B only). **Read `score` against `floor`, never against 0** |
| `heldout_val`, `heldout_test` | Dice on `val:targets.val` and `val:targets.test`. **Both are val-split subjects**; the test split is untouched |
| `delta` | `child − parent` on the `delta_on` metric. Given **only** when both share the corpus *and* the population, compared at **matched epochs** (or best vs best when both ran their full schedule) |
| `benefit` | `yes` / `no` / `mixed` / `not yet` (inside the noise) / `n/a` / `invalid` (the change did not apply) |
| `status` | `done` / `running` / `stopped` / `killed` / `failed` / `invalid` / `adopted` / `superseded` |

**A Dice is not a result by itself.** Before calling a Δ a benefit, check it against the four counterfactuals, the prompt-only and image-replacement tests, and the replicate noise below ([[SpatialVox#16.6 Before a Dice is quoted]]).

## Floors and ceilings

The prompt-blind floor is "take the non-anchor structure nearest the anchors' centroid". The ceiling is how often the anchor identities alone recover the target. Both come from `scripts/corpus_report.py --split val --scenes 600`. On the synthetic corpora structures never overlap, so the floor there is the accuracy of the nearest-centroid rule. A single-class population has a 100% ceiling and tells you nothing.

| corpus | `targets.train` floor / ceiling | `targets.val` | `targets.test` |
|---|---|---|---|
| `data/mri` | 0.3067 / 67.8% | 0.1097 / 83.9% | 0.0630 / 99.2% |
| `data/synthetic-mri` | 0.1383 / 30.6% | 0.2467 / 68.3% | 0.1617 / 66.1% |
| `data/synthetic-hard` | 0.2133 / 29.9% | 0.1500 / 71.5% | 0.2729 / 100% |
| `data/synthetic` | 0.2500 / 29.8% | 0.2167 / 72.5% | 0.1847 / 100% |

## Noise

Nothing sets `cudnn.deterministic`, so the same config and seed do not reproduce a curve. **Every run here is single-seed.**

| estimate | supervised | held-out val | held-out test |
|---|---|---|---|
| `synthetic-mri` replicate ([[B11 arm-noanchor]] vs [[B09 mri-stage-b]], epochs 0–10): mean \|Δ\| per epoch | 0.022 | 0.091 | 0.065 |
| the same, max \|Δ\| | 0.134 | 0.223 | 0.200 |
| within one run: mean \|Δ\| between consecutive epochs ([[B09 mri-stage-b]]) | — | 0.144 | — |
| earlier MRI runs (non-determinism alone) | ~0.01 typical, ~0.03 worst | — | — |

These are **lower bounds** on seed spread. A held-out Δ smaller than about 0.1 at a single epoch is not evidence on `synthetic-mri`.

## Adding an experiment

1. Copy [[Experiment template]] into `Experiments/` as `<ID> <run-name>.md`. The ID prefix is the kind: `A` Stage A, `B` Stage B, `P` pretraining, `D` diagnostic or decision, `L` legacy. Zero-pad: `B12`.
2. Set `parent` to the run it should be compared against, and write down the single `change` and the `assumption` **before** launching.
3. Launch with `True`/`False` for booleans, then open `runs/<name>/best.json` and confirm that the changed key reads what you intended (in `model` for architecture, in `meta.config.stage` for the schedule).
4. When it ends, fill in `score`, `floor`, `heldout_*`, `delta` (only if comparable) and `delta_on`, and set `benefit` against the noise table. Then write the verdict.
5. Add the node and its edge to the lineage graph above.
