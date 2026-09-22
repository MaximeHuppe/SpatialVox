# Results

What has been measured on this branch, and what has not. Every number here comes
from a script in `scripts/` that can be re-run; the command is given beside it.

`CLAUDE.md` §6 is the reading rule: a Dice is read against its population's
prompt-blind floor and its population's shortcut ceiling, never against zero.

---

## 1. The corpus

`data/mri` — 200 HCP subjects, 128³ at 1.25 mm, 23 labelled structures,
anchor-first prompts. 160/20/20 subjects train/val/test.

```
.venv/bin/python scripts/corpus_report.py --split val --scenes 600
```

| population | examples (val) | prompt-blind Dice | anchor-set ceiling |
|---|---|---|---|
| all target classes | 1983 | **0.2017** | 42.3% |
| `targets.train` — 8 supervised classes, the selection curve | 763 | **0.3067** | 67.8% |
| `targets.val` — 4 held out (caudate, putamen) | 392 | **0.1097** | 83.9% |
| `targets.test` — 2 held out (hippocampus) | 127 | **0.0630** | 99.2% |

The conjunction is unique in **100%** of prompts, by construction.

**The two-class row is nearly unreportable.** With only hippocampi left, the
anchor identities alone recover the target 99.2% of the time, so a good Dice
there says almost nothing about whether the model read a direction word.
`scripts/evaluate.py` prints this ceiling next to the Dice for exactly that
reason.

## 2. The gate (proposal §2) — passed

```
.venv/bin/python scripts/gate_mapper.py --segmenter runs/phase-a/current/best.pt
```

Ground-truth centroids, used only as a check.

| `τ` (mm) | gate: target centroids with `where_raw > 0.5` | same, after one clause is flipped |
|---|---|---|
| 0.25 | 0.9908 | 0.0000 |
| **0.50 (shipped)** | **0.9742** | **0.0000** |
| 1.00 | 0.8917 | 0.0000 |
| 2.00 (proposal's starting value) | 0.6508 | 0.0000 |
| 4.00 | 0.2417 | 0.0000 |

A flipped clause moves the target centroid off the high region in **100%** of
examples, at every `τ`. The full table, and why `τ = 0.5` rather than 2, is
`deviations.md` §1.1.

**Stage A's anchors** (`runs/phase-a/current`, val Dice 0.8158 over 23 classes):
centroid error against ground truth is a **median of 0.84 mm** on a 1.25 mm grid
(95th percentile 2.13 mm, worst 8.14 mm). The mapper consumes the centroid, not
the mask, so this — not anchor Dice — is what decides whether `predicted` anchors
degrade the field. They barely do.

**The null head's ceiling**: `where_mass` alone separates "the clauses name one
structure" from "the clauses name none" at **AUC 0.848**, at every `τ`. That is a
property of the four inputs §3 allows it, not of its width.

**Flip outcomes**, re-scored by the corpus's own rule: names two or more 33.2%
(dropped), names none 65.5% (empty mask, null target invalid), retargets 1.2%.

## 3. Wiring: the overfit (proposal §9)

```
.venv/bin/python scripts/train.py b --overfit 1 --set train.stage_b.epochs=120
```

One scene, 41 prompts over 8 target classes, no flips.

| | Dice | centroid error |
|---|---|---|
| epoch 0 | 0.0000 | 17.32 mm |
| epoch 119 | **0.7877** | **1.22 mm** |

Centroid error ends **below one voxel**, which is the check that the channel
order, the world coordinates and the prompt indices are wired correctly. That is
all this number is for: it is *not* a capacity ceiling — the full run in §4 passes
0.7877 on **training** Dice by epoch 11, with the same carver.

### 3.1 Is 0.79 the carver, or Stage A's anchors eating the target?

`logits = background where A_i > 0.5` uses the *predicted* soft mask, so an
over-segmented anchor deletes target voxels before the loss ever sees them. The
same overfit with `anchor_source: oracle` isolates that:

```
.venv/bin/python scripts/train.py b --overfit 1 --set train.stage_b.anchor_source=oracle
```

**It is the carver.** The two curves track within noise, epoch for epoch. Over
33 matched epochs the mean of (oracle − predicted) is **−0.0072**, with a
standard deviation of 0.0542 — the oracle anchors are, if anything, very slightly
*worse*, which is what indistinguishable looks like:

| epoch | predicted anchors | oracle anchors | centroid, predicted / oracle |
|---|---|---|---|
| 0 | 0.0000 | 0.0000 | 17.32 / 17.37 mm |
| 12 | 0.3211 | 0.3545 | 4.68 / 4.60 mm |
| 18 | 0.4464 | 0.4698 | 3.26 / 3.33 mm |
| 24 | 0.5257 | 0.4399 | 1.57 / 3.17 mm |
| 30 | 0.5698 | 0.5584 | 1.70 / 1.92 mm |

The oracle arm was stopped at epoch 33 once the comparison was unambiguous, to
give the GPU back to the run in §4.

Replacing Stage A's masks with the ground truth does not lift the ceiling, so
`logits = background where A_i > 0.5` is not eating enough of the target to
matter and 0.79 is what a 16-channel carver with a stride-2 stem can memorise for
structures of 300–4000 voxels at 1.25 mm. That is consistent with §2's other
measurement — Stage A's centroid error is *sub-voxel*, so `predicted` and
`oracle` are currently non-discriminating on this corpus.

What this does *not* establish is a capacity ceiling — see §4, where the same
carver passes 0.7877 on training Dice once it has 160 subjects instead of one.

## 4. The relational run

```
.venv/bin/python scripts/train.py b --out runs/relational-seed1 \
    --set train.stage_b.epochs=20 --set train.stage_b.scheduler.warmup_epochs=2
```

One seed, 5855 supervised training prompts over 160 subjects, `B` from scratch,
predicted anchors, 12 min/epoch on one RTX 5090 laptop GPU. 0.27M trainable
parameters beside a frozen 17.0M Stage A.

**Note on two column names.** This run was launched before the two held-out
curves were renamed, so its JSON calls them `val_ood` (= `val:targets.val`, the
four caudate/putamen classes) and `test_ood` (= `val:targets.test`, the two
hippocampi). **Both are val-split subjects**; the test split is untouched until
`scripts/evaluate.py --split test`.

### The curve

| epoch | trained-8 (selection) | centroid | held-out-4 | held-out-2 | `flip_direction` drop | `permute_both` drop |
|---|---|---|---|---|---|---|
| 0 | 0.4368 | 5.16 mm | 0.1375 | 0.0198 | +0.384 | +0.0003 |
| 3 | 0.6002 | 2.58 mm | 0.0099 | 0.0006 | +0.545 | +0.0034 |
| 7 | 0.7206 | 2.05 mm | 0.0050 | 0.0003 | +0.656 | +0.0021 |
| 9 | 0.7412 | 2.10 mm | 0.0092 | 0.0006 | +0.655 | +0.0000 |
| 13 | 0.7753 | 1.86 mm | 0.0055 | 0.0007 | +0.696 | +0.0008 |
| 15 | 0.7794 | 1.76 mm | 0.0068 | 0.0008 | +0.692 | +0.0007 |
| **19** | **0.7812** | **1.94 mm** | **0.0053** | **0.0006** | **+0.702** | **+0.0011** |

Training Dice ends at 0.8385, so there is about 0.06 of train/val gap and the
selection curve has been flat since epoch 15 — 20 epochs was enough for this
configuration.

**The headline, read the way `CLAUDE.md` §6 requires it:**

| | value |
|---|---|
| Dice, held-out subjects, supervised classes | **0.7812** |
| prompt-blind floor for that population | 0.3067 |
| anchor-set shortcut ceiling for that population | 67.8% |
| centroid error (heatmap, not mask) | **1.94 mm** on a 1.25 mm grid |
| `permute_channels` / `permute_clauses` | Dice → **0.0000** |
| `flip_direction` | drop **0.702** |
| `permute_both` (control) | **+0.0011** |
| seeds | **one** — see §6 |

**The one-scene overfit is not a capacity bound.** An earlier draft of this file
read §3's 0.7877 as the carver's ceiling. It is not: by epoch 11 the same carver
reaches a *training* Dice of **0.8048** on 160 subjects, above what it managed
memorising a single scene in 120 epochs. The overfit is a wiring check — it says
the channel order, the world coordinates and the prompt indices are right — and
nothing more. Whether `carver.width` is the binding constraint here is untested;
it is a config leaf and lands in `StageB.config`, so it is one run away.

### The counterfactuals — §7's signature, from the first epoch on

Measured on the selection curve at epoch 0, where the drops are already decisive:

| probe | resulting Dice | drop | should |
|---|---|---|---|
| `permute_channels` | **0.0000** | 0.4368 | fall |
| `permute_clauses` | **0.0000** | 0.4368 | fall |
| `flip_direction` | 0.0528 | 0.3840 | fall |
| `permute_both` (control) | 0.4365 | **0.0003** | not move |

Moving the masks against the words, or the words against the masks, takes the
prediction to **exactly zero overlap**. Moving both together — every relation
preserved — moves it by 0.0003. `deviations.md` §7.3 is the caveat: `where_raw`
is a product and therefore exactly permutation-invariant, so this control is a
weaker statement here than it was under the attention architecture.

### Where the loss goes

Per-term, at epoch 0 (total 1.31): mask **0.79**, `field_centroid` 0.30,
`centroid` 0.14, `null_bce` 0.07, `L_far` 0.0001. Anchor Dice 0.817.

The `field_centroid` share is the §5.2 problem showing up in the objective: 23%
of the loss spent on a target measured to be 20 mm from the right one. Every term
is logged per epoch (`loss_*` in `metrics.jsonl`) because six terms on four
different scales cannot be balanced by reading the total.

## 5. Probes, tests and what they showed

### 5.0 The report path, verified end to end

```
.venv/bin/python scripts/evaluate.py runs/relational-seed1/best.pt \
    --split val --classes val --limit 80
```

Run against the **epoch-0** checkpoint, so the numbers are weak by construction —
what it establishes is that all six blocks of §7 are produced from real data:

```
  anchor Dice                 0.8121
  centroid error (mm)         12.39
  gate: where_raw > 0.5 at the target centroid   0.9125   (from predicted anchor centroids)

  shortcut ceiling for THIS population: the anchor identities alone recover
  the target 85.0% of the time   <- read the Dice above against this, not against 0

  counterfactual                       dice     drop
  permute_channels               0.0000   0.1631
  permute_clauses                0.0000   0.1631
  flip_direction                 0.0126   0.1505
  permute_both                   0.1633  -0.0002  <- control: should not move

  prompts that name nothing (381)
    null head says invalid    0.5433
    emitted any mask          0.3123
    mean false-positive voxels 164.7

  image replacement, 63 pairs of different subjects (17 same-subject pairs skipped)
    dice            0.1609 -> 0.1567   (should fall)
    centroid (mm)   12.62 -> 13.17   (should hold)
```

Two things to note about this *machinery*, before the final numbers replace it:

- **`permute_channels` and `permute_clauses` take the Dice to exactly zero**, and
  `permute_both` moves it by −0.0002. That is §7's signature, already at epoch 0.
- **Image replacement is the test that needs a trained model.** At epoch 0 the
  carver barely uses `B(I)`, so swapping the volume changes almost nothing
  (0.1609 → 0.1567). A null result *here* is uninformative; a null result at the
  end would mean the mask is a spatial prior, and it is the number to watch.

### 5.1 The two mandatory tests (proposal §7)

```
.venv/bin/python scripts/evaluate.py runs/relational-seed1/best.pt \
    --split val --classes train --limit 400
```

**Image replacement — passes.** Keep this subject's anchors and every field;
feed `B` a *different subject's* MRI. 360 pairs of genuinely different subjects
(40 same-subject pairs skipped, which is why the pairing is checked rather than
assumed — see the note in `scripts/evaluate.py`):

| | with this subject's MRI | with another subject's | §7 expects |
|---|---|---|---|
| Dice | 0.7922 | **0.4747** | should fall |
| centroid error | 1.72 mm | **5.47 mm** | should hold |

Dice loses 40% of itself while the centroid, though it moves, stays far better
than the geometry alone could manage (§5.2: the field's own centre of mass is
19.2 mm off on this population). That is the pattern §7 calls *"the signature
that the words placed the structure and the image drew it"* — the prompt is
still placing it, and the swapped image is no longer able to draw it.

**Prompt-only carver — passes, and says something the proposal did not
anticipate.** `B(I)` removed; anchors and fields kept; same 20 epochs, same seed.
0.03M trainable parameters against 0.27M.

| final epoch | supervised eight | held-out four | held-out two | centroid |
|---|---|---|---|---|
| full model | **0.7812** | 0.0053 | 0.0006 | 1.94 mm |
| prompt-only | 0.5232 | **0.1171** | **0.0540** | 4.32 mm |
| *their floors* | *0.3067* | *0.1097* | *0.0630* | |

On the supervised classes the ablation behaves exactly as §7 demands: removing
the image costs **0.258 Dice**, so the mask is *not* a shape redrawn from the
spatial prior. Together with the image-replacement result above, the image is
doing real work.

**But the two rows cross.** On the held-out classes the prompt-only carver is
**22× better** than the full model — 0.1171 against 0.0053 — and it is the only
arm that clears its prompt-blind floor there. Removing the image *improves*
relational transfer.

That is the same fact as §5.3 seen from the other side. A carver with `B(I)` can
learn what each of its eight supervised classes *looks like*, so it does, and
then it has nothing to say about a ninth. A carver without the image cannot do
that — it can only put a blob where the field points — so it stays class-agnostic
and generalises, badly but above chance. The image is what makes the supervised
number good and what makes the transfer number bad.

Note also that the prompt-only arm is *still* prompt-dependent: `permute_channels`
0.528, `permute_clauses` 0.528, `permute_both` −0.0001. It is reading the
relations; it just cannot draw.

### 5.1b The rest of the §7 report, on the supervised eight

400 val examples, predicted anchors:

| | value | read against |
|---|---|---|
| Dice | **0.7943** | prompt-blind floor 0.3067 |
| anchor Dice | 0.8127 | Stage A's own error |
| centroid error | **1.72 mm** | 1.4 voxels |
| emitted an **empty** mask | **0.0%** | 53% on held-out classes (§5.3) |
| predicted / true voxels | 1597 / 1631 | the volume tracks the truth |
| gate, from **predicted** anchor centroids | **0.8275** | 0.9742 from ground-truth ones (§2) |
| shortcut ceiling, this population | 71.0% | not 0 |

| counterfactual | resulting Dice | drop |
|---|---|---|
| `permute_channels` | **0.0000** | 0.7943 |
| `permute_clauses` | **0.0000** | 0.7943 |
| `flip_direction` | 0.0631 | 0.7312 |
| `permute_both` (control) | 0.7934 | **0.0009** |

| prompts that name nothing (382 of them) | |
|---|---|
| null head says invalid | 0.6257 |
| **emitted any mask at all** | **0.0550** |
| mean false-positive voxels | 164.9 |

The empty-prompt row is the check that §2's tiny spike in `where_raw` was not
renormalised into a confident answer, and it is the clearest one here: on a
prompt that names nothing the model emits **any** mask only 5.5% of the time.
Note where that comes from — the *carver*, trained by the empty-mask loss, not
the null head, which only reaches 0.626 against its measured 0.848 ceiling
(`deviations.md` §7.1).

**The gate is 0.83 from predicted anchors against 0.97 from ground-truth ones.**
Those are the two numbers `deviations.md` §7.2 insists on naming differently.
Stage A's sub-voxel *median* centroid error hides a tail: the 95th percentile is
2.13 mm and the worst 8.14 mm, and `where_raw > 0.5` at a point is a sharp
predicate, so the tail costs 14 points of gate.

### 5.2 The field contains the target but does not point at it

The single most consequential measurement on this branch, and it needs no model:

```
.venv/bin/python scripts/gate_mapper.py --examples 600 --tau 0.5 1.0 2.0
```

| `τ` (mm) | gate: target centroid inside `where_raw > 0.5` | field's centre of mass → target's centroid |
|---|---|---|
| 0.50 | 0.9767 | **20.2 mm** |
| 1.00 | 0.8850 | **20.2 mm** |
| 2.00 | 0.6333 | **20.3 mm** |

Nearly independent of `τ`, so it is not a softness effect. The conjunction of
three 45° cones is an **elongated wedge**, and the target sits near its *apex* —
close to the anchors — while the wedge runs away from them.

Per population, taking the field's centre of mass as a prompt-only localiser
(ground-truth anchors, so this is the geometry's own ceiling):

| population | mean | median | p90 |
|---|---|---|---|
| `targets.train` (8) | 19.20 mm | 10.93 mm | 48.14 mm |
| `targets.val` (4) | 26.79 mm | 21.50 mm | 50.98 mm |
| `targets.test` (2) | 30.15 mm | 33.84 mm | 40.11 mm |

**Three things follow, and all of them matter for reading §4 of the run above.**

1. §7's metric "centroid error, mm, on the 4 — *the field located the structure*"
   does **not** follow from the gate passing. The gate is an "is it inside"
   question; localisation is a "where is it" question, and the field answers the
   first well and the second badly.
2. §5's heatmap-against-the-field term therefore pulls the heatmap ~20 mm
   off-target on every *valid* prompt, against the structure-centroid term
   pulling it back. The loss components show exactly that: `field_centroid`
   plateaus at 0.32 and never moves — roughly 40% of the total loss spent on a
   constant opposing gradient. `deviations.md` §3.1b.
3. The transfer collapse in §4 is *not* the model failing to use a good signal.
   On the four held-out classes the model's heatmap reaches 18.4 mm while the
   field's own centre of mass is 26.8 mm — the model is **better** than the
   geometry it is given, and the geometry is the ceiling.

The alternative reading of §5 is runnable as
`train.stage_b.field_centroid_on: empty-only`, and is queued as a third arm.

### 5.3 The transfer collapse, and the one deviation that could explain it

The selection curve rises while the held-out curves *fall*:

| epoch | trained-8 | held-out-4 | held-out-2 |
|---|---|---|---|
| 0 | 0.4368 | 0.1375 | 0.0198 |
| 1 | 0.5508 | 0.0813 | 0.0100 |
| 2 | 0.5718 | 0.0499 | 0.0022 |
| 3 | 0.6002 | 0.0099 | 0.0006 |

Early on the carver puts a generic blob near the field and it overlaps a bit;
as it specialises to the eight classes it is supervised on, it stops producing
anything usable for a class it has never been asked to draw.

**The failure is "says nothing", not "says the wrong thing."** On the best
checkpoint so far, over 200 val prompts per population:

| population | Dice | emitted an **empty** mask | predicted voxels | true voxels | centroid |
|---|---|---|---|---|---|
| `targets.train` (8) | 0.6876 | **0.0%** | 1413 | 1609 | 2.1 mm |
| `targets.val` (4) | 0.0096 | **55.0%** | 135 | 2241 | 14.4 mm |
| `targets.test` (2) | 0.0021 | **51.2%** | 397 | 2322 | 21.0 mm |

On a supervised class the carver's volume tracks the truth (1413 against 1609).
On a held-out one it falls silent about half the time and, when it does speak,
predicts a fifth of the volume. So the carver has learned a **class-conditional
size and shape prior** and defaults to the empty mask outside it — which is also
what 16% of its training examples look like, because a flip that names nothing is
supervised to be empty (§5's middle column, 65.5% of flips).

`scripts/evaluate.py` now reports the empty-prediction rate and the
predicted-against-true voxel count beside every Dice, because the three ways a
Dice can be low need three different fixes.

**At the final checkpoint it is sharper still.** Full report on all 392 val
examples of `targets.val`:

| | supervised eight | held-out four |
|---|---|---|
| Dice | **0.7943** | **0.0052** |
| its prompt-blind floor | 0.3067 | 0.1097 |
| centroid error | 1.72 mm | 24.86 mm |
| emitted an **empty** mask | **0.0%** | **75.0%** |
| predicted / true voxels | 1597 / 1631 | **93** / 2219 |
| anchor Dice | 0.8127 | 0.8046 |
| gate, predicted anchor centroids | 0.8275 | **0.8571** |

The two-class `targets.test` population behaves identically — gate **0.8976**,
empty rate **73.2%**, Dice 0.0006, 389 predicted voxels against 2322 — but its
anchor-set shortcut ceiling is **99.2%**, so nothing said about it either way
would mean much. `scripts/evaluate.py` prints that warning beside the number.

Read the last two rows together. The anchors are as good on the held-out classes
as on the supervised ones, and **the field points at the right region slightly
more often** (0.857 against 0.827). Nothing upstream of the carver has failed.
The carver simply does not answer: three times in four it emits nothing, and the
once it does it draws 93 voxels where 2219 belong.

**So the honest statement of this branch's result is:** on classes it is
supervised on, the model does the relational task and does it from the image
(§5.1). On a class it has never been supervised to draw, it scores **below the
prompt-blind floor** — 0.0052 against 0.1097. Relational transfer to an
unsupervised target class does not happen here.

This made a falsifiable prediction about arm 3: *if a class-agnostic edge prior
is what is missing, the empty rate should fall and the predicted volume should
track the true one.*

### 5.4 Arm 3 falsifies it: pretraining `B` does not restore transfer

`B` pretrained for 30 epochs on the three class-agnostic objectives to a
boundary-map Dice of 0.626, loaded in, and given 0.1x the carver's learning rate
— §4's prescription exactly. Epoch-matched against `B` from scratch:

| epoch | supervised, pretrained | supervised, scratch | held-out-4, pretrained | held-out-4, scratch |
|---|---|---|---|---|
| 0 | 0.4647 | 0.4368 | 0.1123 | 0.1375 |
| 1 | 0.5677 | 0.5508 | 0.0817 | 0.0813 |
| 3 | 0.5870 | 0.6002 | 0.0089 | 0.0099 |
| 5 | 0.6886 | 0.6411 | 0.0173 | 0.0132 |

The supervised trajectories are the same within noise, and **the held-out curve
collapses identically** — 0.112 to 0.017 by epoch 5, exactly as it did without
the pretraining. So `deviations.md` §5.1, the one sequencing deviation this
branch made, was **not** the cause of §5.3. Running it was still the right call:
it was the proposal's own prescription and it is now measured rather than
assumed.

**What that leaves.** The cause is the carver's *objective*, not its features.
Nothing in `Dice + BCE on eight classes` rewards class-agnostic behaviour, and
16% of training examples are supervised to be empty (§5's middle column), so
"when in doubt, say nothing" is a locally optimal policy that costs nothing on
the training distribution. The next thing to try is therefore the supervision,
not the network:

- `train.stage_b.flip_probability` — how much of the objective is empty-mask;
- whether the target-class split can be made to reward drawing *something*
  field-shaped when the class is unfamiliar, which is what the prompt-only arm
  (§5.1) does accidentally and better;
- a wider carver is **not** indicated — §4 shows it is not capacity-bound.

**The most likely cause is §5.1 of `deviations.md` — `B` trained from scratch.**
§4 is explicit that `B` is *pretrained with objectives that carry no class id* and
then given a lower learning rate, precisely so that "a structure Stage A has never
seen is still a boundary in the image". Trained jointly from scratch, `B` has no
class-agnostic prior at all: it becomes a feature extractor for whatever helps the
eight supervised classes, and the only component that was supposed to generalise
does not.

That makes the pretrained-`B` arm the *completion of the specification* rather
than an optional ablation, and it is queued as arm 3. `scripts/train.py boundary`
takes about ten minutes.

It is not the only candidate — §5.2's geometry says the localisation signal the
carver is handed is weak to begin with — and the two are not exclusive.

**The pretraining itself** (`runs/boundary-seed1`, 30 epochs, ~10 min,
0.23M parameters):

| epoch | loss | boundary-map Dice, train | val |
|---|---|---|---|
| 0 | 3.514 | 0.0041 | 0.0163 |
| 6 | 2.047 | 0.5313 | 0.5516 |
| 18 | 1.278 | 0.6067 | 0.6138 |
| 29 | 1.206 | 0.6290 | **0.6258** |

Train and val land on the same number, so `B` is not memorising subjects — it has
learned where edges are, from three objectives that carry no class id: put back
blanked cubes of `I`, mark where two neighbouring voxels differ in label, and
regress `|∇I|`.

**What the pretrained-`B` arm does and does not test.** `B`'s boundary target is
label adjacency over *all 23* structures, so it includes the outlines of caudate,
putamen and hippocampus. `B` therefore learns to find their edges — class-
agnostically, with no class channel and no target indicator, but it has seen
them. The proposal says as much in §6: *"Caudate and putamen are held out as
relational targets and are still anchors... It is not the lesion claim."* So this
arm tests whether a class-agnostic edge prior restores **relational transfer**,
which is what the 8/4 split measures. It does **not** test the lesion claim, and
neither does anything else on this corpus — see `CLAUDE.md` §5.

## 6. Summary

| claim | verdict | evidence |
|---|---|---|
| the mapper agrees with the prompts | **yes** at `τ = 0.5` | gate 0.974 (ground-truth anchors), 0.83–0.90 (predicted); a flip moves the centroid off the region 100% of the time |
| the model reads the prompt rather than a shortcut | **yes** | `permute_channels` and `permute_clauses` take Dice to exactly 0.0000; `permute_both` moves 0.0009; Dice 0.7943 against a 0.3067 floor |
| the mask is drawn from the image | **yes** | another subject's MRI into `B` costs 40% of the Dice (0.7922 → 0.4747) while the centroid holds far better than geometry alone (1.72 → 5.47 mm, against 19.2 mm for the field's own centre) |
| an impossible prompt produces nothing | **yes** | of 382 prompts that name nothing, 5.5% emit any mask at all |
| the null head does that work | **no** | 0.626 accuracy against a measured 0.848 input ceiling; the *carver* and its empty-mask loss are what produce the empty output |
| the field *locates* the target | **no** | its centre of mass is 20.2 mm from the target's centroid, nearly independent of `τ` |
| relational transfer to an unsupervised target class | **no** | 0.0052 against a 0.1097 floor; the carver emits nothing 75% of the time while the anchors and the field are as good as on the supervised classes |
| the mask is a shape redrawn from the spatial prior | **no** (§7's prompt-only test) | removing `B(I)` costs 0.258 Dice on the supervised classes |
| ...but the image is *also* what destroys transfer | **yes**, unanticipated | the prompt-only carver scores 0.1171 on the held-out four against the full model's 0.0053 — 22x better, and the only arm above that population's floor |

## 7. What has not been run

State these beside any number taken from this branch:

- **One seed.** `CLAUDE.md` §6: non-determinism alone moved val Dice by ~0.01
  typical and ~0.03 worst case on this project's earlier runs, and nothing sets
  `cudnn.deterministic`, so that is a *lower* bound on seed spread. A
  single-seed number is not a result.
- **`B` is trained from scratch, not pretrained.** `scripts/train.py boundary`
  implements §4's pretraining and `train.stage_b.boundary_checkpoint` wires it
  in; the sequencing decision is `deviations.md` §5.1. The path is *verified to
  run* at full resolution — 0.229M parameters, 1.6 GB peak, loss 4.70 → 4.24 over
  three steps, with the label-adjacency boundary target covering 1.63% of voxels
  — it has simply not been used to initialise a reported run.
- **The contrastive pretraining term**, which §4 marks "optionally", is not
  implemented. The other three objectives are.
- **The `alpha` ablation** (`model.stage_b.additive_prior: true`) has not been
  run.
- **`carver.full_resolution_skip: false`**, the literal reading of §4, has not
  been run as an ablation.
