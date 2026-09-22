# Deviations from `relational_architecture.md`

Every place the implementation differs from the proposal, with the reason and,
where the reason is a number, the measurement that produced it. The proposal is
the specification; this file is the diff. Nothing here changes what the model is
allowed to see — §1 and §6 of the proposal are implemented literally, and
`tests/test_models.py` pins them.

Measurements below come from `scripts/gate_mapper.py` on `data/mri` (200 HCP
subjects, 128³ at 1.25 mm, 23 structures) unless stated otherwise.

---

## 1. Constants the proposal gave a starting value for

### 1.1 `mapper.tau`: 2 mm → **0.5 mm**

§2 makes one number the precondition for training the carver: the fraction of
target centroids with `where_raw > 0.5`, at an absolute threshold. That fraction
is a function of `τ`, because at the target's own centroid `classify` guarantees
a non-negative margin on every clause, so `F_i → 1` as `τ → 0`.

| `τ` (mm) | gate | flip | field vol | dilated | target voxels in field | in dilation | null AUC |
|---|---|---|---|---|---|---|---|
| 0.25 | 0.9908 | 0.0000 | 0.28% | 2.93% | 0.319 | 0.929 | 0.849 |
| **0.50** | **0.9742** | **0.0000** | **0.33%** | **3.13%** | **0.364** | **0.935** | **0.848** |
| 1.00 | 0.8917 | 0.0000 | 0.43% | 3.55% | 0.448 | 0.945 | 0.848 |
| 2.00 *(proposal)* | 0.6508 | 0.0000 | 0.66% | 4.40% | 0.580 | 0.958 | 0.847 |
| 4.00 | 0.2417 | 0.0000 | 1.20% | 6.08% | 0.735 | 0.973 | 0.843 |

At the proposal's 2 mm the gate passes **65%** of prompts; §2 says the carver is
not trained on top of a mapper that disagrees with the prompts that often. At
0.5 mm it passes 97.4%. The cost is on the right of the table and it is small:
the field is tighter, so `L_far` polices more of the volume, but 93.5% of a
target's voxels still lie inside the 8-voxel dilation the penalty is measured
outside of.

0.25 mm buys another 1.7 points of gate for a field that transitions inside a
fifth of a voxel — effectively a hard pyramid. 0.5 mm keeps the transition at
about half a voxel, which is the smallest width that is still a *soft* field on
this grid. Nothing optimises through the mapper (the masks are detached and it
has no parameters), so sharpness costs no gradient.

The `flip` column is §2's other requirement, and it is unconditional: **a flipped
clause moves the target centroid off the high region in 100% of examples, at
every `τ`.**

### 1.2 `mapper.min_mass`: 1e-3 → **1e-6**

`mass_i = mean(A_i)` is a fraction of the volume, and a rejected channel writes
`F_i = 0`, which zeroes `where_raw` for the whole example. Measured masses from
`runs/phase-a/current`:

| | mass (fraction of 128³) |
|---|---|
| Brain-Stem, the largest | 5.0e-3 |
| Left-Thalamus | 2.0e-3 |
| Left-Accumbens | 1.5e-4 |
| Right-Inf-Lat-Vent, the smallest class | 4.6e-5 |
| smallest single anchor observed | **3.2e-6** |

The proposal's 1e-3 sits above **every structure in the vocabulary except the
brainstem and the thalami**, so it would reject almost every anchor. 1e-6 sits
below the smallest one Stage A legitimately finds, which leaves it doing the job
§1 describes: rejecting a query Stage A answered with nothing.

### 1.3 `far.epsilon`, `far.dilation`, `flip_probability`, `alpha`

Unchanged from the proposal's starting points: 0.05, 8 voxels, 0.25, 0.35.

---

## 2. Quantities that had to be rescaled to be usable

These are monotone reparameterisations. They carry exactly the information the
proposal specifies and no more.

### 2.1 The null head reads `log10` of its four inputs

§3: `valid = MLP(where_mass, mass_0, mass_1, mass_2)`. Measured, `where_mass`
spans **1e-21 to 2e-2** across the populations the head has to separate. A linear
layer cannot resolve a range of nineteen decades; the first layer's weights would
have to differ by that factor between the two ends. `log10` of the clamped value
is the same number, ordered the same way.

### 2.2 The carver's `log(where_raw)` and `where_mass` channels are normalised

§4 feeds the carver `log(where_raw + ε)` and `broadcast(where_mass)`. The other
nine channels are probabilities in `[0, 1]`. A raw `where_mass` of 1e-3 is
indistinguishable from zero after one convolution, and a raw `log(where_raw)`
ranging to −21 dominates every other channel. Both are clamped at `LOG_FLOOR =
1e-9` and divided by `−log(LOG_FLOOR)`, which maps them onto `[-1, 0]`.

The `where_mass` channel is therefore the *log* of the mass rather than the mass.
§4 asks that channel for one thing — "so the carver can see that the conjunction
has no mass without renormalizing the map" — and the log says that more legibly
than the raw value, which is what it is there for.

---

## 3. Where the specification was ambiguous

### 3.1 "the peak of `where_raw`" is its first moment, not its `argmax`

§5 trains the heatmap against the peak of `where_raw`. With `τ` at the gate the
field is near-binary, so its maximum is attained on a **plateau** of thousands of
voxels and `argmax` returns whichever one comes first in raster order — an
arbitrary corner of the region, and an actively misleading regression target.

The implementation uses the mass-weighted centroid of `where_raw`, which is the
unique point a plateau designates and coincides with the peak whenever the field
is unimodal. "If that peak exists" is implemented as `sum(where_raw) > 0`, which
is false exactly when a channel was rejected by `min_mass`.

### 3.1b Both heatmap targets apply on a valid prompt — and that is measurably costly

§5's table puts "peak of `where_raw`" in the *names one* column as well as the
*names none* one, so the heatmap is trained against the structure's centroid and
against the field's centre at the same time whenever a structure exists. That is
the default, `field_centroid_on: always`.

**Measured, the two targets are about 20 mm apart.** `scripts/gate_mapper.py`
reports the distance from the field's centre of mass to the target's centroid:

| `τ` (mm) | gate | field's centre of mass → target's centroid |
|---|---|---|
| 0.50 | 0.9767 | **20.2 mm** |
| 1.00 | 0.8850 | **20.2 mm** |
| 2.00 | 0.6333 | **20.3 mm** |

It barely moves with `τ`, because it is not a softness effect. The conjunction of
three 45° cones is an **elongated wedge**; the target sits near its *apex*, close
to the anchors, while the wedge runs away from them. So the field **contains** the
target — the gate says so, 97.7% of the time — without **pointing at** it.

Two consequences:

1. Under `always`, the field term pulls the heatmap roughly 20 mm off on every
   valid prompt, against the structure-centroid term pulling it back. That is
   visible in the loss components: on the overfit run `centroid` falls to 0.008
   while `field_centroid` plateaus at **0.32** and never moves — a constant
   residual and a constant opposing gradient, about 40% of the total loss.
2. It is also why the field's own centre of mass is *not* a usable
   prompt-only localiser: 19.2 mm mean error on the eight supervised classes,
   26.8 mm on the four held out, 30.2 mm on the two. The proposal's §7 metric
   "centroid error, mm, on the 4 — *the field located the structure*" does not
   follow from the gate passing.

`field_centroid_on: empty-only` is the other reading of §5 — the field target
acts only where there is no structure centroid to use, which is the *names none*
column and §4's "the heatmap is still trained on the peak of `where_raw`" when the
mask is empty. It is a **setting, not an opinion**: `always` stays the default
because the table is explicit, and `results.md` reports both arms.

### 3.2 The heatmap lives on the carver's 64³ working grid

§4 says "a separate 1×1, soft-argmax → centroid" without fixing a resolution.
It is taken from the 16-channel block output, before the upsample. Soft-argmax is
an *expectation*, so the coordinate is continuous and is not quantised to that
grid; only the weights are coarser. It is reported in millimetres against a
centroid computed from the label volume at full resolution.

### 3.3 "logits = background where `A_i > 0.5`" writes −10, not −∞

`model.stage_b.background_logit`. A `−inf` makes `BCEWithLogits` produce `nan`
whenever the target disagrees — which happens when Stage A paints part of the
target as an anchor, a real event worth a finite gradient rather than a crashed
run. −10 is `sigmoid ≈ 4.5e-5`, background by any threshold.

### 3.4 `L_far`'s dilation is a cube, not a ball

An 8-voxel `(2r+1)³` max-pool is 4913 taps at 128³; three separable passes are
51 and give the L∞ ball. The cube contains the sphere, so the penalised region is
the smaller of the two and `L_far` stays the looser reading.

### 3.5 "dropped" is a per-example weight

§5 drops an example whose flipped clauses name two or more structures. A
`Dataset` has to return an item, so the item is emitted with `keep = 0` and
excluded from every loss term *and* from `Metrics` — `weighted_mean` in
`src/engine.py` is that weight, and it runs through Dice, BCE, the null BCE, both
heatmap terms and `L_far`. Measured: a flip names two or more 33.2% of the time,
names none 65.5%, and retargets 1.2%, so at `flip_probability = 0.25` about 8% of
training items are dropped and 16% carry an empty mask.

### 3.6 `StageB.config` holds the architecture; the schedule goes in the checkpoint's `meta`

§8 requires `StageB.config` to contain `mapper.tau`, `mapper.min_mass`, `α`, the
width of `B`, `flip_probability` and the loss weights. The first four are there.
`flip_probability` and the loss weights are **not**, because `StageB.config` is
exactly the keyword arguments `load_model` reconstructs the module from, and a
training-schedule value there would be an architecture parameter the constructor
has to accept and ignore.

They are recorded instead in `save_checkpoint`'s `meta["config"]["stage"]`, which
every run writes beside the weights as both `best.pt` and a readable `best.json`.
The requirement — that a checkpoint records what produced it — is met; the
location differs.

---

## 4. One architectural addition

### 4.1 `carver.full_resolution_skip` (default **on**)

§4 specifies `residual = two 16-channel blocks, stride-2 stem, then 1×1 up to
128³`. Read literally, the stride-2 stem is the only path from `B(I)` to the
output, so every full-resolution boundary detail is destroyed before the first
convolution and the mask is a trilinear upsample of a 2.5 mm grid. That defeats
§4's own claim — "an unseen structure is segmented where `B(I)` carries a
boundary" — and on this corpus the targets are 300–4000 voxel nuclei.

With the flag on, the final 1×1 reads `concat(upsample(residual), B(I))`: still
one 1×1, still up at 128³, but it sees the boundary features at the resolution
they were computed at. It is a flag rather than a silent change so the literal
form stays measurable as an ablation.

### 4.2 `B` has no residual block at full resolution

§4 says "two or three stages, 16–32 channels, small beside Stage A" and does not
fix the block layout. A 16→16 3×3×3 convolution on a 128³ volume is by a wide
margin the most expensive operation in Stage B. Measured: the residual pair at
the finest scale cost **350 ms of a 530 ms** training step — two thirds of `B` —
for 6% more parameters. Residual blocks are kept at every stride-2 stage.

---

## 5. Sequencing, stated rather than skipped

### 5.1 `B` is pretrained by a separate stage, but run 1 trains it jointly

§4 has `B` pretrained and then given a lower learning rate. `scripts/train.py
boundary` implements the pretraining — masked-volume reconstruction, the
label-adjacency boundary map, and `|∇I|` regression — and
`train.stage_b.boundary_checkpoint` plus `boundary_lr_scale` wire it in.

The first end-to-end run trains `B` from scratch alongside the carver. The two
mandatory §7 tests (prompt-only carver, image replacement) are what establish
that the image is used at all; if they pass with `B` from scratch, pretraining is
an improvement to measure rather than a prerequisite to assume. Both are reported.

### 5.2 The optional contrastive term is not implemented

§4 lists it as "optionally". The other three objectives are.

---

## 6. Deletions from the previous branch

The proposal replaces an architecture rather than extending it. Removed here, and
recoverable from git history:

| removed | why |
|---|---|
| the attention Stage B: `RelationPrompt`, `StructureEncoder`, `Evidence`, `Intersection`, FiLM decoder | §8: "no name embedding, pair embedding, or slot embedding after Stage A". The direction is already the shape of `F_i`. |
| `SelectionHead` and `model.stage_b_selection` | §8: "a candidate list is the other method" — `blueprint_confrontation.md` |
| `occupancy_mode` and every arm of it | §8: "There is no `occupancy_mode`" |
| the world-coordinate grid in the encoder and decoder | §6: no coordinate grid in `B` or the carver |
| target-first prompt generation, `data.anchor_pool` | measured leak: the unordered anchor set alone recovered the target 98.9% of the time against 94.5% for solving the conjunction, so ignoring the prompt strictly beat reading it. Gone rather than configurable. |
| rotation augmentation for Stage B | §5 gives Stage B one augmentation, the direction flip. It also makes Stage A's output a function of the scene alone, which is what the anchor cache relies on. Stage A keeps its rotations. |
| `src/synthetic.py`, `scripts/generate_data.py` | the experiment in §7 is on MRI; the synthetic corpus is a different one |
| `tests/test_reference_parity.py` | §8: "Checkpoints of the attention-based Stage B do not load here." The test skipped in this clone anyway, so a green suite never meant what it looked like. |
| `docs/method/`, `docs/experiments_plan.md` | they specify and plan the architecture above |
| `notebooks/` — `demo.py`, `diag_model.py`, `diag_separability.py` | the first two drive the deleted Stage B; the third measures the synthetic corpus's intensity separability. `diag_corpus.py` survives as `scripts/corpus_report.py`, because §7's reporting rules need the prompt-blind floor it computes. |
| dead helpers: `widths_for`, `depth_for`, `world_grid`, `pool_to`, `masked_pool`, `mask_geometry`, `check_scene`, `data.margin`, `data.unique_only`, `train.selection_weight` | nothing in the new forward or the new corpus reads them |

**What was kept and why.** `src/mri.py` and `scripts/import_mri.py` build the
corpus §7 runs on; `scripts/rebuild_manifests.py` changes a prompt knob without
rewriting a NIfTI; `scripts/corpus_report.py` computes the floor every Dice is
read against. `anchor_source: oracle` is kept as an explicitly labelled
diagnostic because §2's gate mandates a ground-truth-anchor path and because
without it a bad number cannot be attributed to Stage A rather than to the
architecture — §3.1 of `results.md` is that attribution being made.

---

## 7. Two things worth knowing before reading a result

### 7.1 The null head's ceiling is a property of its inputs

`where_mass` alone separates "the clauses name one structure" from "the clauses
name none" at **AUC 0.848** on this corpus, at every `τ`. The mapper cannot see
which regions hold tissue, so a roomy conjunction that happens to be empty looks
exactly like a valid one — and §3 forbids showing it the MRI, for the right
reason. A trained head at ~0.85 is working as well as its four numbers permit;
that is not evidence the architecture is sound, and it is not a bug to fix by
widening the MLP. The gate script prints it.

Consequence for inference: §3 says "invalid empties the mask". At a 0.5 threshold
that would empty a sizeable share of valid prompts. `scripts/evaluate.py` reports
Dice both gated and ungated and sweeps the threshold, so the cost is visible
rather than absorbed.

### 7.2 Two populations, two gates, two names

`scripts/gate_mapper.py` measures the §2 gate from **ground-truth** centroids —
that is what "ground-truth anchors used only as a check" means, and it is a
property of the corpus and `tau`. `scripts/evaluate.py` measures the same
quantity from the centroids the model actually used, and therefore reports it as
`gate_fraction_predicted` (or `_oracle`). They are different numbers and must not
share a name in a table.

Likewise, the two per-epoch transfer curves are named `val:targets.val` and
`val:targets.test`: both are drawn from **val-split subjects**, restricted to a
held-out class set. The test split itself is untouched until
`scripts/evaluate.py --split test`.

### 7.3 `permute_both` no longer means what it meant

`where_raw = F_0 · F_1 · F_2` is a product and therefore **exactly**
permutation-invariant. The only order dependence left anywhere in Stage B is the
carver's `cat`. `CLAUDE.md` used to read a flat `permute_both` as evidence that
slot order carried no information; here it is close to a tautology. It is still
run, because §7 asks for it, but a null result is a much weaker statement than it
was under the attention architecture.
