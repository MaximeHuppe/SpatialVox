# Which module to change, and why — an audit of `B(I)`, the carver and the mapper on real MRI

**Date:** 2026-09-23, 18:30 · **Type:** analysis and proposals. **No code, config or run was changed.** · Code: commit `9af745f`, whose `src/` is identical to `d14f201`. Checkpoints analysed: B2 `runs/mri-mask-valid-seed1/best.pt` (real MRI, epoch 17) and B0 `runs/mask-valid-seed1/best.pt` (synthetic-mri). Scripts are in the session scratchpad: `geometry_analysis.py`, `bi_probe*.py`, `mapper_variants.py`, plus the inline "fine term" measurement.

The questions (the user's):
1. Which module is most worth changing, among the BoundaryEncoder, the PositionalMapper3D and the Carver? And what is the most elegant way to improve generalisation?
2. Is the BoundaryEncoder strong enough to find edges on such hard structures? It is not pretrained. Would pretraining create discrepancies?
3. The carver looks shallow: 25 channels in, one channel out almost immediately. Is there a more elegant way to align and combine the information of all the modules?
4. Is the mapper's way of defining the region valid for MRI, where no anchor is perfectly aligned with its direction and a relation is usually a mixture of all three axes? And do touching structures matter?

---

## 0. Answers in short

1. **Change the carver, and specifically how it fuses the image with the geometry.** Four measurements point there:
   - The image features already contain what is needed to separate the held-out target from the neighbour the model paints instead: a linear probe on B2's `B(I)` reaches AUC **0.94–0.96**.
   - The mapper places the region correctly: the target's centre is inside it 86–90% of the time.
   - The carver has a **prompt-independent** full-resolution term. On MRI it scores the 8 trained structures at −0.5 to −5 logits and every held-out structure at **−11 to −21, below background**, in every prompt.
   - On the synthetic baseline B0, the same term is class-agnostic: all 16 shapes sit at −3 to −6, background at −13.

   That term is where "recognise and recall" lives. The elegant fix is to make the fine-scale decision **conditioned on the prompt** instead of fixed.
2. **`B(I)` has enough capacity. What it is taught is the problem.**
   - The hard boundaries (caudate/accumbens, hippocampus/amygdala) are **invisible in intensity**: d' 0.02–0.22, intensity AUC 0.50–0.59.
   - B2's `B(I)` separates them only because its class supervision forced it to.
   - A **class-free** pretrained `B` separates caudate/accumbens only partly (AUC 0.79–0.89), and **hippocampus/amygdala not at all** (0.62–0.64, the same as raw intensity).
   - Pretraining does create discrepancies (§4.3). It is not the lever for transfer.
3. **Yes: the fusion is one early concatenation, a 19-voxel trunk, and a fixed linear read-out of `B(I)` at full resolution.** §5 proposes a prompt-conditioned fusion (FiLM gating at every scale), a multi-scale trunk, and a seed-prototype read-out that is class-agnostic by construction.
4. **The rule is internally consistent, but brittle on MRI, and a centroid rule cannot describe touching, extended bodies.**
   - Median clause angle is 35–39° from its axis (45° is the boundary). 64% of prompts hinge on a clause with a margin under 2 mm.
   - With Stage A's predicted anchors, 16–21% of prompts no longer name exactly the target.
   - Only 22% of a held-out target's voxels satisfy all three clauses.
   - MRI targets touch an anchor in 64–79% of prompts, and always touch a trained structure.
   - No single region definition gets both a high centre-gate and high coverage (§6.4). Keep the region as a **seed and prior**, not as the body.

---

## 1. The run these answers are about: B2, real MRI, `mask_on: valid`

The run is at epoch 29 of 30; `best.pt` is epoch 17 (selected on trained classes). The watcher runs the full evaluation when training ends, and it will be appended here.

| epoch | trained | held-out val (floor 0.110) | held-out test (floor 0.063) | held-out val empty | held-out val centroid |
|---|---|---|---|---|---|
| 0 | 0.501 | 0.079 | 0.013 | 12% | 15 mm |
| 9 | 0.774 | 0.017 | 0.002 | 16% | 24 mm |
| 17 (`best.pt`) | **0.794** | **0.022** | **0.012** | 9% | 25 mm |
| 28 | 0.793 | 0.029 | 0.010 | 4% | 25 mm |

The model reads the prompt (flip drop +0.64) and almost never stays silent now. Yet on held-out prompts it paints small pieces of a **trained neighbour**: caudate → accumbens or pallidum; hippocampus → thalamus, pallidum or amygdala. The diagnostic is in `2026-09-23-b2-mri-no-generalisation-analysis.md`.

---

## 2. What was measured

### 2.1 Relation geometry — is the rule sound on MRI? (`geometry_analysis.py`, 20 val subjects, ground-truth centroids unless stated)

| | MRI trained | MRI held-out val | MRI held-out test | synthetic-mri (all three) |
|---|---|---|---|---|
| clause angle from its axis, p10 / median / p90 (45° = boundary) | 11 / 35 / 47° | 15 / **39** / 48° | 19 / 31 / 43° | 14–18 / 34–36 / 46–47° |
| clauses within 5° of the diagonal (> 40°) | 31% | **47%** | 20% | 32–35% |
| clause margin, median | 4.7 mm | 3.6 mm | 6.4 mm | 6.0–6.3 mm |
| clauses with margin < 2 mm | 27% | 27% | 14% | 18–21% |
| prompts whose **weakest** clause is < 2 mm | **64%** | **64%** | 32% | 44–52% |
| anchor→target distance, median | 19 mm | 24 mm | 23 mm | 28–31 mm |
| clauses that **flip** with Stage A's predicted anchor centroids | 5.5% | 4.9% | 3.4% | — (no cache) |
| prompts still naming **exactly** the target with predicted centroids | **79%** | **83%** | **84%** | — |
| target voxels satisfying each clause (the centroid rule applied voxel by voxel) | 73% | 63% | 64% | 76–81% |
| target voxels satisfying **all three** clauses | 38% | **22%** | **23%** | 42–51% |

### 2.2 Touching structures

| | MRI trained | MRI held-out val | MRI held-out test | synthetic-mri |
|---|---|---|---|---|
| target surface touching another labelled structure | 38% | 36% | 27% | **0.6–1.4%** |
| target touches one of its anchors | 78% | 64% | 79% | 19–25% |
| target touches a trained target class | 50% | **100%** | **100%** | 31–46% |
| target voxels removed by the anchor exclusion (predicted anchors > 0.5) | 4.1% (24% of prompts lose > 5%) | 1.2% | 1.0% | — |

### 2.3 Can the image separate the confused pairs? (`bi_probe.py`)

**Method:**
- For each pair, take the voxels within 3 voxels of the shared interface, on both sides.
- Fit a logistic read-out on 10 val subjects and test it on the other 10.
- Three feature sets: raw intensity alone; intensity plus local mean, std and gradient; or the 16 `B(I)` channels.

| pair (touching) | intensity d' | intensity AUC | intensity + local stats | **B2's `B(I)`** (class-supervised) | **class-free pretrained `B`** (archived `boundary-seed1`) |
|---|---|---|---|---|---|
| Left-Caudate / Left-Accumbens | 0.22 | 0.59 | 0.61 | **0.94** | 0.79 |
| Right-Caudate / Right-Accumbens | 0.06 | 0.50 | 0.55 | **0.94** | 0.89 |
| Left-Putamen / Left-Accumbens | 0.54 | 0.67 | 0.72 | **0.96** | 0.78 |
| Left-Putamen / Left-Pallidum | 1.38 | 0.85 | 0.90 | 0.94 | 0.89 |
| Right-Putamen / Right-Pallidum | 1.64 | 0.89 | 0.93 | 0.95 | 0.89 |
| Left-Hippocampus / Left-Amygdala | **0.03** | 0.51 | 0.64 | **0.95** | **0.62** |
| Right-Hippocampus / Right-Amygdala | **0.02** | 0.54 | 0.64 | **0.96** | **0.64** |
| Left-Caudate / Left-Lateral-Ventricle (a visible edge, for reference) | 3.29 | 0.97 | 0.98 | 0.97 | 0.98 |
| Left-Thalamus / Left-VentralDC (trained / anchor-only) | 0.21 | 0.58 | 0.61 | 0.94 | 0.82 |

### 2.4 The carver's prompt-independent fine term

At full resolution the carver adds `W_B · B(I)`: the second half of its 1×1 head, applied to `B(I)`. That term does **not** depend on the prompt. Below is its mean inside each structure (logits; higher means pushed towards "paint" in every prompt).

| B2, real MRI | | B0, synthetic-mri | |
|---|---|---|---|
| thalamus (trained) | −0.5 | tetrahedron (trained) | −2.9 |
| amygdala (trained) | −1.8 | banana (**held-out**) | −3.7 |
| accumbens (trained) | −3.4 / −4.0 | triangular_prism (**held-out**) | −4.3 |
| pallidum (trained) | −4.2 / −5.0 | cross (**held-out**) | −4.5 |
| **background** | **−9.2** | trained shapes | −4.1 to −5.5 |
| caudate (**held-out**) | **−11.0 / −13.4** | hourglass / crescent (**held-out**) | −5.2 / −5.3 |
| hippocampus (**held-out**) | **−15.7 / −17.5** | hollow_cylinder (**held-out**, weakest at 0.53) | −6.2 |
| putamen (**held-out**) | **−20.2 / −20.8** | **background** | **−12.8** |

### 2.5 Region definitions compared offline (`mapper_variants.py`, 100 held-out prompts per population, predicted anchors, region = `where > 0.05`)

| region | held-out val: centre inside / covered / precision / structures in region | held-out test: centre inside / covered / precision / structures in region |
|---|---|---|
| **current** (centroid pyramid, `tau` 0.5 mm) | **88%** / 33% / 11% / 5 | **90%** / 29% / 7% / 7 |
| softer (`tau` 2.0 mm) | 61% / 54% / 9% / 8 | 76% / 49% / 6% / 8 |
| whole anchor (pyramid averaged over anchor voxels) | 7% / 62% / 8% / 7 | 17% / 64% / 5% / 8 |
| current, grown by 8 voxels | 99% / **92%** / 3% / 14 | 100% / **97%** / 2% / 15 |

---

## 3. Question 1 — which module?

**The carver**, and within it, the way it turns image features into a decision.

1. **The information is present at its input.** B2's `B(I)` separates each confused pair at AUC 0.94–0.96 (§2.3), and the region contains the target's centre 86–90% of the time (§2.5). The carver receives what it needs to paint the caudate rather than the accumbens.
2. **It has a mechanism that ignores the prompt at exactly the resolution where the boundary is decided.**
   - The full-resolution term `W_B·B(I)` is one fixed linear map for all prompts. The prompt reaches full resolution only through an upsampled coarse logit.
   - On MRI that fixed map learned **class identity**: trained structures near 0, held-out ones 6–16 logits *below background* (§2.4). A held-out target therefore starts every prompt with a 10–15-logit handicap that the coarse term must overcome.
   - The measured result: masks of 12–22% of the target's volume, painted on the trained neighbour.
3. **The synthetic baseline shows the mechanism is what separates success from failure.** On B0 the same fixed map learned "structure against background" (all shapes −3 to −6, background −13), and held-out transfer reached 0.72. The weakest held-out shape, hollow_cylinder, has the lowest fine score of all shapes.
4. **Why MRI teaches class identity and synthetic does not.**
   - On synthetic, structures barely touch (surface contact 1%). A negative voxel is almost always background, so the map learns "tissue vs background".
   - On MRI, every held-out target touches a trained target (100%), with a third of its surface in contact, and the touching tissue is intensity-identical (d' 0.02–0.22). To paint the accumbens and not the adjacent caudate, the only solution available to a *prompt-independent* map is to learn "accumbens tissue is positive, caudate tissue is negative". This is the negative-supervision hypothesis (`2026-09-23-negatives-carver-origin-and-scale.md` §1), now located in a specific term.

The mapper matters too (§6), but its limits cap how *much* of the body can be recovered. They do not explain why the carver picks the wrong structure. `B(I)` is not the bottleneck on capacity (§4).

**Why this is the elegant lever.** One design decision, a fixed full-resolution read-out, converts touching-structure supervision into class memorisation. Making that read-out prompt-conditioned attacks the mechanism, keeps every CLAUDE.md invariant, and costs few parameters.

---

## 4. Question 2 — is `B(I)` strong enough? Should it be pretrained?

### 4.1 The hard MRI boundaries are not edges

- The boundaries the model gets wrong have **no intensity edge**:
  - hippocampus / amygdala: d' 0.02–0.03, intensity AUC 0.51–0.54;
  - caudate / accumbens: d' 0.06–0.22, AUC 0.50–0.59.
- Adding local mean, std and gradient only reaches 0.55–0.64. Where the boundary is visible (caudate / lateral ventricle, d' 3.3), everything separates it (0.97).
- **So "edge detection" is the wrong frame.** These are anatomical conventions (the striatum is one grey-matter body; the amygdala–hippocampus junction is a continuum). They can only be drawn from **context and prior knowledge of the anatomy**, not from a local edge.

### 4.2 `B(I)` has the capacity, and uses it only when taught

- **B2's `B(I)`** (trained through the 8 supervised classes) separates every confused pair at 0.94–0.96. With a ~39-voxel receptive field, 16 channels are enough to encode *where you are* relative to visible landmarks (ventricles, white matter), and that is how it separates pairs that intensity cannot.
- **The class-free pretrained `B`** separates caudate/accumbens at 0.79–0.89. It learned some of this, because its pretext target, the unsigned label-boundary map, marks that interface. It separates hippocampus/amygdala at **0.62–0.64**: no better than intensity.
- **Conclusion.** Capacity is sufficient. What `B(I)` learns to separate is exactly what the supervision demands. For a genuinely new structure whose border with a known one is invisible, no class-free image cue exists, and only **geometry** (the prompt) can say where it stops.

### 4.3 Your instinct about pretraining is right: the discrepancies

1. **The objective does not match the task.** The pretext (put back blanked cubes, predict the label-boundary map, regress `|∇I|`) rewards intensity edges. The task needs *grouping across invisible borders*. A pretrained `B` is good where it is not needed (visible edges, which the task learns anyway) and weak where it is (§4.2: hippocampus/amygdala 0.62).
2. **The label-boundary pretext includes the held-out structures' outlines.** That is class-free, but it *is* supervision of where caudate and hippocampus begin (0.79–0.89 on caudate/accumbens comes from there). It is allowed by CLAUDE.md §2, but it weakens the "never supervised" claim, and it must be declared if used.
3. **Fine-tuning drift.** Pretrained at `boundary_lr_scale` 0.1 and then trained end to end, `B` is re-specialised by the 8-class loss. The pretrained-`B` run (archived) gave held-out Δ +0.001: the pretraining was overwritten.
4. **Freezing prevents drift but locks in the mismatch.** A frozen class-free `B` cannot provide the context-based separation (§4.2), and the carver cannot adapt it.
5. **The reconstruction target dominates feature scale.** Reconstruction pushes the 16 channels to encode intensity. The carver's fixed read-out then inherits intensity-heavy features, which are useless across invisible borders.
6. **Normalisation and corpus mismatch.** If the pretext corpus or normalisation differs from the task's (for example synthetic scenes, or no brain z-score), feature statistics shift at the start of Stage B.

**When pretraining would help.** With a *grouping* objective that survives into Stage B: an instance-contrastive embedding over all labelled structures, with no class ids, frozen or regularised. For invisible borders that amounts to learning anatomical context from labels. It is legitimate only if the held-out classes are excluded from the pretext as well.

### 4.4 How to evaluate `B(I)` changes (no training needed for the first two)

- **The pair probe of §2.3** on any candidate `B`. It is the fastest test of whether a `B` can separate the confused pairs.
- **The fine-term audit of §2.4** after any Stage B run. Do held-out structures still score below background?
- **In training,** against B2, two seeds each:
  - frozen class-free `B`;
  - `B` with an instance-contrastive pretext (held-out classes excluded);
  - `B` from scratch (= B2).

  Read held-out Dice per class, the share of painted voxels on another structure, and the fine-term gap between held-out and trained structures.

---

## 5. Question 3 — the carver: what is shallow about it, and how to fuse better

### 5.1 What it does today (`src/models.py`, `Carver`)

```
x = cat[B(I) 16, A_i 3, F_i 3, where_raw, log where_raw, log where_mass]   [B, 25, D³]
stem   = ConvBlock(25 → 16, 3×3×3, stride 2)                             [B, 16, (D/2)³]   ← the ONLY place geometry and image meet
blocks = 2 × ResBlock(16)                                                [B, 16, (D/2)³]   ~19-voxel receptive field
logits = up( W_f · blocks + b )  +  W_B · B(I)                           [B, 1, D³]
          └ prompt-dependent, coarse ┘   └ prompt-INDEPENDENT, full resolution ┘
heatmap = 1×1(blocks) → soft-argmax → centroid
```

The problems, in order of importance:
1. **The full-resolution decision ignores the prompt.** `W_B·B(I)` is identical for every prompt, so boundary placement at the finest scale cannot depend on *which* structure is asked for. On MRI it became a class-identity map (§2.4).
2. **Fusion happens once, early, and by concatenation.** Geometry and image meet in a single 3×3×3 convolution. After that there are 16 mixed channels, and geometry can no longer *re-select* image features at depth.
3. **The trunk has one scale and a small receptive field.** 19 voxels at full resolution (24 mm on MRI), while the caudate is 50–60 mm long and weakly bounded. The body's extent is decided locally and completed by templates.
4. **Its capacity is tiny.** 38k parameters against 229k for `B(I)`. That is fine for synthetic, but it leaves no room to reason about *which* of the 5–7 structures in the region (§2.5) is the target.

### 5.2 Better ways to fuse, all within CLAUDE.md §1

None adds a coordinate grid, a name or a label. The carver still sees only `B(I)` and the mapper's fields.

**A. Prompt-conditioned read-out (FiLM gating). The smallest change that removes the mechanism.**
- Replace `W_B·B(I)` with `Σ_c γ_c(g)·B_c(I)`, where `γ(g)` is a per-voxel gain computed from the geometry channels `g` by a 1×1 (or small) convolution.
- The fine term then depends on *where the prompt points*: the same tissue can be "paint" in one prompt and "don't" in another. There is no longer a fixed class-identity map to learn.
- Cost: about 16 × 9 extra weights.
- Evaluate: the fine-term audit should show no held-out/trained gap; then held-out Dice and the share of painted voxels on another structure.

**B. Multi-scale trunk with geometry injected at every scale.** A 3-level U-Net at D/2, D/4 and D/8.
- The geometry fields are downsampled and re-injected at every level, as FiLM or concatenation.
- The receptive field grows to about 80–100 voxels, enough to follow a caudate.
- Cheap: the added levels are coarse.
- Evaluate: held-out Dice per class, predicted/true volume (should rise from 0.12–0.22), centroid error.

**C. Seed-prototype read-out: "paint what looks like the tissue at the seed". The most elegant option, and class-agnostic by construction.**
- The heatmap already predicts a centroid inside the region. Read `B(I)` there to get a prototype `z` (a soft-argmax-weighted average of `B(I)`).
- Define the fine term as the similarity `⟨B(I)(p), z⟩ / τ`, gated by the geometry prior.
- The decision becomes relative: "the same tissue as the target's centre, inside the region". It no longer asks "one of my 8 classes". This is how one-shot and point-prompted segmenters generalise.
- Evaluate: as for A, plus the prototype's agreement with the true target (similarity at target voxels against neighbour voxels).
- **Caveat:** across an *invisible* border (hippocampus/amygdala), the tissue at the seed looks like the neighbour's. The prototype stops at visible borders; the geometry prior must stop it at invisible ones. That is why C is combined with A or B, not used alone.

**D. Learned affinity propagation.** A differentiable flood fill from the seed.
- `B(I)` predicts affinities between neighbouring voxels, and the mask is grown from the seed by iterated propagation (a random walk, or a few ConvGRU steps), bounded by the region prior.
- Affinities are class-free (trainable on *all* labelled boundaries without class ids). The growth follows the body even outside the narrow region.
- More machinery than A–C: a second step, if A–C are not enough.

**Recommended build order:**
1. A alone: the minimal, falsifiable change.
2. A + B.
3. A + B + C.

Each against B2, two seeds, same evaluation.

### 5.3 How to evaluate the carver changes

| measurement | what it tells | where it comes from |
|---|---|---|
| held-out Dice per class, gated and ungated | the goal | `scripts/evaluate.py` |
| share of painted voxels on the target / another structure / background, and which structure | whether recall is gone | `failure_modes_mri.py` |
| predicted / true volume on held-out targets | whether masks extend over the body | `scripts/evaluate.py` |
| fine-term gap: mean fine logit on held-out structures minus on trained ones | whether class identity is still being learned (B2: about −12) | §2.4 script, per checkpoint |
| trained-class Dice | that nothing was lost (B2: 0.794) | per epoch |
| counterfactuals, image replacement | prompt reading, and use of the image | `scripts/evaluate.py` |
| the same on synthetic-mri (B0 method) | that the change does not break what works | `configs/synthetic-hard.yaml` |

---

## 6. Question 4 — is the mapper's region valid for MRI?

### 6.1 What the rule is

- The corpus names a relation by the **dominant axis of the centroid-to-centroid offset**: the axis with the largest `|Δ|` wins (`classify`). Lateral/medial additionally compares distances to the mid-sagittal plane.
- The mapper writes the same rule voxel by voxel, as a 45° square pyramid: `F_i = σ(margin/τ)`, where `margin = primary − max(other two)` in mm (`src/mapper.py::_margin`).
- The mapper and the labels therefore agree by construction; with ground-truth anchors the gate is 0.97. The question is whether the rule itself suits MRI.

### 6.2 Oblique relations: the rule is brittle on MRI

- **Relations are mostly diagonal.** The median clause is 35–39° off its axis, and 31–47% of clauses sit within 5° of the 45° boundary (§2.1). "Superior to X" usually means "somewhat more superior than anterior".
- **Prompts hinge on a thin margin.** In 64% of trained and held-out-val prompts, the weakest clause has a margin under 2 mm, about 1.5 voxels.
- **Stage A's anchor error is enough to break it.** Stage A's centroids are off by a median of 0.84 mm (CLAUDE.md §7). With the predicted anchors:
  - **5% of clauses change direction;**
  - **16–21% of prompts no longer name exactly the target.** From the model's own view, one prompt in five describes something else, or nothing. That is an error floor the carver cannot fix.
- **Synthetic has the same angles but larger distances**, hence larger margins (6 mm) and fewer fragile prompts (44–52% against 64%).
- **The rule is a centroid rule, and MRI bodies are extended and oblique.** Only 22% of a held-out target's voxels satisfy all three clauses when the rule is applied voxel by voxel (42–51% on synthetic). That is why the region covers only about 30% of the body (§2.5): it is not a thresholding artefact but the rule itself.

### 6.3 Touching structures: yes, it matters, in four ways

1. **The region always contains the neighbours.** Held-out targets touch a trained target in 100% of prompts, so a region wide enough to hold the target's body also holds the neighbour (5–7 labelled structures inside, §2.5). The carver must choose, and it chooses the familiar one (§3).
2. **Negative supervision lands on held-out tissue** exactly at the contact, which is what trained the class-identity fine term (§2.4, §3).
3. **The anchor exclusion eats the target.** When the target touches an anchor (64–79% of prompts), Stage A's mask of the anchor bleeds into it. The exclusion removes 1–4% of target voxels on average, and more than 5% in 24% of trained-class prompts.
4. **Close anchors make the wedge narrow.** Touching anchors are 13–24 mm from the target. The wedge's apex sits in the anchor, its width at the target is small, and a small centroid error moves it by a large fraction of the target's size.

### 6.4 No region definition is both precise and covering (§2.5)

| region | centre inside | covered | precision | reading |
|---|---|---|---|---|
| current | 88–90% | 29–33% | 7–11% | a good **locator**, a poor **outline** |
| softer (`tau` 2) | 61–76% | 49–54% | 6–9% | trades location for coverage |
| whole-anchor average | 7–17% | 62–64% | 5–8% | covers bodies, but disagrees with the centroid-based labels, so the gate collapses |
| grown by 8 voxels | 99–100% | 92–97% | 2–3% | contains the target *and* 14–15 structures |

**Reading.** The region cannot do the carver's job. Every widening that covers the body also takes in more structures (up to 15), with precision falling to 2–3%. The mapper is best used as it is designed, *where the centre can be*, plus a wider prior. The body must come from the image, through a carver that can group from a seed (§5, C/D).

### 6.5 What to change in the mapper, and how to evaluate

All of these can be scored **offline**, since the mapper has no parameters. Report, per population:
- centre gate (ground-truth and predicted anchors);
- coverage and precision;
- the number of structures in the region;
- the **flip rate** of clauses under predicted anchors, and under a 1 mm random perturbation;
- the share of prompts still uniquely naming the target under predicted anchors;
- the null-head separability of `where_mass`.

The proposals:
1. **Drop fragile clauses at generation.** Require a margin of at least `m` mm (e.g. 2–3 mm) or an angle of at most 35° per clause. This is a corpus change, consistent with CLAUDE.md §4 (anchor-first and unique still hold).
   - **Expect:** the prompts that are ill-posed under predicted anchors (16–21% today) to drop towards 0.
   - **Measure:** how many prompts remain, how the floors and ceilings move, and B2's held-out Dice restricted to robust prompts. That last number is computable **now**, on B2's predictions.
2. **An angle-based field instead of a millimetre margin.** Use `F = σ((cos θ − cos 45°)/κ)`, where θ is the angle between `p − c` and the clause axis. It is scale-invariant, so near a touching anchor it is not razor-thin in millimetres, and far away it is not over-confident.
   - **Measure:** the gate and coverage against the current field.
3. **Propagate centroid uncertainty.** Average the field over the anchor centroid's uncertainty, for example a Gaussian of 1 mm, or the spread of the anchor mask. The field then equals the *probability* that the clause holds.
   - **Measure:** calibration. The field at the target centre against the empirical flip rate under predicted anchors.
4. **Offer the body-level prior as an extra channel, not a replacement:** the grown or whole-anchor field. The centroid field stays the locator that the labels are defined on. The carver gets both "where the centre can be" and "where the body can be".
5. **Soften the anchor exclusion near the target.** Exclude only where the anchor probability is above about 0.9, or make the exclusion a large but finite, learnable penalty.
   - **Measure:** target voxels lost (1–4% today), and trained-class Dice.
6. **Relabel relations at the body level** (the largest change). Define "superior to X" as "most of the target is above most of X". It is anatomically closer to how the relations read, but it changes the corpus, the floors and the gate definition, and needs a recorded decision.

---

## 7. What to do first (none launched)

1. **Free and offline:**
   - B2's held-out Dice restricted to prompts that are robust under predicted anchors (§6.5-1);
   - the fine-term audit on every future checkpoint (§2.4).
2. **The carver change A** (a prompt-conditioned full-resolution read-out), against B2, two seeds. It is the smallest change aimed at the located mechanism. Success means:
   - no held-out/trained gap in the fine term;
   - less than half of the painted voxels on another structure (59–62% today);
   - held-out val Dice above its 0.110 floor.
3. **Then:**
   - carver B (multi-scale), then C (seed prototype);
   - on the mapper side: the fragile-clause filter and the body prior as an extra channel;
   - evaluated as §5.3 and §6.5 describe.
4. **Keep the two-track evaluation in mind** (novel prompts for all targets, against never-supervised targets): `2026-09-23` discussion of prompt memorisation, 86% exact prompt repeats on MRI val.
