# The positional mapper covers only a third of an MRI target — measurement and ideas

**Date:** 2026-09-23 · **Type:** measurement and proposals. **No code, config or run was changed.** · Code: commit `7f478f7`, whose `src/` is identical to `d14f201`.

## 1. The measurement

**The question** (the user's): on MRI, does the region `where_raw` actually cover the target mask?

**How it was measured.** The mapper was run alone, with no model, on CPU. Script: session scratchpad `coverage.py`.
- **Prompts:** val-split prompts, 200 per population (127 for hippocampus).
- **Region:** `where_raw` built exactly as the model builds it: `tau` 0.5 mm, `min_mass` 1e-6.
- **Anchors:**
  - MRI: Stage A's cached predicted anchors (`phase-a/current`), the ones the model receives.
  - MRI-like synthetic: ground-truth anchors, because no cache exists; predicted anchors are near-oracle there.

| | MRI trained | MRI held-out val (caudate, putamen) | MRI held-out test (hippocampus) | synthetic-mri trained | synthetic-mri held-out val / test |
|---|---|---|---|---|---|
| target voxels inside `where_raw > 0.05` | 46% | **31%** | **30%** | 65% | 52% / 58% |
| target voxels inside `where_raw > 0.5` | 35% | 22% | 22% | 52% | 41% / 45% |
| inside the `L_far` dilation (8 voxels on MRI, 4 on synthetic) | 99.6% | 93% | 97% | 97% | 91% / 93% |
| target centroid inside the region (`> 0.5`, the gate) | 79% | 86% | 90% | 95% | 92% / 91% |
| Dice of the region against the target (`> 0.05`) | 0.18 | 0.15 | 0.10 | 0.27 | 0.26 / 0.26 |
| region volume / target volume | 7.9× | 4.1× | 5.2× | 4.6× | 3.3× / 4.0× |
| precision (share of region voxels that are target) | 19% | 12% | 7% | 19% | 21% / 18% |

**Reading.**
- **The low Dice is expected.** The region is a wedge of every point where a *centroid* could satisfy the clauses, 4–8× larger than the target. **Coverage (recall) is the number that matters.**
- **On MRI, ~70% of a held-out target lies outside the region**, against ~45% on the MRI-like corpus where the baseline transfers (0.72 / 0.78). The carver must paint that part from the image alone, using features supervised on eight other shapes.
- **The MRI held-out structures are elongated or curved.** The caudate is a long arc, the putamen a lens, the hippocampus bends. A 45° pyramid around a centre catches only a slice of them.
- **The missing part is nearby.** 93–97% of each target lies within 10 mm of the region, so the region points at the right place but is too narrow, and it is anchored on centroids rather than on bodies.
- **The gate itself is imperfect on MRI.** 10–21% of prompts have the target centre *outside* the region at `> 0.5`.

**The likely consequence** for B2 (the baseline method on MRI, training now): even with the shy painting removed, held-out masks may come back **too small**. B2's evaluation will show it: compare predicted and true volume on the held-out classes. If they are too small, the mapper's coverage is what caps MRI transfer.

## 2. Constraints every idea must keep (CLAUDE.md §1–§2)

- The mapper reads **only** the three soft anchor masks `A_i` and the three direction ids. It never reads the image, the label volume, names or the target. A coordinate grid may exist **inside** the mapper only.
- What it emits are fields: geometry, never identity. Any new channel changes the carver's input count, so it must appear in `StageB.config` (§8).
- **The centroid pyramid must stay.** The corpus defines a relation by *centroids* (`classify`), and the gate (§6.4) measures exactly that agreement. Every idea below *adds to* or *reshapes* the fields; none replaces the rule the prompts were generated with.

## 3. Ideas, grouped by what they fix

### A. Make the region describe the target's *body*, not only its centre

1. **A body-sized soft dilation of `where_raw`, as an extra channel.**
   - `where_body = maxpool_or_blur(where_raw, r)`, with `r` about a typical target radius (6–10 voxels on MRI).
   - Coverage within 8 voxels is already 93–97%, so this directly hands the carver "where the body can be" and keeps `where_raw` for the centre.
   - Parameter-free and cheap. One new channel: 25 → 26.
2. **Multi-scale fields.** Emit `where_raw` at several temperatures (e.g. `tau` = 0.5, 2 and 8 mm) as separate channels.
   - The sharp field keeps the gate high; the soft ones widen coverage. The carver learns which to trust.
   - Today's single `tau` is a compromise: 0.5 mm gives a gate of 0.97 but narrow coverage; 2.0 mm gave a gate of only 0.65.
3. **An explicit signed distance to the region, in mm, instead of `log where_raw`.**
   - The log is a monotone proxy for "how far outside", but its scale depends on `tau` and saturates at `1e-9`.
   - A distance transform of `where_raw > 0.5` (negative inside, positive outside, clipped to about 20 mm) says it directly, in the units the body extends in.

### B. Make each clause's region follow the anchor's *shape*, not its centroid

4. **Margins from the anchor's extent.** "Superior to the Left-Thalamus" is about the thalamus as a whole, not its centre point.
   - Measure the margin from the anchor's **extreme point along the clause's axis**: a soft maximum of the projection over `A_i`, instead of from `c_i`.
   - For large, elongated anchors (lateral ventricles, brainstem), today's centroid pyramid starts inside the anchor and misplaces the wedge.
   - **Caution:** the corpus rule is centroid-based, so this must be checked against the gate before adoption, or offered as an extra channel.
5. **A cone swept over the anchor mask.** Take the region as the set of points `p` such that `p − q` satisfies the direction for *some* anchor voxel `q`.
   - That is a morphological dilation of `A_i` by the direction's cone.
   - It follows the anchor's full shape, which helps most where anchors are curved or long. It can be approximated with separable directional max-pooling.
6. **A temperature that grows with the anchor's size.** Set `tau_i` in proportion to the spread of `A_i` (its second moment).
   - A big anchor makes the boundary of "superior to it" genuinely fuzzy; a small one keeps it sharp.
   - This propagates Stage A's own uncertainty instead of one global 0.5 mm.

### C. Loosen the conjunction where it is too strict

7. **A softer combination than the raw product**, as an *extra* channel: a soft minimum, or the geometric mean `(F_0·F_1·F_2)^(1/3)`.
   - The product lets a single clause at 0.3 cut a voxel to near zero, and that happens exactly along the body's flanks.
   - Keep the product as it is: it is what the null head's `where_mass` and the gate are defined on.
8. **A wider cone.** The 45° half-angle is a choice, not a measurement.
   - A wider aperture (e.g. 60°), or one per axis (the brain is not isotropic), raises coverage.
   - Sweep it offline with `scripts/gate_mapper.py` and the coverage script, keeping the gate ≥ 0.95.

### D. Measure before training

9. **Add coverage to the mapper's own report.** `scripts/gate_mapper.py` measures only the centroid gate. Add, per class:
   - the target's coverage (recall) inside `where_raw > 0.05`;
   - the coverage inside its dilation;
   - the precision.
   The mapper has no parameters, so **every idea in A–C can be scored offline in minutes, with no training**. Only those that raise coverage without lowering the gate deserve a training run.
10. **An upper bound per population.** "The Dice of the best mask the carver could draw inside the region" (`region ∩ target` against the target) shows how much of the MRI held-out ceiling is set by the mapper, as opposed to the carver.

## 4. Suggested order

1. **B2 first.** Its evaluation will tell whether held-out masks are too small, i.e. whether coverage really caps MRI transfer.
2. **Ideas 9 and 10, offline.** Score ideas 1, 2, 5 and 8 for coverage against the gate on `data/mri`. No training.
3. **Train the best one or two against B2**, on MRI, two seeds, with the same probes and image replacement.

**Most promising a priori:** idea 1 (body dilation) and idea 2 (multi-scale), because they add information without touching the centroid rule. Then idea 5 (the swept cone), for elongated anchors.
