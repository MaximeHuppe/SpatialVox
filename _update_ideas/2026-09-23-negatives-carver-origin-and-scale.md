# Held-out structures as negatives, where the error comes from, and whether 128³ needs a bigger Stage B

**Date:** 2026-09-23 · **Type:** discussion. **No code, config or run was changed.** · Context: `2026-09-23-b2-mri-no-generalisation-analysis.md`. On real MRI, B2 paints a trained neighbour instead of the held-out target: caudate → accumbens or pallidum; hippocampus → thalamus, pallidum or amygdala.

The hypothesis this note starts from (§4.1 of that analysis): **the held-out structures are trained as background, next to trained targets, in fixed anatomy.** It is a hypothesis, not a proven cause.

## 1. Does this mean the model cannot learn a structure it has never seen?

No. It cannot **under the current training signal**. That is a property of the training setup, not of the architecture.

- **What the training currently teaches.** The mask target is "target = 1, everything else = 0". "Everything else" includes caudate, putamen and hippocampus tissue whenever it falls near a training target. Because the anatomy is fixed, it always falls there, at the same place.
- **What the carver learns from that.** Not only "paint the accumbens", but also "the tissue right next to the accumbens is never paintable". That second lesson targets exactly the held-out structures.
- **In deployment the issue splits in two:**
  - **An absent target, e.g. a lesion:** its tissue does not exist in the training scans, so nothing taught the carver to avoid it. The problem does not arise.
  - **A present but unlabelled target, e.g. an unannotated nucleus:** it is label 0 in every training scan, so it gets the same negative supervision. **The issue is real in practice, not only in this benchmark.**

A fix must therefore make the carver learn "paint whatever structure the region points at", instead of "paint one of my 8 classes, and nothing that touches them".

## 2. Ways around it, from quickest to most thorough

1. **Ignore held-out-class voxels in the mask loss** (weight 0).
   - Makes "never supervised" literally true. CLAUDE.md §2 holds, because the class split is training-time knowledge, not a model input. Small code change.
   - **Limit:** it only works because the held-out classes are known. It does not solve the present-but-unlabelled case. Treat it as a **diagnostic**: if transfer jumps, the hypothesis is confirmed.
2. **Only trust negatives that are clearly negative.**
   - Supervise the target and, as background, only voxels that are either far from the region, or labelled as *another trained* structure.
   - Unlabelled or held-out tissue near the region counts as "unknown", with weight 0.
   - This is the general form of fix 1, and it also holds in deployment.
3. **Many more targets**, so that no single "don't paint X" can be memorised.
   - Add every other labelled region as a target-only class: the full aseg, or the 181 wmparc regions. Stage A never needs to know them, because targets are never named.
   - With dozens of target shapes the carver cannot keep one template per class, and "paint the structure in the region" becomes the only strategy that works.
4. **Self-supervised pseudo-targets.** The most elegant bypass.
   - Generate class-free segments from the image itself (supervoxels, intensity clusters) and train the carver to paint "the segment the region points at".
   - That gives unlimited target diversity with no labels. Caudate-like tissue then appears as a *target* in training, without the caudate class ever being supervised.
   - It is also the closest emulation of the lesion claim.
5. **Break the fixed anatomy.**
   - Elastic deformation, or compositing real structures into new positions: a real-MRI version of what made the synthetic corpus work.
   - It conflicts with a recorded decision (no spatial augmentation in Stage B, which the anchor cache relies on). Stage A would have to run live, and the change would need to be recorded explicitly.
6. **Class-free grouping in `B(I)`** (longer term).
   - An instance-contrastive objective over all labelled structures, with no class ids, so that the features separate *any* two touching structures.
   - The carver then paints "the group at the region" instead of recognising a class.

## 3. Is the error really coming from the carver?

It *shows up* in the carver's output. It does not necessarily *originate* there. Three components are entangled:

| component | its share | evidence |
|---|---|---|
| **The training signal** (what the loss rewards) | probably the main one | the wrong answers are exactly the trained neighbours |
| **`B(I)`** | possible | it is trained end to end through only 8 classes, so its features may have become "accumbens detectors". That knowledge would sit in B, not in the carver |
| **The mapper** | contributes | placement is right (target centre in the region 86–90% of the time), but only ~30% of a held-out target is covered, so the carver must extend the mask from the image alone, and falls back on the templates it knows |

Three cheap tests would locate it:
- **Upper bound with a body-covering region.** Give the carver a dilated true-target region in place of `where_raw`. This is a label leak as a method, so it is only a diagnostic.
  - Still paints the accumbens → the carver or `B(I)` is at fault.
  - Now paints the caudate → the mapper's coverage is the bottleneck.
- **A linear probe on `B(I)`.** Check whether B2's features separate caudate voxels from accumbens voxels (and hippocampus from amygdala). If they cannot, the carver never had a chance, and the fix belongs in B or in the training signal.
- **A frozen, class-free `B`.** The earlier pretrained-`B` run was fine-tuned rather than frozen, and trained under `mask_on: all`, so this was never properly tested.

## 4. Should `B(I)` and the carver grow for 128³?

Probably not first, but one asymmetry is worth fixing.

**What changed between the corpora and what did not.** Measured in voxels, targets are about the same size on both: ~1,600–2,200 voxels on MRI at 1.25 mm, ~2,300–3,200 on the synthetic corpus at 64³. Both modules also have the same receptive field in voxels: `B(I)` about 39, the carver trunk about 19. What does differ:

| | MRI-like synthetic (64³) | real MRI (128³) |
|---|---|---|
| field of view | 64 mm | 160 mm |
| target shapes | compact | elongated: the caudate is 40–60 voxels long, the hippocampus curved |
| deepest level of `B(I)` | 16³ | 32³, so B sees relatively less of each structure |
| edges between target and neighbour | clear | weak: caudate, putamen and accumbens are one continuous body of grey matter |
| Stage A scaled up? | 4 stages, 8M | **yes**: 5 stages, 17M, to keep an 8³ bottleneck |
| Stage B scaled up? | — | **no**: same `boundary_widths [16, 32, 32]`, carver width 16 × 2 blocks |

**Against growing it now:**
- Trained classes reach 0.77, close to Stage A's own Dice for the same structures (~0.80), so capacity is not what fails.
- The same architecture transferred on the synthetic corpus.
- With only 8 target classes, more capacity mainly means better memorised templates.

**For one targeted change:**
- The elongated, weakly bounded MRI targets need longer-range context to follow the structure and decide where it ends.
- Stage A received an extra level for 128³; Stage B did not.
- The cheap way to add range is **one more coarse level**:
  - `boundary_widths: [16, 32, 32, 64]`, going down to 16³ and roughly doubling B's receptive field. The extra level costs little, because the full-resolution layers dominate the cost (`SpatialVox.md` §11).
  - Or one stride-2 level in the carver's trunk.
- Adding depth or width *at full resolution* is expensive and not justified by the evidence.

**Cheapest check first:** train Stage B on the MRI resampled to 64³. That reproduces the geometry of the setup that transferred, with no architecture change.
- Held-out rises noticeably → scale matters, and the extra coarse level is worth adding at 128³.
- It does not → scale is not the issue.

## 5. Suggested order (none launched; one change per run, against B2, two seeds each)

1. **Loss masking of held-out classes** (§2.1). The fastest confirmation or refutation of the hypothesis.
2. **The two offline diagnostics** (§3): the body-region upper bound, and the linear probe on `B(I)`.
3. **Depending on those results:**
   - careful negatives (§2.2), many targets (§2.3) or pseudo-targets (§2.4) for the real solution;
   - 64³ resampling, or the extra coarse level (§4), only if scale shows up.
