# Are we giving the carver too many maps? — questions and answers

**Date:** 2026-09-23 · **Type:** analysis only. **No code, config or run was changed.** · Code referenced: commit `d14f201` (baseline B0).

The questions:
1. Why does the BoundaryEncoder output 16 channels and not a single map?
2. Why does the carver take both the image features and the `F_i` maps, and not only one of them?

Both come down to one question: are we giving the carver too many maps?

What the carver receives today, per voxel (`src/models.py`, `StageB.forward`):

| channels | what | what it tells the carver |
|---|---|---|
| 16 | `B(I)`, the image features | what the tissue looks like here: edges, their side, texture, intensity |
| 3 | `A_i`, the soft anchor masks | where the three landmarks are, and their shape |
| 3 | `F_i`, one pyramid per clause | how well each clause is satisfied here |
| 1 | `where_raw = F_0·F_1·F_2` | how well all three together are satisfied |
| 1 | `log where_raw`, rescaled | the same, with graded values far outside the region |
| 1 | `log where_mass`, broadcast | how big the whole feasible region is: one number per prompt |
| **25** | | |

These 25 channels enter **only the stem**: one 3×3×3 convolution to 16 channels, 10,800 of the carver's 38,498 parameters. After that the carver works on its own 16 features. In addition, `B(I)` reaches the final 1×1 head directly (`full_resolution_skip`).

---

## 1. Why 16 channels from the BoundaryEncoder, not one map?

**Short answer.** `B(I)` is not a boundary detector. It is the carver's only view of the image, and one map cannot carry what the carver needs to decide *inside or outside the target*.

**What a single edge map loses.**
- **Which side of an edge the structure is on.** An unsigned edge map says "there is a boundary here", not "the structure is on this side". Painting a region means deciding inside against outside, and the carver can only reach part of a structure with its receptive field (about 20 voxels, while targets are 15–30 voxels across). So it needs *region* descriptors, like intensity level and texture, not just *contour* descriptors.
- **Which edge belongs to the target.** Several structures touch inside the region. One map gives every boundary the same value, so the carver could not tell the target's edge from its neighbour's.
- **Graded evidence.** On the MRI-like corpus no single threshold separates the structures (threshold IoU 0.21; 0.08 on real MRI). One intensity-derived map would be close to that threshold's failure. Several channels let the encoder encode combinations of cues.

**Evidence from the runs.**
- **The image carries most of the "what".** In B0, swapping in another scene's image drops held-out Dice from 0.72 to 0.16 and trained-class Dice from 0.96 to 0.17. A single map carrying that much would be a severe bottleneck.
- **A one-map target did not help.** `B` was pretrained against a 1-channel boundary target, reaching boundary Dice 0.633 (`boundary-seed1`, archived). That pretrained `B` gave no transfer benefit (held-out Δ +0.001, `pretrained-b-seed1`, archived). The useful information is richer than an edge map.
- **The "one map" design was tried as a limit case.** Without `full_resolution_skip`, the only path from `B(I)` to the output is a trilinear upsample of the stride-2 trunk. The flag was added because a single low-resolution path destroys the boundary detail the mask is drawn from (`SpatialVox.md` §12.2).

**Is 16 too many?**
- **Capacity: no.** The final head reads the 16 channels with 16 weights, and the stem with 16 × 27 weights. The carver has 38k parameters; `B` itself has 229k (85% of Stage B).
- **Compute and memory: they are the real cost.** The full-resolution 16-channel tensor, and `B`'s 16→16 layers at full resolution, dominate the step time (`SpatialVox.md` §11). That is a reason to try **8 channels**, not 1.
- **Not measured.** No run has varied the width (`model.stage_b.boundary_widths`, first entry). An 8-channel run would be the test, with at least two seeds because the held-out noise is 0.09.

---

## 2. Why both the image features and the `F_i` maps?

**Short answer.** They answer different questions, and each ablation that removed one side failed.
- `F_i` and `where_raw` say **which** structure is meant: the prompt reaches the carver only through them.
- `B(I)` says **how far** that structure extends: the geometry cannot know that.

**Image only (no geometry).**
- By design, `B(I)` sees no prompt (CLAUDE.md §1), and the carver receives no names or directions. With the geometric channels removed, the carver could not know *which* structure to paint. It could only segment "something", and the counterfactuals would stay flat.
- This is exactly what the probes rule out. In B0, pairing anchors with the wrong directions drops held-out Dice to 0.008–0.015, and flipping one direction to 0.08–0.11. The prompt reaches the mask through these channels alone.

**Geometry only (no image).**
- This ablation was run on real MRI (`prompt-only-seed1`, archived). Trained-class Dice fell from 0.78 to 0.52, and held-out Dice stayed at the floor (0.117 against 0.110).
- **The reason:** the region describes where a *centroid* can be, not the structure's body. Only about 36% of a target's voxels lie inside `where_raw > 0.05` (CLAUDE.md §7). Without the image, the mask is the region's shape, a spatial prior.

**So both are needed. The real question is whether all *nine* geometric channels are.**

| channel(s) | redundant? | verdict |
|---|---|---|
| `F_0..2` | No. `where_raw` is their product, which cannot say *which* clause fails or by how much. Near the edge of the region, the individual margins tell the carver which boundary plane it is close to. | keep |
| `where_raw` | It is a function of the `F_i`. But a 3×3×3 convolution cannot compute a product of its inputs, so giving it the product is the cheap way to supply the conjunction. | keep |
| `log where_raw` | It is a monotone copy of `where_raw`. At `tau = 0.5` mm, `where_raw` is almost binary: 1 inside and ~0 everywhere outside. The log is what tells the carver *how far outside* a voxel is, and the target's body lies partly outside. | keep; not strictly redundant |
| `log where_mass` | One number broadcast over space. Under the old `mask_on: all` it was how the carver learned "the region is empty, stay silent". Under `mask_on: valid` that job belongs to the null head, so its main use is gone. | **candidate to remove** |
| `A_0..2` | Not needed for geometry: that is already in `F_i`. Their *shape* identifies which landmarks were named, and the unordered anchor set alone recovers the target 67.8% of the time on real MRI. That is a route to "recognise and recall" instead of solving the relation. The anchor exclusion uses the masks without them being carver inputs. | **main candidate to remove** (`carver_sees_anchors: False`), never actually tested |

**Answer to "too many maps?"**
- **Not in capacity or cost.** The 9 geometric channels only reach one 3×3×3 convolution, and removing all of them would save about 3,900 parameters.
- **Possibly in what the maps let the carver do.** The anchor masks are a potential shortcut, and `where_mass` is now mostly redundant with the null head. Those are the maps worth removing. The per-clause `F_i`, the product and its log each carry information the others do not.

---

## If these are to be tested (not done; for a later change)

In order of expected information:
1. `model.stage_b.carver_sees_anchors: False` (25 → 22 channels). This tests the anchor-identity shortcut. It is an existing flag and needs no code change. Pass `False` with a capital F; see the CLI boolean note.
2. Drop the `log where_mass` channel. This needs a small code change.
3. `boundary_widths: [8, 32, 32]`, i.e. `B(I)` 16 → 8 channels. This tests the width and saves memory.

For each: parent B0, same seed schedule, **at least two seeds**, and read the held-out Δ against the 0.09 / 0.07 noise. Keep the prompt probes and image replacement beside every number.
