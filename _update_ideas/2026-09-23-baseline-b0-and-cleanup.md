# B0 becomes the Phase B baseline; runs archived; docs and slides rebuilt around it

**Date:** 2026-09-23 · **Branch:** `dev-SpatialVox-V1` · **Code of the baseline:** commit `d14f201`

## Rationale

`runs/mask-valid-seed1` (formerly B12, now **B0**) is the first run whose held-out transfer clears both its prompt-blind floor and the run-to-run noise. The rationale and evidence are in `2026-09-22-null-head-decides-emptiness.md`. In short:

- trained classes 0.962;
- held-out val 0.724 (floor 0.247) and held-out test 0.775 (floor 0.162);
- 0% empty held-out masks;
- five of the six held-out shapes improved by 0.14–0.51;
- the prompt probes and image replacement both pass.

The user asked for it to become the baseline, for obsolete runs to go, and for the documentation and slides to follow. It is **provisional**, for two reasons:

- It is one seed. The user asked not to run a second one for now.
- 29–32% of impossible prompts still get a mask after the null-head gate.

## What changed

**Commits.**
- `d14f201`: the code that trained B0 (`mask_on: valid`, `null_gated`, the split carver head), with its tests and docs. It was committed after the run, from the exact working tree the run used, so **B0 is replicable from `d14f201`** with `configs/synthetic-hard.yaml`, seed 20260915, and Stage A `runs/mri-stage-a/best.pt`.
- The documentation commit described here follows it.

**Runs.** 23 run folders were moved, not deleted, to `SpatialVox-MRI/runs_archive_2026-09-23/` (1.3 GB):
- `overfit1`, `overfit1-oracle`, `prompt-only-seed1`, `pretrained-b-seed1`, `boundary-seed1`, `field-empty-only-seed1`;
- `arm-loo`, `arm-noanchor`;
- `gate-stage-a`, `control-stage-a`, `gate-long`, `gate-mri-like`;
- `hard-stage-a`, `hard-stage-b`, `synthetic-stage-b`, `mri-stage-b`;
- `phase-a/new-model`;
- all 8 folders in `phase-b/`.

**Kept in `runs/`:**

| folder | role |
|---|---|
| `mask-valid-seed1` | B0 |
| `mri-stage-a` | A09, its Phase A |
| `synthetic-stage-a` | A03, the easy corpus's Phase A |
| `phase-a/current` | A01, real MRI's Phase A |
| `relational-seed1` | B03, the only real-MRI Phase B reference |
| `easy-mask-valid-seed1` | the new B1 |

**New run.** B1 `runs/easy-mask-valid-seed1` is the easy-corpus rerun under the baseline method: code `d14f201`, `configs/synthetic.yaml`, Stage A A03. It was launched on 2026-09-23 as a sanity check, not as evidence, because that corpus is threshold-separable. The three-pass evaluation runs automatically when training ends.

**Result tracker** (`documentation/Result_tracker/`):
- B12 was renamed **B0** (`git mv`), marked `status: adopted`, given `git: d14f201`, and left with no parent.
- The notes of every archived run were deleted: A02, A04–A08, B01, B02, B04–B11, D01–D03, L01 and P01.
- The hub's status box, lineage graph and baseline callout were rewritten. The replicate-noise table keeps its numbers, marked as coming from archived runs.
- The 64 wikilinks that pointed at deleted notes became plain text marked "(archived)". The vault check reports 0 unresolved links.

**Documentation.**
- `SpatialVox.md`: the overview paragraph and §22 now state the baseline, and §22 has a new claims table and "Open" list.
- `Model Info/Phase B/MODEL PHASE B.md`: a table of the baseline's evaluation.
- `Flowchart.md` and `.drawio`: labels only (the carver's mask is trained on valid prompts; the null head is the only judge of emptiness), plus the baseline caption in the `.md`. The `.drawio` layout is untouched.

**Slides** (artifact `GqwCjZ3nmqnSQg8JBuhPKu`, version 11):
- 8 new slides, one per step of the forward pass: data and prompt, frozen Stage A, soft anchors, positional mapper, null head, boundary encoder, carver, output and losses. Each gives the step's role, input shape, output shape and method.
- The performance, strengths/limits and "core limit" slides were rewritten for B0. The last of these is now "The fix: stop rewarding silence".
- The model overview slide notes that the null head decides alone.

## Deliberately not done

- No second seed of B0, as asked.
- No real-MRI run under the new method; it is named as the next milestone.
- No change to the code.

## Log

### 2026-09-23 11:11 — B1 (easy-corpus rerun) finished

- **Scores:** 30 epochs, `best.pt` epoch 23. Trained 0.994; held-out val 0.997 and held-out test 0.998. 0% empty masks.
- **Probes:** they all collapse as they should, and `permute_both` holds.
- **Image replacement:** another scene's image drops Dice from 0.99 to 0.02.
- **Leak on impossible prompts:** 26–29% after the null gate.
- **Reading:** the baseline method does not break the trivial corpus, and its leak matches B0's. The tracker, §22 and slide 15 are updated.
