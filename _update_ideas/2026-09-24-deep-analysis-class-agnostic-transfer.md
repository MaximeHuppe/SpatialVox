# Why SpatialVox does not transfer on real MRI, every limit found, and the program to a class-agnostic model

**Date:** 2026-09-24 · **Type:** analysis and proposals. **No code, config, corpus or run was changed.** No training was launched.

**What was analysed:**
- **Code:** commit `1610e48` (branch `exp/b3-never-supervise-heldout`). Its `src/` is the `d14f201` relational Stage B plus B3's `never_supervise_heldout`. Every file:line below refers to it, unless marked "new branch".
- **Data:** the **pre-filter** `data/mri` manifests, with 14,965 / 1,983 / 1,886 train / val / test rows. Every floor, statistic and Dice in this note is on those manifests.
- **Runs:** B03, B2 and B3 (final, epoch 27), B0/B1, and the archive `runs_archive_2026-09-23`.

**What changed during the analysis.** Read this first.
- **A new carver.** Commit `c418b2d`, branch `carver-geometry-query`, replaced the carver with a geometry-only stem plus a channel-attention read-out of B(I) (§1). `d8749a0` made `field_centroid_on: empty-only` the default. B2/B3 checkpoints do not load there.
- **A rebuilt corpus.** The triple-stability filter (PR #1, `3bdbfcb`/`d5753f3`) rebuilt the shared `data/mri` **in place** on 2026-09-24 at 13:47. It now has 11,875 / 1,562 / 1,482 rows, after dropping 3,915 rows and 553 of 3,736 triples (§2).
- **Consequence:** every floor quoted here is stale for new runs, and any B2/B3 comparison with a new run is now **cross-corpus**.

**Reporting conventions.**
- **Every run is single seed.** No MRI seed replicate exists.
- **"Held-out" means "never supervised as a relational target".** Stage A knows all 23 names.
- **Pre-filter floors (val split):**
  - trained 0.3067 (0.3185 on the full 763-prompt population);
  - held-out val (caudate, putamen) 0.1097;
  - held-out test (hippocampus) 0.0630.
  - Subset rows carry their own floors.

**How it was done.** Three multi-agent passes, then my own synthesis:
- understand and probe: 6 readers, 1 GPU probe, 1 CPU probe;
- B3 final probes;
- design: 6 proposers, 3 adversarial verifiers.

The design workflow's last step, the critic, could not be resumed across sessions, so I did that synthesis myself (§14).

---

## 0. The answer in short

1. **On this corpus the image currently only hurts transfer.**
   - Every image-using MRI carver peaks on held-out at epoch 0, then collapses within 1–3 epochs as trained Dice rises:
     - B2 0.079 → 0.014 after one epoch;
     - B03 0.138 → 0.005.
   - The only carver that does not collapse is image-free: B04 averaged 0.135 (floor 0.1097).
   - B3, which ignores held-out voxels using the class split, prevents the collapse but ends *below* its floors: 0.080 (floor 0.1097) and 0.028 (floor 0.063).
2. **What the model learned: "paint whichever supervised-positive class the region contains".**
   - Both carver paths are class-gated. The coarse term scores held-out targets at −10 to −14 against +12 to +15 for trained ones, and the fine term puts held-out tissue below background.
   - An **oracle region does not help**: the trained neighbour's logit jumps by +16 to +36 instead.
   - It is suppression, not calibration: held-out AUROC is 0.45 / 0.37.
3. **Where it is learned: nonlinear, contextual image computation trained by the relational loss.**
   - Heads that are *linear* on a *frozen class-free* B do not gate. At the trained threshold they reach 0.27–0.29 on held-out val and 0.22–0.27 on held-out test, against subset floors 0.119 / 0.069.
   - A 64-unit MLP with image context on the same features re-creates the gate: val 0.308 → 0.120.
4. **Why MRI teaches it and synthetic did not:**
   - position and context name the class: a 40-subject atlas scores Dice 0.57–0.64, and a clause set names the same target 95% of the time;
   - an image-free "look up the clause set, paste the template" baseline scores **0.446** on trained classes (floor 0.307);
   - held-out tissue touches trained targets 55× more often than on the synthetic corpus;
   - intensity is 8.3× class-coded;
   - there are only 1,480 distinct clause sets;
   - the class split is biased by shape: every held-out class is less convex than every trained one.
5. **B3 settles one question: removing held-out negatives is necessary but not sufficient.**
   - Under a perfect region the coarse term is unchanged: trained neighbour +7 to +15, held-out target −11.
   - Held-out tissue gets painted mainly as a *spill* of trained masks: 19% of their voxels.
6. **The design rule that follows:**
   - the relational loss may train only (i) image-free nonlinear parts and (ii) parts linear in frozen, class-free image features;
   - every nonlinear or contextual image computation is trained by class-free objectives only (pretexts);
   - on top of that, class-free positives, and a grouping mechanism so the mask can follow a whole body: instance-level selection reaches 0.97, per-voxel region painting 0.11–0.19.
7. **The confound, and the price.**
   - About 90% of the offline held-out gain is geometry alone (GEO 0.262 / 0.239 at AUROC 0.83). Judge a model only as **full model minus a trained prompt-only twin**, never against an absolute threshold.
   - Trained Dice will fall toward its floor once class lookup is gone. That is the price of class-agnosticism, not a failure.
8. **Your geometry-query carver (`c418b2d`) removes the coarse image path, which is right. Its `refine` is still route (c)**: a query-modulated, nonlinear, per-voxel read-out of a *trainable* B.
   - Your overfit run shows the same shape on the hold subject: 0.10–0.13 early, then 0.02–0.05. That population has no computed floor and only 20–27 prompts.
   - That run memorised one scene and is single seed, so this is consistent with the gate, not proof of it.
9. **Four blockers before its first full run (§1):**
   - no flip-retarget guard;
   - no trained prompt-only twin;
   - the flag-parsing bug: any `=false` override is silently True;
   - the channel attention saves about **4.7 GB per prompt** and is inferred to be 6–10× slower.
10. **The triple-stability filter merged today has a legitimate motive but three problems (§2):**
    - it is computed over all splits and all target classes, so val/test labels and held-out rows decide which training rows exist;
    - it makes every clause set a corpus-wide name tag for its target, by construction, which is the enabling cause located here;
    - it can remove the held-out prompts where a class detector is most exposed.
11. **Protocol (§9):**
    - select on V (VentralDC), AUROC first and relative to the twin; never gate on R;
    - label caudate, putamen and hippocampus *post-design*;
    - pre-register the method before the 3-fold rotation;
    - two seeds;
    - report AUROC, paint shares, spill, per-scope floors and a clause-drop index. B3 scores 0.49 on val and 0.13 on test on that index, where a relational painter should score about 1.
12. **The program (§8):**
    - Phase 0, code only, 0 GPU-h;
    - Phase 1: **B4** = new-carver baseline + twin, then a 5-epoch {frozen / trainable B} × {channel-attention / linear} factorial, which is the decisive causal run;
    - Phase 2: class-free carver + careful negatives + grouping pretext + class-free positives;
    - Phase 3: folds;
    - Phase 4: the VLM side.
13. **Efficiency, measured on the GPU (§10; old carver):**
    - 90.4 ms/prompt today;
    - scene grouping + GroupNorm/channels-last at S2K4: **34.4 ms** (2.6×, near-exact);
    - frozen, cached B: **25.6 ms** (3.5×);
    - one protocol arm then costs about 22 GPU-h instead of about 69.
14. **Hippocampus is a separate problem (§11):**
    - its suppression is context-coded and forms with **no loss on its voxels**: B3's fine term went from −0.6 to −13 between epochs 1 and 27;
    - its border with the amygdala is invisible;
    - it is largely ill-posed in real anatomy: only 17% of Right-Hippocampus prompts are unique under the full wmparc label set.
15. **The language side is a closed compiler (23 names, 6 directions, 3 clauses), but it did not cause the gap:**
    - B3's word-following index is about 0;
    - pooled clause statistics move Dice by at most 0.003.
    - Build it after the transfer fix (§12).

---

## 1. Before the first full run on `carver-geometry-query`

This is the branch you are working on, so it comes first.

**What the new carver is** (`c418b2d`, `src/models.py` Carver, new branch):
```
stem, blocks  <- geometry only: F_i, where_raw, log where_raw (+ A_i if carver_sees_anchors)
coarse        = up(1x1(geometry_features))                     image-free
q             = up(1x1(geometry_features))    k, v = 1x1(B(I))
retrieved_i   = sum_j softmax_j(q_i * k_j / sqrt C) * v_j        per voxel, over channels
logits        = coarse + refine(retrieved)                       refine zero-initialised
```

**Measured on your `--overfit` checkpoint.** CPU, read-only, on a copy. 1 training scene, single seed, 2 unseen val subjects, 12–20 prompts per population. Verifier scripts `evid-verify/newcarver_decomp.py` and `claim-lens/decompose_newcarver.json`.
- **The image-free coarse term is nearly flat.** It is about −9 on every voxel group, with AUROC 0.44 on held-out and 0.41 on trained prompts. All discrimination on unseen subjects is in `refine`.
- **`refine` boosts the prompted class only when it was supervised.**
  - Trained prompts: target +2.39 over label 0, and +1.71 over the trained neighbour.
  - Held-out prompts: target +0.34 over label 0, the same as held-out *non-target* tissue (+0.35), and −0.20 against the trained neighbour.
  - There is **no** B2-style veto below background yet.
- **q is not constant inside the region** (std 2.2–2.4 there, against 0.7 over the volume). But setting q to 0 gives a negative R², so `refine` is nonlinear in B. It has no cross-voxel prototype, which makes it route (c).
- **The hold curve peaked, then fell.** Hold is the held-out classes scored on another subject: it peaked at 0.10–0.13 around epochs 21–30, then sat at 0.02–0.05 from epoch 42. Memorised-scene Dice rose to about 0.75–0.78 over the same span. That is the B2 shape, **confounded** by memorising one scene; trained classes on the hold subject reach only 0.22. No floor is computed for the hold population.
- **The `permute_both` drop grew** from about 0.005 (epochs 6–11) to 0.12 (epochs 72–77). The model starts to depend on slot order, which carries no information.

**Blockers.** Each one invalidates a result or a claim.

| # | blocker | why it blocks | fix |
|---|---|---|---|
| 1 | **No flip-retarget guard** on the new branch. `flip()` retargets onto any single solution, held-out classes included; `retarget_only_to` exists only on `1610e48` | "Never supervised as a relational target" is **false** for every full run. The known MRI rate is about 0.12% of prompts, 121 held-out-target presentations before B2's best.pt. Your overfit run turns the flip off, so it is clean | Port the whitelist with **default ON** (retarget only onto S) |
| 2 | **The prompt-only path was deleted**, and CLAUDE.md §6.3 was rewritten to "there is no prompt-only carver" | No Dice can show the image was used. Image replacement is non-discriminating on ACPC data: it keeps about 60% of trained Dice, and it *raises* B3's held-out Dice. Switching `refine` off at eval is not a twin, because coarse is nearly flat (AUROC 0.44–0.63) | A **trained twin**: `refine` and q/K/V frozen at their zero init, B not computed, same seeds. Restore §6.3 as a recorded decision |
| 3 | **The flag-parsing bug.** `src/config.py::parse_overrides` keeps `literal_eval('false')` as the *string* `'false'` (`'null'` likewise), and `train.py` casts flags with `bool()` | Any `--set x=false` is silently True; this invalidated B11. Hygiene on top: `carver_sees_anchors` has no config key and defaults to True, so anchor shapes reach the stem and q. The overfit decomposition found them unused for class selection (zeroing A_i changes nothing; the fields select), but the anchor-set shortcut (ceilings 67.8 / 83.9 / 99.2%) stays open | Parse `true/false/null` properly and add `isinstance` checks. Add an explicit `carver_sees_anchors` key, set to False for every transfer arm |
| 4 | **Cost of the channel attention** | Saved tensors: 73.4 MB/prompt at 32³ fp32, so about **4.7 GB/prompt at 128³**; the fp32 softmax output alone is about 2.15 GB. Your run holds 18.2 GB at batch 4. Inferred from the overfit timing: 2.7–4 s/step against 0.40 s for B2, i.e. **6–10×, about 33–50 GPU-h per 30-epoch seed** (to be confirmed by a GPU bench) | Recompute per chunk in backward, compute the attention at 64³, or replace it with a linear read-out (§8, P2.1). All of these change the function, so do it **before** the baseline |
| 5 | **The architecture is route (c)** | See above. Freezing B, or a linear read-out, is what the evidence supports (§7) | The Phase-1 factorial decides it (§8) |

Also on the new branch:
- **Missing plumbing:** `never_supervise_heldout`, `voxel_weight` and `ignore_labels` are absent. Port them, because the careful-negative rule needs them.
- **`StageB.center` is not in `StageB.config`**, so a midline change would be lost on reload.
- **Every checkpoint embeds the frozen Stage A** (69 MB).
- **Every probe script assumes the old split head**, so they need a port to "coarse vs refine" (template: `newcarver_decomp.py`).

---

## 2. The triple-stability filter (PR #1, `data/mri` rebuilt 2026-09-24 13:47)

**What it does** (`src/data.py::stabilize_relational_manifests`, commit `3bdbfcb`):
- **The key** is `relational_triple_key`: the unordered set of (anchor name, direction) pairs, i.e. the clause set, with pairing preserved.
- **The pass** collects, over **every split's manifest** and every target class, the set of targets each key names. It keeps a row only if that set has exactly one element.
- **Result on `data/mri`** (`meta.json: triple_stability: global-unique-target`): 3,183 stable triples against 553 colliding, 3,915 rows dropped.
- **Stated goal** (docstring): "a held-out-target prompt never reuses a triple that supervised a trained target (and the converse)".

**Its motive is real.** On the pre-filter corpus:
- 10.7% of held-out-val prompts (3.9% of test) use a clause set that training supervised onto a **trained** structure in another subject;
- B2 paints that trained structure in 40 of 40 such cases (`probe-gpu/p6b_conflict.py`);
- across subjects the same words sometimes mean a different structure.

**The problems:**
1. **Leakage into set construction.** The loop runs over all splits (`for rows in manifests.values()`) and all target classes. Val- and test-subject labels, and held-out-class rows, therefore decide which *training* rows exist. Compute it on train subjects only and apply it to train rows only, or record it as a deviation.
2. **It strengthens the enabling cause.**
   - Before the filter, a clause set named its majority target in 95.1% of cases (98.7% on supervised rows). After it, this is 100% by construction: every clause set is a corpus-wide name tag for its target.
   - That is exactly the recognise-and-recall key of §5 row 1. Recompute the recall baseline (0.446 before) and the anchor-set ceilings.
3. **It can remove the prompts where a class detector is most exposed.**
   - The dropped held-out prompts include those whose words mean a trained class in another subject.
   - On the pre-filter corpus, B2 painted that trained class in 40 of 40 such cases.
   - **R Dice can therefore rise with no transfer gain.** That is an inference, not a measurement.
   - Always report the removed stratum beside the kept one.
4. **Every floor is stale**: 0.3067 / 0.3185 / 0.1097 / 0.0630, V 0.4250, and the fold floors. Keep a copy of the pre-filter manifests under a separate corpus root for any comparison with B2/B3.

**Recommendation.** Keep the filter's diagnostic value, but change three things:
- define it on train subjects;
- report both strata;
- regenerate every floor, ceiling and baseline.

Treat it as a corpus version with its own row in the tracker.

---

## 3. Where the runs stand (pre-filter corpus, single seed)

| run | corpus | change | trained (floor) | held-out val (floor 0.1097) | held-out test (floor 0.063) | how it fails |
|---|---|---|---|---|---|---|
| B0 | synthetic-mri | `mask_on: valid` | 0.962 (0.138) | **0.724** (own floor 0.247) | **0.775** (own floor 0.162) | transfers, but flips supervised each held-out class 385–729 times (§13) |
| B03 | MRI | `mask_on: all` | 0.794 (0.307) | 0.005 | 0.001 | silent: 75% of held-out masks empty |
| B2 | MRI | `mask_on: valid` | 0.792 | 0.022 | 0.011 | paints the trained neighbour |
| B04 (archived) | MRI | prompt-only, `mask_on: all` | 0.523 | 0.117 (mean over epochs 0.135, peak 0.215) | 0.054 | no image, no collapse |
| **B3** | MRI | `never_supervise_heldout` (a diagnostic: uses the split) | 0.727 (best.pt e27) | **0.080** (per epoch 0.066–0.160, peak e14) | **0.028** | no collapse, never reliably above the floor |

**B3's final evaluation** (`runs/mri-never-heldout-seed1/eval_val_*`):

| | trained (floor 0.3067) | held-out val (floor 0.1097) | held-out test (floor 0.063) |
|---|---|---|---|
| Dice (null-gated) | 0.727 (0.722) | 0.080 (0.079) | 0.028 |
| per class | — | L/R-Caudate 0.078 / 0.065; L/R-Putamen 0.138 / 0.073 | L/R-Hippocampus 0.035 / 0.017 |
| predicted / true volume | 1.12 | **0.15** | 0.33 |
| centroid error | 2.0 mm | 25.7 mm | 30.6 mm |
| another subject's image | 0.728 → 0.455 | **0.081 → 0.096 (rises)** | — |
| impossible prompts that get a mask, ungated / gated | 98% / 36% | 97% / 41% | — |

Two things to note:
- Only Left-Putamen clears its own class floor: 0.139 against 0.120, n = 50.
- Selection on trained classes chose epoch 27 (0.080) over the held-out peak at epoch 14 (0.160). That is the selection problem of §9 in one number.

---

## 4. The failure, located

These readouts use B2 `best.pt` (epoch 17) and B3 `best.pt` (epoch 27), with frozen weights. The probe harness reproduces `StageB.forward` exactly (max |Δlogit| 0.0).

Old carver at `1610e48`: `logits = up(coarse(trunk(cat[B(I), A, F, where, log where, log mass]))) + W_B·B(I)`.

### 4.1 Both paths are class-gated

| population | coarse (target) | fine W_B·B(I) (target) | background near the region, coarse / fine |
|---|---|---|---|
| B2 trained | **+12 to +15** | −2 | about −13 / −9 to −10 |
| B2 held-out val / test | **−10 to −12 / −10 to −14** | **−13 to −14 / −16** | same |
| B0 held-out (synthetic, for contrast) | **+12 to +14** | −4.6 to −5.2 | −8 to −9 / −12.5 |

- **On synthetic data the trunk learned a generic "paint what the region points at" score. On MRI it gives a held-out target none.**
- **The fine term encodes "was this ever a supervised positive".** It scores the never-target landmark structures (ventricles, VentralDC, mean −16.3) as low as the held-out classes.

### 4.2 An oracle region does not help

`where_raw` was replaced by dilate(true target, 3), a deliberate label leak.
- Held-out Dice stays at 0.031 / 0.035 against subset floors 0.073 / 0.063.
- In 98–99% of masks, most voxels land on another structure.
- The trained neighbour's coarse logit jumps by +16 to +36; the target's moves by less than 1.3.

The carver is a **region-gated detector of its eight supervised-positive classes**.

### 4.3 Suppression, not calibration

Inside dilate(where > 0.05, 8) minus anchors. The rows are m10's: n = 40 / 100 / 80, with row floors 0.250 / 0.100 / 0.025.

| | B2 | B3 e1 | B3 e27 |
|---|---|---|---|
| val AUROC of target vs surroundings / share of prompts < 0.5 | **0.454 / 56%** | 0.789 / 0% | 0.723 / 6% |
| test AUROC / share < 0.5 | **0.371 / 97%** | 0.727 / 0% | **0.411 / 93%** |
| val: per-prompt oracle-threshold Dice (upper bound) | 0.081 | 0.256 | 0.209 |
| val: threshold chosen on *trained* prompts, applied to held-out | 0.033 | 0.131 | **0.067** (worse than at logit 0: 0.082) |
| trained: best threshold | −0.25 | −1.75 | **+1.5** |

B3 turns caudate/putamen into a calibration-like problem: the target sits 7–11 logits above its surroundings, but around −15. Hippocampus stays suppressed.

**No global threshold works.** Trained targets want +1.5 and held-out ones about −9, so the positive evidence itself must become class-agnostic.

### 4.4 The image pathway is where class identity is learned

| run | held-out val: epoch 0 → best | r(trained, held-out) over epochs |
|---|---|---|
| B03 | 0.138 → 0.005 | −0.85 |
| pretrained-B arm (fine-tuned, archived) | 0.112 → 0.007 | −0.80 |
| B2 | 0.079 → **0.014 after one epoch** → 0.022 | −0.42 |
| attention-era EXP2 (archived) | 0.218 → 0.078 | −0.88 |
| **B04, no image** | 0.169 → peak 0.215 (e4) → 0.117 | −0.28 |

- **The held-out heatmap drifts.** Its error goes from 14–17 mm to about 25 mm, the error of the field's own centre (26.8 mm). B04 stays at 10–16 mm.
- **Image replacement agrees.** At epoch 0 another image changed nothing; at best.pt it *raises* held-out Dice (B2 0.022 → 0.049).
- **The one exception is closed-set.** The only MRI Stage B whose held-out Dice rose with training was attention-era `rel-seed1` (0.098 → 0.25), and it chose among Stage A's 23 names.

**Which part of the image pathway.** This is new, from design-phase pre-tests on the frozen class-free B: `arc/p8_learned_relative.py`, `arc/p9_context_gate.py`, `rep-lens/probe_*`.
- **Per-voxel *linear* heads on the frozen class-free B plus geometry do not gate.**
  - They were fitted on trained prompts only, with held-out tissue as negatives.
  - At the trained threshold: GEO 0.298 / 0.262 / 0.239; ABS 0.320 / 0.294 / 0.238; ABS+REL 0.359 / 0.279 / 0.270. Subset floors are 0.348 / 0.119 / 0.069.
- **A 64-unit MLP over the same features plus context re-creates B2's signature.** Trained rises from 0.326 to 0.418, and held-out falls from 0.308 / 0.263 to 0.120 / 0.177.
- **B trained through the mask loss becomes class-coded and erases held-out borders.**
  - The 1-voxel trained-vs-held-out gate AUC rises from 0.752 (raw intensity) to 0.904 (B3 e27) and 0.958 (B2).
  - Affinity on held-out borders falls from 0.710 (intensity) to 0.651 (B3 e27).
  - Grouping from the true seed on caudate/putamen falls from 0.398 (intensity) to 0.297 (B3 e27).
- **The archived "class-free" B (`boundary-seed1`) is no better than raw intensity on transfer tests**: affinity 0.685, flood 0.403. It is also *more* class-decodable (balanced accuracy 0.293 against 0.206).

### 4.5 The answer is defined per instance; the carver decides per voxel

| label-derived reference (not a method) | trained | val | test |
|---|---|---|---|
| paint where_raw > 0.05 | 0.188 | 0.147 | 0.106 |
| region ∩ any labelled structure | 0.421 | 0.370 | 0.369 |
| the labelled instance whose centroid has the highest where_raw | 0.968 | 1.000 | 0.969 |

The corpus defines a relation at the target's centroid. The carver has no grouping, and the heatmap never feeds the mask. So the only way to extend a mask to a whole body is a learned class template.

### 4.6 Inside the region the geometry says nothing

At τ = 0.5 mm the fields saturate within 2 mm, and `where_raw` is flat inside the region. The geometry's receptive field is 21 voxels, against 45 (fine) and 65 (coarse) voxels for the image; these were computed from the code, whereas the docs say about 39 and about 19. Appearance therefore picks among the 5–7 structures in the region.

### 4.7 What B3 adds

| | B2 | B3 e27 |
|---|---|---|
| fine term, caudate/putamen (0 = background, 1 = trained tissue) | −0.94 | **+0.33** |
| fine term, hippocampus | −0.92 | −0.52 |
| fine term, landmark control (still a negative in B3) | −0.31 | −0.27 |
| **coarse under a perfect region**, val: target / trained neighbour | −11.9 / +8.0 | **−10.7 / +7.1** |
| same, test | −12.1 / +15.3 | −10.9 / +15.3 |
| recall of the held-out target under a perfect region, val / test | 0.018 / 0.024 | **0.089 / 0.075** |
| share of trained-mask voxels on held-out tissue | 5.2% | **19.4%** |

- **Ignoring caudate/putamen lifts their fine term exactly to the level of label-0 tissue of the same intensity, and no further.** The residual is −0.04 in B3, against −7.89 in B2.
- **The hippocampus stays 4.9 logits below its intensity match**, and label-0 tissue within 3 voxels of it stays 4.7 below (`sup-lens/h_fine_source.txt`). The suppression is **context-coded**.

### 4.8 What is not the cause

| suspect | evidence |
|---|---|
| null head | rejects 0.5% of held-out prompts; gated ≈ ungated Dice |
| under-training | held-out below its floor for all 30 epochs of B2 |
| Stage A anchors | anchor Dice 0.805–0.814 on held-out prompts against 0.817 on trained; centroid error median 0.84 mm |
| mapper location / fragility | gate 0.86–0.90 (B0 the same); robust prompts fail equally (0.0205, floor 0.1128) |
| mapper coverage as the primary cause | the oracle region does not help; painting exactly target ∩ region would give about 0.46 |
| prompt-level memorisation | novel trained clause sets score 0.74–0.78 against 0.80 seen (class-matched Δ −0.023, CI [−0.097, +0.019]). The recall is **class-level** |
| anchor-mask channels | zeroing or swapping A_i moves Dice ≤ 0.04 |
| invisible borders as the primary cause | putamen/pallidum is visible (d′ 1.4–1.6), yet putamen is the worst class |
| `field_centroid: always` | B0 transfers under it (the new branch already uses empty-only) |

---

## 5. Why real MRI teaches this and the synthetic corpus did not

| property | real MRI (pre-filter) | synthetic-mri | why it matters |
|---|---|---|---|
| **1. Position and context identify the class** | atlas Dice 0.57 / 0.64 / 0.60; class from the centroid alone 93%; centroid spread 2.3–3.3 mm; clause set → same target 95% | 0.04–0.08; 9%; 18–27 mm; 64% | Recognise-and-recall is optimal: a clause-set lookup plus another subject's template scores **0.446** on trained classes (floor 0.307), against 0.012 on synthetic. A held-out class has no entry in that lookup |
| **2. Contact** | 17.1% of a trained target's surface faces held-out tissue; held-out targets touch a trained one in 100% of prompts | 0.31% (55× less) | 24% of the soft-Dice push-down on negatives lands on held-out tissue within 2 voxels |
| **3. Intensity is class-coded** | class-mean spread 8.3× the within-class spread; class from intensity alone 37.8% | 0.16× (`class_spread: shared`) | B can recognise tissue. Only a **non-monotone** remap breaks this: the context gate falls 0.938 → 0.639, and a monotone remap does nothing |
| **4. Diversity** | 8 classes, 5,855 prompts, **1,480 clause sets**; +34% per doubling of subjects | 10 classes, 28,479 prompts, 19,584 clause sets | four templates fit the training set |
| **5. Split shape** | held-out solidity 0.49–0.68 against trained 0.80–0.90; held-out 2.4–9× larger | similar sizes | partly a shape extrapolation |
| **6. Unused supervision** | 5,186 landmark-target prompts thrown away; landmarks trained as background | all classes in a split | free non-convex shapes are discarded |

**Translation and mirroring do not remove row 1.** The key is the *relative* layout of anchors, fields and tissue, and a 4-voxel shift moved B2's logits by 0.15 against a within-region spread of 7.8. Only layout-changing interventions break it: composited objects, many target classes, or pseudo-targets. **The triple-stability filter pushes row 1 from 95% to 100%** (§2).

---

## 6. Catalogue of limits, by module

**Severity:** C = on the causal path of the transfer failure; M = caps accuracy or validity, or mis-measures it; m = minor. Line numbers are at `1610e48`.

### 6.1 Carver (old carver, `1610e48`)

| limit | sev | evidence |
|---|---|---|
| Fixed full-resolution read-out W_B·B(I) of a *trainable* B learns class identity | C | fine term: held-out −13 to −21, landmarks −16, background −9.6 |
| The prompt-dependent trunk reads B(I) nonlinearly with 65-voxel context: no generic positive score | C | coarse −10 to −14 vs +12 to +15; P9 |
| No instance or body mechanism; heatmap never reaches the mask | C | 0.11–0.19 vs 0.97 (§4.5) |
| Geometry binary inside the region; RF 21 vs 45–65 voxels | M | §4.6 |
| Heatmap seed lands on held-out targets 8–13% of the time | M | `probe-gpu/p4_prototype.py` |
| InstanceNorm over a volume that is 66% background: global gain, and it erases the broadcast `log where_mass` channel | m | stem moves 0.008 for a 1.2 change of that channel |
| Clause count architectural (3-slot cat, `NullHead Linear(1+n)`) | M (VLM) | `models.py:678, 491, 764` |
| Sub-voxel misregistration (stride-2 stem, `align_corners=True`, `grid_world_axes`) | m | about 0.5 voxel; learned around |
| Anchor exclusion (A_i > 0.5 → −10) erases 4.1% of trained targets (accumbens 9–12%) | m | `m4_anchor_exclusion.py` |

### 6.2 New carver (`c418b2d`)

| limit | sev | evidence |
|---|---|---|
| `refine` = nonlinear per-voxel read-out of a trainable B (route c) | C | §1 |
| Channel attention about 4.7 GB/prompt; 6–10× slower (inferred) | M (blocks) | §1 |
| No trained prompt-only twin; no retarget guard; flag-parsing bug (and `carver_sees_anchors` True by default, as hygiene) | M (blocks) | §1 |
| B's own 45-voxel RF is now the only image context (nothing re-aggregates B spatially) | m | this makes B's RF the lever (REP-6) |

### 6.3 Image encoder B(I)

| limit | sev | evidence |
|---|---|---|
| Trained end to end by the relational loss: class-coded and erases held-out borders | C | gate 0.752 → 0.958; affinity 0.710 → 0.651 (§4.4) |
| No augmentation reaches it (160 images, 1,480 configurations) | M | `data.py:613-656`; anchor cache |
| The archived class-free B is no better than intensity on transfer; its pretext saw held-out outlines (label_boundary, no ids) | M / claim | `rep-lens/probe_a/c` |
| 88.8% of MACs, recomputed per prompt | M (efficiency) | §10 |

### 6.4 Mapper and relation geometry

| limit | sev | evidence |
|---|---|---|
| Centroid rule describes a point: 22% of a held-out body satisfies all three clauses | M (not the gap: thalamus 20% and Dice 0.85) | `vl-audit/p4_decomp.py` |
| 64% of prompts hinge on a margin < 2 mm; with predicted anchors 16–21% of prompts are no longer unique | M | robust prompts fail equally on held-out |
| Midline 0.625 mm off ACPC x = 0; changes 12–22% of prompts' solution sets | M (validity) | `mri.py:100`, `geometry.py:116` |
| 57–76% of lateral/medial clauses cross hemispheres | M (validity) | `data-corpus/manifest_audit.py`, `vlm/p1` |
| Well-posedness only w.r.t. 23 names: under full wmparc only 76 / 63 / 35% of prompts are unique (Right-Hippocampus 17%) | M (validity) | `vlm/p5_wellposed.json` |

### 6.5 Losses and training signal

| limit | sev | evidence |
|---|---|---|
| Every non-target voxel is a negative, including touching tissue of structures asked for later | C | §5 row 2 |
| Only 8 classes are ever positive | C | 5,186 unused landmark prompts |
| BCE ≈ 1% of the mask gradient (λ_bce = 1 acts like 0.01) | m | \|dDice\|/\|dBCE\| = 98–102 |
| `L_far` is 0.000 in every run | m | `metrics.jsonl` |
| `field_centroid: always` is the largest late loss term, a pull 20–30 mm off target (fixed on the new branch) | m (M for hippocampus) | §11 |
| 9.2% of presentations wasted by duplicate-direction flips; the flip adds no mask diversity | m | `m8_flip_replay.py` |

### 6.6 Corpus, generator and split

| limit | sev | evidence |
|---|---|---|
| Fixed aligned anatomy makes the configuration a class key; the filter makes it 100% | C | §2, §5 |
| Low prompt diversity; more subjects saturate | M | +34% per doubling |
| Split bias by shape; one split cannot estimate transfer (per-unit ceilings 86–100%) | M | `probe-cpu/c4b_folds.py` |
| One pool for anchors and targets: any new label becomes an anchor Stage A does not know | M (blocks fixes) | `geometry.py:250-275` |
| `leave_one_out` cannot make the recall route fail, and was never run on MRI | m | `data.py:546-622` |

### 6.7 Selection, evaluation, reporting

| limit | sev | evidence |
|---|---|---|
| best.pt selected on trained-class recognition, anti-correlated with transfer on MRI (Spearman −0.61 on B03; +0.89 on B0) | C (protocol) | `m2_curves.py` |
| Dice@0.5 hides latent transfer | M | §4.3 |
| No MRI seed replicate (subject-bootstrap SE 0.006 held-out, about 0.02 trained) | M | `historian/bootstrap.py` |
| The prompt-blind floor understates an image-free model on trained classes (recall baseline 0.446) | M | `lookup_template_baseline` |
| Image replacement weak on ACPC brains; "the centroid should hold" fails on both corpora | M | `report.json` |
| `flip_direction` counterfactual malformed (duplicate direction) in 24–38% of prompts | m | `evaluate.py:76-82` |
| HD95 drops empty predictions; logged train Dice includes invalid flips | m | `engine.py:245, 936` |

### 6.8 Engineering bugs (both branches unless noted)

| bug | impact |
|---|---|
| `parse_overrides`: `literal_eval('false')` keeps the string, so every `bool()` flag reads it as True | invalidated B11; `carver_sees_anchors=false` would be True |
| Train dataset reads `meta.targets`; curves, ignore masks and retargets read `cfg.targets` (`data.py:353`, `train.py:138`) | any fold silently trains on meta's 8 classes |
| `metrics.jsonl` opened with `"w"` | a relaunch truncates the record (lost B10/B11) |
| `best.json` git revision is HEAD at save time, no dirty flag | B3 records a docs commit |
| `evaluate.py` defaults `--classes` to `meta.targets[split]`; reads schedule keys and the anchor cache from the config, not the checkpoint | silent narrow populations; wrong-segmenter risk |
| No per-epoch checkpoints | no matched-epoch readouts |
| New branch: `StageB.center` not in config; Stage A embedded in every checkpoint | midline lost on reload; 69 MB per file |

### 6.9 Language side

| limit | evidence |
|---|---|
| Closed compiler: 23 names (`nn.Embedding`), 6 directions, exactly 3 conjunctive clauses, one regex template | `vocab.py:99-131` |
| No distance, "between", containment, negation, plural or empty answers | — |
| Anchors are closed-vocabulary through Stage A | `models.py:240-256` |
| **Not a cause of the gap:** B3's word-following index (wmparc ambiguous minus unique) is −0.003 / −0.001; swapping per-slot fields for pooled statistics moves Dice ≤ 0.003 | `vlm/p8b`, `vlm/p7` |

---

## 7. What has to change: the design rule

**Correction to the principle stated during the analysis.** An earlier version said "any absolute per-tissue read-out gates held-out tissue". The pre-tests refute that as a general rule:
- linear heads, absolute or relative, on a *frozen class-free* B do not gate (§4.4);
- a nonlinear head with image context does.

The rule is about what the relational loss is allowed to train.

> **Design rule.** The relational (mask) loss may train only
> (i) image-free nonlinear parts: the geometry trunk, the heatmap, the null head; and
> (ii) parts that are **linear** in **frozen, class-free** image features.
> Every nonlinear or contextual image computation (B itself, affinities, offsets, objectness) is trained **only by class-free objectives**: id-free pretexts under random appearance.

Why each piece is needed:

| piece | the evidence it rests on | what it cannot do alone |
|---|---|---|
| Freeze B at a class-free solution | B trained by the mask loss becomes class-coded (gate 0.958) and erases held-out borders | a frozen B consumed nonlinearly with context re-creates the gate (P9) |
| Linear or relative consumption of B | P8: linear heads keep held-out tissue at label-0 level and transfer at the trained threshold | its image contribution over geometry is small (+0.017 / +0.031) |
| Image-free geometry trunk with graded fields | GEO alone reaches AUROC 0.83 and held-out 0.26 / 0.24; the apex-distance field scores AUROC 0.80–0.81, flat across populations | the trunk could learn the layout key; watch the twin's own held-out curve |
| Class-free **positives** (route b) | ignoring negatives lifts held-out tissue only to label-0 level (§4.7) | objects can form a separate "object mode" |
| A **grouping** mechanism | instance reference 0.97 against per-voxel 0.11–0.19 | an offset head trained with held-out tissue as label 0 does not extrapolate (vote error 13.4 mm against 5.75 on trained bodies), so held-out tissue must appear as class-free bodies: route b, declared |

**The confound.** About 90% of the offline held-out gain over B3 is geometry. P8's image-free GEO head reaches held-out 0.262 / 0.239 at AUROC 0.825 / 0.830, while its trained score (0.298) is below its subset floor (0.348). So:
- **every success bar is "full model minus a trained prompt-only twin", with the same geometry and seeds**, never an absolute Dice or AUROC;
- predictions of held-out Dice above about 0.2 have **no trained support**; B04 is the only trained image-free carver.

**The price.** Trained-class Dice today comes largely from class lookup (recall baseline 0.446, B2 0.792).
- A class-agnostic carver will score trained classes far lower, near their floor in P8 and P9 (0.32–0.36 against 0.348–0.382), unless grouping works.
- That must be reported as the cost of class-agnosticism.
- The class-free ceilings make it concrete: accumbens 0.32–0.44 and amygdala about 0.70 are capped by invisible borders (`rep-lens/probe_f2`).

---

## 8. The program

The 48 proposals merge into one spine. The IDs in brackets refer to the design-phase proposals (listed in §14).
- Items marked **(new)** run on `carver-geometry-query`.
- Items marked **(old)** need a separate `1610e48` worktree, and are optional side studies.
- **GPU-hours are quoted only where measured.** New-carver arms cannot be costed until blocker 4 is fixed and benchmarked.

### Phase 0: code and instruments (0 GPU-h)

| item | content | merged from |
|---|---|---|
| 0.1 | **The four blockers of §1**, plus: fix `parse_overrides` and `isinstance` checks; one split resolver (meta vs cfg) with an assert; `metrics.jsonl` in append mode with a run header; git dirty flag; per-epoch checkpoints of trainable parameters only (Stage A by path and sha); quarter-epoch evaluation in epochs 0–2 (the collapse happens inside epoch 0) that restores train mode; `StageB.center` into config. Apply to both codebases | PRO-1, EFF-2 hook |
| 0.2 | **Port `voxel_weight` / `ignore_labels`** to the new branch | SUP-2 prerequisite |
| 0.3 | **The evaluation report**, per population S / V / R and per class, macro means with subject-bootstrap CIs, each with its own floor recomputed on the *rebuilt* manifests: recall baseline, atlas baseline, region-paint baseline, class-free ceiling range; AUROC in dilate(where, 8) and the share < 0.5; oracle-threshold Dice (grid ≥ −9.5, because anchors are clamped at −10); paint shares (target / S / V / R / landmark / label 0); spill; the **coarse vs refine decomposition per tissue against intensity-matched label 0**; the **trained twin** comparator; the **paired clause-drop index** (B3: 0.49 / 0.13; a relational painter ≈ 1); per-scope floors (vocab-23 / wmparc-unique / wmparc-ambiguous); the null-head AUC; the fixed `flip_direction` | PRO-7, REP-8, VLM-1, VLM-2 |
| 0.4 | **Exact efficiency hygiene**: drop `keep=0` items, per-scene centroid cache for the flip, skip `truth`/`anchor_dice` in training, no per-step host syncs, pinned memory. The GroupNorm/channels-last swap is near-exact only, so it goes with the re-baseline | EFF-3 |
| 0.5 | **The acceptance battery for any B** (CPU, about 10 min per checkpoint). Class decodability (1-voxel and context gate vs V), affinity transfer on S borders, a V canary on invisible borders (must stay ≤ 0.60), a true-seed flood ceiling. Thresholds fixed on train subjects before scoring; extended to k(B) and v(B) for the new carver | REP-4 |
| 0.6 | **The triple-stability filter**: recompute it on train subjects, report the removed stratum, regenerate floors, ceilings and the recall baseline, and keep the pre-filter corpus for F0 comparisons | §2 |

### Phase 1: the reference, then the decisive causal run

| item | content | readouts | cost |
|---|---|---|---|
| **1.1 B4** (new) | The new carver on the full corpus, **2 seeds**, with its **trained twin** (2 seeds). Retarget whitelist ON, `carver_sees_anchors` False, empty-only field centroid. Current S on F0, rebuilt corpus as a recorded corpus version | the whole 0.3 report per quarter-epoch in epochs 0–2, then per epoch; the `refine` decomposition; the clause-drop index | **unknown until 0.1 blocker 4**; do not start before the attention fix and a bench |
| **1.2 Factorial** (new) | 5 epochs each, B2 condition (held-out tissue as negatives, so split-free), per-epoch checkpoints and quarter-epoch evaluation. {channel-attention read-out, **linear ABS+REL read-out** with a prototype pooled from the image-free heatmap's soft seed} × {**B trainable**, **B frozen class-free**}, plus the twin. The frozen B must be declared: its pretext saw held-out outlines | the question is whether the class code is learned in B's weights or in the nonlinear read-out. Decide on **V and the twin**; R is reported only | frozen, cached cells: about 25.6 ms/prompt (measured on the old carver's carver path, which is the same size as a linear read-out's). Channel-attention cells: after blocker 4 |
| 1.3 (old, optional) | **B0-clean** × 2 (retarget whitelist only: is the only positive reference real?); a second seed of B2 and B3 with quarter-epoch evaluation (MRI noise band on the old carver); **SYM-U** (a gradient-matched label-0 relief: is B3's lift held-out-specific?) | whether the old-carver facts still need settling | about 3.6 + 2.2 + 1.6 GPU-h at measured old-carver speed |

Predictions, from the pre-tests:
- **Frozen B with a linear read-out keeps held-out flat.** Its image contribution over the twin is small, +0.02 to +0.05.
- **The trainable-B cells collapse, like B2.**
- **Frozen B with channel attention is the open cell.** P9 says a nonlinear, contextual read-out can gate even on frozen features. The new carver's attention is per-voxel, without spatial context, so it may sit between the two.

### Phase 2: the class-agnostic method (after Phase 1 picks the read-out)

| item | content | key conditions |
|---|---|---|
| **2.1 Class-free carver** | Frozen, cached B; an image-free trunk with **graded geometry** (fields at τ = 0.5 / 2 / 8 mm, clipped signed margins); a read-out **linear** in B (absolute + seed-relative); soft seed from the image-free heatmap; field centroid empty-only. **Twin mandatory** | Distance-to-anchor and apex fields form an anchor-relative coordinate frame: admit them only by a recorded §1 relaxation, and ablate them separately. Prediction: held-out stays at twin level plus a small image term; trained Dice falls (§7) |
| 2.2 **Careful negatives** inside 2.1 (not a standalone arm) | Labelled non-target tissue of *any* class is ignored inside dilate(region, 4) during a 3-epoch warm-up, then only in a 2-voxel band. Label 0 is always a negative. Split-free by construction | route (c) alone: it removes the veto but creates no positive (§4.7). It is the prerequisite for 2.3 and 2.4 |
| **2.3 Grouping pretext for B** | An id-free affinity + discriminative objective over all label regions and label-0 components. Invisible pairs are merged (list fixed on train subjects at AUC 0.60). **Non-monotone** appearance remap per sample. The deployment-faithful **CLEAN** variant relabels R/V to label 0; ignoring R/V would use the split, so it is a diagnostic. Optional offset and objectness heads for the grouping vote | Must pass 0.5 before any Stage B run: affinity on visible held-out and S borders ≥ 0.80 (intensity 0.773 / 0.728), flood ≥ 0.50, gate ≤ intensity's. Pretext about 1–2 GPU-h per variant |
| **2.4 Class-free positives** | Composited objects (the 16 synthetic primitives, solidity 0.45–0.90, 300–4,000 voxels) with "replace" appearance, fed to **B only** through `boundary_image`; the real T1 goes to Stage A and the cache; placement prior toward labelled tissue, never on the prompt's anchors | **Generate anchor-first:** fix the triple first, then keep the object only if that triple names it. Placing the object and then searching for a triple is the target-first pattern CLAUDE.md §4 deleted. Pre-tests: geometric ease (object region-paint Dice within ±0.05 of real prompts); dose (≥ 30% of object voxels on labelled tissue); detectability with a linear probe on frozen B, not 5 hand features. The lesion test must use a *different* generative family. Needs a live B pass (conflicts with the cache). One recorded decision covers `boundary_image` in training, the label-derived composite and the §4 augmentation. Declare the exposure |
| 2.5 Landmarks into S | Brain-Stem, lateral ventricles, Inf-Lat-Vent, 3rd and 4th ventricles; VentralDC stays V | a treatment: its own F0 arm first; needs 2.2 (held-out tissue fills 19–43% of the ventricle shells) |
| 2.6 Grouping vote | Each voxel evaluates the prompt at its predicted body centroid, p + o(p), inside the mapper: a dense per-voxel vote, no candidate list. Ceilings: 0.94–0.99 with oracle bodies; 0.78–0.82 on wmparc-unique prompts | Only if 2.3's offset head passes a full-resolution test relative to trained bodies. A 5-minute CPU pre-test reads as unlearnable even on trained bodies (objectness 0.587). The mapper then takes an image-derived input: a recorded §1 change, route (b) declared |

### Phase 3: protocol at scale

The 3-fold rotation of §9, each fold × 2 seeds, negatives as the primary treatment of withheld tissue. Cost with the frozen, cached B: about 2.1 GPU-h per 30-epoch run (measured step time), about 17 GPU-h per 8-run arm.

### Phase 4: the language side (§12)

Set pooling over clauses; a relation-program IR with a parser outside `StageB`; a relation library gated by per-relation floors; open-vocabulary anchors through a retrained, text-conditioned Stage A.

### Decision tree (gates read V and the twin, never R)

```
Phase 0 done
  B4 (1.1) vs its twin, V AUROC and Dice, 2 seeds
    full − twin <= 0 on V, early peak then decay  -> the channel-attention read-out re-forms the gate
    full − twin > 0, no decay                      -> keep the new carver as the reference for 1.2
  Factorial (1.2)
    frozen-B + linear flat, >= twin + 0.03 on V (2 seeds)      -> 2.1 is the spine; go to 2.3 to raise the image term
    frozen-B + channel-attention also flat                     -> the code lives in B's weights: keep the new carver with frozen B; still 2.3
    every cell collapses, including frozen + linear            -> the gate is in the geometry trunk or the loss (layout key):
                                                                 2.4 (class-free positives) becomes mandatory; inspect the twin's own curve
    twin >= every image cell                                   -> "the image currently only hurts": no image claim until 2.3/2.4
  Pretext (2.3) through the battery (0.5)
    pass -> swap into 2.1, frozen
    fail (visible held-out affinity <= 0.773, or flood <= 0.40) -> B is no better than intensity: use fixed hand features or drop B
  Objects (2.4), 2 seeds
    object Dice >= 0.6 but V flat and refine unchanged -> an object mode formed: fix the detectability cue (a repaint as fallback)
    placement near labelled tissue ≈ label-0-only placement -> context exposure is not the lever
Hippocampus: always its own branch (§11)
```

### Rejected or deprioritised

| proposal | reason |
|---|---|
| ARC-8 dynamic per-prompt kernels | linear hypernetwork no better (P8); a nonlinear one re-creates the gate (P9); the new carver already occupies this family |
| ARC-6 run (drop the fixed skip on a trainable-B run) | its items are already done by `c418b2d`; masked GroupNorm only inside a re-baseline |
| EFF-6 rank-16 `up.1` | measured slower (63.7 against 55.4 ms/prompt); B2 uses ≥ 16 of 32 dims |
| EFF-7 subcortical crop | box derived from all subjects' labels, val/test and R included; the crop faces are a positional cue; conflicts with cortical targets. Prefer attention at 64³ |
| SUP-4 SAFE wmparc classes *as a transfer lever* | the SAFE list consults the split; fixed classes are route (c). Keep only as diversity, split-free "reach" list + 2.2 |
| SUP-5 SynthSeg repaint | the gate is context-coded (class means swapped: trained coarse stays at +10.9 to +12.7); a rendered label partition is a §2 exposure. Only as a 2.4 fallback |
| SUP-6 large deformation / transplant | only 60.5 / 33.9 / 20.3% of prompts stay valid and same-target at 3 / 6 / 10 mm; folding; a warped cache is not Stage A |
| SUP-8 co-training with synthetic packings | the domains are trivially separable, and B0 is itself unverified |
| VLM-8 plural and empty answers | union targets reinforce the class detector; many-vs-one AUC is only 0.71 |
| `carver_sees_anchors=False` *as a transfer lever* | eval-time ablation moves Dice ≤ 0.04. Set it False anyway, for hygiene (blocker 3) |
| Translation or mirror augmentation; raising `data.triples` for rows | the relative key survives rigid motion; 11.6× the rows give only 2.25× the prompts and a lower floor |

---

## 9. Protocol integrity

- **R has been read for design.**
  - Caudate, putamen and hippocampus drove this whole programme: every probe, the B3 flag, the choice of fixes. They come back as R in F1/F2/F3.
  - Either label them **post-design** in every table, or confirm only on classes never inspected as held-out: pallidum, amygdala, thalamus, accumbens. Their fold assignment matters here.
  - **Pre-register** the method, the fold assignment, the candidate-checkpoint grid and the thresholds before F1–F3.
- **Every gate reads V, never R.**
  - V = L/R VentralDC only, trained exactly like R. Its pre-filter floor is **0.4250** (not the 0.29 of VentralDC + Brain-Stem), and its anchor-set ceiling is 93.1%, so V Dice is shortcut-prone.
  - VentralDC also has invisible borders with three S classes (AUC 0.553–0.564).
  - Select AUROC first, relative to the twin, on an a-priori candidate grid. Sub-epoch selection over about 180 candidates inflates V, so V is never reported as transfer.
  - Several earlier proposals gated on R-val. That violates this rule, and they were rewritten to gate on V.
- **Folds.** The 3-fold rotation over bilateral units puts two non-convex pairs in every training set; the current trained set has none:
  - F1: Pallidum + Amygdala + Caudate;
  - F2: Thalamus + Putamen;
  - F3: Accumbens + Hippocampus.
  - Pre-filter R floors are F1 0.2813, F2 0.1677, F3 0.1573. Recompute them on the rebuilt corpus.
- **Two seeds minimum.** A single-seed Δ below about 0.05 is not read. The old carver's noise band does not transfer to the new one.
- **Treatment of withheld tissue.** "Negatives" is primary, as at deployment. "Ignore" (B3-like) uses the split, and is a labelled diagnostic.

---

## 10. Efficiency

Measured on the idle GPU after B3 ended: `scratchpad/efficiency/bench.py`, old carver, B2 weights, one full training step at 128³ in bf16, median of 20 iterations. Loader excluded.

| variant | ms / prompt | speed-up | peak memory | exact? |
|---|---|---|---|---|
| **today** (4 prompts from 4 scenes) | **90.4** | 1.0× | 6.2 GiB | reference |
| scene-grouped S2K2 / S2K4 / S2K8 / S2K16 (B once per scene) | 71.9 / 61.0 / 55.4 / 53.7 | 1.26 / 1.48 / 1.63 / 1.68× | 3.2 / 4.9 / 8.3 / 15.0 GiB | forward exact |
| + `cudnn.benchmark` | no gain (−2 to −3%) | — | — | exact |
| + GroupNorm swap + channels_last, today's batching | 77.8 | 1.16× | 11.9 GiB | **near-exact**: mask agreement 0.9993, max \|Δlogit\| 0.56 |
| + GroupNorm swap + channels_last, **S2K4** | **34.4** | **2.63×** | 15.2 GiB | near-exact |
| + GroupNorm swap + channels_last, S2K8 | 27.8 | 3.25× | 19.8 GiB (S2K16 runs out of memory) | near-exact |
| **frozen B, cached features**, S2K8 / S2K16 | **25.6** | **3.5×** | 6.6 / 12.8 GiB | changes the model |
| subcortical crop 96×120×88, S2K8 | 29.0 | 3.1× | 4.3 GiB | changes the function (rejected, §8) |
| rank-16 `up.1`, S2K8 | 63.7 | slower | 8.3 GiB | rejected |

- **Grouping alone buys far less than the MAC estimate:** 1.48×, not 3.0×. The per-prompt work at 128³ is memory-bound; **the layout change is what unlocks the gain.** Scene grouping needs one recorded §1 decision (`image_index=`) and one re-baseline; its gradient-diversity pre-test passes (excess within-scene correlation +0.05 to +0.10, against a break-even of 0.66).
- **Exact and free:** the dropped-flip filter saves 9% of steps, and the flip-centroid cache takes a flipped item from 84 ms to 1.3 ms.
- **Where the savings go:** into the protocol, not into more rows. One 8-run arm with landmarks supervised (9,691 prompts on F0), on the old carver, takes about **69 GPU-h today**. It takes **about 22 GPU-h** with grouping + layout at S2K4, and **about 17 GPU-h** with a frozen, cached B (a carver path the size of the old one).
- **New carver:** none of these numbers apply until blocker 4 is fixed. Its per-prompt channel attention dominates, so grouping and caching B help less there. `torch.compile` would unroll its 512-iteration chunk loop.
- **Stage A** is cached. Running it live costs 566 GMAC per sample, which is why every augmentation reaches B only, through `boundary_image`.

---

## 11. Hippocampus is a separate problem

- **Nothing rescues it.** B3 test is 0.028 (floor 0.063), trending down.
- **Its suppression forms with no loss on its voxels, and it is contextual.**
  - Under B3 its fine term fell from −0.63 at epoch 1 to −13.1 at epoch 27.
  - Test AUROC went 0.727 → 0.457 → 0.411 (epochs 1, 7, 27).
  - It sits 4.9 logits below intensity-matched label-0 tissue, and label-0 tissue within 3 voxels of it sits 4.7 below. Yet its intensity matches the trained amygdala (d′ 0.21).
- **The heatmap is pulled to the field centre.** Heatmap-to-field-centre distance went 18.5 → 8.6 → 5.3 mm (epochs 1, 7, 27), while heatmap-to-target reached 26–33 mm. The new branch already trains the field centroid on empty prompts only, so its first full run will show whether that was the pull.
- **Its wrong paints match no simple rule.** B2 paints the thalamus in 36 of 68 wrong-structure prompts. The contact, centroid and lookup rules predict only 16–32% of them; on caudate/putamen the most-contact rule predicts 86%.
- **Its border with the amygdala is invisible.** d′ is 0.02–0.13, and every class-free read-out scores ≤ 0.57. Only geometry can say where it stops.
- **It is largely ill-posed in real anatomy.** Under the full wmparc label set only 34.7% of test prompts, and **17% of Right-Hippocampus prompts**, name a single structure: parahippocampal and entorhinal parcels compete.

**How to report it:**
- always on its own row;
- with its floors: pre-filter 0.063 overall, L 0.081 and R 0.038 per class;
- with its class-free ceiling range: 0.79–0.81 with the amygdala merged, region cut k = 8;
- with its per-scope floors;
- labelled post-design.

**The diagnostic arm, if budget allows:** medial-temporal ring relief. It ignores label-0 tissue within 3 voxels of the hippocampus and the Inf-Lat-Vent. It uses the hippocampus identity, so it is a diagnostic only.

---

## 12. Toward a class-agnostic VLM

**None of the language limits caused the gap, so this comes after the transfer fix.** The end state:
- free text goes to a parser **outside `StageB`**, which emits a **target-free relation program**: a list of (relation, anchor ref, params), with no target slot;
- anchors come from Stage A (by name, or by text through a retrained text-conditioned Stage A) or from a user mask through the existing `anchors=` argument;
- a **parameter-free field library** in the mapper gives one field per clause, combined by the unnormalised product and its mass;
- a **set-pooled** carver and a count head accept any clause count.

**What survives from today:** the mask → field compiler, `where_raw` / `where_mass`, names stopping at Stage A, relation ids consumed only by the mapper, anchor-first generation with re-scored flips, and the floor / counterfactual battery.

**What is replaced:** the 3-slot cat, `NullHead(1+n)`, the regex, the closed name table, the volume-centre midline, and centroid-only truth as the only rule.

| step | content | evidence and gate |
|---|---|---|
| Set pooling, n ∈ {2, 3} (VLM-3 = ARC-7) | pooled `where`, log where, log mass, min/mean/max over F_i, n; pooled null head | pooled statistics change Dice ≤ 0.003 on B2/B3; a frozen carver fed n = 2 loses held-out paint 0.39 → 0.16, so n must vary in training. Keep a control that can fail, since `permute_both` becomes vacuous |
| Midline and lateral semantics (VLM-4) | ACPC constant 80.0 mm in `StageB.config` at the next rebuild; hemisphere-aware wording only if the floors hold | hemisphere semantics raise the trained floor by +0.11, so adopt only the constant |
| Relation program + parser (VLM-6) | an LLM with a grammar-constrained schema, or a small seq2seq on paraphrase-augmented renders | a recorded decision that a language model sits on the inference path. Evaluation programs come only from the anchor-first generator, never authored from target names |
| Relation library (VLM-5) | adjacent, just-d, between, not, body rule | each relation needs its image-free floor and shortcut ceiling first: the most-contact neighbour answers 98% of caudate/putamen adjacency programs; `within` conflicts with the anchor exclusion |
| Open-vocabulary anchors (VLM-7) | Stage A retrained on about 150 names (23 aseg + 127 wmparc) with a frozen text encoder and a name-level holdout; laterality as a symbolic token | a text → query ridge fails (centroid error 21–22 mm, 13–24% on target), so a retrain is needed. Pass: median centroid error ≤ 2 mm on held-out names |
| Scope-aware scoring (VLM-1) | report vocab-23 / wmparc-unique / wmparc-ambiguous, each with its floor; never filter on scope | on B3, val scores 0.081 on unique prompts (floor 0.134) and 0.078 on ambiguous ones (floor 0.069) |

---

## 13. Corrections to earlier notes

1. **"59–62% of painted voxels on another structure" (B2).** On best.pt the mean share is **44% / 52%**; 64% / 71% of non-empty masks are mostly on another structure. The old figure came from another checkpoint.
2. **"86% exact prompt repeats".** 86.4% is the *order-free* clause set with the same target; exact ordered repeats are 57–59%. The recall that blocks transfer is class-level.
3. **Flip statistics** (CLAUDE.md §4, `data.py:495`, SpatialVox §7.3). Dropped flips are duplicate directions (37% of flips), not multi-structure (0.1%). Per item: 9.1% dropped, 15.3% empty, 0.30% retargeted.
4. **B0 is not a clean "never supervised" result.** Each held-out class was a supervised target 385–729 times over 30 epochs. B0 was 0.448 / 0.498 at epoch 0, so most of its transfer is probably real, but this is unverified. The B3 note's "at most ~0.3%" used the MRI rate.
5. **The module audit's lever.** Recall does not "live" only in the fixed fine term; the trunk is class-gated just as strongly. Its "proposal A first" order should not be followed.
6. **The principle "any absolute read-out gates"**, from this analysis's own digest §7.1. It is refuted by P8/P9; see the design rule in §7.
7. **Receptive fields.** B(I) is 45 voxels, not about 39. The carver sees 65 voxels of image, and 21 only for geometry.
8. **Image replacement.** "The centroid should hold" fails on both corpora: B2 trained 1.9 → 6.3 mm, B0 held-out 5.0 → 10.0 mm.
9. **`L_far` is inactive** (0.000) in every run, yet `SpatialVox.md:1599` calls it "the only spatial penalty on a valid prompt".
10. **The trained null head's AUC is now measured.** It is 0.845 (B2), *below* `where_mass` alone at 0.874 on the same population (`vlm/p3_click.json`). CLAUDE.md §7's 0.848 is on flip empties. On random-direction empties `where_mass` separates one from none at 0.965 and many from one at only 0.708.
11. **The midline** is 0.625 mm off ACPC x = 0, not "kept at the volume centre".
12. **The "class-free" B.** The shipped B(I) is not "pretrained without class ids" (README:44, CLAUDE.md §1): every run trains it from scratch through the class loss. The archived pretrained B saw held-out outlines in its pretext.
13. **Stale status.**
    - SpatialVox.md §0/§22 and the tracker callout still name B03 as the only MRI reference.
    - The next Stage B ID is **B4** (also in the auto-memory).
    - `config.yaml` says `warmup_epochs: 5`, but every MRI run used 2.
    - The `full_resolution_skip` / `use_image` findings hold at `1610e48` and are moot on the new branch.
14. **"Well-posed by construction" (CLAUDE.md §4, "100% name exactly one structure")** holds only for the 23-name vocabulary: 76 / 63 / 35% under full wmparc.
15. **The new branch's CLAUDE.md §6.3**, "there is no prompt-only carver", removes the mandatory image-use test. Restore it (§1, blocker 2).

---

## 14. Method, provenance and what was not done

**Passes.**
1. **Understand and probe:** readers for models, mapper and language, engine and evaluation, data and corpus, run history and the hypothesis ledger, plus one GPU probe and one CPU corpus probe, on B2 and B3 (epochs 1 and 27).
2. **B3 final probes:** best.pt (epoch 27) and last.pt (epoch 29), with the same scripts and populations as B2.
3. **Design:**
   - 6 proposers, 48 proposals: SUP-1..8, ARC-1..8, REP-1..8, PRO-1..9, VLM-1..8, EFF-1..7;
   - 3 adversarial verifiers (CLAUDE.md intent, evidence, feasibility), 144 verdicts: 42 keep, 87 modify, 15 reject;
   - two verifiers ran new CPU checks on a copy of your `--overfit` best.pt.

**Critic.** The workflow's critic step could not be resumed across sessions (the journal stayed in the previous session's directory). The synthesis in §§7–9 is therefore mine, built from all 48 proposals and 144 verdicts.

**GPU wall-time benchmark.** Run while the GPU was idle, after B3 ended.

**Scripts.** Every script and raw output is in the previous session's scratchpad: `/tmp/claude-1000/-home-imag2-Documents-IMAG2-dev-SpatialVox-V1/713fb484-c465-44c9-abb4-b276c4a390d4/scratchpad/`.

| folder | contents |
|---|---|
| `wf1/` | the eight workflow-1 reports |
| `wf2/` | the six proposal sets |
| `evidence_digest.md` | the consolidated evidence |
| `probe-gpu/`, `b3-final/` | decompositions, oracle region, ranking, spill |
| `arch-audit/`, `ledger/`, `train-eval-audit/`, `probe-cpu/`, `data-corpus/`, `vl-audit/`, `historian/` | the workflow-1 audits |
| `sup-lens/`, `arc/` (P8/P9), `rep-lens/`, `protocol/`, `vlm/`, `efficiency/` (`bench.py`, `bench_gpu.log`) | the design-phase pre-tests |
| `evid-verify/`, `claim-lens/` | the verifier checks |

That directory is temporary, session-scoped storage. **Copy it if you want to keep the scripts.** The verifier verdicts are in `~/.claude/projects/-home-imag2-Documents-IMAG2-dev-SpatialVox-V1/713fb484-…/subagents/workflows/wf_f27df548-956/journal.jsonl`.

**Caveats.**
- Every run is single seed.
- The mechanism readouts use frozen weights; a retrained model may store class identity elsewhere, which is why Phase 0.3 puts the same readouts on every new checkpoint.
- Upper-bound rows use labels on purpose, and are marked.
- All corpus numbers are **pre-filter**.
- The new carver was examined only through one overfit checkpoint (1 training scene); its cost is inferred, not benchmarked.

**Not done:**
- no training run;
- no change to `src/`, `scripts/`, `configs/`, `tests/`, `CLAUDE.md`, `documentation/` or the tracker;
- B3's tracker note is not filled in;
- `torch.compile` was not benchmarked;
- nothing was committed.

The harness wrote an untracked `.claude/workflows/spatialvox-understand-and-probe.js` into the worktree when the first workflow ran; it is safe to delete.
