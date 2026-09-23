# Why the baseline method does not generalise on real MRI (B2, interim)

**Date:** 2026-09-23, 14:30 · **Type:** analysis. **No code, config or run was changed.** · Run: B2 `runs/mri-mask-valid-seed1` (commit `32de4e3`, code identical to `d14f201`), still training.

## 1. Scores so far (epochs 0–9 of 30; one seed)

B2 is the B0 method (`mask_on: valid`) on real MRI, Stage A `phase-a/current`. Its parent, B03, is the same setup under the old `mask_on: all`. Floors: trained 0.307, held-out val 0.110, held-out test 0.063.

| epoch | B2 trained | B2 held-out val | B2 held-out test | B2 held-out val **empty** | B2 held-out val centroid | B03 trained | B03 held-out val | B03 held-out test |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.501 | 0.079 | 0.013 | 12% | 15 mm | 0.437 | 0.138 | 0.020 |
| 1 | 0.643 | 0.014 | 0.005 | 42% | 25 mm | 0.551 | 0.081 | 0.010 |
| 2 | 0.668 | 0.012 | 0.001 | 29% | 17 mm | 0.572 | 0.050 | 0.002 |
| 3 | 0.725 | 0.012 | 0.002 | 16% | 15 mm | 0.600 | 0.010 | 0.001 |
| 4 | 0.735 | 0.010 | 0.004 | 14% | 17 mm | 0.666 | 0.009 | 0.002 |
| 5 | 0.740 | 0.016 | 0.008 | 13% | 24 mm | 0.641 | 0.013 | 0.000 |
| 6 | 0.742 | 0.024 | 0.002 | 7% | 17 mm | 0.683 | 0.003 | 0.001 |
| 7 | 0.742 | 0.007 | 0.001 | 46% | 23 mm | 0.721 | 0.005 | 0.000 |
| 8 | 0.770 | 0.013 | 0.005 | 27% | 24 mm | 0.727 | 0.007 | 0.000 |
| 9 (`best.pt`) | **0.774** | **0.017** | **0.002** | 16% | 24 mm | 0.741 | 0.009 | 0.001 |

- **Trained classes learn faster than under the old rule:** +0.03 to +0.12 at every epoch. With the rejection route gone, the carver spends its capacity on painting.
- **Held-out transfer is absent,** at 0.01–0.02 against a 0.110 floor. It is only marginally above B03 (≈ +0.007 per epoch), far inside the noise, and flat for 10 epochs.
- **Contrast with B0.** On the synthetic corpus, held-out started at 0.45 at epoch 0 and rose, so a late recovery of B2 is unlikely.
- **The model still reads the prompt.** The flip drop is +0.64 of the Dice, and `permute_both` ≤ 0.001.
- **The null head is not the cause:** it rejects only 0.5% of held-out prompts, so the gated Dice matches the ungated one.

## 2. Where it paints: failure-mode diagnostic of `best.pt` (epoch 9)

Val-split subjects, predicted anchors from the cache. Script: session scratchpad `failure_modes_mri.py`.

| | trained (n = 120) | held-out val: caudate, putamen (n = 200) | held-out test: hippocampus (n = 127) |
|---|---|---|---|
| Dice | 0.761 | 0.017 | 0.002 |
| empty masks | 0% | 19% | 17% |
| predicted / true volume (non-empty masks) | 1.00 | **0.12** | **0.22** |
| painted voxels **on the target** | 78% | **16%** | **4%** |
| painted voxels **on another structure** | 7% | **59%** | **62%** |
| painted voxels on background | 15% | 25% | 34% |
| non-empty masks painted mostly on another structure | 4 / 120 | **118 / 162** | **76 / 105** |
| …of which a **trained** target class | 4 | **113** | **68** |
| painted voxels within 10 mm of the region | 99.8% | 99.9% | 97% |
| structure under the predicted centroid | target 87% | target 15%, background 54%, trained class 27% | target 6%, background 74% |
| centroid error | 2.1 mm | 22 mm | 28 mm |

The structures painted instead:

| held-out target | painted instead (count) |
|---|---|
| caudate / putamen | Right-Accumbens 41, Left-Accumbens 36, Right-Pallidum 19, Left-Pallidum 13 |
| hippocampus | Left-Thalamus 32, Right-Pallidum 11, Left-Pallidum 10, Left-Amygdala 6 |

**Interpretation.**
- **The fix worked on what it targeted, and uncovered the next failure.** Silence fell from 75% to 17–19% of held-out prompts. The model now answers, but with **the nearest trained structure that lies in or next to the region**. It paints small pieces of the accumbens, pallidum or thalamus: 12–22% of the target's volume, 60% of it on a trained neighbour.
- **This is the recognise-and-recall failure.** On the MRI-like synthetic corpus it was a minor mode, 31 of 34 wrong-structure masks on held-out val. On real MRI it is the dominant one.
- **It does read the prompt,** because the region is placed correctly (the centre is inside it 86–90% of the time). But inside that region, it only knows how to paint the 8 structures it was supervised on.

## 3. What is ruled out

| suspect | evidence it is not the cause |
|---|---|
| the positional mapper's *location* | the region contains the held-out target's centre 86–90% of the time, and painted voxels fall within 10 mm of it 97–99.9% of the time |
| the null head | 0.5% rejection on held-out prompts; gated ≈ ungated Dice |
| shy painting (the B0 fix) | empty masks fell from 75% to 17–19%; B2 answers |
| not reading the prompt | flip drop +0.64; the control holds |
| Stage A anchors | predicted anchors are within 0.84 mm of the true centroid (CLAUDE.md §7); anchor Dice is as good on held-out prompts as on trained ones |
| under-training | trained classes are at 0.774 by epoch 9; held-out has been flat since epoch 1, while B0's held-out was at 0.45 from epoch 0 |

## 4. Possible origins of the lack of generalisation, most likely first

**1. The held-out structures are trained as background, next to trained targets, in fixed anatomy.**
- **Mechanism.** The mask target is `labels == target`, so every caudate, putamen and hippocampus voxel is a **negative** in every training prompt where it falls in the field. On real MRI the anatomy never moves: the caudate always abuts the accumbens, the putamen the pallidum, the hippocampus the amygdala. So the carver is repeatedly taught, at exactly those positions, "paint the accumbens, not the tissue next to it".
- **Why the synthetic corpus escaped it.** Random packing breaks those fixed adjacencies there.
- **Evidence:** held-out prompts are answered with exactly those neighbours (accumbens, pallidum, thalamus, amygdala).
- **Test:** exclude the voxels of held-out classes from the mask loss (`weight = 0` on them), so they are never supervised, neither as target nor as background. The class split is training-time knowledge, never a model input, so CLAUDE.md §2 holds. It makes "never supervised" literally true.

**2. Too few, too distinctive supervised targets.**
- **Mechanism.** The carver is supervised on 4 bilateral pairs (thalamus, pallidum, amygdala, accumbens) with a total of **5,855 training prompts from 160 subjects**. The synthetic baseline had 10 classes and 28,479 prompts from 400 scenes, **4.9× more**. With so few, the carver learns four templates, not "paint what the region points at".
- **Evidence.** On synthetic, transfer tracked similarity to a trained shape (r ≈ 0.67). The MRI held-out shapes are a long arc (caudate), a lens (putamen) and a curved tube (hippocampus), and none of the four trained templates resembles them.
- **Test:** add supervised targets that are never held out, e.g. the other aseg or wmparc regions, as target-only classes (Stage A need not know them, because targets are never named). This is the "more target shapes" idea.

**3. The region is too narrow, and too crowded, for the MRI held-out shapes.**
- **Mechanism.** The region covers only 30% of a held-out MRI target, against 52–58% on synthetic. It is 4–5× the target's volume, with 7–12% precision. So it contains, or touches, trained neighbours, and the carver picks the familiar one.
- **Evidence:** only 33–36% of painted voxels are inside `where_raw > 0.05`; the rest is just outside it.
- **Test:** the mapper ideas in `2026-09-23-positional-mapper-coverage-and-ideas.md`, scored offline first.

**4. The image cannot separate the held-out target from its trained neighbour.**
- **Mechanism.** The caudate, putamen and accumbens form one continuous grey-matter body, the striatum, and the accumbens is its ventral junction. On T1 there is almost no visible edge between them; the real MRI's threshold IoU is 0.08, against 0.21 synthetic. So `B(I)` cannot tell the carver where the caudate stops and the accumbens starts, and the learned template decides.
- **Test:** per-pair edge contrast on `B(I)` features (analysis only); a `B` pretrained without class ids, then frozen; a second input contrast if available (HCP has T2).

**5. The anchor masks as carver inputs: the recall route.**
- **Mechanism.** On MRI's fixed anatomy the anchor set alone identifies the target 84% of the time on held-out val, and the carver sees the anchor masks. It can learn "with these anchors, the answer was the accumbens".
- **Test:** `model.stage_b.carver_sees_anchors=False` (existing flag, capital `F`), never run properly.

**6. The heatmap is pulled to the region's centre.**
- **Mechanism.** The held-out centroid lands on background 54–74% of the time, 22–28 mm off. `field_centroid_on: always` pulls the heatmap towards the region's centre of mass, which lies ~20 mm from the target. The heatmap does not draw the mask, but it shares the carver's trunk, so this pull competes with locating the target.
- **Test:** `field_centroid_on: empty-only`, an existing setting.

**7. Resolution and receptive field.**
- **Mechanism.** At 128³ the held-out structures are long: the caudate spans ~50–60 mm. The carver trunk works at 64³ over a ~19-voxel window, while `B(I)` sees a theoretical ~39 voxels. The *effective* field is smaller, so the body's extent is decided locally and completed by templates. The synthetic shapes at 64³ fit the field better.
- **Test:** train Stage B on MRI resampled to 64³, as a cheap check; or a deeper or dilated carver trunk.

**8. The evaluation population is small and near-deterministic.**
- **Scale:** held-out val is 4 classes over 20 subjects, and test is 2 classes. Its anchor-set ceiling is 84% (val) and 99% (test).
- **Consequence:** this does not explain a Dice of 0.017, but it means any gain must be read per class and against the floor, never as a single number.

**9. One seed, run still in progress.** Everything above comes from epochs 0–9 of one seed. The full run and its evaluation (image replacement, and the impossible-prompt leak with and without the gate) come at the end.

## 5. Suggested next steps (none launched)

1. **Let B2 finish** and run its evaluation. The watcher does it automatically.
2. **Offline, no training:**
   - measure how often each held-out class appears as a *negative* inside the training fields (§4.1);
   - measure the image contrast between each held-out target and the trained neighbour it gets painted as (§4.4);
   - score the mapper coverage ideas (§4.3).
3. **Then one change per run, against B2:**
   - held-out classes excluded from the mask loss (§4.1; small code change);
   - `carver_sees_anchors=False` (§4.5; flag only);
   - more target-only classes (§4.2; corpus change).

   Two seeds each, read per class against the floors.
