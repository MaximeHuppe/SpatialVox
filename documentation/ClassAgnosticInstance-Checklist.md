---
tags:
  - spatialvox
  - checklist
---
# Class-agnostic instance — assumptions & checks (printable)

Extract of [[ClassAgnosticInstance]] §§4–5 for review meetings. Full discussion lives there.

## Assumptions (must not be silent)

### Task / language
- [ ] T1 Exactly 3 clauses (v1)
- [ ] T2 Closed 6 directions; clause = target rel. anchor
- [ ] T3 Target never an input (name/mask/centroid/id)
- [ ] T4 Distinct anchors & directions
- [ ] T6 Closed 23-name compiler (not VLM yet)

### Corpus / prompts
- [ ] C1 Healthy train only
- [ ] C3 Anchor-first generation
- [ ] C4 **Per-scene** prompts (no shared template per class)
- [ ] C5 Many prompts/structure OK; do not collapse to 2–3 shared
- [ ] C6 Per-scene uniqueness
- [ ] C7 Train-defined triple stability; keep exposure stratum on val/test
- [ ] C8 Truth = centroid satisfies conjunction

### Anchors
- [ ] A1 Stage A frozen in relational stage
- [ ] A3 Names stop at Stage A
- [ ] A5 Held-out names may still be anchors (say so; not full lesion claim)
- [ ] A7 Tumour must not need to be a Stage A class

### Region / instances
- [ ] R1 `where_raw` is prior, not mask
- [ ] R3 Body may extend outside high where — instance must extend
- [ ] P1 Answer is one connected body
- [ ] P2 Proposals class-agnostic (no class ids in G)
- [ ] P3 Separability under features — risk on touching GM
- [ ] S1 Prefer score = `where_raw(centroid(P_k))`
- [ ] S4 No dense “never paint held-out” mask training of G

### Eval
- [ ] E1 Parent = B2, same corpus version
- [ ] E3 Geometry twin mandatory
- [ ] E4 ≥2 seeds for small deltas
- [ ] E5 Counterfactuals retained
- [ ] L1–L4 Tumour leap **not** assumed from healthy R

## Checks before claiming transfer

- [ ] Mapper gate still high
- [ ] Oracle instance ceiling ~0.97 on unique prompts
- [ ] Proposal recall@K ≥0.9 on S
- [ ] Affinity ≥ intensity on visible borders; invisible canary reported
- [ ] R Dice ≫ B2; neighbour-paint down; twin delta reported
- [ ] best.pt not chosen on R
- [ ] Hippocampus own row

## Held-out pools (F0 proposal)

| pool | classes | use |
|---|---|---|
| S | Thalamus, Pallidum, Amygdala, Accumbens L/R | B2 train targets; optional scorer train only |
| V | VentralDC L/R | gate / early stop |
| R | Caudate, Putamen, Hippocampus L/R | report only |
| L | ventricles, Brain-Stem, … | anchors / pretext bodies |
