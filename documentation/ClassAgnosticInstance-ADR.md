---
tags:
  - spatialvox
  - design
  - adr
aliases:
  - Instance selection §7 freeze
---
# ADR — freeze §7 for class-agnostic instance v1

**Status:** accepted (2026-09-24).  
**Branch:** `cursor/class-agnostic-instance-design-f966`.  
**Parent baseline:** B2 (`mask_on: valid`, dense carver) on the same corpus version.

Decisions locked before implementation. Changing one requires a new ADR row.

| # | question | decision |
|---|---|---|
| 1 | Scorer v1 | **Rule:** `score_k = where_raw(centroid(P_k))`. No learned scorer. |
| 2 | Proposals v1 | **Seed-flood** inside `dilate(where_raw, r)`. Multiple local-max seeds → `{P_k}`. |
| 3 | `B` widths | Keep **`[16, 32, 32]`** (B2). Frozen after class-free pretext; **no** relational mask loss into G. |
| 4 | Region dilate `r` | Start at **4** voxels; measure body coverage / neighbour invasion and adjust once. |
| 5 | V (gate) | **L/R VentralDC only.** Never select on R. |
| 6 | S mask loss | **None** for the instance path. |
| 7 | Corpus | **Train-only** triple stability + keep full val/test exposure stratum. |
| 8 | Null | **Score / mass threshold** (no NullHead required on the instance path). Keep B2 NullHead only while dense carver remains for comparison. |
| 9 | Multi-τ fields | **v2** (not v1). |
| 10 | Lesion simulation | **After** healthy R clears bars. |

**Pipeline kept from B2:** Stage A (frozen) → `PositionalMapper3D` → `where_raw`.  
**Pipeline replaced:** dense `Carver` answer path → propose + rule score → emit `P*`.
