# Remove the duplicate `documentation/Models/` notes

**Date:** 2026-09-23

## Rationale

`documentation/Models/` and `documentation/Model Info/` held the same 13 module
and phase notes (`Phase A/MODEL PHASE A.md`, `Phase B/MODEL PHASE B.md`, and
`modules/{BoundaryEncoder, BoundaryPretrainer, Carver, ConvBlock, Decoder,
Encoder, NamePrompt, NullHead, PosEnc3D, PositionalMapper3D, ResBlock}.md`).
Two copies of the architecture notes drift apart, so a reader cannot tell which
one describes the code.

`Models/` is the older copy, and it is out of date:

- git history: `Models/` was touched only by `f2067a6` (introduce documentation);
  `Model Info/` was updated again by `d14f201` (`mask_on: valid`) and `a1fc17a`
  (B0 baseline).
- file times: every `Models/` file dates from 2026-09-22 22:14; `Model Info/`
  runs from 22:23 to 2026-09-23 08:57.

`Model Info/` is the copy kept in sync with the code, so it stays.

## What changed

- Deleted `documentation/Models/` (all 13 files) with `git rm -r`.

## Not changed

- `documentation/Model Info/`: untouched.
- `documentation/.obsidian/workspace.json` still lists `Models/...` paths in its
  recently opened files. This is Obsidian UI state, not a link. Obsidian drops
  missing entries itself, so it was left alone.
- No code, config or test refers to `documentation/Models/`. A repo-wide grep
  found only the Obsidian workspace entries above.

## Recovery

`git checkout f2067a6 -- documentation/Models` restores the folder.
