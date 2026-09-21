#!/usr/bin/env python
"""Rewrite a corpus's manifests from its existing label volumes.

    scripts/rebuild_manifests.py
    scripts/rebuild_manifests.py --set data.selection=anchor-first
    scripts/rebuild_manifests.py --set data.anchor_pool=5 --dry-run

Every prompt-side knob - `data.selection`, `data.anchor_pool`,
`data.shuffle_clauses`, `data.triples`, `data.locality`, `data.unique_only` -
changes only the manifests, never the volumes. `scripts/import_mri.py` re-reads
every HCP subject and rewrites every NIfTI, which is minutes of I/O to change a
line of JSON; this reads `scenes/*/labels.nii.gz` and rewrites the JSONL.

`docs/experiments_plan.md` §5.1 has referred to this script for some time. It
did not exist, which is why `data.anchor_pool` was never actually swept.

The vocabulary, the scene list, the split assignment and the target-class split
are all preserved exactly; only `{train,val,test}.jsonl` and the generation keys
of `meta.json` are rewritten.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.config import load_config, parse_overrides
from src.data import Corpus, build_examples, load_nifti, write_corpus


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, help="corpus directory (default: data.root)")
    parser.add_argument("--dry-run", action="store_true", help="report counts, write nothing")
    parser.add_argument("--set", dest="overrides", action="append", metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(overrides=parse_overrides(args.overrides))
    root = args.root or Path(cfg.data.root)
    corpus = Corpus.load(root)
    data = cfg.data

    selection = str(getattr(data, "selection", "target-first"))
    n_anchors = int(data.n_anchors)
    pool = int(data.anchor_pool)
    shuffle = bool(getattr(data, "shuffle_clauses", True))
    triples = int(getattr(data, "triples", 300))
    locality = int(getattr(data, "locality", 8))
    unique_only = bool(getattr(data, "unique_only", True))

    print(f"corpus {root}  selection={selection}  n_anchors={n_anchors}  pool={pool}")
    print(f"  shuffle_clauses={shuffle}  unique_only={unique_only}", end="")
    print(f"  triples={triples}  locality={locality}" if selection == "anchor-first" else "")

    splits = [p.stem for p in sorted(root.glob("*.jsonl"))]
    manifests: dict[str, list[dict]] = {}
    stats: dict[str, int] = {}
    shape = None
    for split in splits:
        rows = []
        for scene_id in corpus.scene_ids(split):
            labels = load_nifti(root / "scenes" / scene_id / "labels.nii.gz", np.int16)
            shape = shape or labels.shape
            rows += build_examples(
                scene_id, labels, corpus.vocab, corpus.spacing, n_anchors,
                pool=pool, shuffle_clauses=shuffle, selection=selection,
                triples=triples, locality=locality, unique_only=unique_only,
                stats=stats,
            )
        manifests[split] = rows
        before = sum(1 for _ in (root / f"{split}.jsonl").read_text().splitlines() if _.strip())
        print(f"  {split:5s} {before:7d} -> {len(rows):7d} examples")

    if stats.get("dropped_ambiguous"):
        print(f"  dropped {stats['dropped_ambiguous']} examples whose conjunction was not unique")
    if stats.get("pool_fallbacks"):
        print(
            f"  WARNING: {stats['pool_fallbacks']} examples fell back to the deterministic"
            " nearest anchor set (the pool window held no feasible triple)"
        )
    if args.dry_run:
        print("\ndry run: nothing written")
        return 0

    write_corpus(
        root, corpus.vocab, manifests,
        shape=shape or corpus.shape,
        spacing=corpus.spacing,
        n_anchors=n_anchors,
        targets=corpus.meta["targets"],
        anchor_pool=pool,
        selection=selection,
        shuffle_clauses=shuffle,
        extra={k: v for k, v in corpus.meta.items() if k in ("source", "label_scheme", "n_subjects", "origin")},
    )
    print(f"\nwritten to {root}")
    print("NOTE: manifests changed, so every checkpoint trained on the old ones is"
          " not comparable. Recompute the prompt-blind baseline before quoting Dice.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
