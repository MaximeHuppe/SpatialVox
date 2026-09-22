#!/usr/bin/env python
"""Audit a corpus: the numbers CLAUDE.md requires beside any reported result.

    .venv/bin/python scripts/corpus_report.py
    .venv/bin/python scripts/corpus_report.py --split val --scenes 400

Run this after every `import_mri.py` or `rebuild_manifests.py`. It needs no
checkpoint and no GPU.

Three things, none of which can be inferred from a Dice number:

1. **P(target | anchor set)** - how often the target is recoverable from the
   anchor identities alone, without parsing a direction word. This is the
   shortcut the model will take if it can. Target-first generation put it at
   98.9%, which is *higher* than solving the conjunction (94.5%), so ignoring
   the prompt was strictly better than reading it.

2. **The same ceiling per class-filtered curve.** Filtering a curve by target
   class inflates the shortcut, because conditioning on fewer classes makes the
   anchor identities more informative. On the anchor-first corpus: 42.1% over
   all classes, 69.7% over the 8 supervised ones, 82.9% over the 4 held out.
   A selection accuracy must be read against the ceiling of *its own*
   population, never the corpus-level one.

3. **The prompt-blind Dice baseline** - "of the non-anchor structures, take the
   one nearest the anchor centroid". Never reads the prompt. On the target-first
   corpus this project used to carry it scored 0.775-1.000, above what the
   trained model achieved, which is what made every Dice there uninterpretable.
   Anchor-first generation is what brought it down; this reports where it sits
   now, and **no Dice from this project is readable without it**.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.data import Corpus, load_nifti
from src.geometry import (
    AmbiguousDirection, centroids_world, classify, volume_center_world,
)


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0, b > 0
    return float(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))


def ceilings(rows: list[dict], label: str) -> None:
    """How well the anchor identities alone predict the target, on one population."""
    if not rows:
        print(f"  {label:34s} (empty)")
        return
    by_set: dict[frozenset, collections.Counter] = collections.defaultdict(collections.Counter)
    for row in rows:
        by_set[frozenset(row["anchors"])][row["target"]] += 1
    best = sum(c.most_common(1)[0][1] for c in by_set.values()) / len(rows)
    classes = collections.Counter(row["target"] for row in rows)
    print(
        f"  {label:34s} n={len(rows):6d}  anchor-set {best:6.1%}"
        f"   commonest class {classes.most_common(1)[0][1] / len(rows):5.1%}"
        f"   uniform {1 / len(classes):5.1%}"
    )


def prompt_blind(corpus: Corpus, rows: list[dict], label: str, limit: int) -> None:
    """Dice of a rule that never reads the prompt, plus the conjunction's uniqueness."""
    if not rows:
        return
    picked = rows if len(rows) <= limit else [
        rows[int(i)] for i in np.random.default_rng(0).permutation(len(rows))[:limit]
    ]
    cache: dict[str, tuple] = {}
    nearest, unique = [], []
    for row in picked:
        if row["scene"] not in cache:
            labels = load_nifti(corpus.root / "scenes" / row["scene"] / "labels.nii.gz", np.int16)
            cache[row["scene"]] = (
                labels,
                centroids_world(labels, len(corpus.vocab), corpus.spacing),
                volume_center_world(labels.shape, corpus.spacing),
                [int(v) for v in np.unique(labels) if v],
            )
        labels, centroids, middle, present = cache[row["scene"]]
        anchors = list(row["anchors"])
        candidates = [l for l in present if l not in anchors]
        if not candidates:
            continue
        centre = np.mean([centroids[a] for a in anchors], axis=0)
        guess = min(candidates, key=lambda l: float(np.linalg.norm(centroids[l] - centre)))
        nearest.append(overlap(labels == guess, labels == row["target"]))

        def satisfied(label_id: int) -> bool:
            for anchor, direction in zip(anchors, row["directions"]):
                try:
                    if classify(centroids[label_id], centroids[anchor], middle) != direction:
                        return False
                except AmbiguousDirection:
                    return False
            return True

        unique.append(sum(1 for l in candidates if satisfied(l)) == 1)
    print(
        f"  {label:34s} n={len(nearest):6d}  prompt-blind Dice {np.mean(nearest):.4f}"
        f"   conjunction unique {np.mean(unique):6.1%}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="data/mri")
    parser.add_argument("--split", default="val", help="split to audit (default val)")
    parser.add_argument("--scenes", type=int, default=200, help="examples sampled for the Dice baseline")
    args = parser.parse_args()

    corpus = Corpus.load(args.root)
    meta = corpus.meta
    names = list(corpus.vocab.names)
    name_of = lambda label: names[label - 1]

    print(f"corpus {args.root}")
    print(f"  selection      {meta.get('selection', 'anchor-first')}")
    print(f"  shuffle_clauses{'':1s} {meta.get('shuffle_clauses', True)}")
  
    print(f"  examples  {meta.get('examples')}")

    rows = [json.loads(line) for line in (corpus.root / f"{args.split}.jsonl").read_text().splitlines() if line.strip()]
    populations = [
        ("all target classes", set(names)),
        ("val_id (targets.train)", set(meta["targets"]["train"])),
        ("transfer (targets.val)", set(meta["targets"]["val"])),
        ("test classes (targets.test)", set(meta["targets"]["test"])),
    ]

    print(f"\nshortcut ceilings on '{args.split}' -- read every selection accuracy against ITS OWN row")
    for label, classes in populations:
        ceilings([r for r in rows if name_of(r["target"]) in classes], label)

    print(f"\nprompt-blind Dice on '{args.split}' -- the floor any reported Dice must clear")
    for label, classes in populations:
        prompt_blind(corpus, [r for r in rows if name_of(r["target"]) in classes], label, args.scenes)

    print(
        "\nCLAUDE.md: a Dice without its prompt-blind floor is not interpretable,"
        "\nand a shortcut ceiling must be read against its own population's row."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
