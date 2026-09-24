#!/usr/bin/env python
"""How often does one (anchors, directions) name 2+ structure centroids?

    .venv/bin/python scripts/hard_multi_targets.py --demo
    .venv/bin/python scripts/hard_multi_targets.py --scene 100206
    .venv/bin/python scripts/hard_multi_targets.py --scene 100206 --mode generator

This is the HARD centroid rule (``solutions_for`` / ``classify``), not the soft
``where_raw`` region.

Modes:
  exhaustive  — every triple of present structures x every assignment of 3
                pairwise-distinct directions
  generator   — same locality/triple sampling as ``anchor_first_examples``, but
                BEFORE the ``matches.sum() == 1`` filter (what would be dropped)
  manifest    — stored prompts only (should be 100% unique by construction)
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from itertools import combinations, product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.config import load_config
from src.data import Corpus, load_nifti
from src.geometry import (
    DIRECTIONS, centroids_world, direction_matrix, solutions_for, volume_center_world,
)

# Reuse demo corpus helper from the soft-region audit (same directory, not a package).
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "centroids_in_region", Path(__file__).resolve().parent / "centroids_in_region.py"
)
_cir = _ilu.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_cir)
build_demo_corpus = _cir.build_demo_corpus
prompts_for_scene = _cir.prompts_for_scene


def exhaustive(present, centroids, center, vocab, *, max_examples: int = 8):
    hist = collections.Counter()
    examples = []
    for trip in combinations(present, 3):
        for dirs in product(DIRECTIONS, repeat=3):
            if len(set(dirs)) != 3:
                continue
            sols = solutions_for(trip, dirs, centroids, present, center)
            n = len(sols)
            hist[n] += 1
            if n > 1 and len(examples) < max_examples:
                examples.append({
                    "anchors": [vocab.name(a) for a in trip],
                    "directions": list(dirs),
                    "solutions": [vocab.name(s) for s in sols],
                    "n": n,
                })
    return hist, examples


def generator_like(
    present, centroids, center, vocab, *,
    triples: int, locality: int, seed: int = 0, max_examples: int = 8,
):
    """Mirror ``anchor_first_examples`` but tally ``matches.sum()`` before filtering."""
    labels = [int(l) for l in present]
    codes, _ = direction_matrix(centroids, labels, center)
    positions = np.array([centroids[l] for l in labels], dtype=float)
    rng = np.random.default_rng(seed)
    hist = collections.Counter()
    examples = []
    seen: set[tuple[int, ...]] = set()
    n_anchors = 3
    for _ in range(int(triples)):
        seed_i = int(rng.integers(len(labels)))
        order = np.argsort(np.linalg.norm(positions - positions[seed_i], axis=1))
        window = order[: max(int(locality), n_anchors)]
        if len(window) < n_anchors:
            continue
        columns = np.sort(rng.choice(window, n_anchors, replace=False))
        key = tuple(int(c) for c in columns)
        if key in seen:
            continue
        seen.add(key)
        relative = codes[:, columns]
        for row in range(len(labels)):
            if row in columns:
                continue
            wanted = relative[row]
            if (wanted < 0).any() or len(set(wanted.tolist())) != n_anchors:
                continue
            matches = (relative == wanted).all(axis=1).copy()
            matches[list(columns)] = False
            n = int(matches.sum())
            hist[n] += 1
            if n > 1 and len(examples) < max_examples:
                examples.append({
                    "anchors": [vocab.name(labels[int(c)]) for c in columns],
                    "directions": [DIRECTIONS[int(d)] for d in wanted],
                    "solutions": [vocab.name(labels[i]) for i, m in enumerate(matches) if m],
                    "n": n,
                })
    return hist, examples, len(seen)


def manifest_check(rows, present, centroids, center, vocab, *, max_examples: int = 8):
    hist = collections.Counter()
    examples = []
    for row in rows:
        sols = solutions_for(row["anchors"], row["directions"], centroids, present, center)
        n = len(sols)
        hist[n] += 1
        if n != 1 and len(examples) < max_examples:
            examples.append({
                "id": row.get("id", ""),
                "anchors": [vocab.name(a) for a in row["anchors"]],
                "directions": list(row["directions"]),
                "stored_target": vocab.name(row["target"]),
                "solutions": [vocab.name(s) for s in sols],
                "n": n,
            })
    return hist, examples


def print_hist(title: str, hist: collections.Counter, examples: list) -> None:
    total = sum(hist.values()) or 1
    multi = sum(c for n, c in hist.items() if n > 1)
    print(f"\n=== {title} ===")
    print(f"conjunctions: {sum(hist.values())}")
    for n in sorted(hist):
        print(f"  n_solutions={n}: {hist[n]:6d}  ({hist[n] / total:6.2%})")
    print(f"  fraction with n_solutions > 1: {multi / total:.2%}")
    if examples:
        print("  examples:")
        for ex in examples[:5]:
            print(f"    {ex}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--mode", choices=("all", "exhaustive", "generator", "manifest"), default="all")
    parser.add_argument("--triples", type=int, default=None, help="generator mode; default from config data.triples")
    parser.add_argument("--locality", type=int, default=None)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    if args.demo:
        corpus = build_demo_corpus(Path("/tmp/spatialvox-hard-multi-demo"))
        scene = "demo_000"
        triples = int(args.triples or 80)
        locality = int(args.locality or 8)
    else:
        cfg = load_config(args.config)
        corpus = Corpus.load(cfg.data.root)
        scene = args.scene or corpus.scene_ids("train")[0]
        triples = int(args.triples or cfg.data.get("triples", 60))
        locality = int(args.locality or cfg.data.get("locality", 8))

    labels = load_nifti(corpus.root / "scenes" / scene / "labels.nii.gz", np.int16)
    vocab = corpus.vocab
    centroids = centroids_world(labels, len(vocab), corpus.spacing)
    center = volume_center_world(labels.shape, corpus.spacing)
    present = [int(v) for v in np.unique(labels) if v != 0]
    print(f"scene {scene}  structures {len(present)}  locality {locality}  triples {triples}")

    out: dict = {"scene": scene, "n_structures": len(present), "modes": {}}
    modes = ("exhaustive", "generator", "manifest") if args.mode == "all" else (args.mode,)

    if "exhaustive" in modes:
        hist, examples = exhaustive(present, centroids, center, vocab)
        print_hist("exhaustive (all triples x distinct directions)", hist, examples)
        out["modes"]["exhaustive"] = {
            "histogram": {str(k): v for k, v in sorted(hist.items())},
            "fraction_multi": sum(c for n, c in hist.items() if n > 1) / max(sum(hist.values()), 1),
            "examples": examples,
        }

    if "generator" in modes:
        hist, examples, n_trips = generator_like(
            present, centroids, center, vocab, triples=triples, locality=locality,
        )
        print_hist(
            f"generator-like BEFORE uniqueness filter ({n_trips} triples drawn)",
            hist, examples,
        )
        out["modes"]["generator"] = {
            "triples_drawn": n_trips,
            "histogram": {str(k): v for k, v in sorted(hist.items())},
            "fraction_multi": sum(c for n, c in hist.items() if n > 1) / max(sum(hist.values()), 1),
            "examples": examples,
        }

    if "manifest" in modes:
        rows = prompts_for_scene(corpus, scene)
        hist, examples = manifest_check(rows, present, centroids, center, vocab)
        print_hist(f"manifest prompts on this scene ({len(rows)} rows)", hist, examples)
        out["modes"]["manifest"] = {
            "n_prompts": len(rows),
            "histogram": {str(k): v for k, v in sorted(hist.items())},
            "fraction_multi": sum(c for n, c in hist.items() if n > 1) / max(sum(hist.values()), 1),
            "examples": examples,
        }

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
