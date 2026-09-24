#!/usr/bin/env python
"""Cross-patient collision: same relational triple, different targets?

    .venv/bin/python scripts/triple_cross_patient.py
    .venv/bin/python scripts/triple_cross_patient.py --demo-fixed  # N jittered copies of one layout
    .venv/bin/python scripts/triple_cross_patient.py --json /tmp/triple-collisions.json

A triple is identified by the **unordered set of (anchor name, direction) pairs**
— clause order does not matter; pairing does. Example::

    {(Left-Thalamus, superior), (Brain-Stem, medial), (Left-Putamen, anterior)}

For each such key, collect every (scene, target) that uses it. Report keys that
map to **more than one target name** across the whole corpus (and within one
scene, which uniqueness forbids).
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.config import load_config
from src.data import Corpus, build_examples, write_corpus, write_scene
from src.vocab import Vocabulary


def triple_key(row: dict, vocab) -> frozenset[tuple[str, str]]:
    """Order-invariant relational triple: set of (anchor_name, direction)."""
    return frozenset(
        (vocab.name(int(a)), str(d))
        for a, d in zip(row["anchors"], row["directions"])
    )


def load_all_rows(corpus: Corpus) -> list[dict]:
    rows: list[dict] = []
    for split in ("train", "val", "test"):
        path = corpus.root / f"{split}.jsonl"
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def analyse(corpus: Corpus) -> dict:
    vocab = corpus.vocab
    rows = load_all_rows(corpus)
    # key -> target -> list of scenes
    by_triple: dict[frozenset, dict[str, list[str]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    within_scene_multi = []
    for row in rows:
        key = triple_key(row, vocab)
        target = vocab.name(int(row["target"]))
        scene = row["scene"]
        by_triple[key][target].append(scene)

    # within one scene: same key, two targets
    per_scene: dict[tuple[str, frozenset], set[str]] = collections.defaultdict(set)
    for row in rows:
        key = triple_key(row, vocab)
        per_scene[(row["scene"], key)].add(vocab.name(int(row["target"])))
    for (scene, key), targets in per_scene.items():
        if len(targets) > 1:
            within_scene_multi.append({
                "scene": scene,
                "triple": sorted(key),
                "targets": sorted(targets),
            })

    collisions = []
    same_target_reused = 0
    unique_once = 0
    for key, target_map in by_triple.items():
        n_targets = len(target_map)
        n_uses = sum(len(scenes) for scenes in target_map.values())
        n_scenes = len({s for scenes in target_map.values() for s in scenes})
        if n_targets == 1 and n_uses == 1:
            unique_once += 1
        elif n_targets == 1 and n_uses > 1:
            same_target_reused += 1
        elif n_targets > 1:
            collisions.append({
                "triple": sorted(key),
                "targets": {
                    t: {"n": len(scenes), "scenes": sorted(set(scenes))[:12]}
                    for t, scenes in sorted(target_map.items())
                },
                "n_targets": n_targets,
                "n_uses": n_uses,
                "n_scenes": n_scenes,
            })

    collisions.sort(key=lambda c: (-c["n_targets"], -c["n_uses"]))
    return {
        "n_rows": len(rows),
        "n_scenes": len({r["scene"] for r in rows}),
        "n_distinct_triples": len(by_triple),
        "triples_used_once": unique_once,
        "triples_reused_same_target": same_target_reused,
        "triples_with_different_targets": len(collisions),
        "fraction_triples_collide": len(collisions) / max(len(by_triple), 1),
        "within_scene_multi_target": within_scene_multi,
        "collisions": collisions,
    }


def build_fixed_anatomy_corpus(root: Path, *, n_subjects: int = 20, shift: int = 2) -> Corpus:
    """Same 8-box layout as tests, small per-subject jitter — MRI-like fixed anatomy."""
    structures = {
        "alpha": ((7, 8, 6), 3),
        "beta": ((8, 20, 9), 2),
        "gamma": ((14, 9, 22), 3),
        "delta": ((15, 22, 19), 2),
        "sigma": ((21, 7, 11), 2),
        "omega": ((22, 19, 7), 3),
        "kappa": ((25, 12, 24), 2),
        "rho": ((18, 26, 26), 2),
    }
    names = tuple(structures)
    resolution, spacing = 32, (1.0, 1.0, 1.0)
    vocab = Vocabulary(names)
    if root.exists():
        import shutil
        shutil.rmtree(root)
    root.mkdir(parents=True)
    all_rows: list[dict] = []
    for i in range(n_subjects):
        rng = np.random.default_rng(1000 + i)
        labels = np.zeros((resolution,) * 3, dtype=np.uint16)
        image = np.full(labels.shape, 0.1, dtype=np.float32)
        for index, (name, (centre, half)) in enumerate(structures.items()):
            offset = rng.integers(-shift, shift + 1, 3) if shift else np.zeros(3, int)
            low = [int(np.clip(c + o - half, 1, resolution - 2)) for c, o in zip(centre, offset)]
            high = [int(np.clip(l + 2 * half + 1, 2, resolution - 1)) for l in low]
            box = tuple(slice(l, h) for l, h in zip(low, high))
            if labels[box].any():
                # fall back to canonical box if overlap
                low = [int(np.clip(c - half, 1, resolution - 2)) for c in centre]
                high = [int(np.clip(l + 2 * half + 1, 2, resolution - 1)) for l in low]
                box = tuple(slice(l, h) for l, h in zip(low, high))
            labels[box] = index + 1
            image[box] = 0.5 + 0.03 * index
        scene_id = f"subj_{i:03d}"
        write_scene(root, scene_id, image, labels, spacing)
        all_rows += build_examples(
            scene_id, labels, vocab, spacing, 3,
            shuffle_clauses=True, triples=60, locality=8,
        )
    # split subjects 80/10/10 by scene id for a real-looking corpus
    scenes = sorted({r["scene"] for r in all_rows})
    n = len(scenes)
    train_s, val_s = set(scenes[: int(0.8 * n)]), set(scenes[int(0.8 * n): int(0.9 * n)])
    manifests = {
        "train": [r for r in all_rows if r["scene"] in train_s],
        "val": [r for r in all_rows if r["scene"] in val_s],
        "test": [r for r in all_rows if r["scene"] not in train_s and r["scene"] not in val_s],
    }
    write_corpus(
        root, vocab, manifests,
        shape=(resolution,) * 3, spacing=spacing, n_anchors=3,
        targets={"train": list(names), "val": list(names), "test": list(names)},
        extra={"demo": "fixed-anatomy-jitter"},
    )
    return Corpus.load(root)


def print_report(report: dict, *, show: int = 15) -> None:
    print(f"rows {report['n_rows']}  scenes {report['n_scenes']}  "
          f"distinct triples {report['n_distinct_triples']}")
    print(f"  used once:                    {report['triples_used_once']}")
    print(f"  reused, always same target:   {report['triples_reused_same_target']}")
    print(f"  DIFFERENT targets across uses:{report['triples_with_different_targets']}  "
          f"({report['fraction_triples_collide']:.2%} of distinct triples)")
    print(f"  within-scene multi-target:    {len(report['within_scene_multi_target'])}  "
          "(should be 0)")
    if report["within_scene_multi_target"][:3]:
        print("  within-scene examples:", report["within_scene_multi_target"][:3])
    print()
    cols = report["collisions"][:show]
    if not cols:
        print("No cross-patient (or cross-row) triple→different-target collisions.")
        return
    print(f"first {len(cols)} collisions:")
    for c in cols:
        targets = ", ".join(
            f"{t} (n={info['n']}, scenes={info['scenes'][:4]}{'…' if len(info['scenes'])>4 else ''})"
            for t, info in c["targets"].items()
        )
        print(f"  triple={c['triple']}")
        print(f"    → {targets}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--demo-fixed", action="store_true",
                        help="build N jittered copies of a fixed layout (MRI-like)")
    parser.add_argument("--subjects", type=int, default=20)
    parser.add_argument("--shift", type=int, default=2, help="voxel jitter for --demo-fixed")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--show", type=int, default=15)
    args = parser.parse_args()

    if args.demo_fixed:
        corpus = build_fixed_anatomy_corpus(
            Path("/tmp/spatialvox-triple-cross-patient"),
            n_subjects=args.subjects, shift=args.shift,
        )
        print(f"demo-fixed: {args.subjects} subjects, shift±{args.shift}")
    else:
        cfg = load_config(args.config)
        corpus = Corpus.load(cfg.data.root)
        print(f"corpus {corpus.root}")

    report = analyse(corpus)
    print_report(report, show=args.show)
    if args.json:
        # JSON needs listable keys
        payload = dict(report)
        payload["collisions"] = report["collisions"]
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
