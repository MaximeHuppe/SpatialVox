#!/usr/bin/env python
"""Count structure centroids inside each prompt's soft region on one volume.

    .venv/bin/python scripts/centroids_in_region.py --scene train_00000
    .venv/bin/python scripts/centroids_in_region.py --config configs/config.yaml --scene 100206
    .venv/bin/python scripts/centroids_in_region.py --demo   # tiny hand-built volume, no corpus

For every prompt of one scene, build the same soft conjunction the mapper uses
(``where = Π sigmoid(margin_i / tau)``) and evaluate it at every present
structure centroid (anchors excluded). That answers: "how many structure
centres fall inside the region this prompt describes?" — which is a different
question from ``solutions_for`` (hard centroid classify, unique by construction).

Thresholds mirror the project: ``> 0.5`` is the gate; ``> 0.05`` is the field
``L_far`` / coverage measurements use.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.config import load_config
from src.data import Corpus, build_examples, load_nifti, write_corpus, write_scene
from src.geometry import DIRECTIONS, centroids_world, solutions_for, volume_center_world
from src.mapper import margin_at
from src.vocab import Vocabulary


def prompts_for_scene(corpus: Corpus, scene_id: str) -> list[dict]:
    """Every manifest row for this volume, across splits (no target-class filter)."""
    rows: list[dict] = []
    seen: set[str] = set()
    for split in ("train", "val", "test"):
        path = corpus.root / f"{split}.jsonl"
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["scene"] != scene_id:
                continue
            key = row.get("id") or json.dumps(row, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return rows


def where_at_centroids(
    candidate_centroids: np.ndarray,
    anchor_centroids: np.ndarray,
    direction_ids: list[int],
    center: np.ndarray,
    tau: float,
) -> np.ndarray:
    """``where_raw`` at each candidate centroid: ``[K]`` in ``(0, 1)``.

    Closed form — no full volume. Matches ``PositionalMapper3D``'s product of
    per-clause sigmoids at those points.
    """
    k = len(candidate_centroids)
    if k == 0:
        return np.zeros(0, dtype=np.float64)
    a = len(direction_ids)
    points = np.repeat(candidate_centroids[:, None, :], a, axis=1)  # [K, A, 3]
    anchors = np.repeat(anchor_centroids[None, :, :], k, axis=0)  # [K, A, 3]
    ids = torch.as_tensor(direction_ids, dtype=torch.long).expand(k, a)
    margins = margin_at(
        torch.as_tensor(points, dtype=torch.float32),
        torch.as_tensor(anchors, dtype=torch.float32),
        ids,
        center,
    )
    where = torch.sigmoid(margins / float(tau)).prod(dim=-1)
    return where.detach().cpu().numpy()


def analyse_scene(
    corpus: Corpus,
    scene_id: str,
    *,
    tau: float,
    thresholds: tuple[float, ...] = (0.5, 0.05),
    targets_filter: list[str] | None = None,
) -> dict:
    labels = load_nifti(corpus.root / "scenes" / scene_id / "labels.nii.gz", np.int16)
    centroids = centroids_world(labels, len(corpus.vocab), corpus.spacing)
    center = volume_center_world(labels.shape, corpus.spacing)
    present = [int(v) for v in np.unique(labels) if v != 0]
    vocab = corpus.vocab

    rows = prompts_for_scene(corpus, scene_id)
    if targets_filter:
        allowed = {vocab.label(name) for name in targets_filter}
        rows = [r for r in rows if int(r["target"]) in allowed]

    per_threshold = {t: collections.Counter() for t in thresholds}
    multi = {t: 0 for t in thresholds}
    details = []

    for row in rows:
        anchors = [int(a) for a in row["anchors"]]
        directions = list(row["directions"])
        target = int(row["target"])
        hard = solutions_for(anchors, directions, centroids, present, center)
        direction_ids = [DIRECTIONS.index(d) for d in directions]
        anchor_cent = np.stack([centroids[a] for a in anchors], axis=0)

        candidates = [lab for lab in present if lab not in anchors]
        cand_cent = np.stack([centroids[lab] for lab in candidates], axis=0)
        scores = where_at_centroids(cand_cent, anchor_cent, direction_ids, center, tau)

        entry = {
            "id": row.get("id", ""),
            "target": vocab.name(target),
            "anchors": [vocab.name(a) for a in anchors],
            "directions": directions,
            "solutions_for": [vocab.name(s) for s in hard],
            "n_solutions_for": len(hard),
            "inside": {},
        }
        for thr in thresholds:
            inside = [candidates[i] for i, s in enumerate(scores) if s > thr]
            names = [vocab.name(lab) for lab in inside]
            entry["inside"][str(thr)] = {
                "n": len(inside),
                "names": names,
                "target_inside": target in inside,
                "scores": {
                    vocab.name(candidates[i]): float(scores[i])
                    for i, s in enumerate(scores) if s > thr
                },
            }
            per_threshold[thr][len(inside)] += 1
            if len(inside) > 1:
                multi[thr] += 1
        details.append(entry)

    return {
        "scene": scene_id,
        "n_prompts": len(rows),
        "n_structures": len(present),
        "tau": tau,
        "histogram": {str(t): dict(sorted(c.items())) for t, c in per_threshold.items()},
        "fraction_multi": {str(t): (multi[t] / len(rows) if rows else 0.0) for t in thresholds},
        "fraction_solutions_not_one": (
            sum(1 for d in details if d["n_solutions_for"] != 1) / len(details) if details else 0.0
        ),
        "prompts": details,
    }


def build_demo_corpus(root: Path) -> Corpus:
    """Eight boxes — same layout as ``tests/conftest.py``, one scene, all prompts."""
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
    labels = np.zeros((resolution,) * 3, dtype=np.uint16)
    image = np.full(labels.shape, 0.1, dtype=np.float32)
    for index, (name, (centre, half)) in enumerate(structures.items()):
        low = [c - half for c in centre]
        high = [c + half + 1 for c in centre]
        box = tuple(slice(l, h) for l, h in zip(low, high))
        labels[box] = index + 1
        image[box] = 0.5 + 0.03 * index
    vocab = Vocabulary(names)
    if root.exists():
        import shutil
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    write_scene(root, "demo_000", image, labels, spacing)
    rows = build_examples(
        "demo_000", labels, vocab, spacing, 3,
        shuffle_clauses=True, triples=80, locality=8,
    )
    write_corpus(
        root, vocab, {"train": rows, "val": [], "test": []},
        shape=(resolution,) * 3, spacing=spacing, n_anchors=3,
        targets={"train": list(names), "val": [], "test": []},
        extra={"demo": True},
    )
    return Corpus.load(root)


def print_report(report: dict, *, show: int = 8) -> None:
    print(f"scene {report['scene']}  prompts {report['n_prompts']}  "
          f"structures {report['n_structures']}  tau {report['tau']}")
    print(f"solutions_for ≠ 1: {report['fraction_solutions_not_one']:.1%}  "
          "(should be 0% on manifests)")
    print()
    for thr, hist in report["histogram"].items():
        multi = report["fraction_multi"][thr]
        print(f"centroids with where_raw > {thr}  (fraction of prompts with n>1: {multi:.1%})")
        total = report["n_prompts"] or 1
        for n, count in sorted((int(k), v) for k, v in hist.items()):
            print(f"  n={n:2d}  {count:4d}  ({count / total:5.1%})")
        print()
    print(f"first {min(show, len(report['prompts']))} prompts:")
    for entry in report["prompts"][:show]:
        soft = entry["inside"]["0.5"]
        print(
            f"  target={entry['target']:<16}  "
            f"solutions_for={entry['solutions_for']}  "
            f"in_region@0.5 n={soft['n']} {soft['names']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--scene", type=str, default=None, help="scene id; default = first train scene")
    parser.add_argument("--tau", type=float, default=None, help="default: config mapper.tau")
    parser.add_argument("--targets", choices=("all", "train"), default="all",
                        help="all prompts on the volume, or only targets.train rows")
    parser.add_argument("--json", type=Path, default=None, help="write full per-prompt report")
    parser.add_argument("--show", type=int, default=8)
    parser.add_argument("--demo", action="store_true", help="build a tiny volume under /tmp and analyse it")
    args = parser.parse_args()

    if args.demo:
        corpus = build_demo_corpus(Path("/tmp/spatialvox-centroids-in-region-demo"))
        scene = "demo_000"
        tau = float(args.tau if args.tau is not None else 0.5)
        targets_filter = None
    else:
        cfg = load_config(args.config)
        corpus = Corpus.load(cfg.data.root)
        scene = args.scene or corpus.scene_ids("train")[0]
        tau = float(args.tau if args.tau is not None else cfg.model.stage_b.mapper.tau)
        targets_filter = list(cfg.targets.train) if args.targets == "train" else None

    report = analyse_scene(corpus, scene, tau=tau, targets_filter=targets_filter)
    print_report(report, show=args.show)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
