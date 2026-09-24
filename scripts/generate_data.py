#!/usr/bin/env python
"""Write a synthetic corpus: packed primitives, painted, with anchor-first prompts.

    scripts/generate_data.py --config configs/synthetic-hard.yaml
    scripts/generate_data.py --config configs/synthetic-hard.yaml --smoke   # 12 scenes

The geometry is fixed by `src/synthetic.py`; what this script exists to vary is
the **appearance**, because appearance is what decides whether the relational
claim can be tested at all.

The proposal's §4 (`documentation/SpatialVox.md`, The claim) is explicit about the limit:
*"An unseen structure is segmented where `B(I)` carries a boundary... Two nuclei
that share an intensity are not."* On the original synthetic appearance a single
global threshold recovers every structure at **IoU 0.9998**, so `B(I)` gets every
boundary for free and the carver never has to learn a class. Transfer to an
unsupervised class is then trivially perfect, and says nothing.

`synthetic.paint` therefore takes three knobs that move the corpus along exactly
that axis, and `--report` prints where a setting lands:

``background`` / ``structure``  overlapping the two ranges removes the global
                                intensity gap
``texture``                     background lumps at a structure's spatial scale,
                                so the background is not a flat constant
``bias_field``                  a smooth multiplicative inhomogeneity, so no
                                *global* threshold is right anywhere

Measured on 64^3: the shipped easy appearance gives threshold IoU 0.9994, and
`configs/synthetic-hard.yaml` gives **0.0662**, against **0.0796** for the real
MRI corpus. That is the point of the hard setting - it is the same experiment at
MRI-grade separability, with the anatomy prior and Stage A's error removed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.config import load_config, parse_overrides
from src.data import build_examples, stabilize_relational_manifests, write_corpus, write_scene
from src.synthetic import SHAPE_NAMES, generate_scene
from src.vocab import Vocabulary


def separability(
    appearance, shape, spacing, margin, scenes: int = 4, size: float = 1.0
) -> tuple[float, float, float]:
    """The one number that decides what this corpus can test.

    ``(best threshold IoU against labels > 0, mean foreground, mean background)``.
    High means ``B(I)`` gets every boundary from a threshold, so a carver need
    never learn a class and transfer is free.
    """
    scores, foreground, background = [], [], []
    for seed in range(scenes):
        image, labels = generate_scene(900_000 + seed, shape, spacing, margin, appearance, size=size)
        mask = labels > 0
        best = 0.0
        for cut in np.arange(0.05, 0.95, 0.01):
            predicted = image > cut
            best = max(best, (predicted & mask).sum() / max((predicted | mask).sum(), 1))
        scores.append(best)
        foreground.append(image[mask].mean())
        background.append(image[~mask].mean())
    return float(np.mean(scores)), float(np.mean(foreground)), float(np.mean(background))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, help="config file (default configs/config.yaml)")
    parser.add_argument("--smoke", action="store_true", help="12 scenes, for a wiring check")
    parser.add_argument("--report", action="store_true", help="print the separability and stop")
    parser.add_argument("--set", dest="overrides", action="append", metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=parse_overrides(args.overrides))
    data, synthetic = cfg.data, cfg.synthetic
    shape = (int(data.resolution),) * 3
    spacing = tuple(float(v) for v in synthetic.get("spacing", (1.0, 1.0, 1.0)))
    margin = int(synthetic.get("margin", 2))
    appearance = synthetic.appearance.to_dict()

    iou, foreground, background = separability(
        appearance, shape, spacing, margin, size=float(synthetic.get("size", 1.0))
    )
    print(f"appearance: a global threshold recovers labels > 0 at IoU {iou:.4f}")
    print(f"            mean intensity {foreground:.3f} inside a structure, {background:.3f} outside")
    print("            (the real MRI corpus scores 0.0796; the easy synthetic one 0.9994)")
    if args.report:
        return 0

    vocab = Vocabulary(SHAPE_NAMES)
    counts = {split: int(n) for split, n in synthetic.scenes.to_dict().items()}
    if args.smoke:
        counts = {split: max(n // 40, 4) for split, n in counts.items()}
    root = Path(data.root)

    manifests: dict[str, list] = {}
    for split, n in counts.items():
        seed0 = int(synthetic.seeds[split])
        manifests[split] = []
        for index in range(n):
            scene_id = f"{split}_{index:05d}"
            image, labels = generate_scene(
                seed0 + index, shape, spacing, margin, appearance,
                max_attempts=int(synthetic.packing.max_scene_attempts),
                size=float(synthetic.get("size", 1.0)),
            )
            write_scene(root, scene_id, image, labels, spacing)
            manifests[split] += build_examples(
                scene_id, labels, vocab, spacing, int(data.n_anchors),
                shuffle_clauses=bool(data.get("shuffle_clauses", True)),
                triples=int(data.get("triples", 60)),
                locality=int(data.get("locality", 8)),
            )
            print(f"\r{split} {index + 1}/{n}  {len(manifests[split])} examples", end="", flush=True)
        print()

    manifests, stab = stabilize_relational_manifests(manifests, vocab)
    print(
        f"triple stability: kept {stab['examples_kept']}  "
        f"dropped {stab['examples_dropped_unstable_triple']}  "
        f"({stab['triples_colliding']}/{stab['triples_total']} colliding triples)"
    )

    write_corpus(
        root, vocab, manifests,
        shape=shape, spacing=spacing, n_anchors=int(data.n_anchors),
        targets=cfg.targets.to_dict(),
        shuffle_clauses=bool(data.get("shuffle_clauses", True)),
        extra={
            "source": "synthetic",
            "appearance": appearance,
            "threshold_iou": round(iou, 4),
            "triple_stability": "train-unique-target",
            "triple_stability_stats": stab,
        },
    )
    print(f"\nwritten to {root}: " + ", ".join(f"{k} {len(v)}" for k, v in manifests.items()))
    print(f"recorded threshold_iou {iou:.4f} in meta.json - the number that says what this "
          f"corpus can and cannot test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
