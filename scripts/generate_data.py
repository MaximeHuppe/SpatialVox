#!/usr/bin/env python
"""Generate the synthetic corpus.

    scripts/generate_data.py                       # the full corpus from configs/config.yaml
    scripts/generate_data.py --smoke               # 8 / 2 / 2 scenes into data/smoke
    scripts/generate_data.py --set data.resolution=128

Scene seeds are disjoint across splits, and a scene is reproducible from its seed
alone, so a corpus can be rebuilt exactly or extended without touching what is
already there.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, parse_overrides
from src.data import build_examples, check_scene, write_corpus, write_scene
from src.synthetic import SHAPE_NAMES, generate_scene
from src.vocab import Vocabulary

SMOKE = {"synthetic.scenes": {"train": 8, "val": 2, "test": 2}, "data.root": "data/smoke"}
SYNTHETIC_TARGETS = {
    "train": list(SHAPE_NAMES[:7]),
    "val": list(SHAPE_NAMES[7:9]),
    "test": [SHAPE_NAMES[9]],
}


def _targets(cfg) -> dict[str, list[str]]:
    """Honour ``targets`` when it names primitives; otherwise the 7/2/1 split.

    ``configs/config.yaml: targets`` is the MRI class split. Generating synthetic
    data with those names would write an empty manifest, so this falls back.
    """
    targets = {split: list(names) for split, names in cfg.targets.to_dict().items()}
    listed = {name for names in targets.values() for name in names}
    if listed <= set(SHAPE_NAMES):
        return targets
    return SYNTHETIC_TARGETS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="a handful of scenes, into data/smoke")
    parser.add_argument("--set", dest="overrides", action="append", metavar="KEY=VALUE")
    args = parser.parse_args()

    overrides = {**(SMOKE if args.smoke else {}), **parse_overrides(args.overrides)}
    cfg = load_config(overrides=overrides)
    root = Path(cfg.data.root)
    shape = (cfg.data.resolution,) * 3
    spacing = tuple(cfg.data.spacing)
    vocab = Vocabulary(SHAPE_NAMES)

    manifests: dict[str, list[dict]] = {}
    for split, count in cfg.synthetic.scenes.items():
        first = int(cfg.synthetic.seeds[split])
        manifests[split] = []
        for index in range(int(count)):
            scene_id = f"{split}_{index:05d}"
            image, labels = generate_scene(
                first + index, shape, spacing, cfg.data.margin, cfg.synthetic.appearance
            )
            check_scene(labels, len(vocab), cfg.data.margin)
            write_scene(root, scene_id, image, labels, spacing)
            manifests[split] += build_examples(
                scene_id, labels, vocab, spacing, cfg.data.n_anchors, pool=cfg.data.anchor_pool
            )
            print(f"\r{split}: {index + 1}/{count} scenes", end="", flush=True)
        print()

    targets = _targets(cfg)
    write_corpus(
        root,
        vocab,
        manifests,
        shape=shape,
        spacing=spacing,
        n_anchors=cfg.data.n_anchors,
        targets=targets,
        anchor_pool=cfg.data.anchor_pool,
        extra={"source": "synthetic", "config": cfg.to_dict()},
    )
    for split, records in manifests.items():
        kept = [row for row in records if vocab.name(row["target"]) in targets[split]]
        print(f"{split}: {len(records)} examples, {len(kept)} on the split's target classes")
    print(f"corpus written to {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
