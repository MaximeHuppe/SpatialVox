#!/usr/bin/env python
"""Offline instance ceilings (§5.1) — no training.

Measures, on a corpus split:

* mapper gate: fraction of target centroids with ``where_raw > 0.5``;
* oracle instance: Dice of the labelled structure whose centroid maximises
  ``where_raw`` (design ceiling ~0.97 on unique prompts);
* seed-flood recall@K: whether some proposal overlaps the GT body (IoU ≥ 0.5).

    scripts/oracle_instance.py --split val --limit 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.config import load_config, parse_overrides
from src.data import Corpus, ExampleDataset
from src.engine import dice_iou
from src.geometry import volume_center_world
from src.instance import oracle_label_instances, propose_seed_flood, run_instance
from src.mapper import PositionalMapper3D


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--set", dest="overrides", action="append")
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=parse_overrides(args.overrides))
    corpus = Corpus.load(cfg.data.root)
    targets = list(cfg.targets.train) + list(cfg.targets.val) + list(cfg.targets.test)
    dataset = ExampleDataset(
        corpus, args.split, targets=targets, limit=args.limit,
        normalize_mode=cfg.data.normalize,
    )
    mapper = PositionalMapper3D(
        tau=float(cfg.model.stage_b.mapper.tau),
        min_mass=float(cfg.model.stage_b.mapper.min_mass),
    )
    inst = cfg.model.stage_b.get("instance", {})
    dilate_radius = int(inst.get("dilate_radius", 4) if hasattr(inst, "get") else 4)
    score_null = float(inst.get("score_null", 0.5) if hasattr(inst, "get") else 0.5)

    gate, oracle_dice, recall = [], [], []
    for item in dataset:
        labels = item["labels"].numpy()
        target = int(item["target"])
        if target == 0:
            continue
        # Oracle soft anchors from labels (offline ceiling, not Stage A).
        anchors = torch.zeros(1, 3, *labels.shape)
        for slot, label in enumerate(item["anchors"].tolist()):
            anchors[0, slot] = torch.from_numpy((labels == int(label)).astype(np.float32))
        directions = item["direction_ids"].unsqueeze(0)
        center = tuple(volume_center_world(labels.shape, corpus.spacing).tolist())
        with torch.no_grad():
            field = mapper(anchors, directions, corpus.spacing, center)

        z, y, x = np.nonzero(labels == target)
        cx = float(x.mean()) * corpus.spacing[0]
        cy = float(y.mean()) * corpus.spacing[1]
        cz = float(z.mean()) * corpus.spacing[2]
        ix = int(round(cx / corpus.spacing[0]))
        iy = int(round(cy / corpus.spacing[1]))
        iz = int(round(cz / corpus.spacing[2]))
        where = field.where_raw[0, 0]
        gate.append(float(where[iz, iy, ix]) > 0.5)

        _, _, oracle_mask = oracle_label_instances(labels, where, corpus.spacing)
        gt = torch.from_numpy((labels == target).astype(np.float32))[None, None]
        pred = torch.from_numpy(oracle_mask)[None, None]
        oracle_dice.append(float(dice_iou(pred, gt)[0]))

        image = item["image"]
        proposals, _ = propose_seed_flood(
            image, where, dilate_radius=dilate_radius, max_seeds=16, min_voxels=4,
        )
        hit = False
        for k in range(proposals.shape[0]):
            iou = float(dice_iou(proposals[k][None, None], gt)[1])
            if iou >= 0.5:
                hit = True
                break
        recall.append(hit)

    n = max(len(gate), 1)
    print(f"split={args.split} n={len(gate)}")
    print(f"mapper_gate (where>0.5 at target centroid): {sum(gate) / n:.4f}")
    print(f"oracle_instance Dice:                     {sum(oracle_dice) / n:.4f}")
    print(f"seed_flood recall@K (IoU≥0.5):            {sum(recall) / n:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
