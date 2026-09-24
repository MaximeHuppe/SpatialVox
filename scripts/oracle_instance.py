#!/usr/bin/env python
"""Offline instance ceilings (§5.1) — no training.

Measures, on a corpus split:

* mapper gate: fraction of target centroids with ``where_raw > 0.5``;
* oracle instance: Dice of the labelled structure whose centroid maximises
  ``where_raw`` (design ceiling ~0.97 on unique prompts);
* seed-flood recall@K: whether some proposal overlaps the GT body (IoU ≥ 0.5);
* diagnostics: mean best IoU, mean K, GT coverage by the dilated region,
  and flood-from-GT-centroid IoU (upper bound on intensity flood alone).

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
from src.instance import (
    _flood_intensity,
    _resolve_intensity_tol,
    oracle_label_instances,
    propose_seed_flood,
    region_mask,
)


def _cfg_get(block, key, default):
    if block is None:
        return default
    if hasattr(block, "get"):
        value = block.get(key, default)
        return default if value is None else value
    return default


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
    from src.mapper import PositionalMapper3D

    mapper = PositionalMapper3D(
        tau=float(cfg.model.stage_b.mapper.tau),
        min_mass=float(cfg.model.stage_b.mapper.min_mass),
    )
    inst = cfg.model.stage_b.get("instance", {})
    dilate_radius = int(_cfg_get(inst, "dilate_radius", 4))
    intensity_tol = float(_cfg_get(inst, "intensity_tol", 1.0))
    tol_mode = str(_cfg_get(inst, "tol_mode", "local_std"))
    region_threshold = float(_cfg_get(inst, "region_threshold", 0.5))
    max_seeds = int(_cfg_get(inst, "max_seeds", 16))

    gate, oracle_dice, recall = [], [], []
    best_ious, n_props, gt_cover, gt_flood = [], [], [], []
    prop_frac, region_frac = [], []
    for item in dataset:
        labels = item["labels"].numpy()
        target = int(item["target"])
        if target == 0:
            continue
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
        oracle_dice.append(float(dice_iou(pred, gt)[0].reshape(-1)[0]))

        image = item["image"]
        region = region_mask(where, region_threshold, dilate_radius)[0, 0]
        region_np = region.numpy() > 0.5
        gt_np = labels == target
        gt_n = float(gt_np.sum())
        region_n = float(region_np.sum())
        gt_cover.append(float((gt_np & region_np).sum()) / gt_n if gt_n > 0 else 0.0)
        region_frac.append(region_n / float(np.prod(labels.shape)))

        proposals, _ = propose_seed_flood(
            image, where,
            dilate_radius=dilate_radius,
            region_threshold=region_threshold,
            max_seeds=max_seeds,
            intensity_tol=intensity_tol,
            tol_mode=tol_mode,
            min_voxels=4,
        )
        n_props.append(int(proposals.shape[0]))
        best = 0.0
        hit = False
        sizes = []
        for k in range(proposals.shape[0]):
            sizes.append(float(proposals[k].sum()))
            iou = float(dice_iou(proposals[k][None, None], gt)[1].reshape(-1)[0])
            best = max(best, iou)
            if iou >= 0.5:
                hit = True
        best_ious.append(best)
        recall.append(hit)
        if sizes and region_n > 0:
            prop_frac.append(max(sizes) / region_n)
        else:
            prop_frac.append(0.0)

        # Upper bound: flood from the GT centroid with the same tolerance.
        image_np = image[0].numpy() if image.ndim == 4 else image.numpy()
        seed = (iz, iy, ix)
        if region_np[seed]:
            tol = _resolve_intensity_tol(
                image_np, region_np, intensity_tol, tol_mode=tol_mode, seed=seed,
            )
            flooded = _flood_intensity(image_np, seed, region_np, intensity_tol=tol)
            gt_flood.append(float(dice_iou(
                torch.from_numpy(flooded.astype(np.float32))[None, None], gt
            )[1].reshape(-1)[0]))
        else:
            gt_flood.append(0.0)

    n = max(len(gate), 1)
    print(f"split={args.split} n={len(gate)}  tol_mode={tol_mode} intensity_tol={intensity_tol} r={dilate_radius}")
    print(f"mapper_gate (where>0.5 at target centroid): {sum(gate) / n:.4f}")
    print(f"oracle_instance Dice:                     {sum(oracle_dice) / n:.4f}")
    print(f"seed_flood recall@K (IoU≥0.5):            {sum(recall) / n:.4f}")
    print(f"mean best IoU over proposals:             {sum(best_ious) / n:.4f}")
    print(f"mean K proposals:                         {sum(n_props) / n:.2f}")
    print(f"mean GT voxel coverage by dilated region: {sum(gt_cover) / n:.4f}")
    print(f"mean region volume fraction:              {sum(region_frac) / n:.4f}")
    print(f"mean max-proposal / region size:          {sum(prop_frac) / n:.4f}")
    print(f"mean IoU flood-from-GT-centroid:           {sum(gt_flood) / n:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
