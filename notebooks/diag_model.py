#!/usr/bin/env python
"""Audit a Stage B checkpoint: what it uses, and what actually failed.

    .venv/bin/python notebooks/diag_model.py runs/phase-b/EXP2-seed1/best.pt \
        --segmenter runs/phase-a/current/best.pt

Dice alone cannot tell you what went wrong, because it fuses two abilities that
transfer differently:

* **localisation** - did it point at the right structure? Class-agnostic, so it
  can transfer to a target never supervised.
* **delineation**  - did it outline that structure? Learned per class.

A near-zero Dice is equally consistent with "pointed at the wrong structure" and
"pointed at the right one but could not outline it". On the target-first corpus
the split was decisive: on trained classes localisation was 100% correct with 0.9
voxels of centroid error, so 0.760 Dice was pure delineation loss; on unseen
classes localisation collapsed to 31.2%, with 37.5% of predictions landing on one
of the anchors.

Also runs the counterfactuals, which `scripts/train.py` now logs per epoch but
which are worth re-running per split:

* ``flip_direction`` - one clause asks for the opposite side. A model reading the
  prompt must lose Dice. **Flat means it found a shortcut.**
* ``permute_both``   - the control. It preserves every relation, so it must not
  move; if it does, something put information back into slot order.

Read every number against `notebooks/diag_corpus.py`: a Dice against its
prompt-blind floor, a selection accuracy against its population's ceiling.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.config import load_config, parse_overrides
from src.data import Corpus, ExampleDataset, loader
from src.engine import (
    Metrics, StageBTask, dice_iou, load_model, load_stage_a, resolve_device,
    resolve_phase_a_checkpoint, resolve_stage_b_mode,
)
from src.geometry import DIRECTIONS, OPPOSITE


@torch.no_grad()
def audit(checkpoint, corpus, cfg, device, segmenter, split, targets, label, limit):
    model = load_model(checkpoint, device)
    task = StageBTask(
        model, corpus.vocab, mode=resolve_stage_b_mode(cfg.train.stage_b),
        segmenter=segmenter, threshold=cfg.train.threshold,
        occupancy_mode=str(cfg.train.stage_b.occupancy_mode),
    )
    dataset = ExampleDataset(
        corpus, split, targets=targets, sample=limit, seed=int(cfg.train.seed),
        normalize_mode=cfg.data.normalize,
    )
    metrics = Metrics()
    opposites = torch.tensor([DIRECTIONS.index(OPPOSITE[d]) for d in DIRECTIONS])
    hit = on_anchor = elsewhere = empty = chosen = correct = 0
    base, flipped, control, distance = [], [], [], []

    for raw in loader(dataset, batch_size=cfg.train.batch_size, shuffle=False, workers=cfg.train.workers):
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in raw.items()}
        prediction = task(batch)
        target = prediction.target
        metrics.update(prediction.logits, target, prediction.groups, threshold=0.5)
        base += dice_iou(torch.sigmoid(prediction.logits.float()), target)[0].flatten().tolist()

        if prediction.selection is not None:
            pick = prediction.selection.float().nan_to_num(neginf=-1e4).argmax(dim=-1)
            correct += int((pick == prediction.selection_target).sum())
            chosen += int(pick.numel())

        turned = batch["direction_ids"].clone()
        turned[:, 0] = opposites.to(turned.device)[turned[:, 0]]
        flipped += dice_iou(
            torch.sigmoid(task({**batch, "direction_ids": turned}).logits.float()), target
        )[0].flatten().tolist()
        rolled = {
            **batch,
            "anchors": batch["anchors"].roll(1, dims=1),
            "name_ids": (batch["anchors"] - 1).roll(1, dims=1),
            "direction_ids": batch["direction_ids"].roll(1, dims=1),
        }
        control += dice_iou(torch.sigmoid(task(rolled).logits.float()), target)[0].flatten().tolist()

        # Localisation: which ground-truth structure does the prediction cover most?
        masks = (torch.sigmoid(prediction.logits.float()) >= 0.5)[:, 0]
        for i in range(masks.shape[0]):
            mask = masks[i]
            if not bool(mask.any()):
                empty += 1
                continue
            values = batch["labels"][i][mask]
            values = values[values > 0]
            if values.numel() == 0:
                elsewhere += 1
                continue
            dominant = int(torch.bincount(values.flatten()).argmax())
            true = int(batch["target"][i])
            if dominant == true:
                hit += 1
            elif dominant in [int(a) for a in batch["anchors"][i]]:
                on_anchor += 1
            else:
                elsewhere += 1
            centre = torch.nonzero(mask).float().mean(0)
            truth = torch.nonzero(batch["labels"][i] == true).float().mean(0)
            distance.append(float((centre - truth).norm()))

    total = hit + on_anchor + elsewhere + empty
    mean = lambda xs: sum(xs) / max(len(xs), 1)
    print(f"\n=== {label} ===  n={total}")
    print(f"  Dice {mean(base):.4f}   (compare with the prompt-blind floor for this population)")
    if chosen:
        print(f"  selection accuracy {correct / chosen:.4f}   (compare with ITS OWN anchor-set ceiling)")
    print(f"  localisation -- prediction's dominant structure is:")
    print(f"     the target      {hit:5d}  ({hit / max(total,1):6.1%})")
    print(f"     an ANCHOR       {on_anchor:5d}  ({on_anchor / max(total,1):6.1%})")
    print(f"     something else  {elsewhere:5d}  ({elsewhere / max(total,1):6.1%})")
    print(f"     empty           {empty:5d}  ({empty / max(total,1):6.1%})")
    if distance:
        print(f"  centroid error, voxels: median {np.median(distance):.1f}  mean {np.mean(distance):.1f}")
    print(f"  flip_direction drop {mean(base) - mean(flipped):+.4f}   <- must be LARGE, or it is not reading the prompt")
    print(f"  permute_both  drop {mean(base) - mean(control):+.4f}   <- control, must be ~0")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--segmenter", type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=200, help="examples per population")
    parser.add_argument("--set", dest="overrides", action="append", metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(overrides=parse_overrides(args.overrides))
    corpus = Corpus.load(cfg.data.root)
    device = resolve_device(cfg.train.device)
    path = resolve_phase_a_checkpoint(cfg.train.stage_b, args.segmenter)
    segmenter = load_stage_a(path, device) if path else None
    meta = corpus.meta

    for label, targets in (
        ("TRAINED classes, held-out subjects", meta["targets"]["train"]),
        ("UNSEEN classes (transfer)", meta["targets"]["val"]),
    ):
        audit(args.checkpoint, corpus, cfg, device, segmenter, args.split, targets, label, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
