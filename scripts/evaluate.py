#!/usr/bin/env python
"""Score a trained Stage B model on a split, and probe what it actually uses.

    scripts/evaluate.py runs/phase-b/seed1/best.pt
    scripts/evaluate.py runs/phase-b/seed1/best.pt --segmenter runs/phase-a/current/best.pt
    scripts/evaluate.py runs/phase-b/seed1/best.pt --set train.stage_b.mode=oracle
    scripts/evaluate.py runs/phase-b/seed1/best.pt --set train.stage_b.occupancy_mode=anchors-only
    scripts/evaluate.py runs/phase-b/seed1/best.pt --split val --save-masks

Two things are reported, and the second is the one that matters.

**Metrics** - Dice, IoU and 95th-percentile Hausdorff, overall and stratified by
target structure, anchor structure, direction and clause slot. The default
(``train.stage_b.mode: predicted``) takes anchors from Stage A; ``mode: oracle``
uses the ground truth instead. The anchors' own Dice is reported next to the
result so a drop can be attributed.

**Counterfactuals** - a high Dice proves nothing on its own. A model that ignores
the prompt and segments "the nearest thing that is not an anchor" can score well.
Each probe perturbs one correspondence and reports how far Dice falls: a model
that is really using the relations should collapse on the first three and hold on
the fourth, which changes nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.config import load_config, parse_overrides
from src.data import Corpus, ExampleDataset, loader, save_nifti
from src.engine import (
    Metrics, StageBTask, dice_iou, format_table, load_model, load_stage_a,
    resolve_device, resolve_phase_a_checkpoint, resolve_stage_b_mode,
)
from src.geometry import DIRECTIONS, OPPOSITE


#: ``anchors`` chooses the mask channels, ``name_ids`` and ``direction_ids`` the
#: prompt. Keeping them separate here is what lets a probe move one and not the
#: other; in a normal batch they are locked together.
PROBES = ("permute_channels", "permute_clauses", "flip_direction", "permute_both")
#: The control: it preserves every relation, so the prediction should not move.
CONTROL = "permute_both"


def counterfactual(batch: dict, kind: str) -> dict:
    """Perturb one correspondence in a batch, leaving the volumes untouched."""
    batch = dict(batch)
    names, directions = batch["anchors"] - 1, batch["direction_ids"]
    if kind == "permute_channels":  # the masks move, the prompt stays put
        batch["anchors"] = batch["anchors"].roll(1, dims=1)
        batch["name_ids"] = names
    elif kind == "permute_clauses":  # the prompt moves, the masks stay put
        batch["name_ids"] = names.roll(1, dims=1)
        batch["direction_ids"] = directions.roll(1, dims=1)
    elif kind == "flip_direction":  # one clause now asks for the opposite side
        opposites = torch.tensor(
            [DIRECTIONS.index(OPPOSITE[d]) for d in DIRECTIONS], device=directions.device
        )
        flipped = directions.clone()
        flipped[:, 0] = opposites[flipped[:, 0]]
        batch["direction_ids"] = flipped
    elif kind == "permute_both":  # the control: correspondence is preserved
        batch["anchors"] = batch["anchors"].roll(1, dims=1)
        batch["name_ids"] = names.roll(1, dims=1)
        batch["direction_ids"] = directions.roll(1, dims=1)
    else:
        raise ValueError(f"unknown counterfactual {kind!r}")
    return batch


@torch.no_grad()
def run(task: StageBTask, batches, device, spacing, evaluation, *, threshold=0.5, save_to: Path | None = None):
    """Score the split, then re-score it under each counterfactual."""
    metrics = Metrics()
    probes = tuple(evaluation.get("counterfactuals") or PROBES)
    hausdorff = "hausdorff" in evaluation.get("metrics", ())
    percentile = float(evaluation.get("hausdorff_percentile", 95.0))
    scores: dict[str, list[float]] = {kind: [] for kind in probes}
    rows = []
    for raw in batches:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in raw.items()}
        prediction = task(batch)
        logits, target = prediction.logits, prediction.target
        metrics.update(
            logits, target, prediction.groups, strata=prediction.strata, threshold=threshold,
            spacing=spacing if hausdorff else None, percentile=percentile,
        )
        dice = dice_iou(torch.sigmoid(logits.float()), target)[0].flatten().tolist()
        for kind in scores:
            perturbed = task(counterfactual(batch, kind)).logits
            scores[kind] += dice_iou(torch.sigmoid(perturbed.float()), target)[0].flatten().tolist()
        for index, example in enumerate(batch["example_id"]):
            rows.append({"id": example, "prompt": batch["prompt"][index], "dice": dice[index]})
            if save_to is not None:
                mask = (torch.sigmoid(logits[index, 0].float()) >= threshold).cpu().numpy()
                save_nifti(save_to / example / "prediction.nii.gz", mask, spacing, "uint8")
                save_nifti(save_to / example / "target.nii.gz", target[index, 0].cpu().numpy(), spacing, "uint8")
    return (
        metrics.summary(evaluation.get("stratify_by")),
        {kind: sum(values) / len(values) for kind, values in scores.items()},
        rows,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path, help="a Stage B checkpoint")
    parser.add_argument(
        "--segmenter", type=Path,
        help="Stage A checkpoint (overrides train.stage_b.phase_a_checkpoint). Required when mode is predicted; rejected when mode is oracle.",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", type=Path, help="report directory (default next to the checkpoint)")
    parser.add_argument("--save-masks", action="store_true", help="also write every predicted mask")
    parser.add_argument("--set", dest="overrides", action="append", metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(overrides=parse_overrides(args.overrides))
    corpus = Corpus.load(cfg.data.root)
    device = resolve_device(cfg.train.device)
    model = load_model(args.checkpoint, device)
    segmenter_path = resolve_phase_a_checkpoint(cfg.train.stage_b, args.segmenter)
    segmenter = load_stage_a(segmenter_path, device) if segmenter_path else None
    task = StageBTask(
        model, corpus.vocab,
        mode=resolve_stage_b_mode(cfg.train.stage_b),
        segmenter=segmenter,
        threshold=cfg.train.threshold,
        occupancy_mode=str(cfg.train.stage_b.occupancy_mode),
        loss_weights={"lambda_dice": float(cfg.train.loss["lambda_dice"]),
                      "lambda_bce": float(cfg.train.loss["lambda_bce"])},
    )

    dataset = ExampleDataset(corpus, args.split, normalize_mode=cfg.data.normalize)
    batches = loader(dataset, batch_size=cfg.train.batch_size, shuffle=False, workers=cfg.train.workers)
    out_dir = args.out or args.checkpoint.parent / f"eval_{args.split}{'_oracle' if task.mode == 'oracle' else ''}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary, probes, rows = run(
        task, batches, device, corpus.spacing, cfg.evaluation.to_dict(),
        threshold=cfg.train.threshold, save_to=out_dir / "masks" if args.save_masks else None,
    )
    if task.anchor_scores:
        summary["anchor_dice"] = sum(task.anchor_scores) / len(task.anchor_scores)

    source = "segmenter" if segmenter else "gt"
    print(format_table(summary, f"{args.split} ({source} anchors, occupancy={task.occupancy_mode})"))
    if segmenter and task.occupancy_mode == "all":
        print("  note: occupancy_mode=all unions Stage A's prediction for every name, including the"
              " target - a ceiling, not a deployable setting. Use anchors-only or none for that.")
    if "anchor_dice" in summary:
        print(f"\n  anchors from Stage A scored Dice {summary['anchor_dice']:.4f}")
    print("\n  counterfactual                       dice     drop")
    for kind, value in probes.items():
        note = "  <- control: should not move" if kind == CONTROL else ""
        print(f"  {kind:<28} {value:>8.4f} {summary['dice'] - value:>8.4f}{note}")

    report = {
        "checkpoint": str(args.checkpoint),
        "mode": resolve_stage_b_mode(cfg.train.stage_b),
        "phase_a_checkpoint": str(segmenter_path) if segmenter_path else None,
        "occupancy_source": source,
        "occupancy_mode": task.occupancy_mode,
        "split": args.split,
        "metrics": summary,
        "counterfactuals": probes,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    print(f"\nwritten to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
