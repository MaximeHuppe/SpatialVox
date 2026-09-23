#!/usr/bin/env python
"""Score a trained Stage B model, and probe what it actually uses.

    scripts/evaluate.py runs/stage_b/best.pt
    scripts/evaluate.py runs/stage_b/best.pt --split val --classes val
    scripts/evaluate.py runs/stage_b/best.pt --save-masks

The proposal's §7 (``documentation/SpatialVox.md``, Evaluation and reporting rules): *"A single Dice is not the
result."* Six blocks come out, and the Dice is only the third.

1. **Dice, with anchor Dice beside it.** A drop is either a worse outline or a
   Stage A failure, and those are different problems.
2. **Centroid error in millimetres**, from the heatmap - not from the mask - and
   beside it the **empty-prediction rate** and the predicted-against-true voxel
   count. Dice fuses three different failures: pointing at the wrong structure,
   outlining it badly, and saying nothing at all. On a class the carver was never
   supervised on the third is the common one, and it needs a different fix from
   the other two.
3. **The gate, on this split**: the fraction of target centroids with
   ``where_raw > 0.5``. Reported as ``gate_fraction_<anchor source>``, because
   this one is built from the centroids the model actually used -
   ``scripts/gate_mapper.py`` measures the same quantity from **ground-truth**
   centroids, and the two are different numbers that would otherwise share a
   name.
4. **The four counterfactuals.** A high Dice proves nothing on its own - a model
   that segments "the nearest thing that is not an anchor" scores well without
   reading a word. ``permute_channels``, ``permute_clauses`` and
   ``flip_direction`` must fall. ``permute_both`` preserves every relation and
   must not move; see ``documentation/SpatialVox.md`` (The four counterfactuals) for how weak that control now is.
5. **Empty prompts.** A direction is flipped until the clauses name *nothing*,
   then: how often the null head says so, and how much mask is emitted anyway.
   This is the check that the tiny spike in ``where_raw`` was not renormalised
   into a confident answer.
6. **The image-replacement test** (§7, mandatory). The anchors and every field
   stay this subject's; the volume ``B`` reads is another subject's. The centroid
   should hold and the Dice should fall. That pattern is the signature that the
   words placed the structure and the image drew it.

The other mandatory test, the prompt-only carver, is a *training* run:
``scripts/train.py b --prompt-only``. If its Dice approaches the full model's,
the carver is redrawing a shape from the spatial prior.
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
from src.data import Corpus, ExampleDataset, anchor_cache_dir, load_nifti, loader, save_nifti
from src.engine import (
    Metrics, StageBTask, dice_iou, format_table, load_model, mask_centroid_world,
    null_gated, null_summary, resolve_anchor_source, resolve_device, roll_anchors,
    segmentation_loss,
)
from src.geometry import DIRECTIONS, OPPOSITE, centroids_world, solutions_for, volume_center_world

#: The control preserves every relation, so the prediction should not move.
CONTROL = "permute_both"


def counterfactual(batch: dict, kind: str) -> dict:
    """Perturb one correspondence in a batch, leaving the volumes untouched."""
    directions = batch["direction_ids"]
    if kind == "permute_channels":  # the masks move, the prompt stays put
        return roll_anchors(batch, 1)
    if kind == "permute_clauses":  # the prompt moves, the masks stay put
        return {**batch, "direction_ids": directions.roll(1, dims=1)}
    if kind == "flip_direction":  # one clause now asks for the opposite side
        opposites = torch.tensor(
            [DIRECTIONS.index(OPPOSITE[d]) for d in DIRECTIONS], device=directions.device
        )
        flipped = directions.clone()
        flipped[:, 0] = opposites[flipped[:, 0]]
        return {**batch, "direction_ids": flipped}
    if kind == CONTROL:  # correspondence preserved: the control
        return {**roll_anchors(batch, 1), "direction_ids": directions.roll(1, dims=1)}
    raise ValueError(f"unknown counterfactual {kind!r}")


def empty_prompts(corpus: Corpus, rows, seed: int = 0, limit: int = 400):
    """Rewrite prompts until they name *nothing*, by the corpus's own rule.

    §5's middle column, gathered as a population: the mask target is empty and
    the null target is invalid. Built here rather than by the dataset because a
    validation curve must not contain them (an empty prediction against an empty
    target scores Dice 1.0).
    """
    rng = np.random.default_rng(seed)
    by_scene: dict[str, list[dict]] = {}
    for row in rows:
        by_scene.setdefault(row["scene"], []).append(row)
    out = []
    for scene_id, scene_rows in by_scene.items():
        volume = load_nifti(corpus.root / "scenes" / scene_id / "labels.nii.gz", np.int16)
        centroids = centroids_world(volume, len(corpus.vocab), corpus.spacing)
        center = volume_center_world(volume.shape, corpus.spacing)
        present = [int(v) for v in np.unique(volume) if v != 0]
        for row in scene_rows:
            for _ in range(6):  # a few tries; most flips already name nothing
                directions = list(row["directions"])
                slot = int(rng.integers(len(directions)))
                directions[slot] = OPPOSITE[directions[slot]]
                if len(set(directions)) != len(directions):
                    continue
                if solutions_for(row["anchors"], directions, centroids, present, center):
                    continue
                out.append({**row, "directions": directions, "target": 0})
                break
        if len(out) >= limit:
            break
    return out[:limit]


def anchor_set_ceiling(rows) -> float:
    """How often the anchor *identities* alone would recover the target.

    The population's own ceiling, and it has to be printed beside the population's
    own Dice. Conditioning on fewer classes inflates it: measured on ``data/mri``
    it is 42.3% over every class, 67.8% over the eight supervised ones and 99.2%
    over the two of ``targets.test`` - where a good Dice therefore says almost
    nothing, because a model that read only the anchor names would get it too.
    """
    by_set: dict[frozenset, dict[int, int]] = {}
    for row in rows:
        counts = by_set.setdefault(frozenset(row["anchors"]), {})
        counts[row["target"]] = counts.get(row["target"], 0) + 1
    return sum(max(c.values()) for c in by_set.values()) / max(len(rows), 1)


@torch.no_grad()
def score(task, batches, device, spacing, evaluation, *, threshold=0.5, save_to=None):
    """One pass: metrics, the gate, the counterfactuals, and per-example rows."""
    metrics = Metrics()
    probes = tuple(evaluation.get("counterfactuals") or ())
    wants_hausdorff = "hausdorff" in evaluation.get("metrics", ())
    percentile = float(evaluation.get("hausdorff_percentile", 95.0))
    scores: dict[str, list[float]] = {kind: [] for kind in probes}
    gate, nulls, rows, volumes, gated_dice, gated_volumes = [], [], [], [], [], []
    for raw in batches:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in raw.items()}
        prediction = task(batch)
        logits, target = prediction.logits, prediction.target
        error = (prediction.centroid - prediction.centroid_target).norm(dim=-1)
        error = torch.where(prediction.valid_target > 0, error, torch.full_like(error, float("nan")))
        # The gate, on this split: where_raw sampled at the target's own centroid.
        index = (prediction.centroid_target / torch.tensor(spacing, device=device)).round().long()
        index = index.clamp(min=0, max=logits.shape[-1] - 1)
        at_centroid = prediction.where_raw[
            torch.arange(len(index), device=device), 0, index[:, 2], index[:, 1], index[:, 0]
        ]
        # Only where there is a structure: an empty prompt's `centroid_target`
        # is the origin, and sampling `where_raw` there would be a silent zero.
        at_centroid = torch.where(
            prediction.valid_target > 0, at_centroid, torch.full_like(at_centroid, float("nan"))
        )
        gate += at_centroid[prediction.valid_target > 0].tolist()
        nulls += list(zip(prediction.valid.float().tolist(), prediction.valid_target.tolist()))
        metrics.update(
            logits, target, prediction.groups, strata=prediction.strata, threshold=threshold,
            spacing=spacing if wants_hausdorff else None, percentile=percentile,
            keep=prediction.keep,
            extra={"centroid_error": error.tolist(),
                   "anchor_dice": prediction.anchor_dice.mean(-1).tolist()},
        )
        # Is a low Dice "drew the wrong thing" or "drew nothing"? On a class it
        # was never supervised on the carver can simply fall silent, and the two
        # failures have different fixes, so they are reported apart.
        probability = torch.sigmoid(logits.float())
        predicted = (probability >= threshold).flatten(1).sum(-1)
        wanted = target.flatten(1).sum(-1)
        volumes += list(zip(predicted.tolist(), wanted.tolist()))
        dice = dice_iou(probability, target)[0].flatten().tolist()
        # The same answer after the null head's gate - the only place emptiness
        # is decided under `mask_on: valid`, and reported beside the raw one (§7).
        gated = null_gated(probability, prediction.valid)
        gated_dice += dice_iou(gated, target, threshold)[0].flatten().tolist()
        gated_volumes += list(zip((gated >= threshold).flatten(1).sum(-1).tolist(), wanted.tolist()))
        for kind in scores:
            perturbed = task(counterfactual(batch, kind)).logits
            scores[kind] += dice_iou(torch.sigmoid(perturbed.float()), target)[0].flatten().tolist()
        for position, example in enumerate(batch["example_id"]):
            rows.append({
                "id": example, "prompt": batch["prompt"][position], "dice": dice[position],
                "centroid_error": float(error[position]), "where_at_centroid": float(at_centroid[position]),
                "valid_logit": float(prediction.valid[position]),
            })
            if save_to is not None:
                mask = (torch.sigmoid(logits[position, 0].float()) >= threshold).cpu().numpy()
                save_nifti(save_to / example / "prediction.nii.gz", mask, spacing, "uint8")
                save_nifti(save_to / example / "target.nii.gz",
                           target[position, 0].cpu().numpy(), spacing, "uint8")
    summary = metrics.summary(evaluation.get("stratify_by"))
    summary.update(null_summary(nulls))
    summary[f"gate_fraction_{task.anchor_source}"] = (
        float(np.mean([v > 0.5 for v in gate])) if gate else float("nan")
    )
    if volumes:
        kept = [(p, w) for p, w in volumes if w > 0]
        summary["empty_prediction_rate"] = float(np.mean([p == 0 for p, _ in kept])) if kept else 0.0
        summary["predicted_voxels"] = float(np.mean([p for p, _ in kept])) if kept else 0.0
        summary["target_voxels"] = float(np.mean([w for _, w in kept])) if kept else 0.0
        kept = [p for p, w in gated_volumes if w > 0]
        summary["empty_prediction_rate_null_gated"] = float(np.mean([p == 0 for p in kept])) if kept else 0.0
        summary["dice_null_gated"] = float(np.mean(gated_dice))
    return summary, {k: sum(v) / len(v) for k, v in scores.items() if v}, rows


@torch.no_grad()
def empty_prompt_report(task, batches, device, threshold=0.5) -> dict[str, float]:
    """How often the null head calls an impossible prompt empty, and what leaks out.

    Twice: from the carver alone, and after the null head's gate. Under
    ``mask_on: valid`` the carver is no longer trained to fall silent here, so the
    ungated leak rises by design and the gated one is the system's answer.
    """
    nulls, volumes, gated = [], [], []
    for raw in batches:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in raw.items()}
        prediction = task(batch)
        nulls += list(zip(prediction.valid.float().tolist(), prediction.valid_target.tolist()))
        probability = torch.sigmoid(prediction.logits.float())
        volumes += (probability >= threshold).flatten(1).sum(-1).tolist()
        gated += (null_gated(probability, prediction.valid) >= threshold).flatten(1).sum(-1).tolist()
    return {
        "invalid_called_invalid": float(np.mean([s <= 0 for s, _ in nulls])),
        "false_positive_voxels": float(np.mean(volumes)),
        "false_positive_rate": float(np.mean([v > 0 for v in volumes])),
        "false_positive_voxels_null_gated": float(np.mean(gated)),
        "false_positive_rate_null_gated": float(np.mean([v > 0 for v in gated])),
    }


@torch.no_grad()
def image_replacement(task, batches, device) -> dict[str, float]:
    """§7: keep this subject's anchors and fields, feed ``B`` another subject's MRI.

    The donor is the neighbouring row in the batch, and a pair is **used only
    when the two rows are different subjects**. The manifest is written in scene
    order and anchor-first generation emits ~90 prompts per scene, so a batch of
    four consecutive rows is usually four prompts about *the same* volume -
    swapping those would compare a subject with himself and the test would report
    a reassuring null for the wrong reason. Pass a shuffled loader; the pairs
    that still collide are skipped and counted.
    """
    dice, shifted, base_dice, base_error = [], [], [], []
    skipped = 0
    for raw in batches:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in raw.items()}
        if batch["image"].shape[0] < 2:
            continue
        reference = task(batch)
        swapped = task({**batch, "boundary_image": batch["image"].roll(1, dims=0)})
        donor = list(batch["scene"][-1:]) + list(batch["scene"][:-1])
        other = torch.tensor(
            [a != b for a, b in zip(batch["scene"], donor)], device=device
        )
        skipped += int((~other).sum())
        use = (reference.valid_target > 0) & other
        if not bool(use.any()):
            continue
        base_dice += dice_iou(torch.sigmoid(reference.logits.float()), reference.target)[0].flatten()[use].tolist()
        dice += dice_iou(torch.sigmoid(swapped.logits.float()), reference.target)[0].flatten()[use].tolist()
        base_error += (reference.centroid - reference.centroid_target).norm(dim=-1)[use].tolist()
        shifted += (swapped.centroid - reference.centroid_target).norm(dim=-1)[use].tolist()
    if not dice:
        return {}
    mean = lambda xs: float(np.mean(xs))
    return {
        "n": len(dice), "same_subject_pairs_skipped": skipped,
        "dice": mean(base_dice), "dice_swapped": mean(dice),
        "centroid_error": mean(base_error), "centroid_error_swapped": mean(shifted),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path, help="a Stage B checkpoint")
    parser.add_argument("--split", default="test")
    parser.add_argument("--classes", default=None,
                        help="targets.<name> to score; default = every class in the split")
    parser.add_argument("--limit", type=int, default=None, help="score this many examples")
    parser.add_argument("--out", type=Path, help="report directory (default next to the checkpoint)")
    parser.add_argument("--save-masks", action="store_true", help="also write every predicted mask")
    parser.add_argument("--config", type=Path, help="config file (default configs/config.yaml)")
    parser.add_argument("--set", dest="overrides", action="append", metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=parse_overrides(args.overrides))
    corpus = Corpus.load(cfg.data.root)
    device = resolve_device(cfg.train.device)
    model = load_model(args.checkpoint, device)
    task = StageBTask(
        model, corpus.vocab, spacing=corpus.spacing,
        anchor_source=resolve_anchor_source(cfg.train.stage_b),
        loss_weights=cfg.train.stage_b.loss.to_dict(),
        far_epsilon=float(cfg.train.stage_b.far.epsilon),
        far_dilation=int(cfg.train.stage_b.far.dilation),
        field_centroid_on=str(cfg.train.stage_b.field_centroid_on),
        mask_on=str(cfg.train.stage_b.mask_on),
    )
    checkpoint = Path(cfg.train.stage_b.phase_a_checkpoint)
    anchors = anchor_cache_dir(corpus.root, checkpoint) if checkpoint.is_file() else None
    anchors = anchors if anchors is not None and (anchors / "meta.json").is_file() else None

    classes = list(cfg.targets[args.classes]) if args.classes else None
    common = dict(anchor_cache=anchors, normalize_mode=cfg.data.normalize)
    dataset = ExampleDataset(corpus, args.split, targets=classes, limit=args.limit, **common)
    batches = loader(dataset, batch_size=cfg.train.batch_size, shuffle=False, workers=cfg.train.workers)
    out_dir = args.out or args.checkpoint.parent / f"eval_{args.split}{'_' + args.classes if args.classes else ''}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary, probes, rows = score(
        task, batches, device, corpus.spacing, cfg.evaluation.to_dict(),
        threshold=cfg.train.threshold, save_to=out_dir / "masks" if args.save_masks else None,
    )
    ceiling = anchor_set_ceiling(dataset.records)
    summary["anchor_set_ceiling"] = ceiling
    title = f"{args.split}" + (f" / targets.{args.classes}" if args.classes else "")
    print(format_table(summary, f"{title} ({task.anchor_source} anchors)"))
    print(f"\n  anchor Dice                 {summary.get('anchor_dice', float('nan')):.4f}")
    print(f"  centroid error (mm)         {summary.get('centroid_error', float('nan')):.2f}")
    print(f"  emitted an EMPTY mask       {summary.get('empty_prediction_rate', float('nan')):.1%}"
          f"   (of prompts that do name a structure)")
    print(f"  gated by the null head      dice {summary.get('dice_null_gated', float('nan')):.4f}"
          f"   empty {summary.get('empty_prediction_rate_null_gated', float('nan')):.1%}")
    print(f"  voxels: predicted {summary.get('predicted_voxels', 0):.0f}"
          f" against a true {summary.get('target_voxels', 0):.0f}")
    gate_key = f"gate_fraction_{task.anchor_source}"
    print(f"  gate: where_raw > 0.5 at the target centroid   {summary[gate_key]:.4f}"
          f"   (from {task.anchor_source} anchor centroids)")
    print(f"\n  shortcut ceiling for THIS population: the anchor identities alone recover")
    print(f"  the target {ceiling:.1%} of the time"
          + ("   <- read the Dice above against this, not against 0" if ceiling < 0.9
             else "   <- SO HIGH THAT THE DICE ABOVE IS NEARLY UNINFORMATIVE"))

    print("\n  counterfactual                       dice     drop")
    for kind, value in probes.items():
        note = "  <- control: should not move" if kind == CONTROL else ""
        print(f"  {kind:<28} {value:>8.4f} {summary['dice'] - value:>8.4f}{note}")

    # Built from the rows that were actually scored, so the empty-prompt
    # population is the same population as the Dice above - not the whole split.
    impossible = empty_prompts(corpus, dataset.records)
    empty_report = {}
    if impossible:
        empty_set = ExampleDataset(corpus, args.split, targets=classes, **common)
        empty_set.records = impossible
        empty_report = empty_prompt_report(
            task,
            loader(empty_set, batch_size=cfg.train.batch_size, shuffle=False, workers=0),
            device, cfg.train.threshold,
        )
        print(f"\n  prompts that name nothing ({len(impossible)})")
        print(f"    null head says invalid    {empty_report['invalid_called_invalid']:.4f}")
        print(f"    emitted any mask          {empty_report['false_positive_rate']:.4f}"
              f"   gated {empty_report['false_positive_rate_null_gated']:.4f}")
        print(f"    mean false-positive voxels {empty_report['false_positive_voxels']:.1f}"
              f"   gated {empty_report['false_positive_voxels_null_gated']:.1f}")

    # A shuffled loader, so the neighbouring row is usually a different subject.
    swap = image_replacement(
        task,
        loader(dataset, batch_size=max(cfg.train.batch_size, 2), shuffle=True,
               workers=cfg.train.workers, seed=int(cfg.train.seed)),
        device,
    ) if model.use_image else {}
    if swap:
        print(f"\n  image replacement, {swap['n']} pairs of different subjects"
              f" ({swap['same_subject_pairs_skipped']} same-subject pairs skipped)")
        print(f"    dice            {swap['dice']:.4f} -> {swap['dice_swapped']:.4f}   (should fall)")
        print(f"    centroid (mm)   {swap['centroid_error']:.2f} -> {swap['centroid_error_swapped']:.2f}"
              f"   (should hold)")

    report = {
        "checkpoint": str(args.checkpoint),
        "anchor_source": task.anchor_source,
        "anchor_cache": str(anchors) if anchors else None,
        "use_image": bool(model.use_image),
        "split": args.split,
        "classes": classes,
        "metrics": summary,
        "counterfactuals": probes,
        "empty_prompts": empty_report,
        "image_replacement": swap,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    (out_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    print(f"\nwritten to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
