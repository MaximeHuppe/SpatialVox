#!/usr/bin/env python
"""The gate the carver waits on: does the pyramid agree with the prompts?

    .venv/bin/python scripts/gate_mapper.py
    .venv/bin/python scripts/gate_mapper.py --segmenter runs/phase-a/current/best.pt

``docs/proposal/relational_architecture.md`` §2 makes this a precondition, not a
diagnostic: *"A low fraction means the mapper disagrees with the prompts, and the
carver is not trained on top of it."* Ground-truth anchors are used **only as a
check** - they are never an input to anything this script decides.

It also settles the three constants the proposal only gave starting values for,
because each of them is a property of the corpus and not of the method:

``mapper.tau``
    The gate fraction is a function of ``tau``, not a fixed property: at the
    target's own centroid ``classify`` guarantees a non-negative margin on every
    clause, so ``F_i -> 1`` as ``tau -> 0`` and the fraction rises to 1. The
    sweep prints what it costs on the other side - the volume ``L_far`` will
    police, and how much of the target sits outside it.

``mapper.min_mass``
    A rejected channel writes ``F_i = 0`` and zeroes ``where_raw`` for the whole
    example. The threshold has to sit below the smallest structure Stage A
    legitimately finds, which the ``--segmenter`` table measures.

the null head's ceiling
    ``where_mass`` is all it has. The last column is the AUC of that one number
    separating prompts that name a structure from flips that name none. The
    mapper cannot see which regions hold tissue, so a roomy conjunction that
    happens to be empty looks exactly like a valid one; that ceiling is a
    property of the inputs the proposal chose and is worth knowing before the
    head is trained rather than after.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.nn import functional as F

from src.data import Corpus, load_nifti, normalize
from src.engine import load_stage_a, resolve_device
from src.geometry import (
    DIRECTIONS, OPPOSITE, centroids_world, solutions_for, volume_center_world,
)
from src.mapper import margin_at, margins, soft_centroids

TAUS = (0.25, 0.5, 1.0, 2.0, 4.0)


def quantiles(values, points=(0.0, 0.05, 0.5, 0.95, 1.0)) -> str:
    if not len(values):
        return "(empty)"
    q = np.quantile(np.asarray(values, dtype=float), points)
    return "  ".join(f"{p:.0%}:{v:.3g}" for p, v in zip(points, q))


def auc(positive, negative) -> float:
    """P(a random positive scores above a random negative), ties counted as half."""
    a, b = np.asarray(positive)[:, None], np.asarray(negative)[None, :]
    if a.size == 0 or b.size == 0:
        return float("nan")
    return float((a > b).mean() + 0.5 * (a == b).mean())


def scene_geometry(corpus: Corpus, scene_id: str):
    labels = load_nifti(corpus.root / "scenes" / scene_id / "labels.nii.gz", np.int16)
    return (
        labels,
        centroids_world(labels, len(corpus.vocab), corpus.spacing),
        volume_center_world(labels.shape, corpus.spacing),
        [int(v) for v in np.unique(labels) if v != 0],
    )


def flipped_clauses(row, centroids, center, present, rng):
    """One clause replaced by its opposite, then re-scored by the corpus's own rule.

    ``(directions, target)`` with ``target = None`` when the new clauses name
    nothing, or ``None`` when they name more than one structure - the case the
    proposal's loss table drops. This is the function ``ExampleDataset`` applies
    at training time, written out here so the gate measures the same populations
    the loss will see.
    """
    slot = int(rng.integers(len(row["directions"])))
    directions = list(row["directions"])
    directions[slot] = OPPOSITE[directions[slot]]
    if len(set(directions)) != len(directions):
        return None  # two clauses naming one side is not a prompt the corpus admits
    solutions = solutions_for(row["anchors"], directions, centroids, present, center)
    if len(solutions) > 1:
        return None
    return directions, (solutions[0] if solutions else None)


def sweep_scene(rows, labels, centroids, center, present, corpus, taus, rng, device, chunk):
    """Every per-tau quantity for one scene. Returns ``(per_tau, margins, counts)``."""
    shape, spacing = corpus.shape, corpus.spacing
    as_t = lambda a, dtype=torch.float32: torch.as_tensor(np.asarray(a), dtype=dtype, device=device)
    anchors = as_t([[centroids[a] for a in row["anchors"]] for row in rows])
    ids = as_t([[DIRECTIONS.index(d) for d in row["directions"]] for row in rows], torch.long)
    at_target = as_t([np.repeat(centroids[row["target"]][None], anchors.shape[1], 0) for row in rows])
    volume = torch.as_tensor(labels.astype(np.int32), device=device)

    outcomes = [flipped_clauses(row, centroids, center, present, rng) for row in rows]
    counts = collections.Counter()
    counts["dropped (names two or more)"] = sum(1 for o in outcomes if o is None)
    counts["empty (names none)"] = sum(1 for o in outcomes if o is not None and o[1] is None)
    counts["retargeted (names one)"] = sum(1 for o in outcomes if o is not None and o[1] is not None)
    live = [i for i, o in enumerate(outcomes) if o is not None]
    empty = [i for i in live if outcomes[i][1] is None]
    flip_ids = as_t([[DIRECTIONS.index(d) for d in outcomes[i][0]] for i in live], torch.long) if live else None

    per_tau: dict[float, dict[str, list]] = {tau: collections.defaultdict(list) for tau in taus}
    for tau in taus:
        for start in range(0, len(rows), chunk):
            window = slice(start, start + chunk)
            where = torch.sigmoid(
                margins(anchors[window], ids[window], shape, spacing, center) / tau
            ).prod(1)
            region = where > 0.05
            dilated = F.max_pool3d(region[:, None].float(), 17, 1, 8)[:, 0] > 0
            per_tau[tau]["mass_valid"] += where.flatten(1).mean(-1).cpu().tolist()
            # How far the field's own centre of mass is from the target it
            # describes. The gate says the target *is inside* the high region;
            # this says whether the region points at it. A cone intersection is
            # an elongated wedge with the target near its apex, so the two are
            # very different questions - and §5 uses this point as a heatmap
            # target, which makes the answer load-bearing.
            field_mass = where.flatten(1).sum(-1).clamp(min=1e-9)
            axes = [
                torch.arange(n, device=where.device, dtype=torch.float32) * step
                for n, step in zip((shape[2], shape[1], shape[0]), spacing)
            ]
            field_centre = torch.stack(
                [(where.sum((1, 2)) * axes[0]).sum(-1),
                 (where.sum((1, 3)) * axes[1]).sum(-1),
                 (where.sum((2, 3)) * axes[2]).sum(-1)], dim=-1
            ) / field_mass[:, None]
            wanted = as_t([centroids[row["target"]] for row in rows[window]])
            per_tau[tau]["field_centre_error"] += (field_centre - wanted).norm(dim=-1).cpu().tolist()
            per_tau[tau]["field_volume"] += region.flatten(1).float().mean(-1).cpu().tolist()
            per_tau[tau]["dilated_volume"] += dilated.flatten(1).float().mean(-1).cpu().tolist()
            for offset, row in enumerate(rows[window]):
                mask = volume == int(row["target"])
                total = mask.sum().clamp(min=1)
                per_tau[tau]["target_in_field"].append(float((mask & region[offset]).sum() / total))
                per_tau[tau]["target_in_dilation"].append(float((mask & dilated[offset]).sum() / total))
        for start in range(0, len(empty), chunk):
            window = [live.index(i) for i in empty[start:start + chunk]]
            rowset = [empty[start:start + chunk]]
            where = torch.sigmoid(
                margins(anchors[rowset[0]], flip_ids[window], shape, spacing, center) / tau
            ).prod(1)
            per_tau[tau]["mass_empty"] += where.flatten(1).mean(-1).cpu().tolist()

    at_flip = (
        margin_at(at_target[live], anchors[live], flip_ids, center).cpu().numpy()
        if live else np.zeros((0, anchors.shape[1]))
    )
    return per_tau, margin_at(at_target, anchors, ids, center).cpu().numpy(), at_flip, counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="data/mri")
    parser.add_argument("--split", default="train")
    parser.add_argument("--examples", type=int, default=1200, help="0 = every row")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tau", type=float, nargs="+", default=list(TAUS))
    parser.add_argument("--chunk", type=int, default=6, help="prompts per volume batch")
    parser.add_argument("--segmenter", type=Path, help="Stage A checkpoint; adds the predicted-anchor table")
    parser.add_argument("--scenes", type=int, default=20, help="scenes for the predicted-anchor table")
    args = parser.parse_args()

    corpus = Corpus.load(args.root)
    device = resolve_device("auto")
    rows = corpus.records(args.split, targets=corpus.vocab.names)
    rng = np.random.default_rng(args.seed)
    if args.examples and args.examples < len(rows):
        rows = [rows[int(i)] for i in sorted(rng.permutation(len(rows))[: args.examples])]
    by_scene: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        by_scene[row["scene"]].append(row)

    totals: dict[float, dict[str, list]] = {tau: collections.defaultdict(list) for tau in args.tau}
    base, flipped, counts = [], [], collections.Counter()
    for scene_id, scene_rows in by_scene.items():
        labels, centroids, center, present = scene_geometry(corpus, scene_id)
        per_tau, at_target, at_flip, tally = sweep_scene(
            scene_rows, labels, centroids, center, present, corpus, args.tau, rng, device, args.chunk
        )
        for tau, values in per_tau.items():
            for key, value in values.items():
                totals[tau][key] += value
        base.append(at_target)
        flipped.append(at_flip)
        counts += tally

    base, flipped = np.concatenate(base), np.concatenate(flipped)
    clears = lambda m, tau: (
        float(((1 / (1 + np.exp(-np.clip(m / tau, -60, 60)))).prod(1) > 0.5).mean()) if len(m) else float("nan")
    )
    print(f"corpus {corpus.root} | split {args.split} | {len(base)} examples, {len(by_scene)} scenes")
    print(f"spacing {corpus.spacing} world units per voxel - tau is in those units\n")
    print("1. the gate, swept. 'gate' is the proposal's precondition; everything right of")
    print("   it is what buying that fraction costs.")
    print(f"   {'tau':>5} {'gate':>8} {'flip':>8} {'field':>8} {'dilated':>8} "
          f"{'target in':>10} {'in dilation':>12} {'null AUC':>9} {'centre err':>11}")
    for tau in args.tau:
        row = totals[tau]
        print(
            f"   {tau:>5.2f} {clears(base, tau):>8.4f} {clears(flipped, tau):>8.4f}"
            f" {np.mean(row['field_volume']) * 100:>7.2f}% {np.mean(row['dilated_volume']) * 100:>7.2f}%"
            f" {np.mean(row['target_in_field']):>10.4f} {np.mean(row['target_in_dilation']):>12.4f}"
            f" {auc(row['mass_valid'], row['mass_empty']):>9.4f}"
            f" {np.mean(row['field_centre_error']):>8.1f}mm"
        )
    print("\n   gate        target centroids with where_raw > 0.5 (§2: the carver waits on this)")
    print("   flip        the same centroids under one flipped clause. Must fall to ~0.")
    print("   field       volume fraction of where_raw > 0.05")
    print("   dilated     the same after the 8-voxel dilation L_far measures mass outside")
    print("   target in   fraction of the target's own voxels inside the field / its dilation")
    print("   null AUC    where_mass alone, prompts that name one against flips that name none")
    print("   centre err  distance from the FIELD's centre of mass to the target's centroid.")
    print("               Large even where the gate passes: the conjunction is an elongated")
    print("               wedge and the target sits near its apex, so the region CONTAINS the")
    print("               target without POINTING at it. See deviations.md 3.1b - this is what")
    print("               makes the field a poor heatmap target on a prompt that names one.")
    print(f"\n   margin per clause at the target centroid: {quantiles(base.reshape(-1))}")
    print(f"   worst clause of the three:                {quantiles(base.min(1))}")

    print("\n2. one clause flipped, re-scored against the corpus rule")
    total = sum(counts.values())
    for key, value in sorted(counts.items()):
        print(f"   {key:<30} {value:>6}  {value / max(total, 1):.1%}")

    if args.segmenter is None:
        print("\n3. pass --segmenter for the anchor mass and centroid-error table")
        return 0

    model = load_stage_a(args.segmenter, device)
    every = torch.arange(len(corpus.vocab), device=device).unsqueeze(0)
    mass_gt, mass_pred, error = (collections.defaultdict(list) for _ in range(3))
    for scene_id in list(by_scene)[: args.scenes]:
        labels, centroids, _, present = scene_geometry(corpus, scene_id)
        image = normalize(load_nifti(corpus.root / "scenes" / scene_id / "image.nii.gz"), "zscore-brain")
        with torch.no_grad(), torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
            probability = torch.sigmoid(
                model(torch.as_tensor(image, device=device)[None, None], every, deep_supervision=False)
                .logits.float()
            )
        predicted, mass = soft_centroids(probability, corpus.spacing)
        truth = torch.as_tensor(labels.astype(np.int32), device=device)[None, None]
        ids = torch.arange(1, len(corpus.vocab) + 1, device=device).reshape(1, -1, 1, 1, 1)
        _, reference = soft_centroids((truth == ids).float(), corpus.spacing)
        for index in range(len(corpus.vocab)):
            if index + 1 not in present:
                continue
            structure = corpus.vocab.name(index + 1)
            mass_gt[structure].append(float(reference[0, index]))
            mass_pred[structure].append(float(mass[0, index]))
            error[structure].append(
                float(np.linalg.norm(predicted[0, index].cpu().numpy() - centroids[index + 1]))
            )

    print(f"\n3. Stage A anchors on {min(args.scenes, len(by_scene))} scenes ({args.segmenter})")
    print(f"   {'structure':<26} {'mass (gt)':>10} {'mass (pred)':>12} {'centroid error':>15}")
    for structure in sorted(mass_gt):
        print(f"   {structure:<26} {np.mean(mass_gt[structure]):>10.2e}"
              f" {np.mean(mass_pred[structure]):>12.2e}"
              f" {np.mean(error[structure]):>12.2f} mm")
    smallest = min(v for values in mass_pred.values() for v in values)
    print(f"\n   smallest predicted anchor mass {smallest:.3e} -> mapper.min_mass must sit below it")
    print(f"   centroid error, all structures: "
          f"{quantiles([v for values in error.values() for v in values])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
