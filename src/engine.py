"""Losses, metrics, and the one training loop every stage uses.

The stages differ only in how a batch becomes ``(logits, target, groups)``. That
is what a *task* is (:class:`StageATask`, :class:`BoundaryTask`,
:class:`StageBTask`); everything after it - optimiser, schedule, precision,
checkpoint selection, metrics - is shared, which is why there is one
:class:`Trainer` and not three.

``groups`` is the per-channel label the metric breakdown uses: the structure name
for Stage A, the target's name for Stage B. It is what turns one accumulator into
"per-class Dice" for one stage and "Dice on held-out target classes" for another.

Stage B's losses are the table in ``documentation/SpatialVox.md`` (Step 10 -
Losses), and its one structural consequence is that **every term is per-sample**: a
flipped prompt that names two structures is *dropped*, and a dropped example must
contribute to no loss and to no metric. That is the ``keep`` weight, and it runs
through :func:`segmentation_loss`, every relational term and :class:`Metrics`.
"""

from __future__ import annotations

import json
import math
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models import BoundaryPretrainer, StageA, StageB

EPS = 1e-8


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def weighted_mean(values: Tensor, weight: Tensor | None) -> Tensor:
    """Mean of ``values`` over samples, with ``weight`` zeroing dropped examples.

    ``weight`` is ``[B]``; ``values`` is ``[B]`` or ``[B, C]``. An all-dropped
    batch returns zero rather than ``nan``, which keeps a rare unlucky batch from
    poisoning the running loss.
    """
    if weight is None:
        return values.mean()
    weight = weight.to(values.dtype).reshape(-1, *([1] * (values.ndim - 1)))
    return (values * weight).sum() / (weight.sum() * values[0].numel()).clamp(min=EPS)


def segmentation_loss(
    logits: Tensor,
    target: Tensor,
    lambda_dice: float = 1.0,
    lambda_bce: float = 1.0,
    weight: Tensor | None = None,
) -> Tensor:
    """``lambda_dice * soft Dice + lambda_bce * BCEWithLogits``, per sample.

    Always computed in float32: the Dice denominator sums one probability per
    voxel - two million of them at 128^3 - and float16 has neither the range nor
    the resolution for that sum, so the objective would depend on the autocast
    dtype.
    """
    logits, target = logits.float(), target.float()
    probability = torch.sigmoid(logits).flatten(2)
    reference = target.flatten(2)
    intersection = (probability * reference).sum(-1)
    dice = 1 - (2 * intersection + 1) / (probability.sum(-1) + reference.sum(-1) + 1)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none").flatten(2).mean(-1)
    return lambda_dice * weighted_mean(dice, weight) + lambda_bce * weighted_mean(bce, weight)


def downsample_target(target: Tensor, size: Sequence[int]) -> Tensor:
    """Shrink binary targets for a coarse supervision scale, preserving presence.

    Max pooling, not nearest or average: at a quarter resolution a small nucleus
    is about one voxel thick and both alternatives can delete it entirely, which
    would supervise the coarse head towards an empty mask for a structure that is
    there.
    """
    target_size = tuple(int(v) for v in size)
    if tuple(target.shape[2:]) == target_size:
        return target
    factors = [s // t for s, t in zip(target.shape[2:], target_size)]
    if [t * f for t, f in zip(target_size, factors)] != list(target.shape[2:]):
        raise ValueError(f"cannot pool {tuple(target.shape[2:])} down to {target_size}")
    return F.max_pool3d(target.float(), factors, factors)


def deep_supervision_weights(n_scales: int, configured: Sequence[float]) -> list[float]:
    """One weight per decoder scale, coarse to fine.

    ``configured`` is used as given when it has the right length. At another
    depth it cannot be, so a doubling ramp normalised to sum to one stands in,
    keeping the same shape: coarse scales matter least.
    """
    if len(configured) == n_scales:
        return [float(w) for w in configured]
    ramp = [2.0**level for level in range(n_scales)]
    return [w / sum(ramp) for w in ramp]


def deep_supervision_loss(
    scales: Sequence[Tensor], target: Tensor, weights: Sequence[float], **loss_weights: float
) -> Tensor:
    """Weighted sum of :func:`segmentation_loss` over the decoder scales."""
    return sum(
        weight * segmentation_loss(logits, downsample_target(target, logits.shape[2:]), **loss_weights)
        for logits, weight in zip(scales, weights)
    )


def dilate(mask: Tensor, radius: int) -> Tensor:
    """Binary dilation by a cube of half-width ``radius``, separably.

    A ``(2r+1)^3`` max pool is 4913 taps at ``r = 8``; three one-dimensional
    passes are 51 and give the same L-infinity ball. The structuring element is a
    cube, not a sphere - ``L_far`` only needs "well away from the field", and the
    corner voxels of the cube are further out, so the penalty stays the looser of
    the two.
    """
    radius = int(radius)
    if radius <= 0:
        return mask
    size = 2 * radius + 1
    x = mask.float()
    for axis in range(3):
        kernel = [1, 1, 1]
        kernel[axis] = size
        padding = [0, 0, 0]
        padding[axis] = radius
        x = F.max_pool3d(x, tuple(kernel), 1, tuple(padding))
    return x


def mask_centroid_world(mask: Tensor, spacing: Sequence[float]) -> tuple[Tensor, Tensor]:
    """``[B, 1, D, H, W]`` -> ``([B, 3]`` world ``(x, y, z)``, ``[B]`` total mass``)``.

    From the three axis marginals, so it costs one pass and no coordinate grid.
    Empty masks return the origin and a zero mass; the caller decides what that
    means rather than having a ``nan`` decide for it.
    """
    mask = mask.float()
    if mask.ndim == 5:
        mask = mask.squeeze(1)
    total = mask.flatten(1).sum(-1)
    depth, height, width = mask.shape[1:]
    options = dict(device=mask.device, dtype=mask.dtype)
    x = torch.arange(width, **options) * float(spacing[0])
    y = torch.arange(height, **options) * float(spacing[1])
    z = torch.arange(depth, **options) * float(spacing[2])
    centroid = torch.stack(
        [
            (mask.sum(dim=(1, 2)) * x).sum(-1),
            (mask.sum(dim=(1, 3)) * y).sum(-1),
            (mask.sum(dim=(2, 3)) * z).sum(-1),
        ],
        dim=-1,
    ) / total.clamp(min=EPS).unsqueeze(-1)
    return centroid, total


def far_mass(probability: Tensor, where_raw: Tensor, epsilon: float, radius: int) -> Tensor:
    """Mean predicted probability outside ``dilate(where_raw > epsilon)``, per sample.

    ``L_far`` is the *only* spatial penalty on a valid prompt. A coverage term on
    the whole exterior fights the body of a structure that legitimately extends
    past the pyramid - measured on ``data/mri``, only 36% of a target's voxels sit
    inside ``where_raw > 0.05``, and 94% inside its 8-voxel dilation. Dice is
    allowed to follow an MRI boundary a few voxels out; this term only says "not
    on the other side of the head".
    """
    outside = 1.0 - dilate((where_raw > float(epsilon)).float(), radius)
    return (probability.float() * outside).flatten(1).sum(-1) / outside.flatten(1).sum(-1).clamp(min=1.0)


def null_gated(probability: Tensor, valid: Tensor) -> Tensor:
    """The model's answer: the carver's mask, emptied where the null head says the
    clauses name nothing (``valid <= 0``, the threshold ``null_summary`` uses).

    Under ``mask_on: valid`` this gate is the only place emptiness is decided, so
    every Dice is reported with and without it (CLAUDE.md §7).
    """
    named = (valid > 0).to(probability.dtype).reshape(-1, *([1] * (probability.ndim - 1)))
    return probability * named


def dice_iou(prediction: Tensor, target: Tensor, threshold: float = 0.5) -> tuple[Tensor, Tensor]:
    """Per-sample, per-channel Dice and IoU on thresholded probabilities.

    Two empty masks score 1.0 and one empty against one non-empty scores 0.0,
    which is the standard convention and the only sane one when a structure can
    legitimately be absent.
    """
    predicted = (prediction >= threshold).flatten(2).float()
    reference = (target >= 0.5).flatten(2).float()
    intersection = (predicted * reference).sum(-1)
    total = predicted.sum(-1) + reference.sum(-1)
    dice = torch.where(total == 0, torch.ones_like(total), 2 * intersection / total.clamp(min=EPS))
    union = total - intersection
    iou = torch.where(union == 0, torch.ones_like(union), intersection / union.clamp(min=EPS))
    return dice, iou


def surface(mask: Tensor) -> Tensor:
    """Coordinates ``(z, y, x)`` of the surface voxels of one binary mask."""
    occupied = (mask > 0.5).float()[None, None]
    eroded = -F.max_pool3d(-F.pad(occupied, (1,) * 6), 3, 1)
    return torch.nonzero(occupied[0, 0] - eroded[0, 0] > 0).float()


def hausdorff(prediction: Tensor, target: Tensor, spacing=(1.0, 1.0, 1.0), percentile: float = 95.0) -> float:
    """Symmetric ``percentile``-th Hausdorff distance in world units, or ``nan``.

    Measured between mask surfaces - a few hundred voxels rather than a few
    hundred thousand - so the exact pairwise distance is affordable. Two empty
    masks are a perfect match; one empty mask is undefined and returned as
    ``nan`` so it is counted rather than silently averaged in.
    """
    a, b = surface(prediction), surface(target)
    if a.numel() == 0 and b.numel() == 0:
        return 0.0
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    scale = torch.tensor([spacing[2], spacing[1], spacing[0]], device=a.device)  # (z, y, x)
    distances = torch.cdist(a * scale, b * scale)
    quantile = percentile / 100.0
    return float(
        torch.maximum(
            torch.quantile(distances.amin(1), quantile), torch.quantile(distances.amin(0), quantile)
        )
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@dataclass
class Metrics:
    """Per-channel scores with their labels, summarised however you ask.

    Rows are kept rather than folded into running means, so one pass over a split
    can be reported overall and stratified by target, anchor, direction or slot
    without deciding the breakdown in advance.
    """

    rows: list[dict[str, Any]] = field(default_factory=list)

    def update(
        self,
        logits: Tensor,
        target: Tensor,
        groups: Sequence[Sequence[str]],
        *,
        threshold: float = 0.5,
        strata: Sequence[Mapping[str, Sequence[str]]] | None = None,
        spacing: Sequence[float] | None = None,
        percentile: float = 95.0,
        keep: Tensor | None = None,
        extra: Mapping[str, Sequence[float]] | None = None,
    ) -> None:
        """Accumulate one batch. ``groups[i][c]`` names channel ``c`` of sample ``i``.

        ``spacing`` turns on the Hausdorff distance, the one metric here that
        costs a surface extraction per sample; leave it ``None`` to skip it.
        ``keep`` drops a sample entirely - a prompt the loss table drops must not
        appear in a reported mean either. ``extra`` carries per-sample scalars
        (centroid error, field mass) that are averaged alongside Dice.
        """
        dice, iou = dice_iou(torch.sigmoid(logits.float()).cpu(), target.float().cpu(), threshold)
        for sample in range(dice.shape[0]):
            if keep is not None and float(keep[sample]) <= 0:
                continue
            distance = None
            if spacing is not None and logits.shape[1] == 1:
                distance = hausdorff(
                    torch.sigmoid(logits[sample, 0].float()).cpu(),
                    target[sample, 0].float().cpu(), spacing, percentile,
                )
            values = {k: float(v[sample]) for k, v in (extra or {}).items()}
            for channel in range(dice.shape[1]):
                self.rows.append(
                    {
                        "dice": float(dice[sample, channel]),
                        "iou": float(iou[sample, channel]),
                        "hausdorff": distance,
                        "name": groups[sample][channel],
                        "strata": dict(strata[sample]) if strata else {},
                        **values,
                    }
                )

    @staticmethod
    def _mean(rows: list[dict[str, Any]]) -> dict[str, float]:
        summary = {
            "dice": sum(row["dice"] for row in rows) / len(rows),
            "iou": sum(row["iou"] for row in rows) / len(rows),
            "n": len(rows),
        }
        for key in sorted({k for row in rows for k in row} - {"dice", "iou", "name", "strata"}):
            values = [
                row[key] for row in rows
                if row.get(key) is not None and not math.isnan(float(row[key]))
            ]
            if values:
                summary[key] = sum(values) / len(values)
        return summary

    def summary(self, stratify_by: Sequence[str] | None = None) -> dict[str, Any]:
        """Overall and per-name means, plus the requested strata."""
        if not self.rows:
            return {"dice": 0.0, "iou": 0.0, "n": 0, "by_name": {}}
        wanted = None if stratify_by is None else set(stratify_by)
        by_name: dict[str, list] = {}
        strata: dict[str, dict[str, list]] = {}
        for row in self.rows:
            by_name.setdefault(row["name"], []).append(row)
            for stratum, keys in row["strata"].items():
                if wanted is not None and stratum not in wanted:
                    continue
                for key in keys:
                    strata.setdefault(stratum, {}).setdefault(key, []).append(row)
        result = {
            **self._mean(self.rows),
            "by_name": {name: self._mean(rows) for name, rows in sorted(by_name.items())},
        }
        if strata:
            result["strata"] = {
                stratum: {key: self._mean(rows) for key, rows in sorted(buckets.items())}
                for stratum, buckets in sorted(strata.items())
            }
        return result


def format_table(summary: Mapping[str, Any], title: str = "overall") -> str:
    """Render a metric summary for the console."""

    def row(label: str, entry: Mapping[str, Any]) -> str:
        distance = entry.get("hausdorff")
        rendered = "      -" if distance is None else f"{distance:>7.2f}"
        error = entry.get("centroid_error")
        offset = "      -" if error is None else f"{error:>7.2f}"
        return (
            f"  {label:<26} {entry['dice']:>7.4f} {entry['iou']:>7.4f} "
            f"{rendered} {offset} {entry['n']:>5}"
        )

    lines = [
        f"  {title:<26} {'dice':>7} {'iou':>7} {'hd95':>7} {'centr':>7} {'n':>5}",
        row("all", summary),
    ]
    for name, entry in summary.get("by_name", {}).items():
        lines.append(row(f"  {name}", entry))
    for stratum, buckets in summary.get("strata", {}).items():
        lines.append(f"  -- {stratum} " + "-" * max(0, 52 - len(stratum)))
        for key, entry in buckets.items():
            lines.append(row(f"  {key}", entry))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------
def masks_from(labels: Tensor, ids: Tensor) -> Tensor:
    """``labels [B, D, H, W]`` and ``ids [B, K]`` -> ``[B, K, D, H, W]`` float masks.

    Every mask in this project is built here, on the accelerator, from the one
    label volume the dataset returned.
    """
    return (labels.unsqueeze(1) == ids[..., None, None, None]).float()


ANCHOR_SOURCES = ("predicted", "oracle")

#: Which prompts the heatmap's *field* target acts on. ``always`` is §5's table
#: read literally - "peak of ``where_raw``" appears in the *names one* column as
#: well as the *names none* one. ``empty-only`` restricts it to prompts that name
#: nothing, where it is the only target available.
#:
#: The choice is measurable, not stylistic. ``scripts/gate_mapper.py`` reports the
#: distance from the field's centre of mass to the target's centroid as **20.2 mm**
#: on ``data/mri``, almost independent of ``tau``: the conjunction of three cones
#: is an elongated wedge, the target sits near its apex, and so the region
#: *contains* the target without *pointing at* it. Under ``always`` that term
#: therefore pulls the heatmap about 20 mm off on every valid prompt, against the
#: structure-centroid term pulling it back.
FIELD_CENTROID_ON = ("always", "empty-only")

#: Which prompts the MASK term supervises. ``all`` is §5's table read literally:
#: a prompt that names nothing is trained towards the empty mask. Measured on
#: ``data/mri``, that taught the carver a rejection of its own - silent on 94.5%
#: of impossible prompts where the null head flags 63%, and on 75% of held-out
#: prompts that do name a structure. ``valid`` supervises the mask only where
#: the clauses name a structure, so an empty answer always costs the full Dice,
#: and "names nothing" is the null head's decision alone (:func:`null_gated`).
#: ``_update_ideas/2026-09-22-null-head-decides-emptiness.md``.
MASK_ON = ("all", "valid")


def roll_anchors(batch: Mapping[str, Any], shift: int) -> dict[str, Any]:
    """Rotate the anchor slots as a unit: ids, names, and any precomputed masks.

    The three move together or the counterfactual is not the one it claims to be
    - with a cache in play, rolling ``name_ids`` alone would leave channel ``i``
    holding the mask of the structure it *used* to name.
    """
    names = batch.get("name_ids", batch["anchors"] - 1)
    probe = {**batch, "anchors": batch["anchors"].roll(shift, 1), "name_ids": names.roll(shift, 1)}
    if batch.get("anchor_probability") is not None:
        probe["anchor_probability"] = batch["anchor_probability"].roll(shift, 1)
    return probe


def resolve_anchor_source(stage_cfg: Mapping[str, Any]) -> str:
    """``train.stage_b.anchor_source``: ``predicted`` (the method) or ``oracle``."""
    source = str(stage_cfg["anchor_source"] if "anchor_source" in stage_cfg else "predicted")
    if source not in ANCHOR_SOURCES:
        raise ValueError(f"anchor_source must be one of {ANCHOR_SOURCES}, got {source!r}")
    return source


def resolve_segmenter(stage_cfg: Mapping[str, Any], override: Path | str | None = None) -> Path:
    """The Stage A checkpoint Stage B is built around. Always required.

    Stage A is inside :class:`~src.models.StageB` and is what turns the three
    names into masks, so there is no mode in which it is absent -
    ``anchor_source: oracle`` replaces the *masks* it produces for a diagnostic,
    not the module. ``override`` is the CLI ``--segmenter`` and wins.
    """
    raw = override if override is not None else (
        stage_cfg["phase_a_checkpoint"] if "phase_a_checkpoint" in stage_cfg else None
    )
    if raw in (None, "", "null"):
        raise ValueError(
            "Stage B is built around a frozen Stage A. Set "
            "train.stage_b.phase_a_checkpoint or pass --segmenter."
        )
    return Path(raw)


@dataclass
class Prediction:
    """What a task hands back: what to score, what to score it against, and how."""

    logits: Tensor  # full resolution
    target: Tensor
    groups: list[list[str]]
    strata: list[dict[str, list[str]]] | None = None
    scales: list[Tensor] | None = None  # deep-supervision maps, coarse to fine
    keep: Tensor | None = None  # [B] 0 = dropped from every loss and every metric
    valid: Tensor | None = None  # [B] null-head logits
    valid_target: Tensor | None = None  # [B] 1 = the clauses name exactly one structure
    centroid: Tensor | None = None  # [B, 3] predicted, world (x, y, z)
    centroid_target: Tensor | None = None  # [B, 3] the structure's own centroid
    field_centroid: Tensor | None = None  # [B, 3] first moment of where_raw
    has_field: Tensor | None = None  # [B] whether where_raw has any mass at all
    where_raw: Tensor | None = None
    anchor_dice: Tensor | None = None  # [B, A] predicted anchors against ground truth


@dataclass
class StageATask:
    """Segment every prompted structure from the intensity volume."""

    model: StageA
    vocab: Any
    loss_weights: Mapping[str, float] = field(default_factory=dict)
    name: str = "stage_a"

    def __call__(self, batch: Mapping[str, Any]) -> Prediction:
        output = self.model(batch["image"], batch["prompt_ids"], deep_supervision=self.model.training)
        target = masks_from(batch["labels"], batch["prompt_ids"] + 1)
        names = [[self.vocab.names[int(i)] for i in row] for row in batch["prompt_ids"].cpu()]
        return Prediction(logits=output.logits, target=target, groups=names, scales=output.scales)

    def loss(self, prediction: Prediction) -> Tensor:
        weights = deep_supervision_weights(
            len(prediction.scales), self.model.config["deep_supervision"]
        )
        return deep_supervision_loss(
            prediction.scales, prediction.target, weights, **dict(self.loss_weights)
        )


def label_boundary(labels: Tensor) -> Tensor:
    """``[B, D, H, W]`` -> ``[B, 1, D, H, W]``: 1 where a 6-neighbour has a different label.

    No class channel and no target indicator - the map says *there is an edge
    here*, never *whose*. It is a pretraining target for ``B`` and never an
    inference input.
    """
    volume = labels.unsqueeze(1).float()
    different = torch.zeros_like(volume)
    for axis in range(2, 5):
        for shift in (1, -1):
            rolled = volume.roll(shift, dims=axis)
            index = [slice(None)] * 5
            index[axis] = 0 if shift == 1 else -1
            rolled = rolled.clone()
            rolled[tuple(index)] = volume[tuple(index)]  # replicate at the border
            different = torch.maximum(different, (rolled != volume).float())
    return different


@dataclass
class BoundaryTask:
    """Pretrain ``B`` with objectives that carry no class id.

    Three terms, all of which a structure Stage A has never seen would also
    satisfy: put back cubes blanked out of ``I``; mark where two neighbouring
    voxels differ in label; regress ``|grad I|``. The point is that ``B`` learns
    *edges*, not *which named thing this is*, so the carver can draw a lesion.
    """

    model: BoundaryPretrainer
    vocab: Any
    loss_weights: Mapping[str, float] = field(default_factory=dict)
    name: str = "boundary"

    def __call__(self, batch: Mapping[str, Any]) -> Prediction:
        image = batch["image"].float()
        blanked, holes = self.model.blank(image)
        heads = self.model(blanked)
        self._image, self._holes = image, holes
        boundary = label_boundary(batch["labels"])
        # Reported as a segmentation of the boundary map: it is the term whose
        # quality decides whether the carver has an edge to follow.
        return Prediction(
            logits=heads["boundary"], target=boundary,
            groups=[["boundary"]] * image.shape[0],
            scales=[heads["reconstruct"], heads["edge"]],
        )

    def loss(self, prediction: Prediction) -> Tensor:
        weights = dict(self.loss_weights)
        reconstruct, edge = prediction.scales
        image, holes = self._image, self._holes
        # Reconstruction is scored on the blanked voxels only; elsewhere it is
        # a copy, which teaches nothing.
        filled = (reconstruct.float() - image).abs() * holes
        target_edge = self.model.edges(image)
        return (
            float(weights.get("boundary", 1.0))
            * segmentation_loss(prediction.logits, prediction.target, 1.0, 1.0)
            + float(weights.get("reconstruct", 1.0)) * filled.sum() / holes.sum().clamp(min=1.0)
            + float(weights.get("edge", 0.5)) * (edge.float() - target_edge).abs().mean()
        )


@dataclass
class StageBTask:
    """The relational model: image and three clauses in, one mask and a null logit out.

    The batch supplies the clauses, the label volume the losses need, and the
    ``keep`` / ``valid`` flags ``ExampleDataset`` computed when it flipped a
    direction. Nothing that identifies the target reaches :meth:`StageB.forward`:
    the label volume is read *here*, to build a training target and to score, and
    is never an argument to the model.
    """

    model: StageB
    vocab: Any
    spacing: Sequence[float] = (1.0, 1.0, 1.0)
    anchor_source: str = "predicted"
    loss_weights: Mapping[str, float] = field(default_factory=dict)
    far_epsilon: float = 0.05
    far_dilation: int = 8
    field_centroid_on: str = "always"
    mask_on: str = "all"
    name: str = "stage_b"
    components: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.anchor_source not in ANCHOR_SOURCES:
            raise ValueError(f"anchor_source must be one of {ANCHOR_SOURCES}, got {self.anchor_source!r}")
        if self.field_centroid_on not in FIELD_CENTROID_ON:
            raise ValueError(
                f"field_centroid_on must be one of {FIELD_CENTROID_ON}, got {self.field_centroid_on!r}"
            )
        if self.mask_on not in MASK_ON:
            raise ValueError(f"mask_on must be one of {MASK_ON}, got {self.mask_on!r}")
        self.model.segmenter.eval()

    def __call__(self, batch: Mapping[str, Any]) -> Prediction:
        labels = batch["labels"]
        name_ids = batch.get("name_ids", batch["anchors"] - 1)
        truth = masks_from(labels, batch["anchors"])
        anchors = truth if self.anchor_source == "oracle" else batch.get("anchor_probability")
        output = self.model(
            batch["image"], batch["direction_ids"], name_ids,
            anchors=anchors, boundary_image=batch.get("boundary_image"),
        )

        keep = batch["keep"].float()
        valid_target = batch["valid"].float()
        # An empty prompt has no structure, so its mask target is the empty
        # volume - built by zeroing, never by asking for label 0, which is the
        # background and would be almost the whole head.
        target = masks_from(labels, batch["target"].unsqueeze(1)) * valid_target.reshape(-1, 1, 1, 1, 1)
        centroid_target, _ = mask_centroid_world(target, self.spacing)
        field_centroid, field_total = mask_centroid_world(output.where_raw, self.spacing)

        strata = [
            {
                "target": [name],
                "anchor": anchor_names,
                "direction": directions,
                "slot": [f"slot{i + 1}:{d}" for i, d in enumerate(directions)],
            }
            for name, anchor_names, directions in zip(
                batch["target_name"], batch["anchor_names"], batch["directions"]
            )
        ]
        return Prediction(
            logits=output.logits, target=target,
            groups=[[name] for name in batch["target_name"]], strata=strata,
            keep=keep, valid=output.valid, valid_target=valid_target,
            centroid=output.centroid, centroid_target=centroid_target,
            field_centroid=field_centroid, has_field=(field_total > 0).float(),
            where_raw=output.where_raw,
            anchor_dice=dice_iou(output.anchors, truth)[0],
        )

    def loss(self, prediction: Prediction) -> Tensor:
        """§5's table. Every term is per-sample, and ``keep`` is how one is dropped.

        The components are kept in :attr:`components` and logged per epoch. Six
        terms on different scales - a Dice in [0, 1], a BCE in nats, two offsets
        in millimetres - cannot be balanced by reading the total, and a run whose
        loss is 80% heatmap is training a different model from the one intended.
        """
        weights, keep = dict(self.loss_weights), prediction.keep
        valid = prediction.valid_target
        # The heatmap carries two targets and the mask loss reaches neither: the
        # structure's own centroid when there is a structure, and the field's own
        # centre always - which is what keeps a centroid when the mask is empty.
        offset = lambda a, b: F.smooth_l1_loss(a.float(), b.float(), reduction="none", beta=2.0).sum(-1)
        # Under `mask_on: valid` the carver is never rewarded for painting
        # nothing: a prompt that names nothing trains the null head, not the mask.
        mask_weight = keep * valid if self.mask_on == "valid" else keep
        terms = {
            "mask": segmentation_loss(
                prediction.logits, prediction.target,
                float(weights.get("dice", 1.0)), float(weights.get("bce", 1.0)), weight=mask_weight,
            ),
            "null_bce": float(weights.get("null_bce", 0.2)) * weighted_mean(
                F.binary_cross_entropy_with_logits(
                    prediction.valid.float(), valid, reduction="none"
                ),
                keep,
            ),
            "centroid": float(weights.get("centroid", 0.02)) * weighted_mean(
                offset(prediction.centroid, prediction.centroid_target), keep * valid
            ),
            "field_centroid": float(weights.get("field_centroid", 0.01)) * weighted_mean(
                offset(prediction.centroid, prediction.field_centroid),
                keep * prediction.has_field
                * (1.0 if self.field_centroid_on == "always" else 1.0 - valid),
            ),
            # L_far is the only spatial penalty, and only on a valid prompt. On an
            # impossible one, `mask_on: all` makes the empty-mask loss the penalty
            # everywhere; under `valid` nothing penalises the mask there - the null
            # head's gate (`null_gated`) is what empties it.
            "far": float(weights.get("far", 0.2)) * weighted_mean(
                far_mass(
                    torch.sigmoid(prediction.logits), prediction.where_raw,
                    self.far_epsilon, self.far_dilation,
                ),
                keep * valid,
            ),
        }
        self.components = {k: float(v.detach()) for k, v in terms.items()}
        return sum(terms.values())


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def build_optimizer(model: nn.Module, cfg: Mapping[str, Any]) -> torch.optim.Optimizer:
    """``{name, lr, weight_decay}`` -> an optimiser. Only ``adamw`` is supported.

    A model that declares ``trainable_parameters`` hands over only those: Stage B
    carries the frozen Stage A as a submodule, and the relational loss must not
    flow back into it.
    """
    name = str(cfg.get("name", "adamw")).lower()
    if name != "adamw":
        raise ValueError(f"optimizer.name must be 'adamw', got {name!r}")
    lr = float(cfg["lr"])
    if hasattr(model, "parameter_groups"):
        parameters = model.parameter_groups(lr)
    elif hasattr(model, "trainable_parameters"):
        parameters = model.trainable_parameters()
    else:
        parameters = model.parameters()
    return torch.optim.AdamW(parameters, lr=lr, weight_decay=float(cfg.get("weight_decay", 0.0)))


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: Mapping[str, Any], epochs: int
) -> torch.optim.lr_scheduler.LRScheduler:
    """``{name, warmup_epochs}`` -> linear warmup, then ``cosine`` decay or a hold."""
    name = str(cfg.get("name", "cosine")).lower()
    if name not in ("cosine", "constant"):
        raise ValueError(f"scheduler.name must be 'cosine' or 'constant', got {name!r}")
    warmup = int(cfg.get("warmup_epochs", 0))

    def factor(epoch: int) -> float:
        if warmup and epoch < warmup:
            return (epoch + 1) / warmup
        if name == "constant":
            return 1.0
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup) / max(epochs - warmup, 1)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


@dataclass
class EarlyStopping:
    """Stop when the metric has not improved by ``min_delta`` for ``patience`` epochs.

    ``patience = 0`` disables it. The first value is always an improvement, so a
    run cannot stop before it has a baseline. This is a *separate* notion of
    "best" from the one that decides checkpointing: a rise smaller than
    ``min_delta`` still saves a better checkpoint, it just does not reset the wait.
    """

    patience: int = 0
    min_delta: float = 0.0
    best: float = field(default=-float("inf"), init=False)
    waited: int = field(default=0, init=False)

    def update(self, metric: float) -> bool:
        if self.patience <= 0:
            return False
        if metric > self.best + self.min_delta:
            self.best, self.waited = metric, 0
            return False
        self.waited += 1
        return self.waited >= self.patience


def resolve_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_revision() -> str | None:
    """HEAD, with ``-dirty`` when tracked files differ from it."""
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()

    try:
        dirty = git("status", "--porcelain", "--untracked-files=no")
        return git("rev-parse", "HEAD") + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, OSError):
        return None


def save_checkpoint(path: Path | str, model: nn.Module, meta: Mapping[str, Any]) -> Path:
    """Weights plus everything needed to rebuild and trace the model.

    ``model.config`` holds the architecture ``load_model`` reconstructs from;
    ``meta`` holds the training schedule that produced these weights -
    ``flip_probability`` and the loss weights among them. Both are mirrored into
    a ``.json`` sidecar, so a run's settings are readable without torch.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "model": model.config,
        "kind": type(model).__name__,
        "meta": {"git": git_revision(), **dict(meta), "torch": torch.__version__},
    }
    torch.save(payload, path)
    path.with_suffix(".json").write_text(
        json.dumps({k: v for k, v in payload.items() if k != "state_dict"}, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return path


def load_model(path: Path | str, device="cpu") -> nn.Module:
    """Rebuild a model from its checkpoint; the architecture comes from the file."""
    payload = torch.load(path, map_location=device, weights_only=False)
    registry = {"StageA": StageA, "StageB": StageB, "BoundaryPretrainer": BoundaryPretrainer}
    model = registry[payload["kind"]](**payload["model"])
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval()


def load_stage_a(path: Path | str, device="cpu") -> StageA:
    """Load a Stage A checkpoint; anything else is an error."""
    model = load_model(path, device)
    if not isinstance(model, StageA):
        raise TypeError(f"expected a Stage A checkpoint, got {type(model).__name__} from {path}")
    return model


class Trainer:
    """One loop for every stage: schedule, AMP, metrics, best-Dice checkpointing.

    Args:
        task: what turns a batch into a :class:`Prediction` and a loss.
        cfg: the ``train`` block - seed, device, precision, batch/accum, and the
            ``early_stopping`` sub-block.
        stage: that stage's ``{epochs, optimizer, scheduler}`` block.
        evaluation: the ``evaluation`` block - which metrics and which strata.
        logging: the ``logging`` block - backend and its settings.
    """

    def __init__(
        self,
        task: StageATask | StageBTask | BoundaryTask,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        cfg: Mapping[str, Any],
        out_dir: Path | str,
        *,
        stage: Mapping[str, Any],
        evaluation: Mapping[str, Any] | None = None,
        logging: Mapping[str, Any] | None = None,
        spacing: Sequence[float] = (1.0, 1.0, 1.0),
        verbose: bool = True,
    ) -> None:
        self.task, self.cfg = task, dict(cfg)
        self.stage, self.evaluation = dict(stage), dict(evaluation or {})
        self.logging = dict(logging or {})
        self.train_loader, self.val_loader = train_loader, val_loader
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.epochs = int(stage["epochs"])
        self.spacing, self.verbose = tuple(spacing), verbose
        self.device = resolve_device(cfg["device"])
        self.model = task.model.to(self.device)
        self.amp = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg["precision"])
        if self.device.type == "cpu":
            self.amp = None  # conv3d has no fused low-precision CPU path; it is 30x slower
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=cfg["precision"] == "fp16")
        self.optimizer = build_optimizer(self.model, stage["optimizer"])
        self.schedule = build_scheduler(self.optimizer, stage["scheduler"], self.epochs)
        stopping = dict(cfg.get("early_stopping") or {})
        self.stopper = EarlyStopping(
            patience=int(stopping.get("patience", 0)), min_delta=float(stopping.get("min_delta", 0.0))
        )
        self.extra_loaders: dict[str, DataLoader] = {}
        #: A smaller loader for `prompt_dependence`, which costs five passes.
        #: The per-epoch probe is a trend, not a reported number - those come
        #: from `scripts/evaluate.py` over the whole split.
        self.probe_loader: DataLoader | None = None
        self.history: list[dict[str, Any]] = []
        self.best = -1.0
        #: The code this run started from, not whatever HEAD is when a checkpoint is saved.
        self.git = git_revision()

    @property
    def wants_hausdorff(self) -> bool:
        return "hausdorff" in self.evaluation.get("metrics", ())

    def to_device(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        return {k: v.to(self.device) if isinstance(v, Tensor) else v for k, v in batch.items()}

    def _autocast(self):
        return torch.autocast(self.device.type, dtype=self.amp, enabled=self.amp is not None)

    def _progress(self, iterable, *, desc: str):
        """Batch bar for a live run; silent when ``verbose`` is off (tests, scripts)."""
        if not self.verbose:
            return iterable
        return tqdm(iterable, desc=desc, leave=False, dynamic_ncols=True, mininterval=1.0, unit="batch")

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        getattr(self.train_loader.dataset, "set_epoch", lambda _: None)(epoch)
        accumulation = max(int(self.cfg["accum"]), 1)
        self.optimizer.zero_grad(set_to_none=True)
        total_loss, total_dice, steps = 0.0, 0.0, 0
        components: dict[str, float] = {}
        batches = self._progress(self.train_loader, desc=f"epoch {epoch} train")
        for index, raw in enumerate(batches):
            batch = self.to_device(raw)
            with self._autocast():
                prediction = self.task(batch)
            loss = self.task.loss(prediction)
            self.scaler.scale(loss / accumulation).backward()
            if (index + 1) % accumulation == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                dice = dice_iou(torch.sigmoid(prediction.logits.float()), prediction.target)[0]
                total_dice += float(weighted_mean(dice, prediction.keep))
            total_loss += float(loss.detach())
            for key, value in getattr(self.task, "components", {}).items():
                components[key] = components.get(key, 0.0) + value
            steps += 1
            if hasattr(batches, "set_postfix"):
                batches.set_postfix(loss=f"{total_loss / steps:.3f}", dice=f"{total_dice / steps:.3f}", refresh=False)
        if steps % accumulation:  # flush a partial window rather than dropping it
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
        return {
            "loss": total_loss / max(steps, 1),
            "dice": total_dice / max(steps, 1),
            **{f"loss_{k}": v / max(steps, 1) for k, v in components.items()},
        }

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader | None = None,
        *,
        with_hausdorff: bool | None = None,
        desc: str = "eval",
    ) -> dict[str, Any]:
        loader = loader or self.val_loader
        if loader is None:
            return {}
        if with_hausdorff is None:
            with_hausdorff = self.wants_hausdorff
        self.model.eval()
        metrics, total_loss, steps = Metrics(), 0.0, 0
        percentile = float(self.evaluation.get("hausdorff_percentile", 95.0))
        anchor_dice: list[float] = []
        null_scores: list[tuple[float, float]] = []
        gated_dice: list[float] = []
        empty: list[bool] = []
        threshold = float(self.cfg.get("threshold", 0.5))
        for raw in self._progress(loader, desc=desc):
            batch = self.to_device(raw)
            with self._autocast():
                prediction = self.task(batch)
            total_loss += float(self.task.loss(prediction))
            steps += 1
            extra = {}
            if prediction.centroid is not None:
                # Localisation, reported apart from Dice. Dice fuses "did it
                # point at the right structure" with "did it outline it", and
                # those two transfer differently.
                error = (prediction.centroid - prediction.centroid_target).norm(dim=-1)
                extra["centroid_error"] = torch.where(
                    prediction.valid_target > 0, error, torch.full_like(error, float("nan"))
                ).tolist()
            if prediction.anchor_dice is not None:
                anchor_dice += prediction.anchor_dice.flatten().tolist()
            if prediction.valid is not None:
                null_scores += list(
                    zip(prediction.valid.float().tolist(), prediction.valid_target.tolist())
                )
                # The shy-painting trend, every epoch: how often a prompt that
                # names a structure gets nothing back, and the Dice once the null
                # head's gate is applied (CLAUDE.md §7).
                probability = torch.sigmoid(prediction.logits.float())
                kept = (prediction.keep > 0) if prediction.keep is not None else torch.ones_like(
                    prediction.valid, dtype=torch.bool
                )
                gated = dice_iou(null_gated(probability, prediction.valid), prediction.target, threshold)[0]
                gated_dice += gated.flatten()[kept].tolist()
                named = kept & (prediction.target.flatten(1).sum(-1) > 0)
                painted = (probability >= threshold).flatten(1).sum(-1) > 0
                empty += (~painted)[named].tolist()
            metrics.update(
                prediction.logits, prediction.target, prediction.groups,
                strata=prediction.strata, threshold=threshold,
                spacing=self.spacing if with_hausdorff else None, percentile=percentile,
                keep=prediction.keep, extra=extra or None,
            )
        summary = metrics.summary(self.evaluation.get("stratify_by"))
        summary["loss"] = total_loss / max(steps, 1)
        if anchor_dice:
            summary["anchor_dice"] = sum(anchor_dice) / len(anchor_dice)
        if null_scores:
            summary.update(null_summary(null_scores))
        if gated_dice:
            summary["dice_null_gated"] = sum(gated_dice) / len(gated_dice)
        if empty:
            summary["empty_rate"] = sum(empty) / len(empty)
        return summary

    @torch.no_grad()
    def prompt_dependence(self, loader: DataLoader | None = None) -> dict[str, float]:
        """How far Dice falls when the prompt is altered in four different ways.

        The only signal that separates the two ways val Dice can rise. A model
        reading the prompt must lose Dice under ``flip_direction``,
        ``permute_channels`` and ``permute_clauses``.

        ``permute_both`` is the control: it preserves every relation, so it must
        not move. ``where_raw`` is a product and therefore exactly
        permutation-invariant, so the only order dependence left is the carver's
        ``cat``.
        """
        loader = loader or self.probe_loader or self.val_loader
        if loader is None or not isinstance(self.task, StageBTask):
            return {}
        from src.geometry import DIRECTIONS, OPPOSITE

        self.model.eval()
        opposites = torch.tensor([DIRECTIONS.index(OPPOSITE[d]) for d in DIRECTIONS])
        scores: dict[str, list[float]] = {
            k: [] for k in ("base", "flip_direction", "permute_channels", "permute_clauses", "permute_both")
        }
        for raw in self._progress(loader, desc="probes"):
            batch = self.to_device(raw)
            with self._autocast():
                reference = self.task(batch)
                target, keep = reference.target, reference.keep
                directions = batch["direction_ids"]
                turned = directions.clone()
                turned[:, 0] = opposites.to(directions.device)[turned[:, 0]]
                rolled = roll_anchors(batch, 1)
                probes = {
                    "base": batch,
                    "flip_direction": {**batch, "direction_ids": turned},
                    # The masks move, the words stay: channel i stops being the
                    # structure clause i names.
                    "permute_channels": rolled,
                    "permute_clauses": {**batch, "direction_ids": directions.roll(1, 1)},
                    "permute_both": {**rolled, "direction_ids": directions.roll(1, 1)},
                }
                for key, probe in probes.items():
                    logits = reference.logits if key == "base" else self.task(probe).logits
                    dice = dice_iou(torch.sigmoid(logits.float()), target)[0].flatten()
                    scores[key] += dice[keep > 0].tolist() if keep is not None else dice.tolist()
        if not scores["base"]:
            return {}
        mean = lambda xs: sum(xs) / len(xs)
        base = mean(scores["base"])
        return {f"{key}_drop": base - mean(values) for key, values in scores.items() if key != "base"}

    def fit(self) -> list[dict[str, Any]]:
        seed_all(int(self.cfg["seed"]))
        trainable = sum(
            p.numel() for group in self.optimizer.param_groups for p in group["params"]
        ) / 1e6
        total = sum(p.numel() for p in self.model.parameters()) / 1e6
        if self.verbose:
            print(
                f"{self.task.name}: {trainable:.2f}M trainable of {total:.2f}M "
                f"on {self.device} ({self.cfg['precision']})"
            )
        # A relaunch into the same directory extends the log; it never truncates it.
        log = (self.out_dir / "metrics.jsonl").open("a", encoding="utf-8")
        run = _logger(self.logging, self.out_dir.name, {**self.cfg, **self.stage})
        try:
            for epoch in range(self.epochs):
                started = time.perf_counter()
                train = self.train_epoch(epoch)
                self.schedule.step()
                # Hausdorff costs a surface extraction and a cdist per sample and
                # selects nothing; the final pass in `scripts/train.py` reports it.
                val = self.evaluate(desc="val", with_hausdorff=False)
                record = {
                    "epoch": epoch,
                    "train": train,
                    "val": {k: v for k, v in val.items() if k not in ("by_name", "strata")},
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "seconds": round(time.perf_counter() - started, 2),
                }
                record["val"].update(self.prompt_dependence())
                for name, loader in self.extra_loaders.items():
                    scored = self.evaluate(loader, with_hausdorff=False, desc=name)
                    record[name] = {k: v for k, v in scored.items() if k not in ("by_name", "strata")}
                self.history.append({**record, "val_full": val})
                log.write(json.dumps(record) + "\n")
                log.flush()
                if run is not None:
                    run.log(_flatten(record), step=epoch)
                if self.verbose:
                    print(self._line(epoch, train, val, record))
                dice = val.get("dice", 0.0)
                # Checkpointing and early stopping use different notions of
                # "better": any rise is worth keeping, but only a rise of more
                # than min_delta resets the patience counter.
                if dice > self.best:
                    self.best = dice
                    save_checkpoint(self.out_dir / "best.pt", self.model, self._meta(epoch, val))
                if self.stopper.update(dice):
                    print(
                        f"  early stop: val dice has not risen by more than "
                        f"{self.stopper.min_delta} for {self.stopper.patience} epochs "
                        f"(best {self.best:.4f})"
                    )
                    break
        finally:
            log.close()
            if run is not None:
                run.finish()
        save_checkpoint(self.out_dir / "last.pt", self.model, self._meta(self.epochs - 1, {}))
        (self.out_dir / "history.json").write_text(json.dumps(self.history, indent=2) + "\n", encoding="utf-8")
        return self.history

    def _line(self, epoch, train, val, record) -> str:
        parts = [
            f"  epoch {epoch:>3}  loss {train['loss']:.4f}  train dice {train['dice']:.4f}",
            f"  val dice {val.get('dice', 0.0):.4f}",
        ]
        for key, label, fmt in (
            ("centroid_error", "centr", "{:.2f}"), ("null_auc", "null", "{:.3f}"),
            ("empty_rate", "empty", "{:.2f}"),
        ):
            if key in val:
                parts.append(f"  {label} {fmt.format(val[key])}")
        for key in ("flip_direction_drop", "permute_both_drop"):
            if key in record["val"]:
                parts.append(f"  {key.split('_')[0][:4]} {record['val'][key]:+.3f}")
        for name in self.extra_loaders:
            if name in record:
                parts.append(f"  {name} {record[name].get('dice', 0.0):.4f}")
                if "empty_rate" in record[name]:
                    parts.append(f" (empty {record[name]['empty_rate']:.2f})")
        return "".join(parts) + f"  ({record['seconds']}s)"

    def _meta(self, epoch: int, metrics: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "git": self.git,
            "stage": self.task.name,
            "epoch": epoch,
            "best_dice": self.best,
            "metrics": {k: v for k, v in metrics.items() if k != "strata"},
            "config": {**self.cfg, "stage": self.stage, "evaluation": self.evaluation},
        }


def null_summary(scores: Sequence[tuple[float, float]]) -> dict[str, float]:
    """Accuracy and AUC of the null head, plus the rate it calls a prompt empty.

    The AUC is what matters: ``scripts/gate_mapper.py`` measures the ceiling that
    ``where_mass`` alone imposes on this head, and a trained value at that ceiling
    means the head is working as well as its four inputs permit - not that the
    architecture is sound.
    """
    logits = np.array([s for s, _ in scores])
    labels = np.array([t for _, t in scores])
    positive, negative = logits[labels > 0], logits[labels <= 0]
    summary = {
        "null_accuracy": float(((logits > 0) == (labels > 0)).mean()),
        "null_rate": float((logits <= 0).mean()),
    }
    if positive.size and negative.size:
        summary["null_auc"] = float(
            (positive[:, None] > negative[None, :]).mean()
            + 0.5 * (positive[:, None] == negative[None, :]).mean()
        )
    return summary


def _flatten(record: Mapping[str, Any], prefix: str = "") -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, value in record.items():
        if isinstance(value, Mapping):
            flat.update(_flatten(value, f"{prefix}{key}/"))
        elif isinstance(value, (int, float)):
            flat[f"{prefix}{key}"] = float(value)
    return flat


def _logger(logging: Mapping[str, Any], name: str, config: Mapping[str, Any]):
    """The ``logging`` block's backend. ``none`` (or a missing wandb) is a no-op.

    Every run writes ``metrics.jsonl`` regardless; this only mirrors it.
    """
    backend = str(logging.get("backend", "none")).lower()
    if backend in ("none", "", "null"):
        return None
    if backend != "wandb":
        raise ValueError(f"logging.backend must be 'wandb' or 'none', got {backend!r}")
    settings = dict(logging.get("wandb") or {})
    try:
        import wandb
    except ImportError:
        print("logging.backend is wandb but wandb is not installed; metrics.jsonl only")
        return None
    return wandb.init(
        project=settings.get("project"),
        name=settings.get("name") or name,
        tags=list(settings.get("tags") or []),
        config=dict(config),
        reinit=True,
    )
