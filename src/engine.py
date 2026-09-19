"""Losses, metrics, and the one training loop both stages use.

The two stages differ only in how a batch becomes ``(logits, target, groups)``.
That is what a *task* is (:class:`StageATask`, :class:`StageBTask`); everything
after it - optimiser, schedule, precision, checkpoint selection, metrics - is
shared, which is why there is one :class:`Trainer` and not two.

``groups`` is the per-channel label used for the metric breakdown: the structure
name for Stage A, the target's name for Stage B. It is what turns one accumulator
into "per-class Dice" for one stage and "Dice on held-out target classes" for the
other.
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

from src.models import StageA, StageB

EPS = 1e-8


# ---------------------------------------------------------------------------
# Losses and metrics
# ---------------------------------------------------------------------------
def segmentation_loss(
    logits: Tensor, target: Tensor, lambda_dice: float = 1.0, lambda_bce: float = 1.0
) -> Tensor:
    """``lambda_dice * soft Dice + lambda_bce * BCEWithLogits``.

    Always computed in float32: the Dice denominator sums one probability per
    voxel - 262,144 of them at 64^3 - and float16 has neither the range nor the
    resolution for that sum, so the objective would depend on the autocast dtype.
    """
    logits, target = logits.float(), target.float()
    probability = torch.sigmoid(logits).flatten(2)
    reference = target.flatten(2)
    intersection = (probability * reference).sum(-1)
    dice = 1 - ((2 * intersection + 1) / (probability.sum(-1) + reference.sum(-1) + 1)).mean()
    return lambda_dice * dice + lambda_bce * F.binary_cross_entropy_with_logits(logits, target)


def downsample_target(target: Tensor, size: Sequence[int]) -> Tensor:
    """Shrink binary targets for a coarse supervision scale, preserving presence.

    Max pooling, not nearest or average: at a quarter resolution a torus is about
    one voxel thick and both alternatives can delete it entirely, which would
    supervise the coarse head towards an empty mask for a structure that is there.
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

    ``configured`` is used as given when it has the right length - which is the
    case at the resolution the weights were tuned for. At another depth it cannot
    be, so a doubling ramp normalised to sum to one stands in, keeping the same
    shape: coarse scales matter least.
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
    ) -> None:
        """Accumulate one batch. ``groups[i][c]`` names channel ``c`` of sample ``i``.

        ``spacing`` turns on the Hausdorff distance, which is the one metric here
        that costs a surface extraction per sample; leave it ``None`` to skip it.
        """
        dice, iou = dice_iou(torch.sigmoid(logits.float()).cpu(), target.float().cpu(), threshold)
        for sample in range(dice.shape[0]):
            distance = None
            if spacing is not None and logits.shape[1] == 1:
                distance = hausdorff(
                    torch.sigmoid(logits[sample, 0].float()).cpu(),
                    target[sample, 0].float().cpu(), spacing, percentile,
                )
            for channel in range(dice.shape[1]):
                self.rows.append(
                    {
                        "dice": float(dice[sample, channel]),
                        "iou": float(iou[sample, channel]),
                        "hausdorff": distance,
                        "name": groups[sample][channel],
                        "strata": dict(strata[sample]) if strata else {},
                    }
                )

    @staticmethod
    def _mean(rows: list[dict[str, Any]]) -> dict[str, float]:
        distances = [row["hausdorff"] for row in rows if row["hausdorff"] is not None and not math.isnan(row["hausdorff"])]
        summary = {
            "dice": sum(row["dice"] for row in rows) / len(rows),
            "iou": sum(row["iou"] for row in rows) / len(rows),
            "n": len(rows),
        }
        if distances:
            summary["hausdorff"] = sum(distances) / len(distances)
        return summary

    def summary(self, stratify_by: Sequence[str] | None = None) -> dict[str, Any]:
        """Overall and per-name means, plus the requested strata.

        ``stratify_by`` is ``evaluation.stratify_by``; ``None`` reports every
        stratum the rows happen to carry.
        """
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
        return f"  {label:<26} {entry['dice']:>7.4f} {entry['iou']:>7.4f} {rendered} {entry['n']:>5}"

    lines = [f"  {title:<26} {'dice':>7} {'iou':>7} {'hd95':>7} {'n':>5}", row("all", summary)]
    for name, entry in summary.get("by_name", {}).items():
        lines.append(row(f"  {name}", entry))
    for stratum, buckets in summary.get("strata", {}).items():
        lines.append(f"  -- {stratum} " + "-" * max(0, 52 - len(stratum)))
        for key, entry in buckets.items():
            lines.append(row(f"  {key}", entry))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tasks: the only thing that differs between the two stages
# ---------------------------------------------------------------------------
def masks_from(labels: Tensor, ids: Tensor) -> Tensor:
    """``labels [B, D, H, W]`` and ``ids [B, K]`` -> ``[B, K, D, H, W]`` float masks.

    Every mask in this project is built here, on the accelerator, from the one
    label volume the dataset returned.
    """
    return (labels.unsqueeze(1) == ids[..., None, None, None]).float()


OCCUPANCY_MODES = ("all", "distractors-only", "anchors-only", "none")
STAGE_B_MODES = ("predicted", "oracle")


def _take_channels(masks: Tensor, ids: Tensor) -> Tensor:
    """``masks [B, C, ...]`` indexed by ``ids [B, K]`` along the channel axis."""
    extra = (1,) * (masks.ndim - 2)
    index = ids.to(dtype=torch.long).reshape(*ids.shape, *extra)
    return masks.gather(1, index.expand(-1, -1, *masks.shape[2:]))


def occupancy_from(
    batch: Mapping[str, Any],
    mode: str = "all",
    *,
    anchors: Tensor | None = None,
    full: Tensor | None = None,
) -> Tensor:
    """Which masks of an already-chosen source are unioned into occupancy.

    Source is resolved by :class:`StageBTask` — this function never queries a
    segmenter. Omitted ``anchors`` / ``full`` fall back to ground-truth labels
    (the oracle source).

    * ``all`` - every shape the source can name. Ceiling: ground truth includes
      the target, and a segmenter trained on the target's name will paint it too.
    * ``distractors-only`` - ``all`` with the target's own voxels removed: the
      candidate blobs without the answer's silhouette. This is the only mode
      that reads ``batch["target"]``, which makes it a **diagnostic, not a
      deployable setting** - inference cannot exclude a target it has not found
      yet. :class:`StageBTask` allows it under ``mode: oracle`` only.
    * ``anchors-only`` - the union of the prompt's three anchors. After
      :meth:`StageB.forward` subtracts those same channels, the decoder
      occupancy is empty, so this is bit-for-bit ``none`` at the decoder. Kept
      as a named arm because it is the natural thing to reach for; see
      ``notebooks/occupancy_ablation.py`` for the proof.
    * ``none`` - an empty channel; Stage B has to localise from the relations
      alone, and from the intensity volume if the model takes one.
    """
    if mode not in OCCUPANCY_MODES:
        raise ValueError(f"occupancy_mode must be one of {OCCUPANCY_MODES}, got {mode!r}")

    labels: Tensor = batch["labels"]
    batch_size, device, spatial = labels.shape[0], labels.device, labels.shape[1:]
    if mode == "none":
        return torch.zeros(batch_size, 1, *spatial, dtype=torch.float32, device=device)
    if mode == "anchors-only":
        if anchors is None:
            anchors = masks_from(labels, batch["anchors"])
        return anchors.amax(dim=1, keepdim=True)
    union = full if full is not None else (labels > 0).float().unsqueeze(1)
    if mode == "distractors-only":
        return (union - masks_from(labels, batch["target"].unsqueeze(1))).clamp(0, 1)
    return union


def resolve_stage_b_mode(stage_cfg: Mapping[str, Any]) -> str:
    """``train.stage_b.mode``: ``predicted`` (default) or ``oracle``."""
    raw = stage_cfg["mode"] if "mode" in stage_cfg else "predicted"
    mode = str(raw)
    if mode not in STAGE_B_MODES:
        raise ValueError(f"train.stage_b.mode must be one of {STAGE_B_MODES}, got {mode!r}")
    return mode


def resolve_phase_a_checkpoint(
    stage_cfg: Mapping[str, Any], override: Path | str | None = None
) -> Path | None:
    """The Stage A checkpoint Stage B should use, or ``None`` for ground truth.

    ``train.stage_b.mode`` defaults to ``predicted``, which *requires* a Stage A
    checkpoint from ``train.stage_b.phase_a_checkpoint`` or the CLI
    ``--segmenter``. Missing that path is an error. The only bypass is
    ``mode: oracle``, which reads ground-truth anchors and occupancy and
    ignores a leftover checkpoint path in the config.

    ``override`` is the CLI ``--segmenter``. In predicted mode it wins over the
    config path. In oracle mode it is rejected: pick one source.
    """
    mode = resolve_stage_b_mode(stage_cfg)
    if mode == "oracle":
        if override is not None:
            raise ValueError(
                "train.stage_b.mode is 'oracle' (ground-truth anchors); "
                "do not pass --segmenter. Set train.stage_b.mode: predicted "
                "to use a Stage A checkpoint."
            )
        return None
    raw = override if override is not None else (
        stage_cfg["phase_a_checkpoint"] if "phase_a_checkpoint" in stage_cfg else None
    )
    if raw in (None, "", "null"):
        raise ValueError(
            "Stage B mode is 'predicted' and needs a pretrained Stage A checkpoint. "
            "Set train.stage_b.phase_a_checkpoint, pass --segmenter, or set "
            "train.stage_b.mode: oracle to use ground-truth anchors."
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


@dataclass
class StageATask:
    """Segment every prompted structure from the intensity volume."""

    model: StageA
    vocab: Any
    loss_weights: Mapping[str, float] = field(default_factory=dict)
    name: str = "stage_a"
    segmenter: None = None  # Stage A has no anchor source; keeps the two tasks alike

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


@dataclass
class StageBTask:
    """Segment the relational target from one mask source.

    ``mode: predicted`` (default) requires a :class:`~src.models.StageA`
    ``segmenter``; one forward produces both the encoder anchors and the
    occupancy union. ``mode: oracle`` reads ground-truth labels and rejects a
    segmenter. ``occupancy_mode`` then chooses which of that source's masks
    are unioned: ``all``, ``anchors-only``, or ``none``.
    """

    model: StageB
    vocab: Any
    mode: str = "predicted"
    segmenter: StageA | None = None
    threshold: float = 0.5
    occupancy_mode: str = "all"
    loss_weights: Mapping[str, float] = field(default_factory=dict)
    anchor_scores: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.occupancy_mode not in OCCUPANCY_MODES:
            raise ValueError(
                f"occupancy_mode must be one of {OCCUPANCY_MODES}, got {self.occupancy_mode!r}"
            )
        self.mode = resolve_stage_b_mode({"mode": self.mode})
        if self.occupancy_mode == "distractors-only" and self.mode != "oracle":
            raise ValueError(
                "occupancy_mode 'distractors-only' removes the target's own voxels, which needs "
                "ground truth: it is a diagnostic and runs under mode 'oracle' only"
            )
        if self.mode == "oracle":
            if self.segmenter is not None:
                raise ValueError("Stage B mode is 'oracle'; do not pass a segmenter")
            return
        if self.segmenter is None:
            raise ValueError("Stage B mode is 'predicted' and needs a Stage A segmenter")
        if not isinstance(self.segmenter, StageA):
            raise TypeError(f"segmenter must be a StageA, got {type(self.segmenter).__name__}")
        if int(self.segmenter.config["vocab_size"]) != len(self.vocab):
            raise ValueError(
                f"Stage A vocab_size {self.segmenter.config['vocab_size']} "
                f"!= Stage B vocab {len(self.vocab)}"
            )
        self.segmenter.eval()

    @property
    def name(self) -> str:
        return "stage_b_predicted" if self.mode == "predicted" else "stage_b"

    def source(self, batch: Mapping[str, Any]) -> tuple[Tensor, Tensor | None]:
        """Anchor channels and, when needed, the full occupancy union. One source."""
        needs_union = self.occupancy_mode in ("all", "distractors-only")
        oracle = masks_from(batch["labels"], batch["anchors"])
        if self.mode == "oracle":
            full = (batch["labels"] > 0).float().unsqueeze(1) if needs_union else None
            return oracle, full
        image = batch["image"]
        name_ids = batch["anchors"] - 1
        if needs_union:
            vocab_ids = torch.arange(len(self.vocab), device=image.device).unsqueeze(0).expand(image.shape[0], -1)
            all_masks = self.segmenter.masks_for(image, vocab_ids, self.threshold)
            predicted = _take_channels(all_masks, name_ids)
            full = all_masks.amax(dim=1, keepdim=True)
        else:
            predicted = self.segmenter.masks_for(image, name_ids, self.threshold)
            full = None
        self.anchor_scores += dice_iou(predicted, oracle)[0].flatten().tolist()
        return predicted, full

    def __call__(self, batch: Mapping[str, Any]) -> Prediction:
        anchor_masks, full = self.source(batch)
        occupancy = occupancy_from(
            batch, self.occupancy_mode, anchors=anchor_masks, full=full,
        )
        # The anchor names the prompt uses default to the ones the channels hold.
        # A counterfactual overrides `name_ids` to break exactly that link.
        name_ids = batch.get("name_ids", batch["anchors"] - 1)
        # The intensity volume, when the model was built to take one. It is the
        # scene as acquired - never the labels, and never anything derived from
        # the target - so it says what is there without saying which one.
        image = batch["image"] if self.model.config.get("image") else None
        output = self.model(anchor_masks, batch["direction_ids"], name_ids, occupancy, image)
        target = masks_from(batch["labels"], batch["target"].unsqueeze(1))
        strata = [
            {
                "target": [name],
                "anchor": anchors,
                "direction": directions,
                "slot": [f"slot{i + 1}:{d}" for i, d in enumerate(directions)],
            }
            for name, anchors, directions in zip(batch["target_name"], batch["anchor_names"], batch["directions"])
        ]
        return Prediction(
            logits=output.logits, target=target,
            groups=[[name] for name in batch["target_name"]], strata=strata,
        )

    def loss(self, prediction: Prediction) -> Tensor:
        return segmentation_loss(prediction.logits, prediction.target, **dict(self.loss_weights))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def build_optimizer(model: nn.Module, cfg: Mapping[str, Any]) -> torch.optim.Optimizer:
    """``{name, lr, weight_decay}`` -> an optimiser. Only ``adamw`` is supported."""
    name = str(cfg.get("name", "adamw")).lower()
    if name != "adamw":
        raise ValueError(f"optimizer.name must be 'adamw', got {name!r}")
    return torch.optim.AdamW(
        model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg.get("weight_decay", 0.0))
    )


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
    run cannot stop before it has a baseline. Note this is a *separate* notion of
    "best" from the one that decides checkpointing: a rise smaller than
    ``min_delta`` still saves a better checkpoint, it just does not reset the
    wait.
    """

    patience: int = 0
    min_delta: float = 0.0
    best: float = field(default=-float("inf"), init=False)
    waited: int = field(default=0, init=False)

    def update(self, metric: float) -> bool:
        """Record ``metric``; returns True when training should stop."""
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
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return None


def save_checkpoint(path: Path | str, model: nn.Module, meta: Mapping[str, Any]) -> Path:
    """Weights plus everything needed to rebuild and trace the model."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "model": model.config,
        "kind": type(model).__name__,
        "meta": {**dict(meta), "git": git_revision(), "torch": torch.__version__},
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
    model = {"StageA": StageA, "StageB": StageB}[payload["kind"]](**payload["model"])
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval()


def load_stage_a(path: Path | str, device="cpu") -> StageA:
    """Load a Stage A checkpoint; anything else is an error."""
    model = load_model(path, device)
    if not isinstance(model, StageA):
        raise TypeError(f"expected a Stage A checkpoint, got {type(model).__name__} from {path}")
    return model


class Trainer:
    """One loop for both stages: schedule, AMP, metrics, best-Dice checkpointing.

    Args:
        task: what turns a batch into a :class:`Prediction` and a loss.
        cfg: the ``train`` block - seed, device, precision, batch/accum, and the
            ``early_stopping`` sub-block.
        stage: that stage's ``{epochs, optimizer, scheduler}`` block.
        evaluation: the ``evaluation`` block - which metrics to compute and which
            strata to report.
        logging: the ``logging`` block - backend and its settings.
    """

    def __init__(
        self,
        task: StageATask | StageBTask,
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
        self.history: list[dict[str, Any]] = []
        self.best = -1.0

    @property
    def wants_hausdorff(self) -> bool:
        """Whether ``evaluation.metrics`` asks for the one metric that costs anything."""
        return "hausdorff" in self.evaluation.get("metrics", ())

    def to_device(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        return {k: v.to(self.device) if isinstance(v, Tensor) else v for k, v in batch.items()}

    def _autocast(self):
        return torch.autocast(self.device.type, dtype=self.amp, enabled=self.amp is not None)

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        getattr(self.train_loader.dataset, "set_epoch", lambda _: None)(epoch)
        accumulation = max(int(self.cfg["accum"]), 1)
        self.optimizer.zero_grad(set_to_none=True)
        total_loss, total_dice, steps = 0.0, 0.0, 0
        for index, raw in enumerate(self.train_loader):
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
                total_dice += float(
                    dice_iou(torch.sigmoid(prediction.logits.float()), prediction.target)[0].mean()
                )
            total_loss += float(loss.detach())
            steps += 1
        if steps % accumulation:  # flush a partial window rather than dropping it
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
        return {"loss": total_loss / max(steps, 1), "dice": total_dice / max(steps, 1)}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader | None = None, *, with_hausdorff: bool | None = None) -> dict[str, Any]:
        loader = loader or self.val_loader
        if loader is None:
            return {}
        if with_hausdorff is None:
            with_hausdorff = self.wants_hausdorff
        self.model.eval()
        if hasattr(self.task, "anchor_scores"):
            self.task.anchor_scores.clear()
        metrics, total_loss, steps = Metrics(), 0.0, 0
        percentile = float(self.evaluation.get("hausdorff_percentile", 95.0))
        for raw in loader:
            batch = self.to_device(raw)
            with self._autocast():
                prediction = self.task(batch)
            total_loss += float(self.task.loss(prediction))
            steps += 1
            metrics.update(
                prediction.logits, prediction.target, prediction.groups,
                strata=prediction.strata, threshold=float(self.cfg.get("threshold", 0.5)),
                spacing=self.spacing if with_hausdorff else None, percentile=percentile,
            )
        summary = metrics.summary(self.evaluation.get("stratify_by"))
        summary["loss"] = total_loss / max(steps, 1)
        scores = getattr(self.task, "anchor_scores", [])
        if scores:
            summary["anchor_dice"] = sum(scores) / len(scores)
        return summary

    def fit(self) -> list[dict[str, Any]]:
        seed_all(int(self.cfg["seed"]))
        parameters = sum(p.numel() for p in self.model.parameters()) / 1e6
        if self.verbose:
            print(f"{self.task.name}: {parameters:.2f}M parameters on {self.device} ({self.cfg['precision']})")
        log = (self.out_dir / "metrics.jsonl").open("w", encoding="utf-8")
        run = _logger(self.logging, self.out_dir.name, {**self.cfg, **self.stage})
        try:
            for epoch in range(self.epochs):
                started = time.perf_counter()
                train = self.train_epoch(epoch)
                self.schedule.step()
                val = self.evaluate()
                record = {
                    "epoch": epoch,
                    "train": train,
                    "val": {k: v for k, v in val.items() if k not in ("by_name", "strata")},
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "seconds": round(time.perf_counter() - started, 2),
                }
                self.history.append({**record, "val_full": val})
                log.write(json.dumps(record) + "\n")
                log.flush()
                if run is not None:
                    run.log(_flatten(record), step=epoch)
                if self.verbose:
                    print(
                        f"  epoch {epoch:>3}  loss {train['loss']:.4f}  train dice {train['dice']:.4f}"
                        f"  val dice {val.get('dice', 0.0):.4f}  ({record['seconds']}s)"
                    )
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

    def _meta(self, epoch: int, metrics: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "stage": self.task.name,
            "epoch": epoch,
            "best_dice": self.best,
            "metrics": {k: v for k, v in metrics.items() if k != "strata"},
            "config": {**self.cfg, "stage": self.stage, "evaluation": self.evaluation},
        }


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
