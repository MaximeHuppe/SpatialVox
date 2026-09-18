"""Losses, metrics, and that both stages actually train."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.data import ExampleDataset, SceneDataset, collate, loader
from src.engine import (
    Metrics,
    StageATask,
    StageBTask,
    Trainer,
    dice_iou,
    hausdorff,
    load_model,
    masks_from,
    occupancy_from,
    resolve_phase_a_checkpoint,
    resolve_stage_b_mode,
    segmentation_loss,
)
from src.models import StageA, StageB

SMALL = dict(encoder_channels=(4, 8, 16, 16), token_dim=16, num_heads=2)
CFG = {
    "seed": 0, "device": "cpu", "precision": "fp32", "batch_size": 2, "accum": 1,
    "workers": 0, "threshold": 0.5, "early_stopping": {"patience": 0, "min_delta": 0.0},
}
STAGE = {
    "epochs": 2,
    "optimizer": {"name": "adamw", "lr": 3e-3, "weight_decay": 0.0},
    "scheduler": {"name": "cosine", "warmup_epochs": 0},
}


# -- losses and metrics ------------------------------------------------------
def test_a_perfect_prediction_scores_one():
    target = torch.zeros(1, 1, 4, 4, 4)
    target[0, 0, 1:3, 1:3, 1:3] = 1
    dice, iou = dice_iou(target, target)
    assert float(dice) == 1.0 and float(iou) == 1.0


def test_two_empty_masks_agree_and_one_empty_mask_does_not():
    empty, full = torch.zeros(1, 1, 4, 4, 4), torch.ones(1, 1, 4, 4, 4)
    assert float(dice_iou(empty, empty)[0]) == 1.0
    assert float(dice_iou(empty, full)[0]) == 0.0


def test_the_loss_falls_as_the_prediction_improves():
    target = torch.zeros(2, 1, 8, 8, 8)
    target[:, :, 2:6, 2:6, 2:6] = 1
    good = torch.where(target > 0, 5.0, -5.0)
    assert float(segmentation_loss(good, target)) < float(segmentation_loss(-good, target))


def test_the_loss_is_computed_in_float32_under_autocast():
    """The Dice denominator sums one probability per voxel; fp16 cannot hold it."""
    target = torch.zeros(1, 1, 8, 8, 8)
    target[0, 0, 2:6, 2:6, 2:6] = 1
    logits = torch.randn(1, 1, 8, 8, 8)
    assert segmentation_loss(logits.half(), target).dtype == torch.float32


def test_hausdorff_is_a_distance_in_world_units():
    a = torch.zeros(8, 8, 8)
    a[2:5, 2:5, 2:5] = 1
    assert hausdorff(a, a) == 0.0
    shifted = torch.zeros(8, 8, 8)
    shifted[3:6, 2:5, 2:5] = 1
    assert 0 < hausdorff(a, shifted) <= 2.0
    assert hausdorff(a, torch.zeros(8, 8, 8)) != hausdorff(a, torch.zeros(8, 8, 8))  # nan


def test_metrics_report_overall_per_name_and_per_stratum():
    metrics = Metrics()
    target = torch.zeros(2, 1, 4, 4, 4)
    target[:, :, 1:3, 1:3, 1:3] = 1
    metrics.update(
        torch.where(target > 0, 5.0, -5.0), target, [["cube"], ["torus"]],
        strata=[{"direction": ["superior"]}, {"direction": ["medial"]}],
    )
    summary = metrics.summary()
    assert summary["n"] == 2 and summary["dice"] == pytest.approx(1.0)
    assert set(summary["by_name"]) == {"cube", "torus"}
    assert set(summary["strata"]["direction"]) == {"superior", "medial"}


def test_masks_from_labels_matches_an_explicit_comparison():
    labels = torch.randint(0, 5, (2, 4, 4, 4))
    ids = torch.tensor([[1, 3], [2, 4]])
    masks = masks_from(labels, ids)
    assert masks.shape == (2, 2, 4, 4, 4)
    assert torch.equal(masks[0, 0], (labels[0] == 1).float())
    assert torch.equal(masks[1, 1], (labels[1] == 4).float())


def _stage_b_batch(corpus):
    return collate([ExampleDataset(corpus, "train")[0]])


def test_occupancy_all_includes_the_target(corpus):
    """The ceiling: occupancy is labels > 0, target included."""
    batch = _stage_b_batch(corpus)
    occupancy = occupancy_from(batch, "all")
    target = masks_from(batch["labels"], batch["target"].unsqueeze(1))
    assert occupancy.shape == (1, 1, *batch["labels"].shape[1:])
    assert torch.equal(occupancy, (batch["labels"] > 0).float().unsqueeze(1))
    assert (occupancy * target).sum() == target.sum()
    assert occupancy.sum() > target.sum()


def test_occupancy_anchors_only_is_the_union_of_the_anchors(corpus):
    batch = _stage_b_batch(corpus)
    occupancy = occupancy_from(batch, "anchors-only")
    anchors = masks_from(batch["labels"], batch["anchors"])
    target = masks_from(batch["labels"], batch["target"].unsqueeze(1))
    assert torch.equal(occupancy, anchors.amax(1, keepdim=True))
    assert not (occupancy * target).any()
    fake = torch.ones_like(anchors)
    assert torch.equal(occupancy_from(batch, "anchors-only", anchors=fake), fake.amax(1, keepdim=True))


def test_occupancy_none_is_empty(corpus):
    batch = _stage_b_batch(corpus)
    occupancy = occupancy_from(batch, "none")
    assert occupancy.shape == (1, 1, *batch["labels"].shape[1:])
    assert occupancy.dtype == torch.float32
    assert not occupancy.any()


def test_occupancy_mode_is_validated(corpus):
    with pytest.raises(ValueError, match="occupancy_mode"):
        occupancy_from(_stage_b_batch(corpus), "named_union")
    with pytest.raises(ValueError, match="occupancy_mode"):
        StageBTask(
            StageB(len(corpus.vocab), min(corpus.shape), corpus.n_anchors, **SMALL),
            corpus.vocab, mode="oracle", occupancy_mode="predicted",
        )


def test_predicted_mode_queries_stage_a_once_and_ignores_labels(corpus):
    """One forward: occupancy_mode=all queries the vocab; anchors-only queries three names."""
    batch = _stage_b_batch(corpus)
    model = StageB(len(corpus.vocab), min(corpus.shape), corpus.n_anchors, **SMALL)
    segmenter = StageA(len(corpus.vocab), min(corpus.shape), **SMALL).eval()
    calls: list[int] = []

    def fake_masks(image, name_ids, threshold=0.5):
        calls.append(int(name_ids.shape[1]))
        return torch.ones(image.shape[0], name_ids.shape[1], *image.shape[2:], device=image.device)

    segmenter.masks_for = fake_masks
    poisoned = dict(batch)
    poisoned["labels"] = torch.zeros_like(batch["labels"])

    StageBTask(model, corpus.vocab, segmenter=segmenter, occupancy_mode="all")(poisoned)
    assert calls == [len(corpus.vocab)]
    StageBTask(model, corpus.vocab, segmenter=segmenter, occupancy_mode="anchors-only")(poisoned)
    assert calls[-1] == corpus.n_anchors


def test_predicted_all_from_a_perfect_segmenter_still_contains_the_target(corpus, monkeypatch):
    """occupancy_mode=all is the ceiling: a segmenter that knows the target will paint it."""
    batch = _stage_b_batch(corpus)
    model = StageB(len(corpus.vocab), min(corpus.shape), corpus.n_anchors, **SMALL)
    segmenter = StageA(len(corpus.vocab), min(corpus.shape), **SMALL).eval()
    segmenter.masks_for = lambda image, name_ids, threshold=0.5: masks_from(batch["labels"], name_ids + 1)
    seen: dict[str, torch.Tensor] = {}
    decode = model.decoder.forward
    monkeypatch.setattr(
        model.decoder,
        "forward",
        lambda features, *, context=None, occupancy=None: (
            seen.update(occupancy=occupancy) or decode(features, context=context, occupancy=occupancy)
        ),
    )
    StageBTask(model, corpus.vocab, segmenter=segmenter, occupancy_mode="all")(batch)
    target = masks_from(batch["labels"], batch["target"].unsqueeze(1))
    assert (seen["occupancy"] * target).sum() == target.sum()


def test_stage_b_task_rejects_a_mismatched_source(corpus):
    model = StageB(len(corpus.vocab), min(corpus.shape), corpus.n_anchors, **SMALL)
    segmenter = StageA(len(corpus.vocab), min(corpus.shape), **SMALL)
    with pytest.raises(ValueError, match="predicted"):
        StageBTask(model, corpus.vocab)
    with pytest.raises(ValueError, match="oracle"):
        StageBTask(model, corpus.vocab, mode="oracle", segmenter=segmenter)
    with pytest.raises(TypeError, match="StageA"):
        StageBTask(model, corpus.vocab, segmenter=model)


def test_resolve_phase_a_checkpoint_requires_a_path_unless_mode_is_oracle():
    with pytest.raises(ValueError, match="predicted"):
        resolve_phase_a_checkpoint({})
    with pytest.raises(ValueError, match="predicted"):
        resolve_phase_a_checkpoint({"mode": "predicted"})
    with pytest.raises(ValueError, match="predicted"):
        resolve_phase_a_checkpoint({"mode": "predicted", "phase_a_checkpoint": None})
    with pytest.raises(ValueError, match="predicted"):
        resolve_phase_a_checkpoint({"phase_a_checkpoint": ""})
    assert resolve_stage_b_mode({}) == "predicted"
    assert resolve_stage_b_mode({"mode": "oracle"}) == "oracle"
    assert resolve_phase_a_checkpoint({"mode": "oracle"}) is None
    assert resolve_phase_a_checkpoint({"mode": "oracle", "phase_a_checkpoint": None}) is None
    assert resolve_phase_a_checkpoint({"mode": "oracle", "phase_a_checkpoint": "runs/a.pt"}) is None
    assert resolve_phase_a_checkpoint({"phase_a_checkpoint": "runs/a.pt"}) == Path("runs/a.pt")
    assert resolve_phase_a_checkpoint(
        {"mode": "predicted", "phase_a_checkpoint": "runs/a.pt"}
    ) == Path("runs/a.pt")
    assert resolve_phase_a_checkpoint(
        {"phase_a_checkpoint": "runs/a.pt"}, override="runs/other.pt"
    ) == Path("runs/other.pt")
    assert resolve_phase_a_checkpoint({}, override="runs/cli.pt") == Path("runs/cli.pt")
    with pytest.raises(ValueError, match="oracle"):
        resolve_phase_a_checkpoint({"mode": "oracle"}, override="runs/cli.pt")
    with pytest.raises(ValueError, match="mode"):
        resolve_phase_a_checkpoint({"mode": "gt", "phase_a_checkpoint": "runs/a.pt"})


# -- training ----------------------------------------------------------------
def _trainer(task, dataset, tmp_path, epochs=2, lr=3e-3):
    batches = loader(dataset, batch_size=2, shuffle=False)
    stage = {**STAGE, "epochs": epochs, "optimizer": {**STAGE["optimizer"], "lr": lr}}
    return Trainer(task, batches, batches, CFG, tmp_path, stage=stage, verbose=False)


def test_stage_a_trains_and_writes_a_usable_checkpoint(corpus, tmp_path):
    model = StageA(len(corpus.vocab), min(corpus.shape), **SMALL)
    dataset = SceneDataset(corpus, "train")
    trainer = _trainer(StageATask(model, corpus.vocab), dataset, tmp_path)
    history = trainer.fit()

    assert len(history) == 2
    assert history[-1]["train"]["loss"] < history[0]["train"]["loss"]
    assert (tmp_path / "best.pt").is_file() and (tmp_path / "metrics.jsonl").is_file()
    assert json.loads((tmp_path / "metrics.jsonl").read_text().splitlines()[0])["epoch"] == 0

    restored = load_model(tmp_path / "best.pt")
    masks = restored.masks_for(dataset[0]["image"].unsqueeze(0), torch.tensor([[0, 1, 2]]))
    assert masks.shape == (1, 3, *corpus.shape)


def test_stage_b_trains_on_oracle_anchors(corpus, tmp_path):
    model = StageB(len(corpus.vocab), min(corpus.shape), corpus.n_anchors, **SMALL)
    dataset = ExampleDataset(corpus, "train")
    trainer = _trainer(StageBTask(model, corpus.vocab, mode="oracle"), dataset, tmp_path)
    history = trainer.fit()
    assert history[-1]["train"]["loss"] < history[0]["train"]["loss"]
    assert "by_name" in history[-1]["val_full"]


def test_stage_b_can_take_its_anchors_from_stage_a(corpus, tmp_path):
    """The end-to-end path: same weights, anchors predicted instead of given."""
    segmenter = StageA(len(corpus.vocab), min(corpus.shape), **SMALL).eval()
    model = StageB(len(corpus.vocab), min(corpus.shape), corpus.n_anchors, **SMALL)
    task = StageBTask(model, corpus.vocab, segmenter=segmenter)
    trainer = _trainer(task, ExampleDataset(corpus, "train"), tmp_path, epochs=1)
    summary = trainer.evaluate()
    # An untrained segmenter is a bad one, but the number has to be reported.
    assert "anchor_dice" in summary and 0.0 <= summary["anchor_dice"] <= 1.0


def test_stage_b_overfits_a_single_example(corpus, tmp_path):
    """The bug catcher: if this fails, channels, coordinates or indices are wrong."""
    model = StageB(
        len(corpus.vocab), min(corpus.shape), corpus.n_anchors,
        encoder_channels=(8, 16, 32, 32), token_dim=32, num_heads=2,
    )
    dataset = ExampleDataset(corpus, "train", limit=1)
    batches = loader(dataset, batch_size=1, shuffle=False)
    trainer = Trainer(
        StageBTask(model, corpus.vocab, mode="oracle"), batches, batches, CFG, tmp_path,
        # One example, so one optimiser step per epoch.
        stage={**STAGE, "epochs": 150, "optimizer": {**STAGE["optimizer"], "lr": 8e-3}},
        verbose=False,
    )
    history = trainer.fit()
    assert max(record["train"]["dice"] for record in history) > 0.95


def test_early_stopping_halts_a_run_that_stops_improving(corpus, tmp_path):
    model = StageA(len(corpus.vocab), min(corpus.shape), **SMALL)
    dataset = SceneDataset(corpus, "train")
    batches = loader(dataset, batch_size=2, shuffle=False)
    trainer = Trainer(
        StageATask(model, corpus.vocab), batches, batches,
        {**CFG, "early_stopping": {"patience": 1, "min_delta": 0.0}}, tmp_path,
        stage={**STAGE, "epochs": 20, "optimizer": {**STAGE["optimizer"], "lr": 0.0}},
        verbose=False,
    )
    assert len(trainer.fit()) < 20  # a zero learning rate cannot improve
