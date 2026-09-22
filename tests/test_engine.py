"""The losses of §5, the ``keep`` weight that implements "dropped", and the loop."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from src.data import ExampleDataset, collate, loader
from src.engine import (
    BoundaryTask, Metrics, Prediction, StageATask, StageBTask, Trainer,
    build_optimizer, build_scheduler, dice_iou, dilate, far_mass, hausdorff,
    label_boundary, load_model, mask_centroid_world, masks_from, null_summary,
    roll_anchors, save_checkpoint, segmentation_loss, weighted_mean,
)
from src.models import BoundaryPretrainer, StageA, StageB

SPACING = (1.0, 1.0, 1.0)
LOSS = {"dice": 1.0, "bce": 1.0, "null_bce": 0.2, "centroid": 0.1, "field_centroid": 0.05, "far": 0.2}


def tiny_stage_a(vocab_size: int, resolution: int) -> StageA:
    torch.manual_seed(0)
    return StageA(vocab_size, resolution, encoder_channels=(8, 16, 32), token_dim=32,
                  num_heads=2, bottleneck=resolution // 4, prior_foreground=0.01,
                  deep_supervision=(0.3, 0.7))


def tiny_stage_b(corpus) -> StageB:
    torch.manual_seed(0)
    resolution = min(corpus.shape)
    return StageB.from_segmenter(
        tiny_stage_a(len(corpus.vocab), resolution),
        spacing=corpus.spacing, n_anchors=corpus.n_anchors, tau=0.5, min_mass=1e-6,
        boundary_widths=(4, 8), carver_width=4, carver_blocks=1, prior_foreground=0.01,
    )


# ---------------------------------------------------------------------------
# "dropped" is a weight, and it has to reach every term
# ---------------------------------------------------------------------------
def test_a_dropped_sample_changes_no_loss_term():
    """§5: a flip that names two structures is dropped - from *every* column."""
    torch.manual_seed(0)
    logits, target = torch.randn(3, 1, 4, 4, 4), torch.zeros(3, 1, 4, 4, 4)
    target[:, :, 1:3, 1:3, 1:3] = 1.0
    keep = torch.tensor([1.0, 1.0, 0.0])

    kept = segmentation_loss(logits[:2], target[:2], weight=torch.ones(2))
    with_dropped = segmentation_loss(logits, target, weight=keep)
    assert float(with_dropped) == pytest.approx(float(kept), abs=1e-6)

    # ...and moving the dropped sample's logits cannot move the loss at all.
    moved = logits.clone()
    moved[2] += 50.0
    assert float(segmentation_loss(moved, target, weight=keep)) == pytest.approx(
        float(with_dropped), abs=1e-6
    )


def test_an_all_dropped_batch_is_zero_and_not_nan():
    values = torch.tensor([1.0, 2.0, 3.0])
    assert float(weighted_mean(values, torch.zeros(3))) == 0.0
    assert float(weighted_mean(values, None)) == pytest.approx(2.0)


def test_metrics_skip_a_dropped_sample():
    metrics = Metrics()
    logits = torch.full((2, 1, 4, 4, 4), 10.0)
    target = torch.zeros(2, 1, 4, 4, 4)
    target[0] = 1.0  # sample 0 perfect, sample 1 would score 0
    metrics.update(logits, target, [["a"], ["b"]], keep=torch.tensor([1.0, 0.0]))
    summary = metrics.summary()
    assert summary["n"] == 1 and summary["dice"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# The individual terms
# ---------------------------------------------------------------------------
def test_dilate_is_the_l_infinity_ball():
    volume = torch.zeros(1, 1, 9, 9, 9)
    volume[0, 0, 4, 4, 4] = 1.0
    grown = dilate(volume, 2)
    assert float(grown.sum()) == 5**3
    assert float(grown[0, 0, 2, 2, 2]) == 1.0  # a corner of the cube
    assert float(grown[0, 0, 1, 4, 4]) == 0.0


def test_l_far_ignores_mass_inside_the_dilated_field_and_counts_it_outside():
    """§5: "``L_far`` is the only spatial penalty on a valid prompt"."""
    where = torch.zeros(1, 1, 16, 16, 16)
    where[0, 0, 7:9, 7:9, 7:9] = 1.0
    inside = torch.zeros(1, 1, 16, 16, 16)
    inside[0, 0, 5:11, 5:11, 5:11] = 1.0  # entirely within the 8-voxel dilation
    assert float(far_mass(inside, where, 0.05, 8)) == 0.0

    outside = torch.zeros(1, 1, 16, 16, 16)
    outside[0, 0, 0, 0, 0] = 1.0
    assert float(far_mass(outside, where, 0.05, 2)) > 0.0


def test_the_mask_centroid_is_in_world_units():
    mask = torch.zeros(1, 1, 8, 8, 8)
    mask[0, 0, 2, 4, 6] = 1.0
    centroid, total = mask_centroid_world(mask, (1.25, 1.25, 1.25))
    assert torch.allclose(centroid, torch.tensor([[7.5, 5.0, 2.5]]))  # world (x, y, z)
    assert float(total) == 1.0


def test_the_boundary_target_marks_label_changes_and_carries_no_class():
    """§4: "a voxel is positive where two neighbouring voxels differ in label"."""
    labels = torch.zeros(1, 6, 6, 6, dtype=torch.long)
    labels[0, 2:4, 2:4, 2:4] = 3
    boundary = label_boundary(labels)
    assert float(boundary[0, 0, 2, 2, 2]) == 1.0  # a corner of the box
    assert float(boundary[0, 0, 0, 0, 0]) == 0.0  # deep in the background
    # Relabelling the same shape gives the same target: no class information.
    assert torch.equal(boundary, label_boundary(torch.where(labels > 0, 5, 0)))


def test_the_null_summary_reports_an_auc():
    scores = [(2.0, 1.0), (1.0, 1.0), (-1.0, 0.0), (-2.0, 0.0)]
    summary = null_summary(scores)
    assert summary["null_auc"] == pytest.approx(1.0)
    assert summary["null_accuracy"] == pytest.approx(1.0)
    assert summary["null_rate"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# The relational task
# ---------------------------------------------------------------------------
def batch_from(corpus, split="train", n=2, **kwargs):
    dataset = ExampleDataset(corpus, split, normalize_mode="none", **kwargs)
    return collate([dataset[i] for i in range(n)])


def test_an_invalid_prompt_gets_the_empty_mask_not_the_background(corpus):
    """§5: "names none -> the empty mask". Label 0 is the background, not nothing."""
    model = tiny_stage_b(corpus)
    task = StageBTask(model, corpus.vocab, spacing=corpus.spacing, loss_weights=LOSS)
    batch = batch_from(corpus, n=2)
    batch["valid"] = torch.tensor([1, 0])
    batch["target"] = torch.stack([batch["target"][0], torch.tensor(0)])
    prediction = task(batch)
    assert float(prediction.target[1].sum()) == 0.0
    assert float(prediction.target[0].sum()) > 0.0


def test_the_task_never_hands_the_model_anything_from_the_label_volume(corpus, monkeypatch):
    """The labels are read to build a target and to score, never as an input."""
    model = tiny_stage_b(corpus)
    seen = {}
    original = StageB.forward

    def spy(self, image, direction_ids, name_ids, **kwargs):
        seen.update(kwargs, image=image)
        return original(self, image, direction_ids, name_ids, **kwargs)

    monkeypatch.setattr(StageB, "forward", spy)
    task = StageBTask(model, corpus.vocab, spacing=corpus.spacing, loss_weights=LOSS)
    batch = batch_from(corpus)
    task(batch)
    assert set(seen) == {"image", "anchors", "boundary_image"}
    assert seen["anchors"] is None and seen["boundary_image"] is None
    assert torch.equal(seen["image"], batch["image"])


def test_oracle_anchors_are_the_ground_truth_masks(corpus):
    """The diagnostic of §2's gate: the same forward, with Stage A's error removed."""
    model = tiny_stage_b(corpus)
    batch = batch_from(corpus)
    predicted = StageBTask(model, corpus.vocab, spacing=corpus.spacing, loss_weights=LOSS)(batch)
    oracle = StageBTask(model, corpus.vocab, spacing=corpus.spacing, loss_weights=LOSS,
                        anchor_source="oracle")(batch)
    assert float(oracle.anchor_dice.mean()) == pytest.approx(1.0)
    assert float(predicted.anchor_dice.mean()) < 1.0


def test_every_loss_term_is_finite_and_the_weights_matter(corpus):
    model = tiny_stage_b(corpus)
    batch = batch_from(corpus, n=3)
    batch["valid"] = torch.tensor([1, 0, 1])
    batch["keep"] = torch.tensor([1, 1, 0])
    task = StageBTask(model, corpus.vocab, spacing=corpus.spacing, loss_weights=LOSS)
    prediction = task(batch)
    total = task.loss(prediction)
    assert torch.isfinite(total)
    task.loss_weights = {**LOSS, "far": 0.0}
    assert task.loss(prediction).item() != pytest.approx(total.item(), abs=1e-6)


def test_the_field_heatmap_target_can_be_restricted_to_empty_prompts(corpus):
    """§5's table puts "peak of `where_raw`" in the *names one* column too.

    Measured, the field's centre of mass is 20.2 mm from the target's centroid on
    this corpus, so under `always` that term pulls against the structure-centroid
    term on every valid prompt. `empty-only` is the other reading, and it has to
    be a setting rather than an opinion.
    """
    model = tiny_stage_b(corpus)
    batch = batch_from(corpus, n=2)
    batch["valid"] = torch.tensor([1, 0])
    batch["target"] = torch.stack([batch["target"][0], torch.tensor(0)])
    always = StageBTask(model, corpus.vocab, spacing=corpus.spacing, loss_weights=LOSS)
    empty_only = StageBTask(model, corpus.vocab, spacing=corpus.spacing, loss_weights=LOSS,
                            field_centroid_on="empty-only")
    always.loss(always(batch))
    empty_only.loss(empty_only(batch))
    assert empty_only.components["field_centroid"] != always.components["field_centroid"]
    for key in ("mask", "null_bce", "centroid", "far"):
        assert empty_only.components[key] == pytest.approx(always.components[key], rel=1e-5)
    with pytest.raises(ValueError, match="field_centroid_on"):
        StageBTask(model, corpus.vocab, field_centroid_on="sometimes")


def test_rolling_the_anchor_slots_moves_ids_names_and_cached_masks_together():
    """A counterfactual that rolls only the ids is not the one it claims to be."""
    batch = {
        "anchors": torch.tensor([[1, 2, 3]]),
        "name_ids": torch.tensor([[0, 1, 2]]),
        "anchor_probability": torch.arange(3).reshape(1, 3, 1, 1, 1).float(),
    }
    rolled = roll_anchors(batch, 1)
    assert rolled["anchors"].tolist() == [[3, 1, 2]]
    assert rolled["name_ids"].tolist() == [[2, 0, 1]]
    assert rolled["anchor_probability"].flatten().tolist() == [2.0, 0.0, 1.0]
    assert roll_anchors({"anchors": torch.tensor([[1, 2, 3]])}, 1)["name_ids"].tolist() == [[2, 0, 1]]


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------
def stage_b_trainer(corpus, tmp_path, epochs=2, **kwargs):
    model = tiny_stage_b(corpus)
    task = StageBTask(model, corpus.vocab, spacing=corpus.spacing, loss_weights=LOSS, **kwargs)
    train = ExampleDataset(corpus, "train", flip_probability=0.5, limit=12, normalize_mode="none")
    val = ExampleDataset(corpus, "val", targets=list(corpus.meta["targets"]["train"]),
                         limit=8, normalize_mode="none")
    return Trainer(
        task,
        loader(train, batch_size=2, shuffle=True, seed=0),
        loader(val, batch_size=2, shuffle=False),
        {"seed": 0, "device": "cpu", "precision": "fp32", "batch_size": 2, "accum": 1,
         "threshold": 0.5, "early_stopping": {"patience": 0}},
        tmp_path / "run",
        stage={"epochs": epochs, "optimizer": {"name": "adamw", "lr": 1e-3},
               "scheduler": {"name": "cosine", "warmup_epochs": 0},
               "flip_probability": 0.5, "loss": LOSS},
        evaluation={"metrics": ["dice", "iou"], "stratify_by": ["target"]},
        spacing=corpus.spacing, verbose=False,
    )


def test_the_loop_trains_stage_b_and_writes_a_reloadable_checkpoint(corpus, tmp_path):
    trainer = stage_b_trainer(corpus, tmp_path)
    history = trainer.fit()
    assert len(history) == 2
    assert torch.isfinite(torch.tensor(history[-1]["train"]["loss"]))
    for key in ("dice", "centroid_error", "anchor_dice", "null_accuracy", "null_rate"):
        assert key in history[-1]["val"], key
    # No `null_auc`: validation never flips, so it holds no invalid prompts and
    # an AUC has nothing to separate. That is the point of keeping flips out of
    # the selection curve - `scripts/evaluate.py` builds the negatives instead.
    assert "null_auc" not in history[-1]["val"]

    restored = load_model(tmp_path / "run" / "best.pt")
    assert isinstance(restored, StageB)
    meta = json.loads((tmp_path / "run" / "best.json").read_text())["meta"]
    # §8: the schedule that produced the weights is recorded beside them.
    assert meta["config"]["stage"]["flip_probability"] == 0.5
    assert meta["config"]["stage"]["loss"] == LOSS


def test_the_optimiser_never_receives_the_frozen_segmenter(corpus):
    model = tiny_stage_b(corpus)
    optimizer = build_optimizer(model, {"name": "adamw", "lr": 1e-3})
    held = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert held.isdisjoint({id(p) for p in model.segmenter.parameters()})
    assert held == {id(p) for p in model.trainable_parameters()}


def test_the_probes_report_all_four_counterfactuals(corpus, tmp_path):
    """§7: ``permute_channels``, ``permute_clauses`` and ``flip_direction`` must drop."""
    trainer = stage_b_trainer(corpus, tmp_path, epochs=1)
    probes = trainer.prompt_dependence()
    assert set(probes) == {
        "flip_direction_drop", "permute_channels_drop",
        "permute_clauses_drop", "permute_both_drop",
    }
    assert all(np.isfinite(v) for v in probes.values())


def test_permute_both_cannot_move_the_field_at_all(corpus):
    """``where_raw`` is a product, so it is *exactly* permutation-invariant.

    Worth pinning because it changes what a flat ``permute_both`` means: under
    this architecture only the carver's concatenation is order-dependent.
    """
    model = tiny_stage_b(corpus)
    task = StageBTask(model, corpus.vocab, spacing=corpus.spacing, loss_weights=LOSS)
    batch = batch_from(corpus)
    base = task(batch)
    rolled = {**roll_anchors(batch, 1), "direction_ids": batch["direction_ids"].roll(1, 1)}
    assert torch.allclose(base.where_raw, task(rolled).where_raw, atol=1e-6)


def test_the_boundary_pretraining_stage_runs(corpus, tmp_path):
    """§4's three class-agnostic objectives, end to end."""
    from src.data import SceneDataset

    torch.manual_seed(0)
    model = BoundaryPretrainer(widths=(4, 8), mask_fraction=0.5, patch=8)
    task = BoundaryTask(model, corpus.vocab, loss_weights={"boundary": 1.0, "reconstruct": 1.0, "edge": 0.5})
    scenes = SceneDataset(corpus, "train", prompts_per_item=1, normalize_mode="none")
    trainer = Trainer(
        task, loader(scenes, batch_size=1, shuffle=True, seed=0),
        loader(SceneDataset(corpus, "val", prompts_per_item=1, normalize_mode="none"),
               batch_size=1, shuffle=False),
        {"seed": 0, "device": "cpu", "precision": "fp32", "batch_size": 1, "accum": 1,
         "threshold": 0.5, "early_stopping": {"patience": 0}},
        tmp_path / "boundary",
        stage={"epochs": 1, "optimizer": {"name": "adamw", "lr": 1e-3},
               "scheduler": {"name": "constant"}},
        evaluation={"metrics": ["dice"]}, verbose=False,
    )
    history = trainer.fit()
    assert torch.isfinite(torch.tensor(history[-1]["train"]["loss"]))
    restored = load_model(tmp_path / "boundary" / "best.pt")
    assert isinstance(restored, BoundaryPretrainer)


def test_stage_a_still_trains(corpus, tmp_path):
    from src.data import SceneDataset

    model = tiny_stage_a(len(corpus.vocab), min(corpus.shape))
    task = StageATask(model, corpus.vocab, loss_weights={"lambda_dice": 1.0, "lambda_bce": 1.0})
    trainer = Trainer(
        task,
        loader(SceneDataset(corpus, "train", normalize_mode="none"), batch_size=1, shuffle=True, seed=0),
        loader(SceneDataset(corpus, "val", normalize_mode="none"), batch_size=1, shuffle=False),
        {"seed": 0, "device": "cpu", "precision": "fp32", "batch_size": 1, "accum": 1,
         "threshold": 0.5, "early_stopping": {"patience": 0}},
        tmp_path / "a",
        stage={"epochs": 1, "optimizer": {"name": "adamw", "lr": 1e-3},
               "scheduler": {"name": "cosine", "warmup_epochs": 0}},
        evaluation={"metrics": ["dice"]}, verbose=False,
    )
    assert torch.isfinite(torch.tensor(trainer.fit()[-1]["train"]["loss"]))


# ---------------------------------------------------------------------------
# Metric conventions that a reported number depends on
# ---------------------------------------------------------------------------
def test_two_empty_masks_score_one_and_one_empty_against_one_full_scores_zero():
    empty, full = torch.zeros(1, 1, 4, 4, 4), torch.ones(1, 1, 4, 4, 4)
    assert float(dice_iou(empty, empty)[0]) == 1.0
    assert float(dice_iou(empty, full)[0]) == 0.0
    assert hausdorff(empty[0, 0], empty[0, 0]) == 0.0
    assert np.isnan(hausdorff(empty[0, 0], full[0, 0]))


def test_the_scheduler_warms_up_then_decays():
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    schedule = build_scheduler(optimizer, {"name": "cosine", "warmup_epochs": 2}, epochs=10)
    seen = []
    for _ in range(10):
        seen.append(optimizer.param_groups[0]["lr"])
        schedule.step()
    assert seen[0] < seen[1] <= seen[2]
    assert seen[-1] < seen[2]
