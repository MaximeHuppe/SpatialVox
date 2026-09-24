"""The counterfactual probes: that each one perturbs exactly what it claims to."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

import numpy as np

from src.data import ExampleDataset, collate, loader
from src.engine import StageBTask
from src.geometry import DIRECTIONS, OPPOSITE, centroids_world, solutions_for, volume_center_world
from src.models import StageA, StageB

_spec = importlib.util.spec_from_file_location(
    "evaluate_script", Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"
)
evaluate = importlib.util.module_from_spec(_spec)
sys.modules["evaluate_script"] = evaluate
_spec.loader.exec_module(evaluate)


@pytest.fixture
def batch(corpus):
    dataset = ExampleDataset(corpus, "train", normalize_mode="none")
    return collate([dataset[0], dataset[1]])


def tiny_model(corpus) -> StageB:
    torch.manual_seed(0)
    resolution = min(corpus.shape)
    segmenter = StageA(len(corpus.vocab), resolution, encoder_channels=(8, 16, 32),
                       token_dim=32, num_heads=2, bottleneck=resolution // 4,
                       deep_supervision=(0.3, 0.7))
    model = StageB.from_segmenter(
        segmenter, spacing=corpus.spacing, n_anchors=corpus.n_anchors,
        boundary_widths=(4, 8), carver_width=4, carver_blocks=1,
    ).eval()
    # ``coarse`` is zero-initialised, so a geometry probe cannot move the logits
    # until it has a weight. ``refine`` stays zero: these probes do not touch ``B``.
    torch.nn.init.normal_(model.carver.coarse.weight, std=0.05)
    return model


def clauses(batch):
    """The ``(anchor name, direction)`` pairs a batch asks for, per sample.

    ``name_ids`` is what Stage A segments, so it *is* the channel selector -
    there is no separate mask tensor to move independently of it. A probe
    therefore breaks the correspondence by moving the names against the words,
    or the words against the names.
    """
    names = batch.get("name_ids", batch["anchors"] - 1)
    return [
        list(zip(names[i].tolist(), batch["direction_ids"][i].tolist()))
        for i in range(names.shape[0])
    ]


def test_a_normal_batch_pairs_every_name_with_its_own_clause(batch):
    names = batch.get("name_ids", batch["anchors"] - 1)
    assert torch.equal(names, batch["anchors"] - 1)
    assert all(len(set(sample)) == len(sample) for sample in clauses(batch))


def test_permute_channels_moves_the_names_and_leaves_the_words(batch):
    """Channel ``i`` stops being the structure clause ``i`` names."""
    probed = evaluate.counterfactual(batch, "permute_channels")
    assert torch.equal(probed["name_ids"], (batch["anchors"] - 1).roll(1, dims=1))
    assert torch.equal(probed["anchors"], batch["anchors"].roll(1, dims=1))
    assert torch.equal(probed["direction_ids"], batch["direction_ids"])
    for before, after in zip(clauses(batch), clauses(probed)):
        assert before != after


def test_permute_clauses_moves_the_words_and_leaves_the_names(batch):
    probed = evaluate.counterfactual(batch, "permute_clauses")
    assert torch.equal(probed["anchors"], batch["anchors"])
    assert torch.equal(probed["direction_ids"], batch["direction_ids"].roll(1, dims=1))
    for before, after in zip(clauses(batch), clauses(probed)):
        assert before != after


def test_flip_direction_replaces_one_clause_with_its_opposite(batch):
    probed = evaluate.counterfactual(batch, "flip_direction")
    before = batch["direction_ids"][:, 0].tolist()
    after = probed["direction_ids"][:, 0].tolist()
    assert [DIRECTIONS[i] for i in after] == [OPPOSITE[DIRECTIONS[i]] for i in before]
    assert torch.equal(probed["direction_ids"][:, 1:], batch["direction_ids"][:, 1:])
    assert torch.equal(probed["anchors"], batch["anchors"])


def test_permute_both_is_a_control_that_preserves_every_relation(batch):
    probed = evaluate.counterfactual(batch, "permute_both")
    # The same (name, direction) clauses, only in different slots.
    for before, after in zip(clauses(batch), clauses(probed)):
        assert sorted(before) == sorted(after)
        assert before != after  # they really did move


def test_every_probe_actually_reaches_the_model(corpus, batch):
    """Each perturbation has to change the logits, or it is testing nothing."""
    task = StageBTask(tiny_model(corpus), corpus.vocab, spacing=corpus.spacing,
                      anchor_source="oracle")
    with torch.no_grad():
        baseline = task(batch).logits
        for kind in ("permute_channels", "permute_clauses", "flip_direction", "permute_both"):
            perturbed = task(evaluate.counterfactual(batch, kind)).logits
            assert not torch.allclose(perturbed, baseline), kind


def test_the_empty_prompt_population_really_names_nothing(corpus):
    """§7: the false-positive volume is only meaningful if the prompts are empty."""
    rows = corpus.records("train", targets=corpus.vocab.names)
    impossible = evaluate.empty_prompts(corpus, rows, limit=40)
    assert impossible
    from src.data import load_nifti

    for row in impossible:
        labels = load_nifti(corpus.root / "scenes" / row["scene"] / "labels.nii.gz", np.int16)
        centroids = centroids_world(labels, len(corpus.vocab), corpus.spacing)
        center = volume_center_world(labels.shape, corpus.spacing)
        present = [int(v) for v in np.unique(labels) if v != 0]
        assert solutions_for(row["anchors"], row["directions"], centroids, present, center) == []
        assert len(set(row["directions"])) == len(row["directions"])


def test_the_report_runs_end_to_end(corpus, tmp_path):
    """One pass over a split produces the gate, the probes and the two swap tests."""
    task = StageBTask(tiny_model(corpus), corpus.vocab, spacing=corpus.spacing,
                      anchor_source="oracle")
    dataset = ExampleDataset(corpus, "val", limit=6, normalize_mode="none")
    batches = list(loader(dataset, batch_size=2, shuffle=False))
    evaluation = {"metrics": ["dice", "iou"], "stratify_by": ["target"],
                  "counterfactuals": ["permute_channels", "permute_both"]}
    summary, probes, rows = evaluate.score(
        task, batches, "cpu", corpus.spacing, evaluation, save_to=tmp_path / "masks"
    )
    # Named after the anchor source: `scripts/gate_mapper.py` measures the same
    # quantity from ground-truth centroids, and the two must not share a name.
    assert "gate_fraction" not in summary
    assert 0.0 <= summary["gate_fraction_oracle"] <= 1.0
    assert set(probes) == {"permute_channels", "permute_both"}
    assert len(rows) == len(dataset)
    assert {"centroid_error", "where_at_centroid", "valid_logit"} <= set(rows[0])
    assert (tmp_path / "masks" / rows[0]["id"] / "prediction.nii.gz").is_file()
    # §7: every Dice is also reported through the null head's gate, which can
    # only remove masks.
    assert summary["dice_null_gated"] <= summary["dice"] + 1e-6
    assert summary["empty_prediction_rate_null_gated"] >= summary["empty_prediction_rate"] - 1e-6

    empty = evaluate.empty_prompt_report(task, batches, "cpu")
    assert set(empty) == {"invalid_called_invalid", "false_positive_voxels", "false_positive_rate",
                          "false_positive_voxels_null_gated", "false_positive_rate_null_gated"}
    assert empty["false_positive_rate_null_gated"] <= empty["false_positive_rate"]
    # A shuffled loader, so the neighbouring row is usually a different scene;
    # same-scene pairs are skipped rather than compared with themselves.
    shuffled = list(loader(ExampleDataset(corpus, "val", normalize_mode="none"),
                           batch_size=4, shuffle=True, seed=0))
    swap = evaluate.image_replacement(task, shuffled, "cpu")
    assert swap["n"] > 0
    assert set(swap) == {"n", "same_subject_pairs_skipped", "dice", "dice_swapped",
                         "centroid_error", "centroid_error_swapped"}


def test_an_unknown_probe_is_rejected(batch):
    with pytest.raises(ValueError, match="unknown counterfactual"):
        evaluate.counterfactual(batch, "shuffle_everything")


def test_the_shortcut_ceiling_is_computed_on_the_population_it_is_printed_beside(corpus):
    """A two-class population has a near-perfect ceiling, and must say so."""
    everything = corpus.records("val", targets=corpus.vocab.names)
    two = corpus.records("val", targets=list(corpus.meta["targets"]["test"]))
    assert evaluate.anchor_set_ceiling(two) > evaluate.anchor_set_ceiling(everything)
    assert evaluate.anchor_set_ceiling([{"anchors": [1, 2, 3], "target": 4}] * 5) == 1.0
