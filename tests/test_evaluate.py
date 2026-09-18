"""The counterfactual probes: that each one perturbs exactly what it claims to."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from src.data import ExampleDataset, collate
from src.engine import StageBTask
from src.geometry import DIRECTIONS, OPPOSITE
from src.models import StageB

_spec = importlib.util.spec_from_file_location(
    "evaluate_script", Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"
)
evaluate = importlib.util.module_from_spec(_spec)
sys.modules["evaluate_script"] = evaluate
_spec.loader.exec_module(evaluate)


@pytest.fixture
def batch(corpus):
    dataset = ExampleDataset(corpus, "train")
    return collate([dataset[0], dataset[1]])


def pairs(batch):
    """The ``(channel contents, name asked for, direction asked for)`` triples."""
    names = batch.get("name_ids", batch["anchors"] - 1)
    return [
        list(zip(batch["anchors"][i].tolist(), names[i].tolist(), batch["direction_ids"][i].tolist()))
        for i in range(batch["anchors"].shape[0])
    ]


def test_a_normal_batch_has_every_channel_matched_to_its_own_clause(batch):
    for sample in pairs(batch):
        assert all(anchor - 1 == name for anchor, name, _ in sample)


def test_permute_channels_moves_the_masks_and_leaves_the_prompt(batch):
    probed = evaluate.counterfactual(batch, "permute_channels")
    assert torch.equal(probed["anchors"], batch["anchors"].roll(1, dims=1))
    assert torch.equal(probed["name_ids"], batch["anchors"] - 1)  # the prompt did not move
    assert torch.equal(probed["direction_ids"], batch["direction_ids"])
    for sample in pairs(probed):  # correspondence is now broken
        assert any(anchor - 1 != name for anchor, name, _ in sample)


def test_permute_clauses_moves_the_prompt_and_leaves_the_masks(batch):
    probed = evaluate.counterfactual(batch, "permute_clauses")
    assert torch.equal(probed["anchors"], batch["anchors"])
    assert torch.equal(probed["direction_ids"], batch["direction_ids"].roll(1, dims=1))
    for sample in pairs(probed):
        assert any(anchor - 1 != name for anchor, name, _ in sample)


def test_flip_direction_replaces_one_clause_with_its_opposite(batch):
    probed = evaluate.counterfactual(batch, "flip_direction")
    before = batch["direction_ids"][:, 0].tolist()
    after = probed["direction_ids"][:, 0].tolist()
    assert [DIRECTIONS[i] for i in after] == [OPPOSITE[DIRECTIONS[i]] for i in before]
    assert torch.equal(probed["direction_ids"][:, 1:], batch["direction_ids"][:, 1:])
    assert torch.equal(probed["anchors"], batch["anchors"])


def test_permute_both_is_a_control_that_preserves_every_relation(batch):
    probed = evaluate.counterfactual(batch, "permute_both")
    # The same (mask, name, direction) triples, only in different slots.
    for before, after in zip(pairs(batch), pairs(probed)):
        assert sorted(before) == sorted(after)


def test_every_probe_actually_reaches_the_model(corpus, batch):
    """Each perturbation has to change the logits, or it is testing nothing."""
    model = StageB(
        len(corpus.vocab), min(corpus.shape), corpus.n_anchors,
        encoder_channels=(4, 8, 8, 8), token_dim=16, num_heads=2,
    ).eval()
    torch.nn.init.normal_(model.head.weight, std=0.05)  # the head starts at a constant
    task = StageBTask(model, corpus.vocab)
    with torch.no_grad():
        baseline = task(batch).logits
        for kind in evaluate.PROBES:
            perturbed = task(evaluate.counterfactual(batch, kind)).logits
            assert not torch.allclose(perturbed, baseline), kind


def test_an_unknown_probe_is_rejected(batch):
    with pytest.raises(ValueError, match="unknown counterfactual"):
        evaluate.counterfactual(batch, "shuffle_everything")
