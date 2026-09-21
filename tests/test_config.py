"""The shipped config: every key is read, and it builds the reference model."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from src.config import load_config, parse_overrides
from src.engine import EarlyStopping, build_optimizer, build_scheduler, _logger, resolve_phase_a_checkpoint
from src.models import StageA, StageB


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_the_64_cube_widths_still_match_the_reference_counts():
    """8,021,763 / 13,710,913 — exp/realistic-appearance, 64³, four encoder widths."""
    common = dict(
        encoder_channels=(32, 64, 128, 256),
        bottleneck=8,
        token_dim=256,
        num_heads=4,
        prior_foreground=0.0016,
    )
    a = StageA(10, 64, deep_supervision=(0.1, 0.3, 0.6), **common)
    b = StageB(10, 64, 3, intersection_hidden=256, **common)
    assert sum(p.numel() for p in a.parameters()) == 8_021_763
    assert sum(p.numel() for p in b.parameters()) == 13_710_913


def test_the_shipped_config_builds_at_128(cfg):
    """The width list, the resolution and `model.bottleneck` must agree.

    Derived from the config rather than hardcoded: the depth is a live
    experimental knob (each width halves the volume once, so N widths take 128
    down to 128 / 2**(N-1)), and a test that pins one choice only ever reports
    that someone changed it. What must never drift is the arithmetic - a width
    list that disagrees with `model.bottleneck` builds a model whose attention
    grid is not the configured one.
    """
    assert cfg.data.resolution == 128
    widths = list(cfg.model.encoder_channels)
    expected = cfg.data.resolution // 2 ** (len(widths) - 1)
    assert cfg.model.bottleneck == expected, (
        f"{len(widths)} encoder widths reduce {cfg.data.resolution} to {expected}, "
        f"but model.bottleneck is {cfg.model.bottleneck}"
    )
    common = dict(
        encoder_channels=tuple(cfg.model.encoder_channels),
        bottleneck=cfg.model.bottleneck,
        token_dim=cfg.model.token_dim,
        num_heads=cfg.model.num_heads,
        prior_foreground=cfg.model.prior_foreground,
    )
    a = StageA(10, cfg.data.resolution, deep_supervision=tuple(cfg.model.deep_supervision), **common)
    b = StageB(10, cfg.data.resolution, cfg.data.n_anchors,
               intersection_hidden=cfg.model.intersection_hidden, **common)
    assert a.config["bottleneck"] == expected
    assert b.config["bottleneck"] == expected


def test_every_block_the_code_reads_is_present(cfg):
    for block in ("data", "synthetic", "targets", "mri", "model", "train", "logging", "evaluation"):
        assert block in cfg, block
    assert {"source", "output", "n_subjects", "structures"} <= set(cfg.mri)
    listed = {name for names in cfg.targets.to_dict().values() for name in names}
    assert listed <= set(cfg.mri.structures)
    for stage in ("stage_a", "stage_b"):
        assert {"epochs", "optimizer", "scheduler", "augment"} <= set(cfg.train[stage]), stage
        assert {"name", "lr", "weight_decay"} == set(cfg.train[stage]["optimizer"])
        assert {"name", "warmup_epochs"} == set(cfg.train[stage]["scheduler"])
    assert {"mode", "occupancy_mode", "phase_a_checkpoint"} <= set(cfg.train.stage_b)
    assert cfg.train.stage_b.mode == "predicted"
    # `all` is the shipped default *because* `mode` is `predicted`. The verdict
    # that `all` is a silhouette leak was reached under `mode: oracle`, where
    # occupancy came from ground-truth labels; under `predicted` it is Stage A's
    # own output, available at inference, and `output subset of occupancy` is the
    # intended behaviour rather than a leak. On MRI the decoder needs it: no
    # global intensity threshold isolates structures better than IoU 0.0998, so
    # with `none`/`anchors-only` the decoder cannot see that a structure is
    # there at all. Pairing `all` with `oracle` is what must never ship.
    assert cfg.train.stage_b.occupancy_mode == "all"
    assert cfg.model.stage_b_image is True
    assert cfg.train.stage_b.phase_a_checkpoint == "runs/phase-a/current/best.pt"
    assert resolve_phase_a_checkpoint(cfg.train.stage_b) == Path("runs/phase-a/current/best.pt")
    assert {"name", "lambda_dice", "lambda_bce"} == set(cfg.train.loss)
    assert {"patience", "min_delta"} == set(cfg.train.early_stopping)


def test_overrides_reach_a_nested_leaf():
    cfg = load_config(overrides=parse_overrides(["train.stage_b.optimizer.lr=1e-4"]))
    assert cfg.train.stage_b.optimizer.lr == 1e-4
    assert cfg.train.stage_a.optimizer.lr == 0.001  # untouched
    cfg = load_config(overrides=parse_overrides(["train.stage_b.occupancy_mode=anchors-only"]))
    assert cfg.train.stage_b.occupancy_mode == "anchors-only"
    cfg = load_config(overrides=parse_overrides(["train.stage_b.mode=oracle"]))
    assert cfg.train.stage_b.mode == "oracle"
    assert resolve_phase_a_checkpoint(cfg.train.stage_b) is None


# -- the factories reject what they cannot do --------------------------------
def test_the_optimizer_block_is_honoured():
    model = nn.Linear(2, 2)
    optimizer = build_optimizer(model, {"name": "adamw", "lr": 0.01, "weight_decay": 0.5})
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == 0.01
    assert optimizer.param_groups[0]["weight_decay"] == 0.5
    with pytest.raises(ValueError, match="adamw"):
        build_optimizer(model, {"name": "sgd", "lr": 0.01})


def test_the_scheduler_block_is_honoured():
    model = nn.Linear(2, 2)

    def rates(cfg, epochs=10):
        optimizer = build_optimizer(model, {"name": "adamw", "lr": 1.0})
        schedule = build_scheduler(optimizer, cfg, epochs)
        out = []
        for _ in range(epochs):
            out.append(optimizer.param_groups[0]["lr"])
            schedule.step()
        return out

    cosine = rates({"name": "cosine", "warmup_epochs": 2})
    assert cosine[0] == pytest.approx(0.5) and cosine[1] == pytest.approx(1.0)  # warmup
    assert cosine[-1] < cosine[2]  # then decays
    constant = rates({"name": "constant", "warmup_epochs": 0})
    assert all(rate == pytest.approx(1.0) for rate in constant)
    with pytest.raises(ValueError, match="cosine"):
        build_scheduler(build_optimizer(model, {"name": "adamw", "lr": 1.0}), {"name": "step"}, 10)


def test_early_stopping_needs_a_rise_bigger_than_min_delta():
    stopper = EarlyStopping(patience=2, min_delta=0.01)
    assert not stopper.update(0.50)  # the first value is always a baseline
    assert not stopper.update(0.505)  # a rise, but under min_delta -> counts as waiting
    assert stopper.update(0.505)  # second stall -> stop
    assert EarlyStopping(patience=0).update(0.0) is False  # disabled


def test_early_stopping_resets_on_a_real_improvement():
    stopper = EarlyStopping(patience=2, min_delta=0.01)
    stopper.update(0.50)
    stopper.update(0.50)  # waited = 1
    assert not stopper.update(0.60)  # a real rise resets the counter
    assert not stopper.update(0.60)
    assert stopper.update(0.60)


def test_the_logging_backend_is_validated():
    assert _logger({"backend": "none"}, "run", {}) is None
    assert _logger({}, "run", {}) is None
    with pytest.raises(ValueError, match="wandb"):
        _logger({"backend": "tensorboard"}, "run", {})
