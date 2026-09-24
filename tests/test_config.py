"""The shipped config: every key is read, and it builds the reference model."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from src.config import load_config, parse_overrides
from src.engine import (
    EarlyStopping, build_optimizer, build_scheduler, _logger, resolve_anchor_source,
    resolve_segmenter,
)
from src.models import StageA, StageB


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_the_shipped_config_builds_both_stages(cfg):
    """Every architectural key is read, and the two of them agree on the cube."""
    resolution = int(cfg.data.resolution)
    widths = list(cfg.model.stage_a.encoder_channels)
    assert cfg.model.stage_a.bottleneck == resolution // 2 ** (len(widths) - 1)
    segmenter = StageA(
        23, resolution, encoder_channels=tuple(widths),
        bottleneck=cfg.model.stage_a.bottleneck, token_dim=cfg.model.stage_a.token_dim,
        num_heads=cfg.model.stage_a.num_heads,
        prior_foreground=cfg.model.stage_a.prior_foreground,
        deep_supervision=tuple(cfg.model.stage_a.deep_supervision),
    )
    block = cfg.model.stage_b
    model = StageB.from_segmenter(
        segmenter, spacing=(1.25, 1.25, 1.25), n_anchors=cfg.data.n_anchors,
        tau=block.mapper.tau, min_mass=block.mapper.min_mass,
        answer_mode=str(block.get("answer_mode", "instance")),
        boundary_widths=tuple(block.boundary_widths), carver_width=block.carver.width,
        carver_blocks=block.carver.blocks,
        full_resolution_skip=block.carver.full_resolution_skip,
        use_image=block.use_image,
        carver_sees_anchors=bool(block.get("carver_sees_anchors", False)),
        additive_prior=block.additive_prior, alpha=block.alpha,
        background_logit=block.background_logit, prior_foreground=block.prior_foreground,
        dilate_radius=int(block.get("instance", {}).get("dilate_radius", 4)),
        region_threshold=float(block.get("instance", {}).get("region_threshold", 0.5)),
        max_seeds=int(block.get("instance", {}).get("max_seeds", 16)),
        intensity_tol=float(block.get("instance", {}).get("intensity_tol", 1.0)),
        tol_mode=str(block.get("instance", {}).get("tol_mode", "local_std")),
        feature_tol=float(block.get("instance", {}).get("feature_tol", 0.30)),
        score_null=float(block.get("instance", {}).get("score_null", 0.5)),
    )
    # The relational half is meant to be small beside the frozen segmenter.
    trainable = sum(p.numel() for p in model.trainable_parameters())
    assert trainable < sum(p.numel() for p in segmenter.parameters()) / 10


def test_the_shipped_constants_are_the_measured_ones(cfg):
    """`documentation/SpatialVox.md` (Deviations): both differ from the proposal's start."""
    assert cfg.model.stage_b.mapper.tau == 0.5          # 2.0 fails §2's gate at 0.65
    assert cfg.model.stage_b.mapper.min_mass == 1e-6    # 1e-3 rejects every structure
    assert cfg.train.stage_b.flip_probability == 0.25
    assert cfg.train.stage_b.far.epsilon == 0.05 and cfg.train.stage_b.far.dilation == 8


def test_every_block_the_code_reads_is_present(cfg):
    for block in ("data", "targets", "mri", "model", "train", "logging", "evaluation"):
        assert block in cfg
    for key in ("mapper", "boundary_widths", "carver", "use_image", "additive_prior",
                "alpha", "background_logit", "prior_foreground", "answer_mode", "instance"):
        assert key in cfg.model.stage_b, key
    for key in ("epochs", "optimizer", "scheduler", "phase_a_checkpoint", "anchor_source",
                "flip_probability", "loss", "far", "field_centroid_on", "mask_on",
                "boundary_checkpoint", "boundary_lr_scale"):
        assert key in cfg.train.stage_b, key
    for term in ("dice", "bce", "null_bce", "centroid", "field_centroid", "far"):
        assert term in cfg.train.stage_b.loss, term
    # Anything the architecture no longer has must be gone from the config too.
    for gone in ("occupancy_mode", "mode", "selection_weight"):
        assert gone not in cfg.train.stage_b, gone
    assert "stage_b_selection" not in cfg.model and "stage_b_image" not in cfg.model


def test_overrides_reach_a_nested_leaf():
    cfg = load_config(overrides=parse_overrides(["train.stage_b.optimizer.lr=1e-4"]))
    assert cfg.train.stage_b.optimizer.lr == 1e-4
    assert cfg.train.stage_a.optimizer.lr == 0.001  # untouched
    cfg = load_config(overrides=parse_overrides(["train.stage_b.anchor_source=oracle"]))
    assert resolve_anchor_source(cfg.train.stage_b) == "oracle"
    with pytest.raises(ValueError, match="anchor_source"):
        resolve_anchor_source({"anchor_source": "guess"})


def test_yaml_words_parse_as_bool_and_none_in_any_case():
    """`--set x=false` used to stay the string "false", which bool() reads as True."""
    parsed = parse_overrides(["a=false", "b=True", "c=null", "d=NONE", "e=1e-4", "f=oracle"])
    assert parsed == {"a": False, "b": True, "c": None, "d": None, "e": 1e-4, "f": "oracle"}


def test_stage_b_always_needs_a_segmenter():
    """Stage A is inside Stage B; there is no mode in which it is absent."""
    cfg = load_config()
    assert resolve_segmenter(cfg.train.stage_b) == Path(cfg.train.stage_b.phase_a_checkpoint)
    assert resolve_segmenter(cfg.train.stage_b, "other.pt") == Path("other.pt")
    with pytest.raises(ValueError, match="frozen Stage A"):
        resolve_segmenter({"phase_a_checkpoint": None})


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
