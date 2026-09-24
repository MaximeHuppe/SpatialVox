"""What the two networks are allowed to see, and what they may never see.

These are the contract tests. Breaking one of them does not make a run crash -
it makes the result mean something other than what the paper would claim, which
is why each test says which line of the architecture
(``documentation/SpatialVox.md``) it is holding.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from src.engine import load_model, save_checkpoint
from src.models import (
    BoundaryEncoder, BoundaryPretrainer, Carver, NullHead, StageA, StageB,
    bottleneck_for, grid_world_axes, prior_bias, soft_argmax,
)

RESOLUTION = 16
SPACING = (1.25, 1.25, 1.25)
VOCAB = 6
SEGMENTER = dict(
    vocab_size=VOCAB, resolution=RESOLUTION, encoder_channels=(8, 16, 32),
    token_dim=32, num_heads=2, bottleneck=4, prior_foreground=0.01,
    deep_supervision=(0.3, 0.7),
)


@pytest.fixture
def model() -> StageB:
    torch.manual_seed(0)
    return StageB(
        SEGMENTER, spacing=SPACING, n_anchors=3, tau=0.5, min_mass=1e-6,
        answer_mode="carver", carver_sees_anchors=True,
        boundary_widths=(4, 8), carver_width=4, carver_blocks=2, prior_foreground=0.01,
    ).eval()


def inputs(batch: int = 2, n_anchors: int = 3):
    torch.manual_seed(1)
    return (
        torch.randn(batch, 1, *(RESOLUTION,) * 3),
        torch.randint(0, 6, (batch, n_anchors)),
        torch.randint(0, VOCAB, (batch, n_anchors)),
    )


def soft_anchors(batch: int = 2, n_anchors: int = 3) -> torch.Tensor:
    """Three plausible soft masks, so a test can hold them fixed across prompts."""
    anchors = torch.zeros(batch, n_anchors, *(RESOLUTION,) * 3)
    for sample in range(batch):
        for slot in range(n_anchors):
            z, y, x = 3 + 4 * slot, 5 + 2 * sample, 4 + 3 * slot
            anchors[sample, slot, z:z + 3, y:y + 3, x:x + 3] = 0.9
    return anchors


# ---------------------------------------------------------------------------
# §6: what is withheld
# ---------------------------------------------------------------------------
def test_stage_b_signature_admits_nothing_that_identifies_the_target():
    """§6: not the label volume, the target mask, its name, or its centroid.

    A signature test rather than a behavioural one, because the failure this
    guards against is somebody *adding* an argument.
    """
    parameters = set(inspect.signature(StageB.forward).parameters) - {"self"}
    assert parameters == {"image", "direction_ids", "name_ids", "anchors", "boundary_image"}
    for banned in ("label", "target", "occupancy", "candidate", "mask", "centroid"):
        assert not any(banned in name for name in parameters), banned


def test_the_boundary_encoder_sees_the_image_and_nothing_else():
    """§4: "no prompt, name, direction, coordinate, or label"."""
    assert set(inspect.signature(BoundaryEncoder.forward).parameters) - {"self"} == {"image"}


def test_names_reach_stage_a_and_stop_there(model):
    """§1: "Names are used only to obtain the three anchor masks."

    With the masks supplied, the name ids are unused - so every output must be
    *bit-identical* under any renaming. A name embedding anywhere downstream
    would break this, and a triple of anchor names could then stand in for the
    target.
    """
    image, directions, names = inputs()
    anchors = soft_anchors()
    with torch.no_grad():
        first = model(image, directions, names, anchors=anchors)
        second = model(image, directions, names.roll(1, 1) * 0 + 3, anchors=anchors)
    assert torch.equal(first.logits, second.logits)
    assert torch.equal(first.where_raw, second.where_raw)
    assert torch.equal(first.valid, second.valid)


def test_the_carver_takes_exactly_the_ten_declared_channels(model):
    """§4: ``B(I)``, three masks, three fields, ``where_raw``, its log, the mass.

    Counted off the first convolution's weight, so an added coordinate grid or a
    smuggled extra map changes the number and fails here.
    """
    expected = model.boundary.out_channels + 2 * model.n_anchors + 3
    assert model.carver.stem[0].in_channels == expected
    assert expected == 4 + 3 + 3 + 1 + 1 + 1  # B(I), A_i, F_i, where, log(where), mass


def test_no_module_in_stage_b_builds_a_coordinate_grid(model):
    """§6: "a coordinate grid in ``B`` or the carver" is not an input.

    The world coordinates that exist live inside the mapper, which consumes them
    to place a pyramid and emits only fields.
    """
    import src.models as models

    assert not hasattr(models, "world_grid")
    source = inspect.getsource(Carver) + inspect.getsource(BoundaryEncoder)
    assert "grid_world_axes" not in source and "world_grid" not in source


# ---------------------------------------------------------------------------
# §1, §3: the anchors and the null head
# ---------------------------------------------------------------------------
def test_stage_a_is_frozen_and_out_of_the_optimiser(model):
    """§1: "then frozen. The relational loss does not flow back into it"."""
    assert all(not p.requires_grad for p in model.segmenter.parameters())
    trainable = {id(p) for p in model.trainable_parameters()}
    assert trainable.isdisjoint({id(p) for p in model.segmenter.parameters()})
    assert trainable == {id(p) for p in model.parameters() if p.requires_grad}

    model.train()
    assert not model.segmenter.training  # `.train()` must not wake it up
    image, directions, names = inputs()
    model(image, directions, names).logits.sum().backward()
    assert all(p.grad is None for p in model.segmenter.parameters())


def test_the_anchors_the_carver_sees_are_detached_probabilities(model):
    """§1: ``A_i = stop_gradient(sigmoid(anchor_logits_i))``, not a threshold."""
    image, directions, names = inputs()
    with torch.no_grad():
        out = model(image, directions, names)
    assert not out.anchors.requires_grad
    assert float(out.anchors.min()) >= 0.0 and float(out.anchors.max()) <= 1.0
    assert not torch.equal(out.anchors, (out.anchors > 0.5).float())  # soft, not binary


def test_the_null_head_reads_four_scalars_and_no_image():
    """§3: "The head cannot see the MRI"."""
    assert set(inspect.signature(NullHead.forward).parameters) - {"self"} == {"where_mass", "masses"}
    head = NullHead(3)
    assert head.mlp[0].in_features == 4


def test_an_impossible_prompt_does_not_produce_a_peak_of_one(model):
    """§9 Step 2, at random init: the product is not renormalised anywhere."""
    image = torch.randn(1, 1, *(RESOLUTION,) * 3)
    anchors = torch.zeros(1, 3, *(RESOLUTION,) * 3)
    anchors[0, :, 7:9, 7:9, 7:9] = 1.0  # all three on one spot
    directions = torch.tensor([[2, 3, 0]])  # superior, inferior, anterior
    with torch.no_grad():
        out = model(image, directions, torch.zeros(1, 3, dtype=torch.long), anchors=anchors)
    assert float(out.where_raw.max()) < 0.2
    assert float(out.where_mass) < 0.05


def test_replacing_the_image_moves_the_mask_and_not_the_field(model):
    """§7's image-replacement test, as an invariant of the forward.

    ``where_raw`` is computed from detached masks and direction ids; ``B`` is the
    only thing that reads pixels. Swapping the volume ``B`` sees must therefore
    leave every geometric quantity untouched.
    """
    image, directions, names = inputs()
    anchors = soft_anchors()
    other = torch.randn_like(image)
    # The mask head is zero-initialised (it starts at the foreground prior), so
    # at init no input can move the logits at all. Give it a weight first: the
    # claim under test is the architecture's dependency structure.
    torch.nn.init.normal_(model.carver.head.weight, std=0.1)
    with torch.no_grad():
        base = model(image, directions, names, anchors=anchors)
        swapped = model(image, directions, names, anchors=anchors, boundary_image=other)
    assert torch.equal(base.where_raw, swapped.where_raw)
    assert torch.equal(base.valid, swapped.valid)
    assert not torch.equal(base.logits, swapped.logits)


# ---------------------------------------------------------------------------
# §4: the carver
# ---------------------------------------------------------------------------
def test_anchor_voxels_above_a_half_are_written_to_background(model):
    """§4: "logits = background where A_i > 0.5", from the predicted soft mask."""
    image, directions, names = inputs()
    anchors = soft_anchors()
    with torch.no_grad():
        out = model(image, directions, names, anchors=anchors)
    inside = anchors.amax(dim=1, keepdim=True) > 0.5
    assert bool(inside.any())
    assert torch.allclose(out.logits[inside], torch.full_like(out.logits[inside], -10.0))
    assert not torch.allclose(out.logits[~inside], torch.full_like(out.logits[~inside], -10.0))


def test_the_prompt_only_carver_builds_without_b():
    """§7: "``B(I)`` is removed. Anchors and ``where_raw`` stay"."""
    model = StageB(SEGMENTER, spacing=SPACING, answer_mode="carver",
                   boundary_widths=(4, 8), carver_width=4,
                   use_image=False, carver_sees_anchors=True).eval()
    assert model.boundary is None
    assert model.carver.stem[0].in_channels == 2 * model.n_anchors + 3
    image, directions, names = inputs()
    with torch.no_grad():
        out = model(image, directions, names)
    assert out.logits.shape == (2, 1, *(RESOLUTION,) * 3)


def test_the_additive_prior_is_one_scalar_with_no_other_input():
    """§4: "``alpha`` is a single scalar... It cannot depend on the MRI, a class, or a name"."""
    off = StageB(SEGMENTER, spacing=SPACING, answer_mode="carver",
                 boundary_widths=(4, 8), carver_width=4)
    on = StageB(SEGMENTER, spacing=SPACING, answer_mode="carver",
                boundary_widths=(4, 8), carver_width=4,
                additive_prior=True, alpha=0.35)
    assert off.alpha is None
    assert on.alpha.shape == () and on.alpha.item() == pytest.approx(0.35)
    assert sum(p.numel() for p in on.trainable_parameters()) == (
        sum(p.numel() for p in off.trainable_parameters()) + 1
    )


@pytest.mark.parametrize("skip", [True, False])
def test_the_split_mask_head_is_exactly_the_1x1_on_the_upsampled_concatenation(skip):
    """The efficient head is the same function as the literal one, to rounding.

    ``head(cat[up(f), B]) = up(W_f f + b) + W_B B`` because a 1x1 convolution and
    a trilinear upsample are both linear and the upsample's weights sum to one.
    Checked in float64 with a non-zero head, since the shipped head starts at zero
    and would make the comparison vacuous.
    """
    torch.manual_seed(0)
    boundary_channels = 4 if skip else 0
    carver = Carver(boundary_channels + 9, boundary_channels, width=4, blocks=1,
                    full_resolution_skip=skip).double().eval()
    torch.nn.init.normal_(carver.head.weight, std=0.5)
    torch.nn.init.normal_(carver.head.bias, std=0.5)
    x = torch.randn(2, boundary_channels + 9, 10, 12, 14, dtype=torch.float64)
    boundary = x[:, :boundary_channels] if skip else None
    with torch.no_grad():
        logits, heatmap = carver(x, boundary)
        features = carver.blocks(carver.stem(x))
        full = torch.nn.functional.interpolate(features, size=x.shape[2:], mode="trilinear",
                                               align_corners=True)
        reference = carver.head(torch.cat([full, boundary], dim=1) if skip else full)
    assert logits.shape == (2, 1, 10, 12, 14)
    assert torch.allclose(logits, reference, atol=1e-10)
    assert torch.equal(heatmap, carver.heatmap(features))


def test_the_heatmap_is_a_separate_head_not_a_reading_of_the_mask(model):
    """§4: "a separate 1x1, soft-argmax -> centroid"; §5: not trained through the mask."""
    assert model.carver.heatmap is not model.carver.head
    image, directions, names = inputs()
    with torch.no_grad():
        out = model(image, directions, names)
    assert out.centroid.shape == (2, 3)
    extent = [(RESOLUTION - 1) * s for s in SPACING]
    assert bool(((out.centroid >= 0) & (out.centroid <= torch.tensor(extent))).all())


def test_soft_argmax_is_an_expectation_in_world_units():
    """A one-hot heatmap must report that voxel's world centre, at any grid ratio."""
    heat = torch.full((1, 1, 4, 4, 4), -30.0)
    heat[0, 0, 3, 1, 0] = 30.0
    centroid = soft_argmax(heat, (8, 8, 8), (2.0, 2.0, 2.0))
    # a cell of a grid twice as coarse covers indices 2i..2i+1, so it sits at 2i + 0.5
    assert torch.allclose(centroid, torch.tensor([[1.0, 5.0, 13.0]]), atol=1e-3)


# ---------------------------------------------------------------------------
# §8: the checkpoint
# ---------------------------------------------------------------------------
def test_the_config_carries_every_architectural_constant(model):
    """§8: ``mapper.tau``, ``mapper.min_mass``, ``alpha``, and the width of ``B``."""
    for key in ("tau", "min_mass", "alpha", "boundary_widths", "carver_width",
                "full_resolution_skip", "use_image", "additive_prior", "spacing",
                "answer_mode"):
        assert key in model.config, key
    assert model.config["segmenter"] == SEGMENTER
    assert model.config["answer_mode"] == "carver"


def test_a_checkpoint_round_trips_through_load_model(tmp_path, model):
    """``StageB.config`` is what ``load_model`` rebuilds from - including Stage A."""
    image, directions, names = inputs()
    with torch.no_grad():
        before = model(image, directions, names).logits
    save_checkpoint(tmp_path / "b.pt", model, {"stage": "test"})
    restored = load_model(tmp_path / "b.pt")
    assert isinstance(restored, StageB)
    with torch.no_grad():
        assert torch.equal(restored(image, directions, names).logits, before)


def test_a_pretrained_boundary_encoder_loads_into_stage_b():
    """§4: ``B`` is pretrained, then its weights continue inside Stage B."""
    pretrainer = BoundaryPretrainer(widths=(4, 8))
    model = StageB(SEGMENTER, spacing=SPACING, answer_mode="carver",
                   boundary_widths=(4, 8), carver_width=4)
    model.boundary.load_state_dict(pretrainer.encoder.state_dict())
    for a, b in zip(model.boundary.parameters(), pretrainer.encoder.parameters()):
        assert torch.equal(a, b)


def test_b_gets_its_own_learning_rate_only_once_it_is_pretrained(model):
    assert len(model.parameter_groups(1e-3)) == 1
    model.boundary_lr_scale = 0.1
    groups = model.parameter_groups(1e-3)
    assert [g["lr"] for g in groups] == [1e-3, 1e-4]
    assert sum(len(g["params"]) for g in groups) == len(model.trainable_parameters())


# ---------------------------------------------------------------------------
# Stage A, which the rest of the pipeline depends on being unchanged
# ---------------------------------------------------------------------------
def test_stage_a_masks_do_not_depend_on_which_other_names_were_asked_for():
    """What lets Stage A be trained on the vocabulary and queried for three names."""
    torch.manual_seed(0)
    model = StageA(VOCAB, RESOLUTION, encoder_channels=(8, 16, 32), token_dim=32,
                   num_heads=2, bottleneck=4).eval()
    image = torch.randn(1, 1, *(RESOLUTION,) * 3)
    with torch.no_grad():
        alone = model(image, torch.tensor([[2]]), deep_supervision=False).logits
        crowd = model(image, torch.tensor([[0, 2, 5]]), deep_supervision=False).logits
    assert torch.allclose(alone[:, 0], crowd[:, 1], atol=1e-5)


def test_the_bottleneck_is_checked_against_the_width_list():
    assert bottleneck_for(128, [32, 64, 128, 256, 256]) == 8
    with pytest.raises(ValueError, match="model.bottleneck to 16"):
        bottleneck_for(128, [32, 64, 128, 256], expected=8)
    with pytest.raises(ValueError, match="attention tokens"):
        bottleneck_for(512, [32, 64, 128, 256])


def test_the_head_bias_starts_at_the_foreground_prior():
    assert prior_bias(0.0016) == pytest.approx(-6.4362, abs=1e-3)
    assert torch.sigmoid(torch.tensor(prior_bias(0.01))).item() == pytest.approx(0.01)


def test_a_coarse_grid_maps_to_the_world_centre_of_the_block_it_covers():
    axes = grid_world_axes((2, 2, 2), (8, 8, 8), (1.0, 1.0, 1.0), "cpu", torch.float32)
    assert torch.allclose(axes[0], torch.tensor([1.5, 5.5]))
