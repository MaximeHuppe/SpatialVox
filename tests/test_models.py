"""Model contracts: shapes, scaling, and what Stage B is and is not allowed to see."""

from __future__ import annotations

import inspect

import pytest
import torch

from src import models
from src.engine import StageBTask, masks_from
from src.models import (
    StageA,
    StageB,
    bottleneck_for,
    depth_for,
    mask_geometry,
    widths_for,
    world_grid,
)

SMALL = dict(encoder_channels=(4, 8, 16), token_dim=16, num_heads=2)


@pytest.fixture(scope="module")
def stage_a():
    return StageA(vocab_size=10, resolution=16, **SMALL)


@pytest.fixture(scope="module")
def stage_b():
    model = StageB(vocab_size=10, resolution=16, n_anchors=3, **SMALL)
    # The mask head is deliberately zero-initialised (it starts at the foreground
    # prior), so an untrained model outputs a constant. Give it a real weight so
    # the sensitivity tests below measure the network and not that initialisation.
    torch.nn.init.normal_(model.head.weight, std=0.05)
    return model


# -- sizing scales with the data --------------------------------------------
@pytest.mark.parametrize("resolution, expected_depth", [(32, 2), (64, 3), (128, 4), (256, 5)])
def test_depth_follows_the_resolution(resolution, expected_depth):
    assert depth_for(resolution, 8) == expected_depth


def test_a_resolution_that_does_not_reduce_cleanly_is_rejected():
    with pytest.raises(ValueError, match="power-of-two"):
        depth_for(100, 8)


def test_the_bottleneck_follows_from_the_width_list():
    assert bottleneck_for(64, [32, 64, 128, 256]) == 8
    assert bottleneck_for(128, [32, 64, 128, 256, 256]) == 8


def test_a_width_list_that_disagrees_with_the_resolution_says_what_to_write():
    """The 128^3 trap: same widths, so the bottleneck quadruples in token count."""
    with pytest.raises(ValueError, match="set model.bottleneck to 16"):
        bottleneck_for(128, [32, 64, 128, 256], expected=8)
    with pytest.raises(ValueError, match="attention tokens"):
        bottleneck_for(512, [32, 64, 128, 256])


def test_widths_double_but_stay_under_the_cap():
    assert widths_for(32, 4, 256) == [32, 64, 128, 256, 256]


@pytest.mark.parametrize("resolution", [16, 32])
def test_stage_b_returns_the_input_resolution_at_any_scale(resolution):
    # One width per scale, so the list length is what takes 16 or 32 down to 4.
    channels = widths_for(4, depth_for(resolution, 4), 8)
    model = StageB(
        vocab_size=10, resolution=resolution, n_anchors=3,
        encoder_channels=channels, token_dim=16, num_heads=2, bottleneck=4,
    )
    volume = (resolution,) * 3
    output = model(
        torch.rand(1, 3, *volume) > 0.9,
        torch.tensor([[0, 2, 4]]),
        torch.tensor([[1, 2, 3]]),
        torch.zeros(1, 1, *volume),
    )
    assert output.logits.shape == (1, 1, *volume)
    # The bottleneck stays inside the attention budget however big the volume is.
    assert all(e.shape[-1] == 4 for e in output.evidence)


def test_the_bottleneck_stays_at_512_tokens_from_64_to_128():
    for resolution in (64, 128):
        assert (resolution // 2 ** depth_for(resolution, 8)) ** 3 == 512


# -- forward contracts -------------------------------------------------------
def test_stage_a_emits_one_mask_per_requested_name(stage_a):
    output = stage_a(torch.randn(2, 1, 16, 16, 16), torch.tensor([[0, 3, 7], [1, 2, 9]]))
    assert output.logits.shape == (2, 3, 16, 16, 16)
    # One supervised map per decoder scale, coarse to fine, the last full size.
    assert [tuple(s.shape[2:]) for s in output.scales] == [(8, 8, 8), (16, 16, 16)]


def test_stage_a_logits_do_not_depend_on_the_other_names_in_the_request(stage_a):
    """Why Stage B can ask for exactly the three anchors a prompt names."""
    stage_a.eval()
    image = torch.randn(1, 1, 16, 16, 16)
    with torch.no_grad():
        many = stage_a(image, torch.tensor([[0, 3, 7]])).logits
        one = stage_a(image, torch.tensor([[3]])).logits
    assert torch.allclose(many[:, 1], one[:, 0], atol=1e-5)


def test_stage_b_output_changes_when_the_prompt_changes(stage_b):
    stage_b.eval()
    anchors = (torch.rand(1, 3, 16, 16, 16) > 0.9).float()
    occupancy = torch.ones(1, 1, 16, 16, 16)
    names = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        first = stage_b(anchors, torch.tensor([[0, 2, 4]]), names, occupancy).logits
        second = stage_b(anchors, torch.tensor([[1, 3, 5]]), names, occupancy).logits
    assert not torch.allclose(first, second)


def test_stage_b_output_changes_when_the_channels_move(stage_b):
    stage_b.eval()
    anchors = (torch.rand(1, 3, 16, 16, 16) > 0.9).float()
    occupancy = torch.ones(1, 1, 16, 16, 16)
    directions, names = torch.tensor([[0, 2, 4]]), torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        first = stage_b(anchors, directions, names, occupancy).logits
        second = stage_b(anchors.roll(1, dims=1), directions, names, occupancy).logits
    assert not torch.allclose(first, second)


def test_stage_b_rejects_the_wrong_number_of_channels(stage_b):
    with pytest.raises(ValueError, match="anchor channels"):
        stage_b(
            torch.zeros(1, 2, 16, 16, 16), torch.tensor([[0, 2, 4]]),
            torch.tensor([[1, 2, 3]]), torch.zeros(1, 1, 16, 16, 16),
        )


def test_stage_b_subtracts_the_anchors_from_the_occupancy_it_is_given(stage_b):
    """Occupancy says 'something is here'; the anchors are the given, not the answer."""
    source = inspect.getsource(StageB.forward)
    assert "1 - anchors.amax" in source


def test_stage_b_signature_admits_nothing_that_identifies_the_target():
    forbidden = {"labels", "target", "target_mask", "target_name", "instance", "centroid"}
    assert not forbidden & set(inspect.signature(StageB.forward).parameters)


def test_stage_b_sees_the_anchors_and_an_anonymous_occupancy_and_nothing_else(corpus, monkeypatch):
    """The end-to-end leak check, on the real forward pass.

    Whatever a batch happens to carry, exactly two volumes reach the network: the
    ordered anchor channels (encoder) and one binary occupancy map (decoder).
    """
    from src.data import ExampleDataset, collate

    dataset = ExampleDataset(corpus, "train")
    batch = collate([dataset[0], dataset[1]])
    model = StageB(
        len(corpus.vocab), min(corpus.shape), corpus.n_anchors,
        encoder_channels=(4, 8, 8, 8), token_dim=16, num_heads=2,
    )
    seen: dict[str, torch.Tensor] = {}
    model.encoder.register_forward_pre_hook(lambda _, args: seen.update(encoder=args[0]))
    decode = model.decoder.forward
    monkeypatch.setattr(
        model.decoder,
        "forward",
        lambda features, *, context=None, occupancy=None, image=None: (
            seen.update(occupancy=occupancy) or decode(features, context=context, occupancy=occupancy, image=image)
        ),
    )
    prediction = StageBTask(model, corpus.vocab, mode="oracle")(batch)

    target, anchors = prediction.target, masks_from(batch["labels"], batch["anchors"])
    # The encoder sees the anchor channels and nothing else - not the target, not
    # the occupancy: grounding queries that could see the target's own voxels
    # would let the model ignore the prompt and pick "a blob that is not an anchor".
    assert torch.equal(seen["encoder"], anchors)
    assert not (seen["encoder"] * target).any()
    # The decoder's occupancy has the anchors removed - they are the given, not
    # the answer - and is binary, so the target is one unmarked structure among
    # several. Occupancy says "something is here", never "this one".
    assert not (seen["occupancy"] * anchors.amax(1, keepdim=True)).any()
    assert set(seen["occupancy"].unique().tolist()) <= {0.0, 1.0}
    assert float(seen["occupancy"].sum()) > float(target.sum()) * 1.5
    assert prediction.groups == [[n] for n in batch["target_name"]]


def test_stage_b_image_widens_the_guidance_projection_and_nothing_else(corpus):
    """The scene enters at the decoder, so the encoder stays anchors-only.

    Turning it on must not touch the encoder, the prompt tables or the head:
    only the 1x1 convolutions that mix the volumetric guidance in gain a plane.
    A checkpoint written before the flag existed therefore still rebuilds, which
    is why the projection keeps the name ``occupancy``.
    """
    small = dict(encoder_channels=(4, 8, 8, 8), token_dim=16, num_heads=2)
    blind = StageB(len(corpus.vocab), min(corpus.shape), corpus.n_anchors, **small)
    seeing = StageB(len(corpus.vocab), min(corpus.shape), corpus.n_anchors, image=True, **small)

    assert blind.config["image"] is False and seeing.config["image"] is True
    assert set(blind.state_dict()) == set(seeing.state_dict())
    widened = {
        key for key in blind.state_dict()
        if blind.state_dict()[key].shape != seeing.state_dict()[key].shape
    }
    assert widened == {f"decoder.occupancy.{level}.weight" for level in range(len(widened))}
    assert all(key.startswith("decoder.occupancy.") for key in widened)
    for before, after in zip(blind.decoder.occupancy, seeing.decoder.occupancy):
        assert after.in_channels == before.in_channels + 1
    assert (seeing.encoder.stages[0][0][0].in_channels
            == blind.encoder.stages[0][0][0].in_channels)


def test_stage_b_image_reaches_the_decoder_but_never_the_encoder(corpus, monkeypatch):
    """The leak check again, with the scene switched on.

    The image says *what* is there; the relations say *which one*. So it may
    reach the decoder, where shape is reconstructed, and must not reach the
    encoder, whose features are what the grounding attention queries.
    """
    from src.data import ExampleDataset, collate

    dataset = ExampleDataset(corpus, "train")
    batch = collate([dataset[0], dataset[1]])
    model = StageB(
        len(corpus.vocab), min(corpus.shape), corpus.n_anchors, image=True,
        encoder_channels=(4, 8, 8, 8), token_dim=16, num_heads=2,
    )
    seen: dict[str, torch.Tensor] = {}
    model.encoder.register_forward_pre_hook(lambda _, args: seen.update(encoder=args[0]))
    decode = model.decoder.forward
    monkeypatch.setattr(
        model.decoder,
        "forward",
        lambda features, *, context=None, occupancy=None, image=None: (
            seen.update(image=image) or decode(features, context=context, occupancy=occupancy, image=image)
        ),
    )
    prediction = StageBTask(model, corpus.vocab, mode="oracle", occupancy_mode="none")(batch)

    anchors = masks_from(batch["labels"], batch["anchors"])
    assert torch.equal(seen["encoder"], anchors)          # encoder: anchors, still
    assert torch.equal(seen["image"], batch["image"])     # decoder: the volume, unaltered
    assert prediction.logits.shape == (2, 1, *batch["labels"].shape[1:])


def test_stage_b_image_is_the_acquired_volume_not_a_label_derivative(corpus):
    """Whatever else it is, the image must not be a function of the labels."""
    from src.data import ExampleDataset, collate

    batch = collate([ExampleDataset(corpus, "train")[0]])
    labels, image = batch["labels"], batch["image"]
    assert image.dtype.is_floating_point
    assert image.shape == (1, 1, *labels.shape[1:])
    # Background is noisy, so the volume is nowhere near a binary occupancy map.
    assert float(image[(labels == 0).unsqueeze(1)].std()) > 0
    assert len(torch.unique(image)) > 2


def test_the_image_is_area_pooled_not_max_pooled(corpus):
    """`max` on intensities returns the brightest voxel per block and saturates.

    Occupancy wants `max` - a thin structure must survive the downsample. An
    intensity volume wants the anti-aliased average, or every coarse voxel
    reports the brightest thing near it and the guidance says "structure
    everywhere".
    """
    from src.models import pool_to

    volume = torch.zeros(1, 1, 8, 8, 8)
    volume[0, 0, 0, 0, 0] = 1.0                  # one bright voxel in the first block
    assert float(pool_to(volume, (4, 4, 4), "max")[0, 0, 0, 0, 0]) == 1.0
    assert float(pool_to(volume, (4, 4, 4), "avg")[0, 0, 0, 0, 0]) == pytest.approx(1 / 8)


def test_stage_b_anchors_only_occupancy_vanishes_after_subtract(corpus, monkeypatch):
    """The decoder never sees the anchors twice: they are the given, not occupancy."""
    from src.data import ExampleDataset, collate

    dataset = ExampleDataset(corpus, "train")
    batch = collate([dataset[0], dataset[1]])
    model = StageB(
        len(corpus.vocab), min(corpus.shape), corpus.n_anchors,
        encoder_channels=(4, 8, 8, 8), token_dim=16, num_heads=2,
    )
    seen: dict[str, torch.Tensor] = {}
    decode = model.decoder.forward
    monkeypatch.setattr(
        model.decoder,
        "forward",
        lambda features, *, context=None, occupancy=None, image=None: (
            seen.update(occupancy=occupancy) or decode(features, context=context, occupancy=occupancy, image=image)
        ),
    )
    StageBTask(model, corpus.vocab, mode="oracle", occupancy_mode="anchors-only")(batch)
    assert not seen["occupancy"].any()


def test_stage_b_none_occupancy_is_an_empty_channel(corpus, monkeypatch):
    from src.data import ExampleDataset, collate

    dataset = ExampleDataset(corpus, "train")
    batch = collate([dataset[0], dataset[1]])
    model = StageB(
        len(corpus.vocab), min(corpus.shape), corpus.n_anchors,
        encoder_channels=(4, 8, 8, 8), token_dim=16, num_heads=2,
    )
    seen: dict[str, torch.Tensor] = {}
    decode = model.decoder.forward
    monkeypatch.setattr(
        model.decoder,
        "forward",
        lambda features, *, context=None, occupancy=None, image=None: (
            seen.update(occupancy=occupancy) or decode(features, context=context, occupancy=occupancy, image=image)
        ),
    )
    StageBTask(model, corpus.vocab, mode="oracle", occupancy_mode="none")(batch)
    assert not seen["occupancy"].any()


# -- geometry features -------------------------------------------------------
def test_mask_geometry_reads_position_and_size_off_the_mask():
    masks = torch.zeros(1, 3, 16, 16, 16)
    masks[0, 0, 8, 8, 12:16] = 1  # right of centre in x
    masks[0, 1, 0:4, 8, 8] = 1  # inferior in z
    # channel 2 is deliberately empty
    geometry = mask_geometry(masks)
    assert geometry.shape == (1, 3, 8)
    assert geometry[0, 0, 0] > 0.3  # centroid x
    assert geometry[0, 1, 2] < -0.3  # centroid z
    assert geometry[0, 0, 7] == 1 and geometry[0, 1, 7] == 1  # present
    assert geometry[0, 2, 7] == 0 and geometry[0, 2].abs().sum() == 0  # absent, and zeroed


def test_world_coordinates_agree_across_scales():
    """A coarse cell must sit at the world centre of the fine cells it covers."""
    fine = world_grid((8, 8, 8), (8, 8, 8), "cpu", torch.float32)
    coarse = world_grid((2, 2, 2), (8, 8, 8), "cpu", torch.float32)
    assert torch.allclose(coarse[0, 0, 0, 0, 0], fine[0, 0, 0, 0, 0:4].mean(), atol=1e-6)
    assert float(fine.min()) == -1.0 and float(fine.max()) == 1.0


def test_world_coordinates_are_ordered_x_y_z():
    grid = world_grid((2, 4, 8), (2, 4, 8), "cpu", torch.float32)
    assert grid[0, 0, 0, 0, :].tolist() == pytest.approx([-1.0, -5 / 7, -3 / 7, -1 / 7, 1 / 7, 3 / 7, 5 / 7, 1.0])
    assert grid[0, 2, :, 0, 0].tolist() == pytest.approx([-1.0, 1.0])  # z varies along axis 0


def test_a_checkpoint_rebuilds_the_same_architecture(tmp_path):
    from src.engine import load_model, save_checkpoint

    model = StageB(vocab_size=7, resolution=16, n_anchors=3, **SMALL)
    save_checkpoint(tmp_path / "m.pt", model, {"stage": "test"})
    restored = load_model(tmp_path / "m.pt")
    assert restored.config == model.config
    assert all(torch.equal(a, b) for a, b in zip(restored.state_dict().values(), model.state_dict().values()))


# ---------------------------------------------------------------------------
# The selection head: the class-agnostic pathway
# ---------------------------------------------------------------------------
#: Deliberately unlike any width below, so "sized by the vocabulary" is testable.
SELECT_VOCAB = 23


def _stage_b_selecting(resolution=32):
    return StageB(
        SELECT_VOCAB, resolution, 3, encoder_channels=(8, 16, 32), bottleneck=8,
        token_dim=32, num_heads=2, image=True, selection=True,
    )


def _selection_inputs(model, batch=2, resolution=32, candidates=8):
    torch.manual_seed(0)
    shape = (batch, 1, resolution, resolution, resolution)
    return dict(
        anchors=torch.rand(batch, 3, resolution, resolution, resolution).round(),
        direction_ids=torch.zeros(batch, 3, dtype=torch.long),
        name_ids=torch.zeros(batch, 3, dtype=torch.long),
        occupancy=torch.rand(*shape).round(),
        image=torch.randn(*shape),
        candidates=torch.rand(batch, candidates, resolution, resolution, resolution).round(),
    )


def test_the_selection_head_has_no_per_class_parameters():
    """This is the whole reason it transfers.

    The dense head has to synthesise a silhouette, which is learned per class -
    measured on the val split, localisation is 100% on trained classes and 31.2%
    on unseen ones. Selection reads the same two numbers off every candidate, so
    a structure never seen as a target is scored by exactly the same weights.
    Any parameter whose size tracks the vocabulary would break that.
    """
    head = _stage_b_selecting().selector
    for name, parameter in head.named_parameters():
        assert SELECT_VOCAB not in parameter.shape, (
            f"{name} is sized by the vocabulary: {tuple(parameter.shape)}"
        )
    # And it must accept a candidate count it was never built for.
    model = _stage_b_selecting().eval()
    with torch.no_grad():
        scores = model(**_selection_inputs(model, candidates=5)).selection
    assert scores.shape[-1] == 5


def test_selection_scores_follow_the_candidates_not_their_position():
    """Reordering the candidates must reorder the scores, nothing else.

    A head that leaned on candidate index rather than candidate content would
    look fine on the training classes and be meaningless on any other ordering.
    """
    model = _stage_b_selecting().eval()
    inputs = _selection_inputs(model)
    with torch.no_grad():
        base = model(**inputs).selection
        permutation = torch.tensor([3, 1, 0, 2, 5, 4, 7, 6])
        shuffled = model(**{**inputs, "candidates": inputs["candidates"][:, permutation]}).selection
    torch.testing.assert_close(shuffled, base[:, permutation], rtol=1e-4, atol=1e-5)


def test_an_empty_candidate_can_never_be_selected():
    """Stage A scores 0.000 on two of 23 classes, so empty masks do occur."""
    model = _stage_b_selecting().eval()
    inputs = _selection_inputs(model)
    inputs["candidates"][:, 2] = 0.0
    with torch.no_grad():
        scores = model(**inputs).selection
    assert torch.isinf(scores[:, 2]).all() and (scores[:, 2] < 0).all()
    assert scores.argmax(dim=-1).ne(2).all()


def test_selection_is_off_by_default_and_recorded_in_the_config():
    """`load_model` rebuilds from `config`, so a new knob must live there."""
    assert StageB(SELECT_VOCAB, 32, 3, encoder_channels=(8, 16, 32), bottleneck=8).selector is None
    assert _stage_b_selecting().config["selection"] is True


def test_stage_b_without_candidates_still_returns_logits():
    """Evaluation paths that pass no candidates must not crash."""
    model = _stage_b_selecting().eval()
    inputs = _selection_inputs(model)
    inputs.pop("candidates")
    with torch.no_grad():
        output = model(**inputs)
    assert output.selection is None and output.logits.shape[1] == 1
