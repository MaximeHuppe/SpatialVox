"""The direction rule, anchor-first generation and the prompt language."""

from __future__ import annotations

import numpy as np
import pytest

from src.geometry import (
    AmbiguousDirection,
    anchor_first_examples,
    solutions_for,
    DIRECTIONS,
    OPPOSITE,
    centroid_world,
    centroids_world,
    classify,
    volume_center_world,
)
from src.vocab import Vocabulary

CENTER = np.array([31.5, 31.5, 31.5])


def at(x, y, z):
    return np.array([float(x), float(y), float(z)])


@pytest.mark.parametrize(
    "target, expected",
    [
        (at(31.5, 31.5, 40.0), "superior"),
        (at(31.5, 31.5, 20.0), "inferior"),
        (at(31.5, 40.0, 31.5), "anterior"),
        (at(31.5, 20.0, 31.5), "posterior"),
        (at(50.0, 31.5, 31.5), "lateral"),  # farther from the mid-sagittal plane
        (at(35.0, 31.5, 31.5), "medial"),  # closer than the anchor at x = 40
    ],
)
def test_every_direction_token_is_reachable(target, expected):
    anchor = at(40.0, 31.5, 31.5) if expected in ("medial", "lateral") else CENTER
    assert classify(target, anchor, CENTER) == expected


def test_medial_lateral_is_measured_from_the_centre_plane_not_the_sign_of_x():
    # Both structures sit left of centre; the farther one is still the lateral one.
    assert classify(at(10.0, 31.5, 31.5), at(20.0, 31.5, 31.5), CENTER) == "lateral"
    assert classify(at(20.0, 31.5, 31.5), at(10.0, 31.5, 31.5), CENTER) == "medial"


def test_axis_ties_break_z_then_y_then_x():
    assert classify(at(35.0, 35.0, 35.0), at(31.5, 31.5, 31.5), CENTER) == "superior"
    assert classify(at(35.0, 35.0, 31.5), at(31.5, 31.5, 31.5), CENTER) == "anterior"


def test_undecidable_pairs_raise_rather_than_inventing_a_direction():
    with pytest.raises(AmbiguousDirection):
        classify(CENTER, CENTER, CENTER)  # coincident centroids
    with pytest.raises(AmbiguousDirection):
        classify(at(20.0, 31.5, 31.5), at(43.0, 31.5, 31.5), CENTER)  # equidistant from centre


def test_directions_and_opposites_are_a_closed_involution():
    assert set(OPPOSITE) == set(DIRECTIONS)
    assert all(OPPOSITE[OPPOSITE[d]] == d for d in DIRECTIONS)


def test_centroids_world_matches_the_per_mask_computation(scene):
    _, labels = scene
    everything = centroids_world(labels, int(labels.max()), (1.0, 1.0, 1.0))
    for label in range(1, int(labels.max()) + 1):
        assert np.allclose(everything[label], centroid_world(labels == label, (1.0, 1.0, 1.0)))


def test_centroids_world_reports_absent_labels_as_nan():
    labels = np.zeros((4, 4, 4), dtype=np.uint8)
    labels[1, 1, 1] = 2
    centroids = centroids_world(labels, 3, (1.0, 1.0, 1.0))
    assert np.allclose(centroids[2], [1.0, 1.0, 1.0])
    assert np.isnan(centroids[1]).all() and np.isnan(centroids[3]).all()


def test_anisotropic_spacing_moves_the_dominant_axis():
    labels = np.zeros((8, 8, 8), dtype=np.uint8)
    labels[0, 0, 0], labels[2, 3, 0] = 1, 2
    isotropic = centroids_world(labels, 2, (1.0, 1.0, 1.0))
    # 3 voxels in y beats 2 in z, until z voxels are twice as long.
    assert classify(isotropic[2], isotropic[1], volume_center_world(labels.shape)) == "anterior"
    anisotropic = centroids_world(labels, 2, (1.0, 1.0, 2.0))
    assert classify(anisotropic[2], anisotropic[1], volume_center_world(labels.shape, (1.0, 1.0, 2.0))) == "superior"
def test_a_manifest_clause_always_describes_its_own_mask_channel(scene, vocab):
    """Clause ``i`` names ``anchors[i]`` with ``directions[i]``, in the prompt too.

    Anchor order is randomised, but it is randomised *once* and then shared:
    the mask channels are built from ``anchors`` in order and the prompt is
    rendered from the same list, so channel ``i`` is the structure clause ``i``
    talks about. Breaking that is what the ``permute_channels`` counterfactual
    is designed to detect, so it must hold by construction here.
    """
    from src.data import build_examples

    _, labels = scene
    centroids = centroids_world(labels, len(vocab), (1.0, 1.0, 1.0))
    center = volume_center_world(labels.shape, (1.0, 1.0, 1.0))
    for example in build_examples("s", labels, vocab, (1.0, 1.0, 1.0), 3):
        clauses = vocab.parse(example["prompt"])
        assert len(clauses) == len(example["anchors"])
        for clause, anchor, direction in zip(clauses, example["anchors"], example["directions"]):
            assert clause["anchor"] == vocab.name(anchor)
            assert clause["direction"] == direction
            # and the relation is the true one for that pair, not a relabelling
            assert classify(centroids[example["target"]], centroids[anchor], center) == direction


def test_a_manifest_anchor_order_is_not_sorted_by_distance(scene, vocab):
    """The stored order must not rank the anchors by proximity - see the leak."""
    from src.data import build_examples

    _, labels = scene
    centroids = centroids_world(labels, len(vocab), (1.0, 1.0, 1.0))
    ordered = 0
    examples = build_examples("s", labels, vocab, (1.0, 1.0, 1.0), 3)
    for example in examples:
        distances = [
            float(np.linalg.norm(centroids[anchor] - centroids[example["target"]]))
            for anchor in example["anchors"]
        ]
        ordered += distances == sorted(distances)
    assert ordered < len(examples), "every example is still stored nearest-first"


def test_prompt_round_trips(vocab):
    clauses = [
        {"direction": "superior", "anchor": "alpha"},
        {"direction": "medial", "anchor": "beta"},
        {"direction": "anterior", "anchor": "gamma"},
    ]
    prompt = vocab.render(clauses)
    assert prompt == (
        "segment the structure that is superior to the alpha, medial to the beta, "
        "and anterior to the gamma."
    )
    assert vocab.parse(prompt) == clauses
    assert vocab.clauses_from_ids(*vocab.clause_ids(clauses)) == clauses


def test_prompt_rejects_what_the_vocabulary_does_not_contain(vocab):
    with pytest.raises((ValueError, KeyError)):
        vocab.render([{"direction": "above", "anchor": "alpha"}] * 3)
    with pytest.raises((ValueError, KeyError)):
        vocab.render([{"direction": "superior", "anchor": "hippocampus"}] * 3)
    with pytest.raises(ValueError):
        vocab.parse("find the thing near the alpha.")


def test_prompt_requires_distinct_directions_and_anchors(vocab):
    with pytest.raises(ValueError):
        vocab.render(
            [
                {"direction": "superior", "anchor": "alpha"},
                {"direction": "superior", "anchor": "beta"},
                {"direction": "medial", "anchor": "gamma"},
            ]
        )


def test_the_vocabulary_scales_to_anatomical_names():
    vocab = Vocabulary(("Left-Hippocampus", "Right-Thalamus", "Brain-Stem", "Left-Putamen"))
    clauses = [
        {"direction": "superior", "anchor": "Brain-Stem"},
        {"direction": "lateral", "anchor": "Right-Thalamus"},
    ]
    assert vocab.parse(vocab.render(clauses)) == clauses
    assert vocab.name(vocab.label("Left-Putamen")) == "Left-Putamen"


# ---------------------------------------------------------------------------
# Anchor ordering and the sampling pool
# ---------------------------------------------------------------------------
def _ring(n_structures=9):
    """A target at the origin with `n_structures` neighbours at growing radii."""
    centroids = [np.array([np.nan] * 3), np.array([0.0, 0.0, 0.0])]
    for i in range(n_structures):
        offset = np.zeros(3)
        offset[i % 3] = (1.0 + i) * (1 if (i // 3) % 2 == 0 else -1)
        centroids.append(offset * 4.0 + np.array([0.3, 0.2, 0.1]) * i)
    return np.stack(centroids), list(range(1, n_structures + 2))


# ---------------------------------------------------------------------------
# Anchor-first generation
# ---------------------------------------------------------------------------
def _scatter(rng, n=12, size=64.0):
    """A scene of well-separated structures, as centroids."""
    centroids = np.full((n + 1, 3), np.nan)
    centroids[1:] = rng.uniform(4, size - 4, size=(n, 3))
    return centroids, list(range(1, n + 1)), np.array([size / 2] * 3)


def test_anchor_first_only_emits_prompts_that_describe_one_structure():
    """Well-posedness by construction, not by luck.

    The target-first generator emitted examples whose three clauses matched more
    than one structure (5.5% of them, and 29% at anchor_pool 8). Supervising on
    those teaches the model to prefer one defensible reading over another.
    """
    rng = np.random.default_rng(0)
    centroids, present, center = _scatter(rng)
    found = anchor_first_examples(
        centroids, present, center, 3, triples=200, locality=12, rng=rng
    )
    assert found, "expected this scene to support some prompts"
    for anchors, directions, target in found:
        assert solutions_for(anchors, directions, centroids, present, center) == [target]


def test_anchor_first_stops_the_anchor_set_from_naming_the_target():
    """The reason for the change: identities must not be a lookup key.

    Target-first draws the anchors nearest the target, so on fixed anatomy the
    unordered anchor set recovers the target 98.9% of the time on `data/mri`,
    and reading the prompt is strictly worse than ignoring it. Anchor-first
    makes one triple serve several targets.
    """
    rng = np.random.default_rng(1)
    centroids, present, center = _scatter(rng)
    found = anchor_first_examples(
        centroids, present, center, 3, triples=400, locality=12, rng=rng
    )
    by_set: dict[frozenset, set[int]] = {}
    for anchors, _, target in found:
        by_set.setdefault(frozenset(anchors), set()).add(target)
    shared = [targets for targets in by_set.values() if len(targets) > 1]
    assert shared, "no anchor set served more than one target; directions stay decorative"


def test_anchor_first_clauses_stay_aligned_with_their_anchors():
    """Clause i must describe anchor i even after the order is shuffled."""
    rng = np.random.default_rng(2)
    centroids, present, center = _scatter(rng)
    for anchors, directions, target in anchor_first_examples(
        centroids, present, center, 3, triples=100, locality=12, rng=rng
    ):
        for anchor, direction in zip(anchors, directions):
            assert classify(centroids[target], centroids[anchor], center) == direction


def test_anchor_first_directions_are_pairwise_distinct():
    rng = np.random.default_rng(3)
    centroids, present, center = _scatter(rng)
    for _, directions, _ in anchor_first_examples(
        centroids, present, center, 3, triples=200, locality=12, rng=rng
    ):
        assert len(set(directions)) == len(directions)
