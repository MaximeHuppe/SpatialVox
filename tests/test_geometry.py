"""The direction rule, anchor selection and the prompt language."""

from __future__ import annotations

import numpy as np
import pytest

from src.geometry import (
    AmbiguousDirection,
    DIRECTIONS,
    OPPOSITE,
    bbox_extent_world,
    centroid_world,
    centroids_world,
    classify,
    select_anchors,
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


def test_bbox_extent_counts_a_single_voxel_as_one_spacing_unit():
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[1, 1, 1] = True
    assert np.allclose(bbox_extent_world(mask), [1.0, 1.0, 1.0])


def test_select_anchors_takes_the_nearest_feasible_set_not_the_nearest_structures():
    # Target 5 sits at the centre. Structures 1 and 2 are both superior to it and
    # are its two nearest neighbours, but only one direction may be used once, so
    # the second-nearest structure is skipped for the farther but feasible ones.
    centroids = np.array(
        [
            [np.nan] * 3,
            [31.5, 31.5, 20.0],  # 1: superior, distance 11.5
            [31.5, 31.5, 18.0],  # 2: superior, distance 13.5 -> skipped
            [31.5, 15.0, 31.5],  # 3: anterior, distance 16.5
            [50.0, 31.5, 31.5],  # 4: medial,   distance 18.5
            [31.5, 31.5, 31.5],  # 5: the target
        ]
    )
    chosen = select_anchors(5, centroids, [1, 2, 3, 4, 5], CENTER, n_anchors=3)
    assert [label for label, _ in chosen] == [1, 3, 4]
    assert [direction for _, direction in chosen] == ["superior", "anterior", "medial"]


def test_select_anchors_returns_none_when_no_feasible_set_exists():
    centroids = np.array([[np.nan] * 3, [31.5, 31.5, 0.0], [31.5, 31.5, 10.0], [31.5, 31.5, 20.0]])
    assert select_anchors(3, centroids, [1, 2, 3], CENTER, n_anchors=3) is None


def test_anchor_order_is_ascending_distance(scene, vocab):
    from src.data import build_examples

    _, labels = scene
    for example in build_examples("s", labels, vocab, (1.0, 1.0, 1.0), 3):
        centroids = centroids_world(labels, len(vocab), (1.0, 1.0, 1.0))
        distances = [
            float(np.linalg.norm(centroids[anchor] - centroids[example["target"]]))
            for anchor in example["anchors"]
        ]
        assert distances == sorted(distances)


def test_prompt_round_trips(vocab):
    clauses = [
        {"direction": "superior", "anchor": "cube"},
        {"direction": "medial", "anchor": "torus"},
        {"direction": "anterior", "anchor": "sphere"},
    ]
    prompt = vocab.render(clauses)
    assert prompt == (
        "segment the structure that is superior to the cube, medial to the torus, "
        "and anterior to the sphere."
    )
    assert vocab.parse(prompt) == clauses
    assert vocab.clauses_from_ids(*vocab.clause_ids(clauses)) == clauses


def test_prompt_rejects_what_the_vocabulary_does_not_contain(vocab):
    with pytest.raises((ValueError, KeyError)):
        vocab.render([{"direction": "above", "anchor": "cube"}] * 3)
    with pytest.raises((ValueError, KeyError)):
        vocab.render([{"direction": "superior", "anchor": "hippocampus"}] * 3)
    with pytest.raises(ValueError):
        vocab.parse("find the thing near the cube.")


def test_prompt_requires_distinct_directions_and_anchors(vocab):
    with pytest.raises(ValueError):
        vocab.render(
            [
                {"direction": "superior", "anchor": "cube"},
                {"direction": "superior", "anchor": "torus"},
                {"direction": "medial", "anchor": "sphere"},
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
