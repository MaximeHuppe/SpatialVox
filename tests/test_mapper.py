"""The WHERE: the pyramid is ``classify``, and the product is not renormalised.

The proposal's §9 Step 1 (``documentation/SpatialVox.md``, Invariants and the tests that pin
them) lists what has to hold before the carver is trained on top of the mapper. Each of those
is a test here.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.geometry import DIRECTIONS, classify, volume_center_world
from src.mapper import (
    PositionalMapper3D, margin_at, margins, soft_centroids, world_axes,
)

SIZE = 24
SHAPE = (SIZE,) * 3
SPACING = (1.0, 1.0, 1.0)
CENTER = tuple(volume_center_world(SHAPE, SPACING).tolist())
MIDDLE = (SIZE - 1) / 2


def ball(centre_zyx, radius: float = 2.0) -> np.ndarray:
    """A soft blob, so the centroid is the confidence-weighted one it is meant to be."""
    grid = np.stack(np.meshgrid(*[np.arange(SIZE)] * 3, indexing="ij"), -1).astype(float)
    distance = np.linalg.norm(grid - np.array(centre_zyx, float), axis=-1)
    return np.clip(1.5 - distance / radius, 0.0, 1.0).astype(np.float32)


def fields_for(centres_zyx, directions, tau=0.5, min_mass=1e-6):
    masks = torch.from_numpy(np.stack([ball(c) for c in centres_zyx]))[None]
    ids = torch.tensor([[DIRECTIONS.index(d) for d in directions]])
    mapper = PositionalMapper3D(tau=tau, min_mass=min_mass)
    return mapper(masks, ids, SPACING, CENTER)


def at(volume: torch.Tensor, point_zyx) -> float:
    z, y, x = (int(round(v)) for v in point_zyx)
    return float(volume[..., z, y, x])


# ---------------------------------------------------------------------------
# The six regions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "direction, anchor, inside_zyx, outside_zyx",
    [
        ("superior", (12, 12, 12), (18, 12, 12), (6, 12, 12)),
        ("inferior", (12, 12, 12), (6, 12, 12), (18, 12, 12)),
        ("anterior", (12, 12, 12), (12, 18, 12), (12, 6, 12)),
        ("posterior", (12, 12, 12), (12, 6, 12), (12, 18, 12)),
        # Medial and lateral are *distances to the midline* (x = 11.5), not the
        # half-spaces +x and -x, so the anchor has to sit off-centre for either
        # word to be decidable: at x = 16 it is 4.5 out, x = 22 is further out
        # and x = 13 is nearer in - all on the same side.
        ("lateral", (12, 12, 16), (12, 12, 22), (12, 12, 13)),
        ("medial", (12, 12, 16), (12, 12, 13), (12, 12, 22)),
    ],
)
def test_each_region_is_the_half_pyramid_classify_names(direction, anchor, inside_zyx, outside_zyx):
    """``F`` is high where ``classify`` would emit this word, and low where it would not.

    The two probes are one axis-offset apart from the anchor, which is exactly
    the question ``classify`` answers for a centroid.
    """
    out = fields_for([anchor], [direction])
    field = out.fields[0, 0]

    to_world = lambda p: np.array([p[2] * SPACING[0], p[1] * SPACING[1], p[0] * SPACING[2]])
    assert classify(to_world(inside_zyx), to_world(anchor), np.array(CENTER)) == direction
    assert at(field, inside_zyx) > 0.99
    assert at(field, outside_zyx) < 0.01


def test_a_point_off_the_dominant_axis_falls_outside_the_45_degree_pyramid():
    """The predicate is that one axis *dominates*, not merely that it is positive."""
    field = fields_for([(12, 12, 12)], ["superior"]).fields[0, 0]
    assert at(field, (16, 12, 12)) > 0.99  # straight up: z leads by 4
    assert at(field, (16, 20, 12)) < 0.01  # up and well forward: y leads instead


# ---------------------------------------------------------------------------
# The product
# ---------------------------------------------------------------------------
#: Three anchors and a point that is superior to the first, anterior to the
#: second and lateral to the third - the conjunction, by construction.
ANCHORS = [(4, 12, 12), (12, 4, 12), (12, 12, 6)]
CLAUSES = ["superior", "anterior", "lateral"]
SATISFYING = (18, 18, 20)


def centre_of_mass(volume: torch.Tensor) -> np.ndarray:
    grid = np.stack(np.meshgrid(*[np.arange(SIZE)] * 3, indexing="ij"), -1).astype(float)
    weights = volume.numpy()[..., None]
    return (grid * weights).sum((0, 1, 2)) / max(weights.sum(), 1e-9)


def test_the_product_sits_on_the_satisfying_point_not_on_any_anchor():
    """Three clauses intersect on the region their conjunction describes.

    Measured on the field's centre of mass rather than its ``argmax``: with
    ``tau`` at the gate the maximum is a plateau, and ``argmax`` would return
    whichever corner of it comes first in raster order. That is the same reason
    ``src/engine.py`` regresses the heatmap onto the first moment.
    """
    out = fields_for(ANCHORS, CLAUSES)
    where = out.where_raw[0, 0]
    assert at(where, SATISFYING) > 0.9
    for anchor in ANCHORS:
        # An anchor sits on the *surface* of its own clause's pyramid, where the
        # margin is exactly zero and F is 0.5, so the product there cannot be
        # arbitrarily small - only decisively below the region.
        assert at(where, anchor) < 0.5

    centre = centre_of_mass(where)
    assert np.linalg.norm(centre - np.array(SATISFYING, float)) < min(
        np.linalg.norm(centre - np.array(a, float)) for a in ANCHORS
    )


def test_flipping_one_clause_moves_the_high_region_off_the_old_point():
    """§2's other gate: a flip must take the satisfying centroid out of the field."""
    base = fields_for(ANCHORS, CLAUSES)
    flipped = fields_for(ANCHORS, ["inferior", *CLAUSES[1:]])
    assert at(base.where_raw, SATISFYING) > 0.9
    assert at(flipped.where_raw, SATISFYING) < 0.01


def test_an_impossible_conjunction_keeps_a_tiny_peak_and_is_not_renormalised():
    """A sigmoid is never zero, so the product has a peak. It must stay small.

    Dividing by that peak would turn an impossible prompt into a confident map;
    the null head reads ``where_mass`` precisely so that it does not have to.
    """
    # Above one anchor and below another that is already below it: no point in
    # the volume can satisfy both.
    out = fields_for(
        [(18, 12, 12), (6, 12, 12), (12, 18, 12)], ["superior", "inferior", "anterior"]
    )
    assert float(out.where_raw.max()) < 1e-6
    assert float(out.where_mass) < 1e-6

    # The degenerate case is the three-way tie: with all three anchors on one
    # point, every margin is exactly zero there and the product is 1/8. Still far
    # below 1, which is the property that matters - renormalising would make it 1.
    tied = fields_for([(12, 12, 12)] * 3, ["superior", "inferior", "anterior"])
    assert float(tied.where_raw.max()) == pytest.approx(0.125, abs=1e-3)


def test_a_rejected_anchor_zeroes_its_field_and_the_whole_product():
    """``min_mass`` is what a failed Stage A query looks like downstream."""
    masks = torch.zeros(1, 3, *SHAPE)
    masks[0, 0] = torch.from_numpy(ball((12, 12, 12)))
    masks[0, 1] = torch.from_numpy(ball((6, 12, 12)))
    # channel 2 left empty: mass 0, below any positive min_mass
    ids = torch.tensor([[DIRECTIONS.index(d) for d in ("superior", "anterior", "lateral")]])
    out = PositionalMapper3D(tau=0.5, min_mass=1e-6)(masks, ids, SPACING, CENTER)
    assert float(out.masses[0, 2]) == 0.0
    assert float(out.fields[0, 2].abs().max()) == 0.0
    assert float(out.where_raw.abs().max()) == 0.0
    assert float(out.where_mass) == 0.0


def test_a_target_voxel_on_the_near_side_may_score_low():
    """The pyramid is about the *centroid*. Part of the body lies outside it.

    This is why ``where_raw`` is a channel and a weak bias, never a crop: a
    structure centred inside the field still has voxels that are not.
    """
    out = fields_for([(12, 12, 12)], ["superior"])
    field = out.fields[0, 0]
    assert at(field, (16, 12, 12)) > 0.99  # the centroid of a body sitting above
    assert at(field, (13, 12, 12)) > 0.5  # still inside near the apex
    assert at(field, (13, 12, 17)) < 0.5  # ...but its lateral edge is not


# ---------------------------------------------------------------------------
# Invariants of the module itself
# ---------------------------------------------------------------------------
def test_the_mapper_has_no_parameters():
    """§2: "It has no parameters." A learned gain is the shortcut it exists to close."""
    assert list(PositionalMapper3D().parameters()) == []


def test_the_soft_centroid_is_confidence_weighted_not_thresholded():
    """A dim blob still moves the centroid; a 0.5 cut would discard it."""
    masks = torch.zeros(1, 1, *SHAPE)
    masks[0, 0] = torch.from_numpy(ball((12, 12, 6)))
    bright, _ = soft_centroids(masks, SPACING)
    masks[0, 0, 12, 12, 18] = 0.3  # one dim voxel, far to the other side
    dim, _ = soft_centroids(masks, SPACING)
    assert float(dim[0, 0, 0]) > float(bright[0, 0, 0])


def test_mass_is_the_volume_fraction_so_min_mass_is_corpus_independent():
    masks = torch.zeros(1, 1, *SHAPE)
    masks[0, 0, :2, :2, :2] = 1.0
    _, mass = soft_centroids(masks, SPACING)
    assert float(mass[0, 0]) == pytest.approx(8 / SIZE**3)


def test_the_point_form_of_the_margin_matches_the_volume_form():
    """``margin_at`` is what the manifest gate sweeps; it must be the same predicate.

    The gate would otherwise be a gate on a different function from the one the
    carver is handed.
    """
    centroids = torch.tensor([[[5.0, 9.0, 13.0], [17.0, 3.0, 8.0], [11.0, 11.0, 11.0]]])
    ids = torch.tensor([[DIRECTIONS.index(d) for d in ("superior", "medial", "posterior")]])
    volume = margins(centroids, ids, SHAPE, SPACING, CENTER)

    x, y, z = world_axes(SHAPE, SPACING, volume.device, torch.float32)
    probes = torch.tensor(
        [[[float(x[i]), float(y[j]), float(z[k])] for _ in range(3)]
         for i, j, k in [(3, 5, 7), (11, 11, 11), (20, 2, 17), (0, 23, 9)]]
    )
    for row, (i, j, k) in enumerate([(3, 5, 7), (11, 11, 11), (20, 2, 17), (0, 23, 9)]):
        point = margin_at(probes[row][None], centroids, ids, CENTER)[0]
        assert torch.allclose(point, volume[0, :, k, j, i], atol=1e-4)


def test_tau_only_sharpens_it_never_moves_the_boundary():
    """``tau`` is a softness, not a gain: the sign of the margin decides the region."""
    anchors = [(12, 12, 12)]
    sharp = fields_for(anchors, ["superior"], tau=0.1).fields[0, 0]
    soft = fields_for(anchors, ["superior"], tau=4.0).fields[0, 0]
    assert ((sharp > 0.5) == (soft > 0.5)).float().mean() > 0.99
    assert float(sharp.std()) > float(soft.std())


def test_classify_and_margin_agree_on_random_pairs():
    """Corpus truth (:func:`classify`) and the soft field share one predicate.

    Random distinct world points: the direction ``classify`` returns must have
    positive ``margin_at``, and every other direction must be non-positive. That
    is the clause/relation contract the instance scorer inherits.
    """
    rng = np.random.default_rng(0)
    center = np.array(CENTER, float)
    for _ in range(80):
        target = rng.uniform(1.0, SIZE - 2.0, size=3)
        anchor = rng.uniform(1.0, SIZE - 2.0, size=3)
        if np.linalg.norm(target - anchor) < 1.0:
            continue
        try:
            word = classify(target, anchor, center)
        except Exception:
            continue
        ids = torch.tensor([[DIRECTIONS.index(d) for d in DIRECTIONS]])
        points = torch.tensor([[target.tolist()] * len(DIRECTIONS)], dtype=torch.float32)
        centroids = torch.tensor([[anchor.tolist()] * len(DIRECTIONS)], dtype=torch.float32)
        values = margin_at(points, centroids, ids, CENTER)[0]
        winner = DIRECTIONS.index(word)
        assert float(values[winner]) > 0.0
        for i, name in enumerate(DIRECTIONS):
            if i == winner:
                continue
            assert float(values[i]) <= 0.0 + 1e-5, (word, name, float(values[i]))


def test_conjunction_product_matches_solutions_for_on_a_toy_scene():
    """Three hard masks: the unique ``solutions_for`` label is the only soft peak.

    Builds six labelled balls, picks an anchor-first unique conjunction, and
    checks ``where_raw`` is high at that target's centroid and low at every
    other structure's.
    """
    from src.geometry import AmbiguousDirection, solutions_for

    labels = np.zeros(SHAPE, dtype=np.int16)
    centres = {
        1: (6, 12, 12),
        2: (12, 6, 12),
        3: (12, 12, 12),  # near midline; lateral term grows with target x
        4: (18, 18, 22),  # superior to 1, anterior to 2, lateral to 3
        5: (18, 6, 6),
        6: (6, 18, 6),
    }
    for label, c in centres.items():
        ball_mask = ball(c, radius=1.5) > 0.5
        labels[ball_mask] = label

    spacing = SPACING
    center = np.array(CENTER, float)
    from src.geometry import centroids_world

    cents = centroids_world(labels, 6, spacing)
    anchors, directions, target = [1, 2, 3], ["superior", "anterior", "lateral"], 4
    assert solutions_for(anchors, directions, cents, list(centres), center) == [target]

    masks = torch.zeros(1, 3, *SHAPE)
    for slot, label in enumerate(anchors):
        masks[0, slot] = torch.from_numpy((labels == label).astype(np.float32))
    ids = torch.tensor([[DIRECTIONS.index(d) for d in directions]])
    out = PositionalMapper3D(tau=0.5, min_mass=1e-6)(masks, ids, spacing, CENTER)

    to_zyx = lambda xyz: (xyz[2] / spacing[2], xyz[1] / spacing[1], xyz[0] / spacing[0])
    peak = at(out.where_raw[0, 0], to_zyx(cents[target]))
    assert peak > 0.9, peak
    for label, c in centres.items():
        if label == target or label in anchors:
            continue
        assert at(out.where_raw[0, 0], c) < peak
