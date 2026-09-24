"""Instance selection: seed-flood proposals scored by ``where_raw(centroid)``."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.instance import (
    local_maxima_seeds,
    oracle_label_instances,
    propose_seed_flood,
    region_mask,
    run_instance,
    score_proposals,
    select_instance,
)
from src.models import StageB

RESOLUTION = 24
SPACING = (1.0, 1.0, 1.0)
SEGMENTER = dict(
    vocab_size=6, resolution=RESOLUTION, encoder_channels=(8, 16, 32),
    token_dim=32, num_heads=2, bottleneck=6, prior_foreground=0.01,
    deep_supervision=(0.3, 0.7),
)


def test_region_mask_dilates_the_high_where_support():
    where = torch.zeros(1, 8, 8, 8)
    where[0, 4, 4, 4] = 1.0
    region = region_mask(where, threshold=0.5, radius=1)[0, 0]
    assert bool(region[4, 4, 4])
    assert bool(region[4, 4, 5])
    assert not bool(region[0, 0, 0])


def test_local_maxima_prefer_the_highest_peak_inside_the_region():
    where = torch.zeros(16, 16, 16)
    where[4, 4, 4] = 0.6
    where[10, 10, 10] = 0.95
    region = torch.ones_like(where)
    seeds = local_maxima_seeds(where, region, max_seeds=2)
    assert seeds[0] == (10, 10, 10)


def test_seed_flood_grows_a_homogeneous_blob_and_stops_at_contrast():
    image = torch.zeros(1, 20, 20, 20)
    image[0, 5:12, 5:12, 5:12] = 1.0
    image[0, 5:12, 5:12, 14:18] = 0.0  # dark neighbour
    where = torch.zeros(1, 20, 20, 20)
    where[0, 4:13, 4:13, 4:16] = 0.9
    proposals, region = propose_seed_flood(
        image, where, dilate_radius=0, max_seeds=8, intensity_tol=0.2,
        tol_mode="absolute", min_voxels=4,
    )
    assert proposals.shape[0] >= 1
    # At least one proposal covers the bright cube without invading the dark neighbour.
    good = [
        p for p in proposals
        if float(p[8, 8, 8]) == 1.0 and float(p[8, 8, 15]) == 0.0
    ]
    assert good, f"K={proposals.shape[0]} none matched bright-only"


def test_rule_scorer_picks_the_proposal_whose_centroid_sits_in_where():
    proposals = torch.zeros(2, 16, 16, 16)
    proposals[0, 2:5, 2:5, 2:5] = 1.0
    proposals[1, 10:13, 10:13, 10:13] = 1.0
    where = torch.zeros(16, 16, 16)
    where[10:13, 10:13, 10:13] = 1.0
    scores, _ = score_proposals(proposals, where, SPACING)
    winner, mask = select_instance(proposals, scores, score_null=0.1)
    assert winner == 1
    assert float(mask[11, 11, 11]) == 1.0


def test_score_null_emits_an_empty_mask():
    proposals = torch.ones(1, 8, 8, 8)
    scores = torch.tensor([0.1])
    winner, mask = select_instance(proposals, scores, score_null=0.5)
    assert winner == -1
    assert float(mask.sum()) == 0.0


def test_oracle_label_pick_matches_the_design_ceiling_rule():
    labels = np.zeros((16, 16, 16), dtype=np.int16)
    labels[2:5, 2:5, 2:5] = 1
    labels[10:13, 10:13, 10:13] = 2
    where = torch.zeros(16, 16, 16)
    where[10:13, 10:13, 10:13] = 0.99
    where[2:5, 2:5, 2:5] = 0.2
    label, score, mask = oracle_label_instances(labels, where, SPACING)
    assert label == 2
    assert score > 0.9
    assert float(mask[11, 11, 11]) == 1.0


def test_stage_b_instance_mode_has_no_carver_and_names_still_stop_at_stage_a():
    torch.manual_seed(0)
    model = StageB(
        SEGMENTER, spacing=SPACING, answer_mode="instance",
        boundary_widths=(4, 8), dilate_radius=1, max_seeds=4, score_null=0.01,
    ).eval()
    assert model.carver is None
    assert model.null is None
    image = torch.randn(1, 1, *(RESOLUTION,) * 3)
    # Soft anchor blobs that make a superior / anterior / lateral conjunction around a bright blob.
    anchors = torch.zeros(1, 3, *(RESOLUTION,) * 3)
    anchors[0, 0, 4, 12, 12] = 1.0
    anchors[0, 1, 12, 4, 12] = 1.0
    anchors[0, 2, 12, 12, 4] = 1.0
    image[0, 0, 16:20, 16:20, 16:20] = 2.0
    directions = torch.tensor([[2, 0, 5]])  # superior, anterior, lateral
    names = torch.zeros(1, 3, dtype=torch.long)
    with torch.no_grad():
        out = model(image, directions, names, anchors=anchors)
    assert out.logits.shape == (1, 1, *(RESOLUTION,) * 3)
    assert out.where_raw.shape == (1, 1, *(RESOLUTION,) * 3)
    # Renaming the Stage A vocabulary ids must not change the answer when anchors are given.
    with torch.no_grad():
        again = model(image, directions, names + 1, anchors=anchors)
    assert torch.equal(out.logits, again.logits)


def test_barrier_flood_stops_at_a_wall():
    image = torch.zeros(1, 16, 16, 16)
    image[0, 4:12, 4:12, 4:12] = 1.0
    where = torch.ones(1, 16, 16, 16) * 0.9
    barrier = torch.zeros(16, 16, 16)
    barrier[:, :, 8] = 1.0  # wall separating left/right
    proposals, _ = propose_seed_flood(
        image, where, barrier=barrier, dilate_radius=0, max_seeds=4,
        barrier_tol=0.35, min_voxels=4,
    )
    assert proposals.shape[0] >= 1
    # A body seeded on the left must not cross x=8.
    left = [p for p in proposals if float(p[8, 8, 5]) == 1.0]
    assert left and all(float(p[8, 8, 10]) == 0.0 for p in left)
    """Cosine flood follows a constant feature body and stops at a different code."""
    image = torch.zeros(1, 16, 16, 16)
    image[0, 4:10, 4:10, 4:10] = 1.0
    where = torch.zeros(1, 16, 16, 16)
    where[0, 3:11, 3:11, 3:11] = 0.9
    features = torch.zeros(4, 16, 16, 16)
    features[0, 4:10, 4:10, 4:10] = 1.0
    features[1, 4:10, 4:10, 11:14] = 1.0  # different code next door
    proposals, _ = propose_seed_flood(
        image, where, features=features, dilate_radius=0, max_seeds=4,
        feature_tol=0.2, min_voxels=4,
    )
    assert proposals.shape[0] >= 1
    good = [p for p in proposals if float(p[6, 6, 6]) == 1.0 and float(p[6, 6, 12]) == 0.0]
    assert good


def test_run_instance_end_to_end_on_a_toy_conjunction():
    image = torch.zeros(1, RESOLUTION, RESOLUTION, RESOLUTION)
    image[0, 16:20, 16:20, 16:20] = 1.0
    # Put a peak of where_raw on the bright blob so seeding+scoring prefer it.
    where = torch.zeros(1, RESOLUTION, RESOLUTION, RESOLUTION)
    where[0, 14:22, 14:22, 14:22] = 0.95
    result = run_instance(
        image, where, SPACING,
        dilate_radius=1, max_seeds=8, intensity_tol=0.25,
        tol_mode="absolute", min_voxels=4, score_null=0.1,
    )
    assert result.winner >= 0
    assert float(result.mask[17, 17, 17]) == 1.0
