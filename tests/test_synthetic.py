"""The generator, and the one property of it that decides what a run can claim.

A synthetic corpus is only worth training on if it withholds the shortcuts the
architecture is supposed to do without. Two of those live in the appearance, and
both were measured the hard way:

* **a global intensity threshold.** On the shipped easy appearance one cut
  recovers every structure at IoU 0.994, so ``B(I)`` gets every boundary for free,
  the carver never has to learn a class, and transfer to an unsupervised class is
  perfect and meaningless - 0.9490 supervised against 0.9504 held out.
* **brightness as a class label.** If each class has its own bright band, "the
  bright blob is the capsule" is available and the prompt is not needed.

``class_spread`` and the background knobs are what remove them, and these tests
pin the removal rather than the numbers, which belong to a config.
"""

from __future__ import annotations

import numpy as np

from src.synthetic import (
    SHAPES, SHAPE_NAMES, bias_field, class_range, generate_scene, pack_scene,
)

#: The real resolution, deliberately. ``texture_scale`` is in **voxels**, so a
#: smaller test cube is not a smaller version of this corpus - at 32^3 a scale of
#: 12 spans 38% of the volume against 19% here, and the local contrast these
#: tests are about drops by a factor of three. The appearance has to be checked
#: at the size it will be generated at.
SHAPE = (64, 64, 64)
SPACING = (1.0, 1.0, 1.0)
SIZE = 1.8
EASY = dict(background=[0.12, 0.04], structure=[0.45, 0.75], class_spread=0.06, noise=0.03, blur=0.6)
HARD = dict(
    background=[0.45, 0.04], structure=[0.45, 0.75], class_spread="shared",
    noise=0.035, blur=0.6, texture=0.90, texture_scale=12.0, bias_field=0.60,
)


def threshold_iou(image: np.ndarray, labels: np.ndarray) -> float:
    """The best a single global cut can do against ``labels > 0``."""
    mask = labels > 0
    return max(
        float((image > cut)[mask].sum() / max(((image > cut) | mask).sum(), 1))
        for cut in np.arange(0.05, 0.95, 0.02)
    )


# ---------------------------------------------------------------------------
# Per-class intensity: coherent across the corpus, or shared by every class
# ---------------------------------------------------------------------------
def test_a_class_keeps_its_intensity_range_across_the_whole_corpus():
    """A cube is cube-bright in every scene, which is how tissue behaves."""
    a = class_range(1, 10, 0.45, 0.75, 0.10)
    assert class_range(1, 10, 0.45, 0.75, 0.10) == a
    assert a != class_range(2, 10, 0.45, 0.75, 0.10)
    low, high = a
    assert 0.45 - 0.05 <= low < high <= 0.75 + 0.05


def test_class_ranges_overlap_so_brightness_never_fixes_an_identity():
    """Spacing between centres is (high - low) / n; a wider spread must overlap."""
    ranges = [class_range(k, 10, 0.45, 0.75, 0.10) for k in range(1, 11)]
    pairs = [
        (i, j) for i in range(10) for j in range(i + 1, 10)
        if ranges[i][0] < ranges[j][1] and ranges[j][0] < ranges[i][1]
    ]
    assert len(pairs) > 20, "a spread of 0.10 over centres 0.030 apart must overlap widely"


def test_shared_spread_gives_every_class_the_same_range():
    """§4's mechanism: a held-out shape must look exactly like a trained one.

    With one band for all ten classes, brightness carries no class information,
    so the relational prompt is the only way to pick the target.
    """
    assert {class_range(k, 10, 0.45, 0.75, None) for k in range(1, 11)} == {(0.45, 0.75)}


def test_shared_intensity_really_does_erase_the_class_signal():
    """Measured on volumes, not on the formula: class means must not separate."""
    per_class = {k: [] for k in range(1, len(SHAPES) + 1)}
    for seed in range(3):
        image, labels = generate_scene(seed, SHAPE, SPACING, 2, HARD, size=SIZE)
        for k in per_class:
            mask = labels == k
            if mask.sum() > 20:
                per_class[k].append(image[mask].mean())
    means = [np.mean(v) for v in per_class.values() if v]
    spread_between = float(np.std(means))
    spread_within = float(np.mean([np.std(v) for v in per_class.values() if len(v) > 1]))
    assert spread_between <= 2.0 * spread_within, (
        f"class means vary by {spread_between:.3f} against {spread_within:.3f} within a class, "
        "so brightness still narrows the identity down"
    )


# ---------------------------------------------------------------------------
# The global threshold, which is the shortcut that matters
# ---------------------------------------------------------------------------
def test_the_easy_appearance_is_trivially_separable_and_says_so():
    """Kept as a *negative* reference: this is the corpus a result cannot come from."""
    image, labels = generate_scene(0, SHAPE, SPACING, 2, EASY)
    assert threshold_iou(image, labels) > 0.9


def test_the_hard_appearance_defeats_any_single_threshold():
    image, labels = generate_scene(0, SHAPE, SPACING, 2, HARD, size=SIZE)
    assert threshold_iou(image, labels) < 0.6


def local_contrast(image: np.ndarray, labels: np.ndarray, label: int) -> float:
    """One structure's mean against the background in the box just around it.

    *Local*, deliberately. The global difference is the thing this corpus exists
    to destroy, so measuring visibility with it would report the opposite of what
    is wanted.
    """
    where = np.argwhere(labels == label)
    low, high = where.min(0), where.max(0) + 1
    box = tuple(slice(max(l - 4, 0), h + 4) for l, h in zip(low, high))
    patch, mask = image[box], labels[box] == label
    return float(patch[mask].mean() - patch[~mask].mean())


def test_the_hard_appearance_keeps_the_structures_visible():
    """The other half of the requirement, and the one an earlier attempt failed.

    Pushing the noise up and the background texture down to a structure's own
    scale made a corpus whose shapes a person could not see, and Stage A stalled
    at Dice 0.27 where it reaches 0.88 here.

    **Threshold IoU does not catch that**, which is the whole reason this test
    exists: the rejected appearance scored 0.192 and the shipped one 0.219 - all
    but identical. What separates them is local contrast, 0.4x the noise against
    2.2x. A corpus can fail by being unreadable as easily as by being trivial,
    and only one of the two shows up in the separability number.
    """
    image, labels = generate_scene(0, SHAPE, SPACING, 2, HARD, size=SIZE)
    contrasts = [
        local_contrast(image, labels, k)
        for k in range(1, len(SHAPES) + 1) if (labels == k).sum() > 20
    ]
    assert contrasts
    assert float(np.median(contrasts)) > 1.5 * float(HARD["noise"])


def test_the_background_varies_across_the_volume_rather_than_being_flat():
    """What kills the global threshold is a slow background, not a noisy one.

    Measured as the spread of BLOCK means over the volume, not the difference
    between two halves. Two halves test one arbitrary direction of a random
    field, and a field can swing hard while its two half-means happen to
    coincide: that statistic held for only 7 seeds in 10, so it failed the first
    time a change upstream shifted which realisation seed 0 draws. The block
    spread measures the same property - slow, spatially structured variation -
    and stayed in 0.081-0.149 across every seed tried.
    """
    for seed in (0, 1, 2):
        image, labels = generate_scene(seed, SHAPE, SPACING, 2, HARD, size=SIZE)
        background = np.where(labels == 0, image, np.nan)
        blocks = background.reshape(4, SHAPE[0] // 4, 4, SHAPE[1] // 4, 4, SHAPE[2] // 4)
        assert float(np.nanstd(np.nanmean(blocks, axis=(1, 3, 5)))) > 2 * float(HARD["noise"])


def test_the_bias_field_is_smooth_and_centred_on_one():
    field = bias_field((32, 32, 32), np.random.default_rng(0), 0.4)
    assert field.shape == (32, 32, 32)
    assert 0.5 < field.mean() < 1.5
    step = float(abs(np.diff(field, axis=0)).max())
    assert step < 0.25 * (field.max() - field.min()), "a bias field must vary slowly"
    assert np.array_equal(bias_field((8, 8, 8), np.random.default_rng(0), 0.0), np.ones((8, 8, 8)))


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def test_size_scales_every_class_without_spreading_them_apart():
    """A size that identifies a class is the same shortcut as a brightness that does.

    It is also the one the MRI carver took - it learned a class-conditional size
    prior and, on a class it had never been supervised on, predicted 93 voxels
    where 2219 belonged. So ``size`` must scale, not spread.
    """
    volumes = {}
    for size in (1.0, 1.8):
        counts = {k: [] for k in range(1, len(SHAPES) + 1)}
        for seed in range(2):
            labels = pack_scene(np.random.default_rng(seed), SHAPE, SPACING, 2, size=size)
            for k in counts:
                counts[k].append(int((labels == k).sum()))
        volumes[size] = [np.mean(v) for v in counts.values()]
    assert np.mean(volumes[1.8]) > 2 * np.mean(volumes[1.0])
    ratio = lambda v: max(v) / max(min(v), 1)
    assert ratio(volumes[1.8]) < 2 * ratio(volumes[1.0]), "size must not widen the class spread"


def test_a_scene_holds_one_of_every_primitive_and_they_never_overlap():
    labels = pack_scene(np.random.default_rng(0), SHAPE, SPACING, 2)
    assert set(np.unique(labels)) == set(range(len(SHAPE_NAMES) + 1))


def test_a_scene_is_reproducible_from_its_seed():
    first = generate_scene(7, SHAPE, SPACING, 2, HARD, size=SIZE)
    second = generate_scene(7, SHAPE, SPACING, 2, HARD, size=SIZE)
    assert np.array_equal(first[0], second[0]) and np.array_equal(first[1], second[1])
    assert not np.array_equal(generate_scene(8, SHAPE, SPACING, 2, HARD, size=SIZE)[1], first[1])
