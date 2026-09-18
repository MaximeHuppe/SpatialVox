"""Spatial relations: the geometric core of the project.

Conventions, used everywhere and never re-derived elsewhere:

* voxel arrays are indexed ``(z, y, x)``, i.e. shape ``(D, H, W)``;
* world coordinates are ordered ``(x, y, z)`` in a RAS frame - ``x`` right /
  lateral, ``y`` anterior, ``z`` superior;
* ``spacing`` is world units per voxel, ordered ``(x, y, z)``.

A relation always describes the TARGET relative to an ANCHOR, from
``delta = centroid(target) - centroid(anchor)``:

1. the main axis is ``argmax(|dx|, |dy|, |dz|)``, ties broken ``z > y > x``;
2. axis ``z``: ``superior`` / ``inferior``;
3. axis ``y``: ``anterior`` / ``posterior``;
4. axis ``x``: the structure farther from the mid-sagittal plane is
   ``lateral``, the closer one ``medial``.

A direction is never invented. Coincident centroids and exact medial/lateral
ties raise :class:`AmbiguousDirection`, and the caller drops that pair.
"""

from __future__ import annotations

import numpy as np

#: The closed direction vocabulary. No synonyms, no ordinals.
DIRECTIONS: tuple[str, ...] = (
    "anterior",
    "posterior",
    "superior",
    "inferior",
    "medial",
    "lateral",
)

OPPOSITE: dict[str, str] = {
    "anterior": "posterior",
    "posterior": "anterior",
    "superior": "inferior",
    "inferior": "superior",
    "medial": "lateral",
    "lateral": "medial",
}

#: World axis each direction pair is decided on.
AXIS_OF: dict[str, str] = {
    "anterior": "y",
    "posterior": "y",
    "superior": "z",
    "inferior": "z",
    "medial": "x",
    "lateral": "x",
}

ATOL = 1e-9


class AmbiguousDirection(ValueError):
    """Raised instead of inventing a direction for an undecidable pair."""


def centroids_world(
    labels: np.ndarray, num_labels: int, spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> np.ndarray:
    """Centroid of every label in one pass over the volume.

    Args:
        labels: ``(D, H, W)`` integer label volume, 0 = background.
        num_labels: highest label id; row ``i`` of the result is label ``i``.
        spacing: world units per voxel, ordered ``(x, y, z)``.

    Returns:
        ``(num_labels + 1, 3)`` world centroids ``(x, y, z)``. Rows of absent
        labels (including row 0) are ``nan``.

    Three ``bincount`` reductions cost O(voxels) regardless of how many
    structures there are, which is what keeps per-item augmentation affordable
    at 128^3 with a large anatomical vocabulary.
    """
    flat = np.asarray(labels).reshape(-1)
    size = int(num_labels) + 1
    counts = np.bincount(flat, minlength=size)[:size].astype(np.float64)
    shape = labels.shape
    means = []
    for axis in range(3):  # array axes are (z, y, x)
        index = np.arange(shape[axis], dtype=np.float64)
        broadcast = index.reshape([-1 if a == axis else 1 for a in range(3)])
        weights = np.broadcast_to(broadcast, shape).reshape(-1)
        totals = np.bincount(flat, weights=weights, minlength=size)[:size]
        means.append(np.divide(totals, counts, out=np.full(size, np.nan), where=counts > 0))
    # means is (z, y, x); emit (x, y, z) in world units.
    return np.stack([means[2] * spacing[0], means[1] * spacing[1], means[0] * spacing[2]], axis=1)


def centroid_world(mask: np.ndarray, spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> np.ndarray:
    """World centroid ``(x, y, z)`` of one binary mask."""
    indices = np.nonzero(mask)
    if indices[0].size == 0:
        raise ValueError("cannot take the centroid of an empty mask")
    z, y, x = (float(idx.mean()) for idx in indices)
    return np.array([x * spacing[0], y * spacing[1], z * spacing[2]])


def bbox_extent_world(mask: np.ndarray, spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> np.ndarray:
    """Inclusive bounding-box extent ``(x, y, z)``; one voxel spans one spacing unit."""
    indices = np.nonzero(mask)
    if indices[0].size == 0:
        raise ValueError("cannot take the bounding box of an empty mask")
    spans = [int(idx.max()) - int(idx.min()) + 1 for idx in indices]  # (z, y, x)
    return np.array([spans[2] * spacing[0], spans[1] * spacing[1], spans[0] * spacing[2]])


def volume_center_world(
    shape: tuple[int, int, int], spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> np.ndarray:
    """World centre ``(x, y, z)`` of a ``(D, H, W)`` volume."""
    depth, height, width = shape
    return np.array(
        [(width - 1) / 2 * spacing[0], (height - 1) / 2 * spacing[1], (depth - 1) / 2 * spacing[2]]
    )


def classify(target: np.ndarray, anchor: np.ndarray, center: np.ndarray) -> str:
    """The direction token for ``target`` relative to ``anchor``.

    All three points are world ``(x, y, z)``; ``center`` fixes the mid-sagittal
    plane the medial/lateral comparison is made against.
    """
    delta = np.asarray(target, float) - np.asarray(anchor, float)
    magnitude = np.abs(delta)
    largest = float(magnitude.max())
    if largest <= ATOL:
        raise AmbiguousDirection("centroids coincide; no direction is defined")
    # Tie priority z > y > x: the first of those axes within tolerance wins.
    axis = next(a for a in (2, 1, 0) if magnitude[a] >= largest - ATOL)
    if axis == 2:
        return "superior" if delta[2] > 0 else "inferior"
    if axis == 1:
        return "anterior" if delta[1] > 0 else "posterior"
    target_offset = abs(float(target[0] - center[0]))
    anchor_offset = abs(float(anchor[0] - center[0]))
    if abs(target_offset - anchor_offset) <= ATOL:
        raise AmbiguousDirection("target and anchor are equidistant from the mid-sagittal plane")
    return "lateral" if target_offset > anchor_offset else "medial"


def select_anchors(
    target: int,
    centroids: np.ndarray,
    present: list[int],
    center: np.ndarray,
    n_anchors: int = 3,
) -> list[tuple[int, str]] | None:
    """Nearest-feasible anchor selection for one target.

    Candidates are ranked by centroid distance (label id as tie-break) and taken
    in order, keeping a candidate only when its direction has not been used yet.
    The result is the nearest *feasible* set - not necessarily the nearest
    structures - and its order is shared by the prompt clauses and the mask
    channels.

    Returns ``None`` when no set of ``n_anchors`` distinct directions exists, in
    which case the caller drops this target.
    """
    ranked = sorted(
        (label for label in present if label != target),
        key=lambda label: (float(np.linalg.norm(centroids[label] - centroids[target])), label),
    )
    chosen: list[tuple[int, str]] = []
    used: set[str] = set()
    for label in ranked:
        try:
            direction = classify(centroids[target], centroids[label], center)
        except AmbiguousDirection:
            continue
        if direction in used:
            continue
        used.add(direction)
        chosen.append((label, direction))
        if len(chosen) == n_anchors:
            return chosen
    return None
