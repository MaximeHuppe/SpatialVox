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

from typing import Sequence

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


def _distinct_directions(
    candidates: Sequence[tuple[int, str]], n_anchors: int
) -> list[tuple[int, str]] | None:
    """Take candidates in the order given, keeping each new direction.

    No two anchors may share a direction: two clauses naming the same side would
    not narrow the conjunction, and the prompt would under-determine the target.
    """
    chosen: list[tuple[int, str]] = []
    used: set[str] = set()
    for label, direction in candidates:
        if direction in used:
            continue
        used.add(direction)
        chosen.append((label, direction))
        if len(chosen) == n_anchors:
            return chosen
    return None


def select_anchors(
    target: int,
    centroids: np.ndarray,
    present: list[int],
    center: np.ndarray,
    n_anchors: int = 3,
    *,
    pool: int | None = None,
    rng: np.random.Generator | None = None,
    shuffle: bool = True,
    fallbacks: list[int] | None = None,
) -> list[tuple[int, str]] | None:
    """Anchor selection for one target, with pairwise-distinct directions.

    Candidates are ranked by centroid distance (label id as tie-break). ``pool``
    is how many of those nearest candidates are eligible; ``n_anchors`` are then
    drawn from that window. ``pool <= n_anchors`` (the default) reproduces the
    deterministic nearest-feasible set. A wider pool trades realism - a reader
    names *nearby* landmarks - for a target that is no longer simply "the
    structure closest to its anchors", which is the shortcut a prompt-blind
    baseline exploits.

    **The returned order is randomised whenever ``rng`` is given**, and it is the
    order the prompt clauses and the anchor mask channels both take. Ranking by
    distance and then storing that ranking made the slot index a perfect proxy
    for "how close is this anchor" - a leak the model can read without parsing a
    single direction word. Order must carry no information.

    Feasibility never depends on ``pool``: if the window holds no set of
    ``n_anchors`` distinct directions, the scan falls back to the full ranking,
    so widening the pool never drops a target that a narrower one would keep.

    Returns ``None`` when no such set exists anywhere, in which case the caller
    drops this target.
    """
    ranked = sorted(
        (label for label in present if label != target),
        key=lambda label: (float(np.linalg.norm(centroids[label] - centroids[target])), label),
    )
    candidates: list[tuple[int, str]] = []
    for label in ranked:
        try:
            candidates.append((label, classify(centroids[target], centroids[label], center)))
        except AmbiguousDirection:  # never invent a direction; drop the pair
            continue

    width = max(int(n_anchors if pool is None else pool), int(n_anchors))
    chosen = None
    if rng is not None and width > n_anchors:
        window = list(candidates[:width])
        rng.shuffle(window)  # which of the k nearest, not just the nearest
        chosen = _distinct_directions(window, n_anchors)
    if chosen is None:
        # The pool window held no feasible triple, so this example silently
        # reverts to the deterministic nearest set. Counted, because a corpus
        # that is part-deterministic without saying so is a leak in waiting.
        if fallbacks is not None and width > n_anchors:
            fallbacks.append(1)
        chosen = _distinct_directions(candidates, n_anchors)
    if chosen is not None and rng is not None and shuffle:
        rng.shuffle(chosen)  # slot order must not encode distance rank
    return chosen


def directions_for(
    target: int,
    anchors: Sequence[int],
    centroids: np.ndarray,
    center: np.ndarray,
) -> list[str] | None:
    """The clause directions describing ``target`` against each of ``anchors``.

    ``None`` when any pair is undecidable, or when two clauses would name the
    same side - two identical directions do not narrow the conjunction, which is
    the ``_distinct_directions`` rule applied to a triple that is already fixed.
    """
    directions: list[str] = []
    for anchor in anchors:
        try:
            directions.append(classify(centroids[target], centroids[anchor], center))
        except AmbiguousDirection:
            return None
    return directions if len(set(directions)) == len(directions) else None


def solutions_for(
    anchors: Sequence[int],
    directions: Sequence[str],
    centroids: np.ndarray,
    present: Sequence[int],
    center: np.ndarray,
) -> list[int]:
    """Every present non-anchor structure that satisfies *all* the clauses.

    This is the relational conjunction read literally. A prompt is well posed
    exactly when this returns one label; anything else and the sentence
    describes more than one structure, or none.
    """
    solutions = []
    for label in present:
        if label in anchors:
            continue
        for anchor, direction in zip(anchors, directions):
            try:
                if classify(centroids[label], centroids[anchor], center) != direction:
                    break
            except AmbiguousDirection:
                break
        else:
            solutions.append(int(label))
    return solutions


def direction_matrix(
    centroids: np.ndarray, present: Sequence[int], center: np.ndarray
) -> tuple[np.ndarray, dict[int, int]]:
    """Direction code of every ordered (target, anchor) pair among ``present``.

    ``classify`` is pure Python, and anchor-first generation asks the same
    question about the same pairs thousands of times per scene. Answering each
    pair once turns ~360k calls per scene into ~500, which is the difference
    between a manifest rebuild taking minutes and taking hours.

    Returns ``(codes, index)`` where ``codes[i, j]`` is the index into
    :data:`DIRECTIONS` describing ``present[i]`` relative to ``present[j]``, or
    ``-1`` when the pair is undecidable, and ``index`` maps a label to its row.
    """
    labels = [int(l) for l in present]
    index = {label: i for i, label in enumerate(labels)}
    codes = np.full((len(labels), len(labels)), -1, dtype=np.int8)
    for i, target in enumerate(labels):
        for j, anchor in enumerate(labels):
            if i == j:
                continue
            try:
                codes[i, j] = DIRECTIONS.index(
                    classify(centroids[target], centroids[anchor], center)
                )
            except AmbiguousDirection:
                continue
    return codes, index


def anchor_first_examples(
    centroids: np.ndarray,
    present: Sequence[int],
    center: np.ndarray,
    n_anchors: int,
    *,
    triples: int,
    locality: int,
    rng: np.random.Generator,
    shuffle: bool = True,
) -> list[tuple[list[int], list[str], int]]:
    """Anchor-first generation: fix a landmark set, then ask what it determines.

    The target-first generator picks the anchors *nearest the target*, which
    makes the anchor identities a near-perfect name tag for the target - on fixed
    anatomy the three nearest neighbours of a structure are the same in every
    subject. Measured on ``data/mri``: the unordered anchor set alone recovers
    the target in 98.9% of examples, while solving the conjunction is right 94.5%
    of the time, so ignoring the prompt strictly beats reading it. No pool width
    inverts that; the selection rule itself is the leak.

    Here the triple comes first and the targets follow, so one anchor set serves
    several targets and only the direction words tell them apart. Examples whose
    conjunction is *not* unique are dropped, which makes well-posedness hold by
    construction rather than by luck. Measured: P(target | anchors) falls from
    98.9% to 42.3%, and the prompt-blind floor from 0.800 to 0.205.

    ``locality`` keeps the prompts sayable: the triple is drawn from the
    ``locality`` structures nearest a randomly chosen seed, not from the whole
    volume, so clauses name landmarks a reader would actually name together.

    Returns ``(anchors, directions, target)``. Anchor order is shuffled when
    ``shuffle`` - slot order must carry no information (see
    :func:`select_anchors`).
    """
    labels = [int(l) for l in present]
    if len(labels) <= n_anchors:
        return []
    codes, index = direction_matrix(centroids, labels, center)
    positions = np.array([centroids[l] for l in labels], dtype=float)

    out: list[tuple[list[int], list[str], int]] = []
    seen: set[tuple[int, ...]] = set()
    for _ in range(int(triples)):
        seed = int(rng.integers(len(labels)))
        order = np.argsort(np.linalg.norm(positions - positions[seed], axis=1))
        window = order[: max(int(locality), n_anchors)]
        if len(window) < n_anchors:
            continue
        columns = np.sort(rng.choice(window, n_anchors, replace=False))
        key = tuple(int(c) for c in columns)
        if key in seen:
            continue
        seen.add(key)
        # [K, n_anchors]: how every structure sits relative to this triple.
        relative = codes[:, columns]
        for row in range(len(labels)):
            if row in columns:
                continue
            wanted = relative[row]
            if (wanted < 0).any() or len(set(wanted.tolist())) != n_anchors:
                continue  # undecidable pair, or two clauses naming the same side
            matches = (relative == wanted).all(axis=1)
            matches[columns] = False
            if int(matches.sum()) != 1:
                continue  # the sentence does not pick out exactly one structure
            clauses = [
                (labels[int(c)], DIRECTIONS[int(d)]) for c, d in zip(columns, wanted)
            ]
            if shuffle:
                rng.shuffle(clauses)
            out.append(
                ([a for a, _ in clauses], [d for _, d in clauses], labels[row])
            )
    return out
