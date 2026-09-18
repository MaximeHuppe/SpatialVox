"""The synthetic corpus: ten axis-aligned primitives in an MRI-like image.

This module produces exactly what a real dataset produces - an intensity volume
and an integer label volume - and nothing else. Everything downstream (relations,
prompts, manifests, training) is written against that pair, so swapping this file
for real MRI changes no other code. See ``docs/method/08_scaling.md``.

A scene packs one instance of every primitive by rejection sampling: draw size
and centre, voxelise, reject if empty, out of bounds or overlapping, retry. Sizes
are written for a 64-voxel axis and rescale linearly with the requested
resolution, so a 128^3 corpus holds the same structures at twice the sampling.

Intensities are painted afterwards: noisy background, per-structure means drawn
from one shared, overlapping range, intra-structure noise, and a light blur that
makes edges partial-volume rather than a 0/1 cut. Because the means overlap,
brightness identifies a structure no better than chance and the segmenter has to
use shape and position - the situation it will face on real MRI. Geometry stays
exact in the labels.

Each entry of :data:`SHAPES` is ``(parameter ranges, solid, half extent)``:
``solid(d, p)`` is the membership predicate on world offsets ``d = (dx, dy, dz)``
from the structure centre, and ``half_extent(p)`` is half its analytic bounding
box ``(x, y, z)``, which is what keeps a sampled centre in bounds.
"""

from __future__ import annotations

from typing import Callable, Mapping

import numpy as np

REFERENCE_AXIS = 64.0  # the axis length the size ranges below are written for

SHAPES: dict[str, tuple[dict[str, tuple[float, float]], Callable, Callable]] = {
    "cube": (
        {"side": (6.0, 10.0)},
        lambda d, p: (abs(d[0]) <= p["side"] / 2) & (abs(d[1]) <= p["side"] / 2) & (abs(d[2]) <= p["side"] / 2),
        lambda p: (p["side"] / 2,) * 3,
    ),
    "cuboid": (
        {"x": (5.2, 11.0), "y": (5.2, 11.0), "z": (5.2, 11.0)},
        lambda d, p: (abs(d[0]) <= p["x"] / 2) & (abs(d[1]) <= p["y"] / 2) & (abs(d[2]) <= p["z"] / 2),
        lambda p: (p["x"] / 2, p["y"] / 2, p["z"] / 2),
    ),
    "sphere": (
        {"radius": (4.0, 6.0)},
        lambda d, p: d[0] ** 2 + d[1] ** 2 + d[2] ** 2 <= p["radius"] ** 2,
        lambda p: (p["radius"],) * 3,
    ),
    "ellipsoid": (
        {"x": (3.5, 6.5), "y": (3.5, 6.5), "z": (3.5, 6.5)},
        lambda d, p: (d[0] / p["x"]) ** 2 + (d[1] / p["y"]) ** 2 + (d[2] / p["z"]) ** 2 <= 1.0,
        lambda p: (p["x"], p["y"], p["z"]),
    ),
    "cylinder": (  # circular section in x-y, extruded along z
        {"radius": (3.0, 5.0), "height": (7.0, 13.0)},
        lambda d, p: (d[0] ** 2 + d[1] ** 2 <= p["radius"] ** 2) & (abs(d[2]) <= p["height"] / 2),
        lambda p: (p["radius"], p["radius"], p["height"] / 2),
    ),
    "cone": (  # circular base at low z, apex at high z
        {"radius": (4.5, 6.5), "height": (10.0, 14.0)},
        lambda d, p: (abs(d[2]) <= p["height"] / 2)
        & (d[0] ** 2 + d[1] ** 2 <= (p["radius"] * _taper(d[2], p["height"])) ** 2),
        lambda p: (p["radius"], p["radius"], p["height"] / 2),
    ),
    "pyramid": (  # square base at low z, apex at high z
        {"side": (8.0, 12.0), "height": (10.0, 14.0)},
        lambda d, p: (abs(d[2]) <= p["height"] / 2)
        & (np.maximum(abs(d[0]), abs(d[1])) <= p["side"] / 2 * _taper(d[2], p["height"])),
        lambda p: (p["side"] / 2, p["side"] / 2, p["height"] / 2),
    ),
    "triangular_prism": (  # isoceles triangle in x-z, extruded along y
        {"width": (8.0, 12.0), "height": (8.0, 12.0), "depth": (7.0, 11.0)},
        lambda d, p: (abs(d[2]) <= p["height"] / 2)
        & (abs(d[0]) <= p["width"] / 2 * _taper(d[2], p["height"]))
        & (abs(d[1]) <= p["depth"] / 2),
        lambda p: (p["width"] / 2, p["depth"] / 2, p["height"] / 2),
    ),
    "torus": (  # ring in x-y, hole along z
        {"major": (3.0, 4.5), "minor": (2.0, 2.5)},
        lambda d, p: (np.sqrt(d[0] ** 2 + d[1] ** 2) - p["major"]) ** 2 + d[2] ** 2 <= p["minor"] ** 2,
        lambda p: (p["major"] + p["minor"], p["major"] + p["minor"], p["minor"]),
    ),
    "capsule": (  # cylinder segment along z with hemispherical caps
        {"radius": (3.5, 4.5), "segment": (3.0, 5.0)},
        lambda d, p: d[0] ** 2 + d[1] ** 2 + (d[2] - np.clip(d[2], -p["segment"] / 2, p["segment"] / 2)) ** 2
        <= p["radius"] ** 2,
        lambda p: (p["radius"], p["radius"], p["segment"] / 2 + p["radius"]),
    ),
}

SHAPE_NAMES: tuple[str, ...] = tuple(SHAPES)

#: Classes that must stay anisotropic or they degenerate into another class
#: (a cuboid that is a cube, an ellipsoid that is a sphere).
ANISOTROPIC = {"cuboid": 1.35, "ellipsoid": 1.35}


def _taper(dz, height):
    """Linear taper: 1 at the base (``dz = -h/2``), 0 at the apex (``dz = +h/2``)."""
    return 0.5 - dz / height


def _offsets(shape, spacing, center) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Broadcastable world offsets ``(dx, dy, dz)`` of every voxel from ``center``."""
    depth, height, width = shape
    return (
        (np.arange(width) * spacing[0] - center[0]).reshape(1, 1, -1),
        (np.arange(height) * spacing[1] - center[1]).reshape(1, -1, 1),
        (np.arange(depth) * spacing[2] - center[2]).reshape(-1, 1, 1),
    )


def voxelize(name: str, params: Mapping[str, float], center, shape, spacing) -> np.ndarray:
    """Rasterise one primitive: a voxel belongs to it when its centre is inside."""
    solid = SHAPES[name][1](_offsets(shape, spacing, center), params)
    return np.broadcast_to(solid, tuple(shape))


def half_extent(name: str, params: Mapping[str, float]) -> np.ndarray:
    """Half of the analytic bounding box, in world units ``(x, y, z)``."""
    return np.asarray(SHAPES[name][2](params), dtype=float)


def draw_params(name: str, rng: np.random.Generator, scale: float = 1.0) -> dict[str, float]:
    """Draw one size parameter set, honouring the per-class degeneracy rules."""
    for _ in range(64):
        params = {key: float(rng.uniform(*bounds)) * scale for key, bounds in SHAPES[name][0].items()}
        if name == "torus" and params["minor"] >= params["major"]:
            continue
        ratio = ANISOTROPIC.get(name)
        if ratio is not None and max(params.values()) / min(params.values()) < ratio:
            continue
        return params
    raise RuntimeError(f"could not draw valid parameters for {name!r}")


def _mean_params(name: str) -> dict[str, float]:
    return {key: (low + high) / 2 for key, (low, high) in SHAPES[name][0].items()}


#: Bulky classes first: placing them early keeps rejection sampling cheap.
PLACEMENT_ORDER: tuple[str, ...] = tuple(
    sorted(SHAPE_NAMES, key=lambda name: -float(np.prod(half_extent(name, _mean_params(name)))))
)


def pack_scene(rng: np.random.Generator, shape, spacing, margin: int, attempts: int = 200) -> np.ndarray:
    """Rejection-sample one instance of every primitive into a ``uint8`` label volume.

    Raises when an object cannot be placed within the attempt budget; the caller
    retries the whole scene with a fresh draw.
    """
    labels = np.zeros(tuple(shape), dtype=np.uint8)
    occupied = np.zeros(tuple(shape), dtype=bool)
    spacing = np.asarray(spacing, dtype=float)
    scale = min(shape) / REFERENCE_AXIS
    last_center = (np.array([shape[2], shape[1], shape[0]]) - 1) * spacing  # world (x, y, z)

    for name in PLACEMENT_ORDER:
        for _ in range(attempts):
            params = draw_params(name, rng, scale)
            half = half_extent(name, params)
            low, high = margin * spacing + half, last_center - margin * spacing - half
            if np.any(low > high):
                continue  # too big for this grid: draw a smaller one
            candidate = voxelize(name, params, rng.uniform(low, high), shape, spacing)
            if not candidate.any() or (candidate & occupied).any():
                continue
            labels[candidate] = SHAPE_NAMES.index(name) + 1
            occupied |= candidate
            break
        else:
            raise RuntimeError(f"could not place {name!r} in {attempts} attempts")
    return labels


def gaussian_blur(volume: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur by shifted sums; ``sigma <= 0`` is a no-op."""
    if sigma <= 0:
        return volume.astype(np.float32, copy=False)
    radius = max(1, round(3 * sigma))
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    for axis in range(3):
        padding = [(radius, radius) if a == axis else (0, 0) for a in range(3)]
        padded = np.pad(volume, padding, mode="edge")
        blurred = np.zeros_like(volume, dtype=np.float32)
        window = [slice(None)] * 3
        for offset, weight in enumerate(kernel):
            window[axis] = slice(offset, offset + volume.shape[axis])
            blurred += float(weight) * padded[tuple(window)]
        volume = blurred
    return volume


def class_bias(label: int, amplitude: float, n_classes: int = len(SHAPES)) -> float:
    """A small deterministic per-class offset, roughly in ``[-amplitude, amplitude]``.

    T1-like: brightness carries a hint of identity, but the per-structure jitter
    is comparable to it and the ranges overlap, so it is never a lookup table.
    """
    if amplitude == 0:
        return 0.0
    return float(amplitude * (label - (n_classes + 1) / 2) / ((n_classes - 1) / 2))


def paint(labels: np.ndarray, rng: np.random.Generator, appearance: Mapping) -> np.ndarray:
    """Paint an MRI-like float image over a label volume, leaving labels untouched."""
    background_mean, background_std = appearance["background"]
    low, high = appearance["structure"]
    jitter = float(appearance.get("jitter", 0.0))
    image = rng.normal(background_mean, background_std, labels.shape).astype(np.float32)
    for label in range(1, int(labels.max()) + 1):
        mask = labels == label
        if not mask.any():
            continue
        mean = 0.5 * (low + high) + class_bias(label, float(appearance.get("class_bias", 0.0)))
        mean += float(rng.uniform(-jitter, jitter))
        mean = float(np.clip(mean, low, high))
        image[mask] = mean + rng.normal(0.0, appearance["noise"], int(mask.sum()))
    return np.clip(gaussian_blur(image, appearance["blur"]), 0.0, 1.0)


def generate_scene(
    seed: int, shape, spacing, margin: int, appearance: Mapping, max_attempts: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    """One reproducible synthetic scene: ``(image, labels)``.

    Every draw derives from ``seed``, so a scene is bit-for-bit reproducible
    from its seed alone.
    """
    for attempt in range(max_attempts):
        rng = np.random.default_rng([seed, attempt])
        try:
            labels = pack_scene(rng, shape, spacing, margin)
        except RuntimeError:
            continue
        return paint(labels, rng, appearance), labels
    raise RuntimeError(f"scene seed {seed} failed to pack in {max_attempts} attempts")
