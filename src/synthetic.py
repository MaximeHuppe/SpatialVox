"""The synthetic corpus: ten axis-aligned primitives in an MRI-like image.

This module produces exactly what a real dataset produces - an intensity volume
and an integer label volume - and nothing else. Everything downstream (relations,
prompts, manifests, training) is written against that pair, so swapping this file
for real MRI changes no other code. See ``documentation/SpatialVox.md`` (Synthetic corpora).

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

from typing import Callable, Mapping, Sequence

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
    "tetrahedron": (  # regular tetrahedron: four half-spaces of a cube's alternate corners
        {"side": (11.0, 13.5)},
        lambda d, p: (d[0] + d[1] + d[2] <= p["side"] / 2)
        & (d[0] - d[1] - d[2] <= p["side"] / 2)
        & (-d[0] + d[1] - d[2] <= p["side"] / 2)
        & (-d[0] - d[1] + d[2] <= p["side"] / 2),
        lambda p: (p["side"] / 2,) * 3,
    ),
    "hollow_cylinder": (  # a tube: annular section in x-y, bore along z
        {"radius": (4.6, 5.8), "bore": (2.3, 3.4), "height": (7.5, 11.0)},
        lambda d, p: (d[0] ** 2 + d[1] ** 2 <= p["radius"] ** 2)
        & (d[0] ** 2 + d[1] ** 2 >= p["bore"] ** 2)
        & (abs(d[2]) <= p["height"] / 2),
        lambda p: (p["radius"], p["radius"], p["height"] / 2),
    ),
    "cross": (  # three orthogonal bars through a common centre - NON-CONVEX
        {"arm": (11.0, 14.0), "thickness": (3.6, 4.8)},
        lambda d, p: (
            ((abs(d[0]) <= p["thickness"] / 2) & (abs(d[1]) <= p["thickness"] / 2))
            | ((abs(d[1]) <= p["thickness"] / 2) & (abs(d[2]) <= p["thickness"] / 2))
            | ((abs(d[0]) <= p["thickness"] / 2) & (abs(d[2]) <= p["thickness"] / 2))
        )
        & (abs(d[0]) <= p["arm"] / 2)
        & (abs(d[1]) <= p["arm"] / 2)
        & (abs(d[2]) <= p["arm"] / 2),
        lambda p: (p["arm"] / 2,) * 3,
    ),
    "crescent": (  # a sphere with a spherical bite taken out - CONCAVE boundary
        {"radius": (5.5, 7.0), "bite": (4.8, 6.2), "offset": (3.2, 4.6)},
        lambda d, p: (d[0] ** 2 + d[1] ** 2 + d[2] ** 2 <= p["radius"] ** 2)
        & ((d[0] - p["offset"]) ** 2 + d[1] ** 2 + d[2] ** 2 > p["bite"] ** 2),
        lambda p: (p["radius"],) * 3,
    ),
    "hourglass": (  # two cones apex to apex: radius 0 at the waist - NON-CONVEX
        {"radius": (5.5, 7.0), "height": (11.0, 14.0)},
        lambda d, p: (abs(d[2]) <= p["height"] / 2)
        & (d[0] ** 2 + d[1] ** 2 <= (p["radius"] * 2 * abs(d[2]) / p["height"]) ** 2),
        lambda p: (p["radius"], p["radius"], p["height"] / 2),
    ),
    "banana": (  # a tube on a CURVED medial axis - the hippocampus of this vocabulary.
        # The bow is a length, not an angle, so it scales with `size` like everything
        # else; the -1/3 recentres the parabola so the shape stays near its centroid.
        {"thickness": (3.4, 4.4), "length": (13.0, 16.5), "bow": (5.0, 7.0)},
        lambda d, p: (abs(d[2]) <= p["length"] / 2)
        & (
            (d[0] - p["bow"] * ((2 * d[2] / p["length"]) ** 2 - 1 / 3.0)) ** 2 + d[1] ** 2
            <= p["thickness"] ** 2
        ),
        lambda p: (p["bow"] + p["thickness"], p["thickness"], p["length"] / 2),
    ),
}

SHAPE_NAMES: tuple[str, ...] = tuple(SHAPES)

#: Classes that must stay anisotropic or they degenerate into another class
#: (a cuboid that is a cube, an ellipsoid that is a sphere).
ANISOTROPIC = {"cuboid": 1.35, "ellipsoid": 1.35}

#: Per-class draws that must be rejected because they collapse the class into a
#: different one. Keeping them here rather than in ``draw_params`` means adding a
#: shape is one entry in one table, and the reason travels with the rule.
DEGENERATE: dict[str, Callable[[Mapping[str, float]], bool]] = {
    # a torus whose tube is fatter than its ring has no hole
    "torus": lambda p: p["minor"] >= p["major"],
    # too small a bore is a cylinder; too large leaves a shell with no body
    "hollow_cylinder": lambda p: not 0.35 * p["radius"] < p["bore"] < 0.72 * p["radius"],
    # the bite must reach inside the sphere without swallowing it
    "crescent": lambda p: not abs(p["radius"] - p["bite"]) < p["offset"] < p["radius"] + p["bite"],
    # arms as thick as they are long is a cube
    "cross": lambda p: p["thickness"] >= 0.6 * p["arm"],
}


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
        rule = DEGENERATE.get(name)
        if rule is not None and rule(params):
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


def head_mask(shape, spacing, fraction: float, exponent: float = 3.0) -> np.ndarray:
    """An ellipsoidal envelope inscribed in the grid: the 'head'.

    Everything outside it is air. A real MRI has a hard, closed outer boundary
    and a consistent global frame; a featureless cube does not, and that absence
    makes the synthetic corpus harder than the thing it stands in for in a way
    that has nothing to do with intensity. ``fraction <= 0`` disables it.
    """
    if fraction <= 0:
        return np.ones(tuple(shape), dtype=bool)
    axes = [(np.arange(n) - (n - 1) / 2) * sp for n, sp in zip(shape, spacing[::-1])]
    dz, dy, dx = (a.reshape(sh) for a, sh in
                  zip(axes, ((-1, 1, 1), (1, -1, 1), (1, 1, -1))))
    radii = [fraction * (n - 1) / 2 * sp for n, sp in zip(shape, spacing[::-1])]
    return (
        abs(dz / radii[0]) ** exponent
        + abs(dy / radii[1]) ** exponent
        + abs(dx / radii[2]) ** exponent
    ) <= 1.0


def pack_scene(
    rng: np.random.Generator, shape, spacing, margin: int, attempts: int = 200, size: float = 1.0,
    region: np.ndarray | None = None,
) -> np.ndarray:
    """Rejection-sample one instance of every primitive into a ``uint8`` label volume.

    Raises when an object cannot be placed within the attempt budget; the caller
    retries the whole scene with a fresh draw.

    ``size`` scales every structure. It matters more than it looks: at the
    default the primitives are about 8 voxels across in a 64^3 volume, and the
    carver's stride-2 stem sees them at four - so Dice measures the discretisation
    rather than the shape. What ``size`` must **not** do is widen the spread
    *between* classes, because a size that identifies a class is the same kind of
    shortcut as a brightness that does, and it is the one the MRI carver actually
    took: it learned a class-conditional size prior and, on a class it had never
    been supervised on, predicted 93 voxels where 2219 belonged. The ten classes
    here sit within 1.5x of each other and this multiplier keeps them there.
    """
    labels = np.zeros(tuple(shape), dtype=np.uint8)
    occupied = np.zeros(tuple(shape), dtype=bool)
    spacing = np.asarray(spacing, dtype=float)
    scale = min(shape) / REFERENCE_AXIS
    last_center = (np.array([shape[2], shape[1], shape[0]]) - 1) * spacing  # world (x, y, z)

    for name in PLACEMENT_ORDER:
        for _ in range(attempts):
            params = draw_params(name, rng, scale * float(size))
            half = half_extent(name, params)
            low, high = margin * spacing + half, last_center - margin * spacing - half
            if np.any(low > high):
                continue  # too big for this grid: draw a smaller one
            candidate = voxelize(name, params, rng.uniform(low, high), shape, spacing)
            if not candidate.any() or (candidate & occupied).any():
                continue
            if region is not None and (candidate & ~region).any():
                continue  # a structure must lie inside the head, never in air
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


def bias_field(shape, rng: np.random.Generator, amplitude: float) -> np.ndarray:
    """A smooth multiplicative inhomogeneity, like an MRI receive field.

    Three random low-order polynomials, one per axis, multiplied together and
    normalised to ``1 +- amplitude``. Its job is to make a *global* intensity
    threshold fail without touching the *local* edge: after this the same tissue
    is brighter on one side of the volume than the other, so "foreground is above
    t" stops being true anywhere but locally - while a structure still stands out
    against what immediately surrounds it, which is what a convolution, and an
    eye, actually read. ``amplitude <= 0`` is a no-op.
    """
    if amplitude <= 0:
        return np.ones(tuple(shape), dtype=np.float32)
    field = np.ones(tuple(shape), dtype=np.float32)
    for axis in np.meshgrid(*[np.linspace(-1.0, 1.0, int(n)) for n in shape], indexing="ij"):
        a, b, c = rng.uniform(-1.0, 1.0, 3)
        field *= (1.0 + a * axis + b * axis**2 + c * axis**3).astype(np.float32)
    field -= field.mean()
    return (1.0 + amplitude * field / max(float(np.abs(field).max()), 1e-6)).astype(np.float32)


def correlated_field(shape, rng: np.random.Generator, scale: float) -> np.ndarray:
    """Smooth Gaussian noise of **unit variance** and the given correlation length.

    Normalised by its own realised standard deviation, so an amplitude in the
    config means exactly that amplitude in the image. ``gaussian_blur`` reduces
    the variance of white noise by a factor that depends on the kernel, which is
    why the older ``texture`` path carries a hand-fitted ``* 6.0``; this does not.
    """
    field = rng.normal(0.0, 1.0, tuple(shape)).astype(np.float32)
    if scale > 0:
        field = gaussian_blur(field, scale)
    deviation = float(field.std())
    return field / deviation if deviation > 1e-6 else field


def octave_texture(shape, rng: np.random.Generator, length: float, octaves: int = 4) -> np.ndarray:
    """Unit-variance texture with a roughly EXPONENTIAL autocorrelation.

    One Gaussian-blurred field decays as ``exp(-d^2)`` and so has no tail: it is
    smooth lumps and nothing else. Measured on ``data/mri``, brain tissue decays
    as roughly ``exp(-d/4)`` - 0.76 at one voxel, 0.57 at two, 0.36 at four, 0.21
    at eight - which is the signature of structure at *many* scales at once, not
    one. Summing octaves with ``1/scale`` weights reproduces that, and it is what
    makes the result read as tissue rather than as blobs.
    """
    field = np.zeros(tuple(shape), dtype=np.float32)
    for k in range(octaves):
        scale = length * (0.5 ** k)
        field += correlated_field(shape, rng, scale) / (k + 1.0)
    deviation = float(field.std())
    return field / deviation if deviation > 1e-6 else field


def tissue_volume(
    shape, rng: np.random.Generator, fractions: Sequence[float], shades: Sequence[float],
    length: float,
) -> np.ndarray:
    """A brain-like tissue background: a few classes at measured proportions.

    ``fractions`` are volume shares and ``shades`` their intensities, both taken
    from ``data/mri``: CSF 11%, grey 50%, white 39%, and grey->white is the
    separation every other amplitude in this file is quoted against. The classes
    are cut out of an octave texture by quantile, so they interdigitate the way
    grey and white matter do instead of forming a few smooth lobes.
    """
    field = octave_texture(shape, rng, length)
    cuts = np.quantile(field, np.cumsum(np.asarray(fractions, dtype=float))[:-1])
    index = np.digitize(field, cuts)
    out = np.zeros(tuple(shape), dtype=np.float32)
    for k, shade in enumerate(shades):
        out[index == k] = float(shade)
    return out


def class_range(
    label: int, n_classes: int, low: float, high: float, spread: float | None
) -> tuple[float, float]:
    """The intensity range one class is always drawn from. ``(lo, hi)``.

    Every structure of a given class is painted from the *same* range in every
    scene of the corpus - a cube is always cube-bright - while the ranges of
    different classes overlap. That is how tissue behaves: grey matter, white
    matter and CSF each have a characteristic intensity, they overlap, and no
    single global threshold separates one structure from the rest.

    Class ``i`` of ``n`` is centred at ``low + (high - low) * (i + 0.5) / n`` and
    spans ``spread``. When ``spread`` exceeds the spacing between centres the
    ranges overlap, so brightness narrows the identity down without ever fixing
    it. Setting ``spread`` to 0 pins each class to a single value.

    ``spread = None`` is the other regime, and the stronger test: **every class
    draws from the whole band**, so brightness carries no class information at
    all and the relational prompt becomes the only way to pick the target. It
    also makes ``B(I)`` genuinely class-agnostic - a held-out shape looks exactly
    like a trained one - which is the mechanism §4 claims. The per-class form
    reproduces the MRI-like regime, where intensity partly identifies a class.
    """
    if spread is None:
        return low, high
    index = (int(label) - 1) % max(int(n_classes), 1)
    centre = low + (high - low) * (index + 0.5) / max(int(n_classes), 1)
    return centre - 0.5 * spread, centre + 0.5 * spread


def paint(
    labels: np.ndarray, rng: np.random.Generator, appearance: Mapping,
    envelope: np.ndarray | None = None,
) -> np.ndarray:
    """Paint an MRI-like float image over a label volume, leaving labels untouched.

    What surrounds a structure is the part that was wrong for a long time, and it
    is worth saying exactly how. A real MRI's background is *anatomy*: connected
    white matter, a grey-matter ribbon, dark ventricles, and hard black air
    outside the head. Its intensity varies at two scales - whole lobes, far
    coarser than any target, and tissue speckle, far finer - and at **no scale
    resembling a target**. A field of random lumps the size of the structures is
    not a harder version of that; it is a different and strictly harder problem,
    because every lump is a decoy that looks exactly like the thing being asked
    for. Stage A scored Dice 0.0000 on all sixteen classes against such a
    background while the same architecture reaches 0.88 against this one.

    So the background here is built from compartments, not noise: a very coarse
    field quantised into ``tissues`` levels, each level one intensity, which
    gives large connected regions with real edges between them and no power at
    the structure scale. ``tissues < 2`` restores the older single-mean
    background exactly.
    """
    # `background` is the pre-tissue-model single mean. When `separation` is set
    # the tissue volume replaces it entirely, so it is optional; `grey` then
    # stands in wherever a scalar background level is still wanted (the air noise).
    background_mean, background_std = appearance.get(
        "background", (float(appearance.get("grey", 0.45)), 0.02)
    )
    low, high = appearance["structure"]
    spread = appearance.get("class_spread", 0.0)
    # `shared` (or null): every class draws from the whole band, so brightness
    # carries no class information at all and the relational prompt is the only
    # way to pick the target. A number instead gives each class its own narrower,
    # overlapping range - the MRI-like regime, where intensity narrows identity
    # down without fixing it.
    spread = None if spread in (None, "shared") else float(spread)
    image = rng.normal(background_mean, background_std, labels.shape).astype(np.float32)
    # Background TISSUE, measured from data/mri rather than invented. `separation`
    # is grey->white and every other amplitude is quoted as a multiple of it, so
    # the whole appearance scales with one number.
    separation = float(appearance.get("separation", 0.0))
    texture_sd = float(appearance.get("texture_sd", 0.39)) * separation
    length = float(appearance.get("tissue_length", 4.0))
    if separation > 0:
        grey = float(appearance.get("grey", background_mean))
        fractions = appearance.get("tissue_fractions", (0.112, 0.496, 0.392))
        offsets = appearance.get("tissue_offsets", (-1.33, 0.0, 1.0))
        shades = [grey + separation * float(o) for o in offsets]
        image = tissue_volume(labels.shape, rng, fractions, shades, length)
        # tissue is not flat either: the same texture runs through it
        image = image + texture_sd * octave_texture(labels.shape, rng, length)
    # Background texture, at the same spatial scale as a structure. Without it
    # the background is white noise, which a blur flattens to a constant - and a
    # constant background is what makes a single global threshold recover every
    # structure at IoU 0.9998. `texture <= 0` reproduces that easy corpus.
    texture = float(appearance.get("texture", 0.0))
    if texture > 0:
        lumps = rng.normal(0.0, 1.0, labels.shape).astype(np.float32)
        image += texture * gaussian_blur(lumps, float(appearance.get("texture_scale", 2.0))) * 6.0
    for label in range(1, int(labels.max()) + 1):
        mask = labels == label
        if not mask.any():
            continue
        # The class's own range, identical in every scene of the corpus.
        lo, hi = class_range(int(label), len(SHAPES), low, high, spread)
        mean = float(rng.uniform(lo, hi))
        mean = float(np.clip(mean, low, high))
        image[mask] = mean + rng.normal(0.0, appearance["noise"], int(mask.sum()))
    # A structure is not a constant. Measured on data/mri its internal spread is
    # 0.525 against a grey->white separation of 1.35, i.e. 0.39 of it - the same
    # texture amplitude the surrounding tissue carries. Painting structures flat
    # is what made every earlier synthetic corpus separable by one threshold.
    if separation > 0 and texture_sd > 0:
        interior = labels > 0
        image[interior] = image[interior] + (
            texture_sd * octave_texture(labels.shape, rng, length)
        )[interior]
    # Correlated detail, added to structures AND background alike, after the
    # structures are painted so they inherit it. This is the knob that sets the
    # local contrast-to-noise ratio: it leaves the intensity STEP at a boundary
    # untouched and buries it in variance, which is the regime `data/mri` is in.
    #
    # Measured on data/mri, a structure's local CNR against a 3-voxel shell is a
    # median of 0.40 while its own internal variation is 0.388 - the structure is
    # *not* locally separable from what surrounds it, and an eye finds it through
    # continuity and extent instead. Without this the synthetic CNR is 3.4, nine
    # times too generous, and `B(I)` gets a boundary no real image offers.
    #
    # It must be CORRELATED, not white: white noise of the same amplitude
    # destroys the edge rather than camouflaging it, which is a harder and
    # different problem. `detail <= 0` reproduces the older corpora exactly.
    detail = float(appearance.get("detail", 0.0))
    if detail > 0:
        image += detail * correlated_field(labels.shape, rng, float(appearance.get("detail_scale", 1.5)))
    image = gaussian_blur(image, appearance["blur"])
    image = image * bias_field(labels.shape, rng, float(appearance.get("bias_field", 0.0)))
    # Air outside the head, last, so nothing smears the outer boundary. A closed
    # dark envelope is most of what gives a real volume its global frame.
    if envelope is not None:
        air = float(appearance.get("air", 0.02))
        image = np.where(envelope, image,
                         rng.normal(air, background_std / 2, labels.shape).astype(np.float32))
    return np.clip(image, 0.0, 1.0)


def generate_scene(
    seed: int, shape, spacing, margin: int, appearance: Mapping, max_attempts: int = 50,
    size: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """One reproducible synthetic scene: ``(image, labels)``.

    Every draw derives from ``seed``, so a scene is bit-for-bit reproducible
    from its seed alone.
    """
    for attempt in range(max_attempts):
        rng = np.random.default_rng([seed, attempt])
        envelope = head_mask(shape, spacing, float(appearance.get("head", 0.0)),
                             float(appearance.get("head_exponent", 4.0)))
        inside = envelope if appearance.get("head", 0.0) else None
        try:
            labels = pack_scene(rng, shape, spacing, margin, size=size, region=inside)
        except RuntimeError:
            continue
        return paint(labels, rng, appearance, envelope=inside), labels
    raise RuntimeError(f"scene seed {seed} failed to pack in {max_attempts} attempts")
