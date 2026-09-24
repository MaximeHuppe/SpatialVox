"""Class-agnostic instance selection: propose bodies, score by the relational region.

Replaces the dense carver answer path. The mapper's ``where_raw`` stays the
locate signal; this module only turns a soft region into one binary body.

v1 (ADR ``ClassAgnosticInstance-ADR``):

* restrict to ``dilate(where_raw > region_threshold, r)``;
* seed at local maxima of ``where_raw`` inside that region;
* flood by intensity affinity;
* score each proposal by ``where_raw`` at its centroid;
* emit the argmax body (empty if the best score is below ``score_null``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from src.mapper import soft_centroids

EPS = 1e-8


def _dilate(mask: Tensor, radius: int) -> Tensor:
    """Binary cubic dilation (same separable max-pool as ``engine.dilate``)."""
    radius = int(radius)
    if radius <= 0:
        return mask
    size = 2 * radius + 1
    x = mask.float()
    for axis in range(3):
        kernel = [1, 1, 1]
        kernel[axis] = size
        padding = [0, 0, 0]
        padding[axis] = radius
        x = F.max_pool3d(x, tuple(kernel), 1, tuple(padding))
    return x


@dataclass
class InstanceResult:
    """One batch item's proposals and the selected mask."""

    proposals: Tensor  # [K, D, H, W]
    scores: Tensor  # [K]
    centroids: Tensor  # [K, 3] world (x, y, z)
    winner: int  # index into K, or -1 if null
    mask: Tensor  # [D, H, W] float {0,1}
    region: Tensor  # [D, H, W]


def region_mask(where_raw: Tensor, threshold: float = 0.5, radius: int = 4) -> Tensor:
    """``dilate(where_raw > threshold, radius)`` — hard support for proposals."""
    volume = where_raw
    while volume.ndim > 3:
        volume = volume[0]
    hard = (volume.unsqueeze(0).unsqueeze(0) > float(threshold)).float()
    return _dilate(hard, int(radius))


def local_maxima_seeds(
    where_raw: Tensor,
    region: Tensor,
    *,
    max_seeds: int = 16,
    window: int = 3,
    image: Tensor | None = None,
) -> list[tuple[int, int, int]]:
    """Local maxima of ``where_raw`` inside ``region``, highest first.

    When ``image`` is given, peaks are ranked by ``where_raw * |I - median(I in
    region)|`` so a flat background plateau under a roomy field does not outrank
    a structure that actually sits under the peak.
    """
    volume = where_raw.float()
    support = region.float() > 0.5
    if not bool(support.any()):
        return []
    padded = volume.unsqueeze(0).unsqueeze(0)
    pooled = F.max_pool3d(padded, kernel_size=window, stride=1, padding=window // 2)
    is_peak = (padded == pooled) & support.unsqueeze(0).unsqueeze(0)
    coords = is_peak[0, 0].nonzero(as_tuple=False)
    if coords.numel() == 0:
        flat = (volume * support.float()).flatten()
        idx = int(flat.argmax())
        d, h, w = volume.shape
        return [(idx // (h * w), (idx // w) % h, idx % w)]
    values = volume[coords[:, 0], coords[:, 1], coords[:, 2]]
    if image is not None:
        intensity = image.float()
        while intensity.ndim > 3:
            intensity = intensity[0]
        median = intensity[support].median()
        contrast = (intensity[coords[:, 0], coords[:, 1], coords[:, 2]] - median).abs()
        values = values * (contrast + 1e-3)
    order = torch.argsort(values, descending=True)
    picked = coords[order[: int(max_seeds)]]
    return [(int(z), int(y), int(x)) for z, y, x in picked.tolist()]


def _resolve_intensity_tol(
    image: np.ndarray,
    region: np.ndarray,
    intensity_tol: float,
    *,
    tol_mode: str = "std",
) -> float:
    """Absolute intensity band, or a multiple of ``std(I | region)``.

    Fixed ``0.15`` is fine on synthetic unit-range blobs and far too tight on
    z-scored MRI, where within-structure texture routinely exceeds that. The
    shipped default is therefore ``tol_mode='std'`` with ``intensity_tol`` as a
    scale on the region's intensity std.
    """
    scale = max(float(intensity_tol), 0.0)
    if tol_mode == "absolute":
        return max(scale, 1e-6)
    if tol_mode != "std":
        raise ValueError(f"tol_mode must be 'std' or 'absolute', got {tol_mode!r}")
    vals = image[region]
    if vals.size == 0:
        return max(scale, 1e-6)
    return max(scale * float(vals.std()), 1e-4)


def _flood_intensity(
    image: np.ndarray,
    seed: tuple[int, int, int],
    region: np.ndarray,
    *,
    intensity_tol: float,
) -> np.ndarray:
    """6-connected flood from ``seed`` while ``|I - I_seed| <= intensity_tol``."""
    depth, height, width = image.shape
    out = np.zeros((depth, height, width), dtype=bool)
    if not region[seed]:
        return out
    seed_value = float(image[seed])
    stack = [seed]
    out[seed] = True
    while stack:
        z, y, x = stack.pop()
        for dz, dy, dx in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            nz, ny, nx = z + dz, y + dy, x + dx
            if not (0 <= nz < depth and 0 <= ny < height and 0 <= nx < width):
                continue
            if out[nz, ny, nx] or not region[nz, ny, nx]:
                continue
            if abs(float(image[nz, ny, nx]) - seed_value) > intensity_tol:
                continue
            out[nz, ny, nx] = True
            stack.append((nz, ny, nx))
    return out


def propose_seed_flood(
    image: Tensor,
    where_raw: Tensor,
    *,
    dilate_radius: int = 4,
    region_threshold: float = 0.5,
    max_seeds: int = 16,
    intensity_tol: float = 1.0,
    tol_mode: str = "std",
    min_voxels: int = 8,
) -> tuple[Tensor, Tensor]:
    """Seed-flood proposals inside the relational region.

    ``intensity_tol`` is an absolute band when ``tol_mode='absolute'``, otherwise
    a multiple of ``std(I)`` inside the region (MRI default).
    """
    if image.ndim == 4:
        image = image[0]
    if where_raw.ndim == 4:
        where_raw = where_raw[0]
    region = region_mask(where_raw, region_threshold, dilate_radius)[0, 0]
    seeds = local_maxima_seeds(where_raw, region, max_seeds=max_seeds, image=image)
    # Always keep the region's where_raw argmax — local-max pool can miss a flat peak.
    flat = (where_raw.float() * (region > 0.5).float()).flatten()
    if flat.numel() and float(flat.max()) > 0:
        idx = int(flat.argmax())
        d, h, w = where_raw.shape
        peak = (idx // (h * w), (idx // w) % h, idx % w)
        if peak not in seeds:
            seeds = [peak, *seeds][:max_seeds]
    image_np = image.detach().float().cpu().numpy()
    region_np = region.detach().cpu().numpy() > 0.5
    tol = _resolve_intensity_tol(image_np, region_np, intensity_tol, tol_mode=tol_mode)
    # Drop near-median seeds: a roomy where_raw plateau over CSF/background
    # otherwise grows a large empty body whose centroid still sits in the field
    # and outscores the real structure under the pure where(centroid) rule.
    if region_np.any():
        median = float(np.median(image_np[region_np]))
        spread = float(np.std(image_np[region_np])) + 1e-6
        seeds = [
            s for s in seeds
            if abs(float(image_np[s]) - median) >= 0.25 * spread
        ] or seeds
    bodies: list[np.ndarray] = []
    for seed in seeds:
        body = _flood_intensity(image_np, seed, region_np, intensity_tol=tol)
        if int(body.sum()) < int(min_voxels):
            continue
        if any(float((body & other).sum()) / max(float(body.sum()), 1.0) > 0.9 for other in bodies):
            continue
        bodies.append(body)
    if not bodies:
        empty = torch.zeros(0, *where_raw.shape, dtype=torch.float32, device=where_raw.device)
        return empty, region
    stacked = torch.from_numpy(np.stack(bodies).astype(np.float32)).to(where_raw.device)
    return stacked, region


def score_proposals(
    proposals: Tensor,
    where_raw: Tensor,
    spacing: Sequence[float],
) -> tuple[Tensor, Tensor]:
    """``score_k = where_raw(centroid(P_k))`` — the corpus truth rule."""
    if proposals.numel() == 0:
        device = where_raw.device
        return torch.zeros(0, device=device), torch.zeros(0, 3, device=device)
    field = where_raw
    while field.ndim > 3:
        field = field[0]
    centroids, masses = soft_centroids(proposals.unsqueeze(0), spacing)
    centroids = centroids[0]
    masses = masses[0]
    depth, height, width = field.shape
    sx, sy, sz = (float(v) for v in spacing)
    ix = (centroids[:, 0] / sx).round().long().clamp(0, width - 1)
    iy = (centroids[:, 1] / sy).round().long().clamp(0, height - 1)
    iz = (centroids[:, 2] / sz).round().long().clamp(0, depth - 1)
    scores = field[iz, iy, ix] * (masses > 0).to(field.dtype)
    return scores, centroids


def select_instance(
    proposals: Tensor,
    scores: Tensor,
    *,
    score_null: float = 0.5,
) -> tuple[int, Tensor]:
    """Hard argmax over scores; empty mask when best score ``< score_null``."""
    shape = proposals.shape[-3:] if proposals.ndim >= 3 else (0, 0, 0)
    device = scores.device if scores.numel() else (
        proposals.device if proposals.numel() else torch.device("cpu")
    )
    if scores.numel() == 0 or float(scores.max()) < float(score_null):
        return -1, torch.zeros(shape, device=device, dtype=torch.float32)
    winner = int(scores.argmax())
    return winner, proposals[winner].float()


def run_instance(
    image: Tensor,
    where_raw: Tensor,
    spacing: Sequence[float],
    *,
    dilate_radius: int = 4,
    region_threshold: float = 0.5,
    max_seeds: int = 16,
    intensity_tol: float = 1.0,
    tol_mode: str = "std",
    min_voxels: int = 8,
    score_null: float = 0.5,
) -> InstanceResult:
    """End-to-end propose → score → pick for one sample."""
    proposals, region = propose_seed_flood(
        image, where_raw,
        dilate_radius=dilate_radius,
        region_threshold=region_threshold,
        max_seeds=max_seeds,
        intensity_tol=intensity_tol,
        tol_mode=tol_mode,
        min_voxels=min_voxels,
    )
    scores, centroids = score_proposals(proposals, where_raw, spacing)
    winner, mask = select_instance(proposals, scores, score_null=score_null)
    return InstanceResult(
        proposals=proposals,
        scores=scores,
        centroids=centroids,
        winner=winner,
        mask=mask,
        region=region,
    )


def oracle_label_instances(
    labels: np.ndarray,
    where_raw: Tensor | np.ndarray,
    spacing: Sequence[float],
    *,
    present: Sequence[int] | None = None,
) -> tuple[int, float, np.ndarray]:
    """Pick the labelled structure whose centroid maximises ``where_raw``.

    Design ceiling (~0.97 on unique prompts), not a method.
    """
    if torch.is_tensor(where_raw):
        field = where_raw.detach().float().cpu().numpy()
    else:
        field = np.asarray(where_raw, dtype=np.float32)
    while field.ndim > 3:
        field = field[0]
    labels = np.asarray(labels)
    ids = [int(v) for v in (present if present is not None else np.unique(labels)) if int(v) != 0]
    best_label, best_score, best_mask = 0, -1.0, np.zeros(labels.shape, dtype=np.float32)
    sx, sy, sz = (float(v) for v in spacing)
    for label in ids:
        mask = labels == label
        if not mask.any():
            continue
        z, y, x = np.nonzero(mask)
        cx = float(x.mean()) * sx
        cy = float(y.mean()) * sy
        cz = float(z.mean()) * sz
        ix = min(max(int(round(cx / sx)), 0), field.shape[2] - 1)
        iy = min(max(int(round(cy / sy)), 0), field.shape[1] - 1)
        iz = min(max(int(round(cz / sz)), 0), field.shape[0] - 1)
        score = float(field[iz, iy, ix])
        if score > best_score:
            best_label, best_score, best_mask = label, score, mask.astype(np.float32)
    return best_label, best_score, best_mask
