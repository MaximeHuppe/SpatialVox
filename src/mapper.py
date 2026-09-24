"""``PositionalMapper3D`` - the WHERE, written as a soft field.

The mapper is :func:`src.geometry.classify` turned into a differentiable map.
``classify`` answers "which single word describes this target relative to this
anchor"; the mapper answers the inverse question, "for every point in the
volume, how well would a structure centred there satisfy this clause", and
answers it with the same predicate: *one axis of the centroid offset dominates,
inside a square pyramid of 45 degrees*.

It has **no parameters**. It never sees the image, and it never sees a name.
The direction id is consumed here and nowhere downstream - a learned field that
could move itself onto a blob is exactly the shortcut a fixed pyramid closes.

Three properties are load-bearing and each is pinned by a test:

* **The product is not renormalised.** ``where_raw = F_0 * F_1 * F_2`` and that
  is what leaves the module. A sigmoid is never exactly zero, so an impossible
  conjunction still has a tiny peak; dividing by that peak would turn it into
  1.0 and manufacture a confident answer out of nothing. The null head reads
  ``where_mass`` instead, and a meaningless spike stays small.
* **The centroid is confidence-weighted, not thresholded.** ``c_i`` is the
  first moment of the *soft* mask. A cut at 0.5 makes the centroid jump and can
  delete a dim but real anchor in one step.
* **The pyramid describes where a centroid would satisfy the clause**, not
  where the target's voxels are. Part of a structure legitimately lies outside
  it. ``where_raw`` is a channel into the carver and a weak bias - never a crop,
  never a mask, never the initial segmentation.

Conventions are the project's: arrays are ``(z, y, x)``, world coordinates are
``(x, y, z)`` in a RAS frame, and ``spacing`` is world units per voxel ordered
``(x, y, z)``. ``tau`` is therefore in world units, and on a 1.25 mm corpus it
means millimetres, not voxels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from src.geometry import DIRECTIONS

EPS = 1e-8

#: For each direction, ``(axis, sign)``. ``axis`` indexes world ``(x, y, z)``;
#: ``sign`` is the half-space on that axis, and for the ``x`` pair it is the
#: sign of the *midline* comparison rather than of ``dx`` itself.
_AXIS_SIGN: dict[str, tuple[int, float]] = {
    "lateral": (0, +1.0),
    "medial": (0, -1.0),
    "anterior": (1, +1.0),
    "posterior": (1, -1.0),
    "superior": (2, +1.0),
    "inferior": (2, -1.0),
}

#: Row ``i`` is ``DIRECTIONS[i]`` as ``(is_x, is_y, is_z, sign)``.
DIRECTION_TABLE: tuple[tuple[float, float, float, float], ...] = tuple(
    (
        float(_AXIS_SIGN[name][0] == 0),
        float(_AXIS_SIGN[name][0] == 1),
        float(_AXIS_SIGN[name][0] == 2),
        _AXIS_SIGN[name][1],
    )
    for name in DIRECTIONS
)


@dataclass
class MapperOutput:
    """What the mapper hands the carver and the null head.

    ``fields`` and ``where_raw`` go to the carver; ``where_mass`` and ``masses``
    are the null head's *only* four numbers.
    """

    fields: Tensor  # [B, A, D, H, W] one soft pyramid per clause
    where_raw: Tensor  # [B, 1, D, H, W] their product, never renormalised
    where_mass: Tensor  # [B, 1] its mean over the volume
    masses: Tensor  # [B, A] mean of each soft anchor mask
    centroids: Tensor  # [B, A, 3] world (x, y, z), confidence-weighted


def world_axes(
    shape: Sequence[int], spacing: Sequence[float], device, dtype
) -> tuple[Tensor, Tensor, Tensor]:
    """The three world coordinate axes of a ``(D, H, W)`` volume, as ``(x, y, z)``.

    Kept separable on purpose. A dense ``[3, D, H, W]`` coordinate grid costs
    three full volumes; every quantity the mapper needs is a function of one
    axis at a time until the final ``max``, so broadcasting three vectors is
    both exact and an order of magnitude cheaper.
    """
    depth, height, width = (int(v) for v in shape)
    options = dict(device=device, dtype=dtype)
    x = torch.arange(width, **options) * float(spacing[0])
    y = torch.arange(height, **options) * float(spacing[1])
    z = torch.arange(depth, **options) * float(spacing[2])
    return x, y, z


def soft_centroids(
    masks: Tensor, spacing: Sequence[float]
) -> tuple[Tensor, Tensor]:
    """Confidence-weighted centroids and masses of ``[B, A, D, H, W]`` soft masks.

    ``mass_i = mean(A_i)`` - a fraction of the volume, so it is comparable
    across corpora and is what ``mapper.min_mass`` is expressed in.
    ``c_i = sum(A_i * p) / (sum(A_i) + eps)``, the first moment of the soft
    mask, in world ``(x, y, z)``.

    Computed from the three axis marginals rather than a dense coordinate grid:
    ``sum(A * p_x)`` is ``sum_x p_x * (sum over D, H of A)``, which is exact and
    touches the volume once per axis.
    """
    masks = masks.float()
    x, y, z = world_axes(masks.shape[2:], spacing, masks.device, masks.dtype)
    total = masks.flatten(2).sum(-1)  # [B, A]
    along_z = masks.sum(dim=(3, 4))  # [B, A, D]
    along_y = masks.sum(dim=(2, 4))  # [B, A, H]
    along_x = masks.sum(dim=(2, 3))  # [B, A, W]
    centroids = torch.stack(
        [(along_x * x).sum(-1), (along_y * y).sum(-1), (along_z * z).sum(-1)], dim=-1
    ) / (total + EPS).unsqueeze(-1)
    voxels = int(masks.shape[2] * masks.shape[3] * masks.shape[4])
    return centroids, total / voxels


def direction_terms(direction_ids: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """``[B, A]`` direction ids -> per-slot ``(is_x, is_y, is_z, sign)`` selectors.

    Exactly one indicator is 1 per slot, which lets every margin below be one
    broadcast sum over the three mutually exclusive axis cases rather than six
    margins of which five are discarded.
    """
    table = torch.tensor(
        DIRECTION_TABLE, device=direction_ids.device, dtype=torch.float32
    )
    is_x, is_y, is_z, sign = table[direction_ids.long()].unbind(-1)
    return is_x, is_y, is_z, sign


def _margin(
    is_x: Tensor,
    is_y: Tensor,
    is_z: Tensor,
    sign: Tensor,
    dx: Tensor,
    dy: Tensor,
    dz: Tensor,
    voxel_offset: Tensor,
    anchor_offset: Tensor,
) -> Tensor:
    """The margin formula itself, on whatever shapes broadcast together.

    Written once and called from both :func:`margins` (a whole volume, kept
    separable per axis) and :func:`margin_at` (explicit points). The two share
    this body so the gate cannot drift from the field it is a gate on;
    ``tests/test_mapper.py`` pins that they agree at voxel centres.
    """
    ax, ay, az = dx.abs(), dy.abs(), dz.abs()
    primary = is_z * (sign * dz) + is_y * (sign * dy) + is_x * ax
    rival = (
        is_z * torch.maximum(ax, ay)
        + is_y * torch.maximum(ax, az)
        + is_x * torch.maximum(ay, az)
    )
    margin = primary - rival
    lateral = sign * (voxel_offset - anchor_offset)
    return torch.where(is_x.bool(), torch.minimum(margin, lateral), margin)


def margin_at(
    points: Tensor, centroids: Tensor, direction_ids: Tensor, center: Sequence[float]
) -> Tensor:
    """The margin of explicit world points: ``[B, A, 3]``, ``[B, A, 3]``, ``[B, A]`` -> ``[B, A]``.

    The manifest gate asks one question - "does ``where_raw`` clear 0.5 *at the
    target's centroid*" - and that is a single point, so the whole tau sweep is
    a closed form over the manifests and never materialises a volume.
    """
    is_x, is_y, is_z, sign = direction_terms(direction_ids)
    delta = points.float() - centroids.float()
    midline = float(center[0])
    return _margin(
        is_x, is_y, is_z, sign,
        delta[..., 0], delta[..., 1], delta[..., 2],
        (points[..., 0].float() - midline).abs(),
        (centroids[..., 0].float() - midline).abs(),
    )


def margins(
    centroids: Tensor,
    direction_ids: Tensor,
    shape: Sequence[int],
    spacing: Sequence[float],
    center: Sequence[float],
) -> Tensor:
    """Signed world-unit margin of every voxel against every clause.

    ``[B, A, 3]`` centroids and ``[B, A]`` directions -> ``[B, A, D, H, W]``.

    For ``d = p - c`` the margin is *how far the dominant axis leads the other
    two*, which is positive exactly inside the 45-degree square pyramid and is
    zero on its surface:

    ======================  ===================================================
    superior / inferior     ``+-d_z - max(|d_x|, |d_y|)``
    anterior / posterior    ``+-d_y - max(|d_x|, |d_z|)``
    lateral / medial        ``min(|d_x| - max(|d_y|, |d_z|),``
                            ``    +-(|p_x - m| - |c_x - m|))``
    ======================  ===================================================

    Lateral and medial are *distances to the mid-sagittal plane* ``m``, not the
    half-spaces ``+x`` and ``-x``; the second term is that comparison, and
    taking the ``min`` makes the clause true only where both hold. This is
    ``classify``'s rule, voxel by voxel: there, the axis wins by being the
    largest of the three, which is the same inequality written as a difference.

    There is no axial fade and no learned gain. A direction is a relation, not a
    distance, so a point twice as far superior is not twice as superior.
    """
    dtype = torch.float32
    x, y, z = world_axes(shape, spacing, centroids.device, dtype)
    centroids = centroids.to(dtype)
    slots = centroids.shape[:2]
    view = lambda t: t.reshape(*slots, 1, 1, 1)
    is_x, is_y, is_z, sign = (view(t) for t in direction_terms(direction_ids))

    # d along each axis, kept separable: [B, A, 1, 1, W] / [B, A, 1, H, 1] / [B, A, D, 1, 1].
    # Only the max and the sum below ever expand to a full volume.
    dx = x.reshape(1, 1, 1, 1, -1) - view(centroids[..., 0])
    dy = y.reshape(1, 1, 1, -1, 1) - view(centroids[..., 1])
    dz = z.reshape(1, 1, -1, 1, 1) - view(centroids[..., 2])
    midline = float(center[0])
    return _margin(
        is_x, is_y, is_z, sign, dx, dy, dz,
        (x - midline).abs().reshape(1, 1, 1, 1, -1),
        view((centroids[..., 0] - midline).abs()),
    )


class PositionalMapper3D(torch.nn.Module):
    """Soft anchor masks and direction ids in, ``F``, ``where_raw`` and the masses out.

    Stateless: ``tau`` and ``min_mass`` are buffers so that a checkpoint records
    the geometry it was trained with, but the module has no learnable weight and
    ``forward`` is a pure function of its inputs.

    ``min_mass`` rejects an anchor Stage A failed to find. A rejected channel
    writes ``F_i = 0``, so ``where_raw`` is identically zero and the null head -
    not a threshold inside the mapper - is what declares the prompt empty.
    """

    def __init__(self, tau: float = 0.5, min_mass: float = 1e-6) -> None:
        super().__init__()
        if float(tau) <= 0:
            raise ValueError(f"mapper.tau must be positive, got {tau!r}")
        self.tau = float(tau)
        self.min_mass = float(min_mass)

    def extra_repr(self) -> str:  # pragma: no cover - debugging aid
        return f"tau={self.tau}, min_mass={self.min_mass}"

    def forward(
        self,
        masks: Tensor,
        direction_ids: Tensor,
        spacing: Sequence[float],
        center: Sequence[float],
    ) -> MapperOutput:
        """``[B, A, D, H, W]`` soft masks + ``[B, A]`` directions -> :class:`MapperOutput`."""
        if masks.ndim != 5:
            raise ValueError(f"masks must be [B, A, D, H, W], got {tuple(masks.shape)}")
        if direction_ids.shape != masks.shape[:2]:
            raise ValueError(
                f"expected [B, A] direction ids for {tuple(masks.shape[:2])} slots, "
                f"got {tuple(direction_ids.shape)}"
            )
        masks = masks.float()
        centroids, masses = soft_centroids(masks, spacing)
        margin = margins(centroids, direction_ids, masks.shape[2:], spacing, center)
        gate = (masses >= self.min_mass).to(margin.dtype).reshape(*masses.shape, 1, 1, 1)
        fields = torch.sigmoid(margin / self.tau) * gate
        where_raw = fields.prod(dim=1, keepdim=True)
        return MapperOutput(
            fields=fields,
            where_raw=where_raw,
            where_mass=where_raw.flatten(1).mean(-1, keepdim=True),
            masses=masses,
            centroids=centroids,
        )
