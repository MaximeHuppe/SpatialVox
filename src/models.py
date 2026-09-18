"""Both networks, and the blocks they share.

**Stage A** (:class:`StageA`) is a promptable segmenter: an intensity volume and
a set of structure names in, one mask per name out. On synthetic data it segments
primitives; on MRI it is the anatomy segmenter that supplies Stage B's anchors.

**Stage B** (:class:`StageB`) is the relational segmenter: ordered anchor masks
and a relational prompt in, the target mask out. The target is never an input -
it has to be inferred from the intersection of the relations.

The architecture is the one from ``exp/realistic-appearance``, parameter for
parameter (``tests/test_reference_parity.py`` loads a checkpoint from that branch
into these classes and compares outputs). What changed is only how it is sized:
``model.encoder_channels`` lists one width per scale, with a stride-2 stage
between each pair, so its length sets the depth and the bottleneck follows:
``resolution / 2^(len - 1)``, checked against ``model.bottleneck``. A 128^3 corpus
wants one more width than a 64^3 one, and the bottleneck stays at ``8^3`` - 512
tokens, small enough for global cross-attention. Embedding tables are sized from
the vocabulary, so ten primitives and eighty anatomical labels are the same code.
See ``docs/method/``.

The two stages differ in more than their inputs, and the differences are
deliberate:

* Stage A sees an *image*, so it uses ReLU, no coordinate channels, and deep
  supervision at three decoder scales;
* Stage B sees *masks*, so it uses leaky ReLU and needs normalised world
  coordinates at every scale - the anchor channels alone are
  translation-ambiguous, and `medial`/`lateral` is defined against the volume
  centre plane, which exists only in world coordinates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

EPS = 1e-6
#: centroid (3) + bounding-box extent (3) + linear size (1) + presence flag (1)
GEOMETRY_FEATURES = 8
#: Number of coordinate channels appended per scale: (x, y, z).
COORDS = 3


# ---------------------------------------------------------------------------
# Shared blocks
# ---------------------------------------------------------------------------
def activation(name: str) -> nn.Module:
    """The two activations the project uses: ReLU (Stage A), leaky (Stage B)."""
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "leaky_relu":
        return nn.LeakyReLU(0.01, inplace=True)
    raise ValueError(f"activation must be 'relu' or 'leaky_relu', got {name!r}")


def conv(in_channels: int, out_channels: int, kernel: int = 3, stride: int = 1) -> nn.Conv3d:
    """The project's standard convolution: no bias, padding preserves the size."""
    return nn.Conv3d(in_channels, out_channels, kernel, stride, kernel // 2, bias=False)


def norm_act(channels: int, act: str) -> nn.Sequential:
    return nn.Sequential(nn.InstanceNorm3d(channels, affine=False), activation(act))


class ConvBlock(nn.Sequential):
    """``Conv -> InstanceNorm -> activation``."""

    def __init__(self, in_channels: int, out_channels: int, act: str, stride: int = 1) -> None:
        super().__init__(conv(in_channels, out_channels, stride=stride), *norm_act(out_channels, act))


class ResBlock(nn.Module):
    """Two convolutions plus an identity shortcut; size and width are unchanged."""

    def __init__(self, channels: int, act: str) -> None:
        super().__init__()
        self.body = nn.Sequential(
            ConvBlock(channels, channels, act),
            conv(channels, channels),
            nn.InstanceNorm3d(channels, affine=False),
        )
        self.act = activation(act)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.body(x) + x)


class PosEnc3D(nn.Module):
    """Learned positional encoding factorised over the three spatial axes.

    Holds ``pos_z``, ``pos_y`` and ``pos_x`` and returns their sum flattened to
    ``[1, D*H*W, C]``, so the parameter count is ``(D + H + W) * C`` rather than
    ``D*H*W*C``. The grid is fixed at construction; a mismatch is an error rather
    than a silent interpolation.
    """

    def __init__(self, grid: Sequence[int], channels: int) -> None:
        super().__init__()
        depth, height, width = (int(v) for v in grid)
        self.grid, self.channels = (depth, height, width), int(channels)
        self.pos_z = nn.Parameter(torch.zeros(1, depth, 1, 1, channels))
        self.pos_y = nn.Parameter(torch.zeros(1, 1, height, 1, channels))
        self.pos_x = nn.Parameter(torch.zeros(1, 1, 1, width, channels))
        for parameter in (self.pos_z, self.pos_y, self.pos_x):
            nn.init.trunc_normal_(parameter, std=0.02)

    def forward(self, grid: Sequence[int] | None = None) -> Tensor:
        if grid is not None and tuple(int(v) for v in grid) != self.grid:
            raise ValueError(f"positional encoding is built for {self.grid}, got {tuple(grid)}")
        return (self.pos_z + self.pos_y + self.pos_x).reshape(1, -1, self.channels)


#: Hard ceiling on bottleneck attention tokens. The design point is 8^3 = 512;
#: beyond this the global cross-attention stops being affordable.
MAX_ATTENTION_TOKENS = 4096


def widths_for(base: int, depth: int, cap: int) -> list[int]:
    """Channel width per scale: doubling from ``base``, capped at ``cap``.

    A convenience for sweeps; ``configs/config.yaml`` lists the widths outright.
    """
    return [min(base * 2**level, cap) for level in range(depth + 1)]


def depth_for(resolution: int, bottleneck: int) -> int:
    """Number of stride-2 stages that takes ``resolution`` down to ``bottleneck``."""
    depth = round(math.log2(resolution / bottleneck))
    if bottleneck * 2**depth != resolution:
        raise ValueError(f"resolution {resolution} is not a power-of-two multiple of {bottleneck}")
    return depth


def bottleneck_for(resolution: int, channels: Sequence[int], expected: int | None = None) -> int:
    """The bottleneck grid side implied by one width per scale.

    ``channels`` is the encoder's widths, finest first; there is a stride-2 stage
    between each pair, so ``len(channels) - 1`` halvings. ``expected`` is the
    configured ``model.bottleneck``, checked rather than assumed - which is what
    turns "I changed the resolution and forgot the widths" into a message that
    says what to write.
    """
    depth = len(channels) - 1
    if depth < 1:
        raise ValueError(f"encoder_channels needs at least two widths, got {list(channels)}")
    if resolution % 2**depth:
        raise ValueError(
            f"{len(channels)} encoder widths give {depth} stride-2 stages, which do not divide "
            f"a resolution of {resolution}"
        )
    bottleneck = resolution // 2**depth
    if expected is not None and bottleneck != expected:
        raise ValueError(
            f"{len(channels)} encoder widths give {depth} stride-2 stages, so {resolution}^3 "
            f"reduces to {bottleneck}^3, not the configured bottleneck of {expected}^3. "
            f"Add or drop an encoder width, or set model.bottleneck to {bottleneck}."
        )
    if bottleneck**3 > MAX_ATTENTION_TOKENS:
        raise ValueError(
            f"a {bottleneck}^3 bottleneck is {bottleneck ** 3} attention tokens, over the "
            f"{MAX_ATTENTION_TOKENS} budget; add a width to encoder_channels"
        )
    return bottleneck


def world_grid(grid: Sequence[int], full: Sequence[int], device, dtype) -> Tensor:
    """Normalised world coordinates ``(x, y, z)`` on a possibly coarse grid.

    Returns ``[1, 3, *grid]`` in ``[-1, 1]`` at the corners of the *full* volume.
    A cell of a grid ``f`` times coarser covers input indices ``f*i .. f*i+f-1``,
    so its world position is ``f*i + (f-1)/2``. Feeding local indices instead
    would make the same anatomical position take different values at different
    scales. Voxel spacing cancels in the normalisation, which is why no module
    here takes a spacing argument.
    """
    axes = []
    for size, full_size in zip(grid, full):
        factor = full_size / size
        index = torch.arange(size, device=device, dtype=dtype)
        world = index * factor + (factor - 1.0) / 2.0
        half = (full_size - 1) / 2.0
        # Written as (world - half) / half rather than the algebraically equal
        # 2*world/(full-1) - 1, so the rounding matches the reference
        # implementation exactly and a ported checkpoint reproduces it bitwise.
        axes.append((world - half) / max(half, 1e-8))
    z, y, x = (axis.reshape([-1 if a == i else 1 for a in range(3)]) for i, axis in enumerate(axes))
    return torch.stack(torch.broadcast_tensors(x, y, z), dim=0).unsqueeze(0)


def pool_to(x: Tensor, size: Sequence[int], mode: str = "max") -> Tensor:
    """Resize a mask-like tensor to a coarser grid.

    ``max`` preserves presence (a thin structure vanishes under averaging);
    ``avg`` gives the occupancy fraction, which is what pooling weights want.
    """
    target = tuple(int(v) for v in size)
    if tuple(x.shape[2:]) == target:
        return x
    factors = [s // t for s, t in zip(x.shape[2:], target)]
    if [t * f for t, f in zip(target, factors)] != list(x.shape[2:]):
        return F.interpolate(x, size=target, mode="nearest" if mode == "max" else "area")
    return (F.max_pool3d if mode == "max" else F.avg_pool3d)(x, factors, factors)


def prior_bias(fraction: float) -> float:
    """``log(p / (1 - p))``: the head bias that makes sigmoid output ``p`` at init.

    A zero-initialised head predicts 0.5 everywhere, but one structure covers
    well under 1% of a volume, so training then begins by pushing hundreds of
    thousands of background logits down before Dice carries usable gradient.
    """
    return math.log(fraction / (1.0 - fraction))


# ---------------------------------------------------------------------------
# Encoder / decoder
# ---------------------------------------------------------------------------
class Encoder(nn.Module):
    """Stem plus ``depth`` stride-2 stages, optionally with world coordinates."""

    def __init__(self, in_channels: int, widths: Sequence[int], act: str, coords: bool) -> None:
        super().__init__()
        self.widths, self.coords = list(widths), bool(coords)
        extra = COORDS if coords else 0
        inputs = [in_channels] + self.widths[:-1]
        self.stages = nn.ModuleList(
            nn.Sequential(
                ConvBlock(channels + extra, width, act, stride=1 if level == 0 else 2),
                ResBlock(width, act),
            )
            for level, (channels, width) in enumerate(zip(inputs, self.widths))
        )

    def forward(self, x: Tensor) -> list[Tensor]:
        """``[B, C, D, H, W]`` -> one feature map per scale, finest first."""
        full = x.shape[2:]
        features = []
        for stage in self.stages:
            if self.coords:
                grid = world_grid(x.shape[2:], full, x.device, x.dtype)
                x = torch.cat([x, grid.expand(x.shape[0], -1, -1, -1, -1)], dim=1)
            x = stage(x)
            features.append(x)
        return features


class FiLM(nn.Module):
    """``y = (1 + gamma(context)) * x + beta(context)``, starting as the identity."""

    def __init__(self, context_dim: int, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.project = nn.Linear(context_dim, 2 * channels)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        gamma, beta = self.project(context).chunk(2, dim=-1)
        shape = (x.shape[0], self.channels, 1, 1, 1)
        return (1 + gamma.reshape(shape)) * x + beta.reshape(shape)


class Decoder(nn.Module):
    """Upsample, concatenate the skip, optionally inject occupancy and context.

    Returns every stage's features, coarse to fine, so Stage A can supervise all
    of them. Conditioning is FiLM rather than attention: at half resolution a
    64^3 volume already has 32,768 query tokens, sixty-four times the bottleneck
    budget. FiLM is applied at every stage except the finest, which is where the
    reference architecture puts it (16^3 and 32^3 of 64^3).
    """

    def __init__(
        self,
        widths: Sequence[int],
        act: str,
        *,
        coords: bool = False,
        refine: bool = False,
        context_dim: int = 0,
        occupancy: bool = False,
    ) -> None:
        super().__init__()
        widths = list(widths)
        outputs = widths[-2::-1]  # coarse to fine: w_{D-1} ... w_0
        inputs = widths[:0:-1]  # w_D ... w_1
        extra = COORDS if coords else 0
        self.coords = bool(coords)
        self.fuse = nn.ModuleList(ConvBlock(i + extra + s, s, act) for i, s in zip(inputs, outputs))
        self.refine = nn.ModuleList(ConvBlock(w, w, act) for w in outputs) if refine else None
        self.occupancy = (
            nn.ModuleList(nn.Conv3d(w + 1, w, 1) for w in outputs) if occupancy else None
        )
        # Every stage but the finest, which keeps the reference's 16^3/32^3 at
        # 64^3 and generalises to any resolution.
        self.film = (
            nn.ModuleDict({str(i): FiLM(context_dim, w) for i, w in enumerate(outputs[:-1])})
            if context_dim
            else None
        )

    def forward(
        self,
        features: Sequence[Tensor],
        *,
        context: Tensor | None = None,
        occupancy: Tensor | None = None,
    ) -> list[Tensor]:
        x, full = features[-1], features[0].shape[2:]
        stages = []
        for level, skip in enumerate(features[-2::-1]):
            if self.coords:
                grid = world_grid(x.shape[2:], full, x.device, x.dtype)
                x = torch.cat([x, grid.expand(x.shape[0], -1, -1, -1, -1)], dim=1)
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=True)
            x = self.fuse[level](torch.cat([x, skip], dim=1))
            if self.occupancy is not None and occupancy is not None:
                x = self.occupancy[level](torch.cat([x, pool_to(occupancy, x.shape[2:])], dim=1))
            if self.film is not None and context is not None and str(level) in self.film:
                x = self.film[str(level)](x, context)
            if self.refine is not None:
                x = self.refine[level](x)
            stages.append(x)
        return stages


# ---------------------------------------------------------------------------
# Prompt encoders
# ---------------------------------------------------------------------------
class NamePrompt(nn.Module):
    """A structure name as a learned embedding, projected to the token width.

    The vocabulary is closed, so an embedding table is the exact and
    deterministic representation - there is no open-vocabulary text to
    generalise over. Swap this module for a frozen sentence encoder to accept
    free-text names.
    """

    def __init__(self, vocab_size: int, dim: int, text_dim: int | None = None) -> None:
        super().__init__()
        self.table = nn.Embedding(vocab_size, text_dim or dim)
        self.projection = nn.Linear(text_dim or dim, dim)
        nn.init.trunc_normal_(self.table.weight, std=0.02)

    def forward(self, name_ids: Tensor) -> Tensor:
        return self.projection(self.table(name_ids))


class RelationPrompt(nn.Module):
    """One token per clause, never pooled::

        token_i = direction_i + name_i + pair(direction_i, name_i) + slot_i

    The pair table is what lets a relation be more than the sum of its parts
    (``lateral to the thalamus`` need not behave like ``lateral`` plus
    ``thalamus``). Keeping the clauses separate all the way into the grounding
    branches is what makes correspondence learnable at all.
    """

    def __init__(self, vocab_size: int, n_clauses: int, dim: int, n_directions: int = 6) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.name = NamePrompt(vocab_size, dim)
        self.direction = nn.Embedding(n_directions, dim)
        self.pair = nn.Embedding(n_directions * vocab_size, dim)
        self.slot = nn.Embedding(n_clauses, dim)
        self.norm = nn.LayerNorm(dim)
        for table in (self.direction, self.pair, self.slot):
            nn.init.trunc_normal_(table.weight, std=0.02)

    def forward(self, direction_ids: Tensor, name_ids: Tensor) -> Tensor:
        """``[B, A]`` + ``[B, A]`` -> ``[B, A, dim]``, clause ``i`` in slot ``i``."""
        slots = torch.arange(direction_ids.shape[1], device=direction_ids.device)
        tokens = (
            self.slot(slots)[None]
            + self.direction(direction_ids)
            + self.name(name_ids)
            + self.pair(direction_ids * self.vocab_size + name_ids)
        )
        return self.norm(tokens)


class StructureEncoder(nn.Module):
    """One token per anchor channel: what it looks like, and where it is.

    Every geometric feature is measured **from the mask channel itself**, never
    read out of the manifest. That is what makes ground-truth and predicted
    anchors interchangeable: when the segmenter supplies a slightly wrong mask,
    the token describes *that* mask, so an oracle-versus-predicted gap measures
    Stage A's error rather than a change of interface.
    """

    def __init__(self, visual_channels: int, vocab_size: int, n_clauses: int, dim: int) -> None:
        super().__init__()
        self.visual = nn.Linear(visual_channels, dim)
        self.geometry = nn.Sequential(
            nn.Linear(GEOMETRY_FEATURES, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim)
        )
        self.name = nn.Embedding(vocab_size, dim)
        self.slot = nn.Embedding(n_clauses, dim)
        self.norm = nn.LayerNorm(dim)
        for table in (self.name, self.slot):
            nn.init.trunc_normal_(table.weight, std=0.02)

    def forward(self, masks: Tensor, bottleneck: Tensor, name_ids: Tensor) -> Tensor:
        slots = torch.arange(masks.shape[1], device=masks.device)
        tokens = (
            self.visual(masked_pool(bottleneck, masks))
            + self.geometry(mask_geometry(masks))
            + self.slot(slots)[None]
            + self.name(name_ids)
        )
        return self.norm(tokens)


def masked_pool(features: Tensor, masks: Tensor) -> Tensor:
    """Pool ``[B, C, ...]`` features under each of ``[B, A, ...]`` mask channels.

    Masks are average-pooled onto the feature grid, so a structure smaller than
    one cell still contributes a fractional weight instead of vanishing. An
    empty channel falls back to the globally pooled features rather than a NaN.
    """
    weights = pool_to(masks.to(features.dtype), features.shape[2:], mode="avg").flatten(2)
    flat = features.flatten(2)
    totals = weights.sum(-1, keepdim=True)
    pooled = torch.bmm(weights, flat.transpose(1, 2)) / totals.clamp(min=EPS)
    return torch.where(totals > EPS, pooled, flat.mean(-1).unsqueeze(1))


def mask_geometry(masks: Tensor) -> Tensor:
    """``[B, A, D, H, W]`` -> ``[B, A, 8]`` normalised geometry per channel.

    Normalised centroid ``(x, y, z)`` in ``[-1, 1]``, bounding-box extent as a
    fraction of each axis, the cube root of the occupied volume fraction (a
    linear size, which keeps the feature O(0.1) rather than O(0.001)), and a
    presence flag so the network can tell "tiny structure at the origin" from
    "no structure at all".
    """
    occupancy = (masks > 0.5).to(torch.float32)
    counts = occupancy.flatten(2).sum(-1)
    present = (counts > 0).to(counts.dtype)
    centroids, extents = [], []
    for axis, size in enumerate(masks.shape[2:]):  # array axes are (z, y, x)
        others = tuple(a for a in (2, 3, 4) if a != axis + 2)
        profile, weights = occupancy.amax(dim=others), occupancy.sum(dim=others)
        index = torch.arange(size, device=masks.device, dtype=counts.dtype)
        centre = (weights * index).sum(-1) / counts.clamp(min=1.0)
        # (centre - half) / half, not the equal 2*centre/(size-1) - 1: same value,
        # same rounding as the reference, so a ported checkpoint matches bitwise.
        half = (size - 1) / 2.0
        centroids.append((centre - half) / max(half, EPS) * present)
        occupied = profile > 0
        low = torch.where(occupied, index, torch.full_like(profile, size)).amin(-1)
        high = torch.where(occupied, index, torch.zeros_like(profile)).amax(-1)
        extents.append((high - low + 1) / size * present)
    voxels = float(masks.shape[2] * masks.shape[3] * masks.shape[4])
    size_fraction = (counts / voxels).clamp(min=0.0) ** (1 / 3)
    return torch.cat(
        [
            torch.stack(centroids[::-1], -1),
            torch.stack(extents[::-1], -1),
            size_fraction[..., None],
            present[..., None],
        ],
        dim=-1,
    )


# ---------------------------------------------------------------------------
# Stage A: promptable segmenter
# ---------------------------------------------------------------------------
class MaskHead(nn.Module):
    """Per-prompt mask logits: a scaled dot product plus a per-query bias.

    Visual features are only *read*, never modulated by the prompt set, and
    queries never interact. So the logits for one name do not depend on which
    other names were requested - which is what lets Stage A be trained on the
    whole vocabulary and queried for just the three anchors a relational prompt
    names.
    """

    def __init__(self, dim: int, visual_channels: int, prior_foreground: float) -> None:
        super().__init__()
        self.visual_channels = visual_channels
        self.project = nn.Sequential(
            nn.Linear(dim, visual_channels),
            nn.ReLU(inplace=True),
            nn.Linear(visual_channels, visual_channels),
        )
        self.bias = nn.Linear(visual_channels, 1)
        nn.init.zeros_(self.bias.weight)
        nn.init.constant_(self.bias.bias, prior_bias(prior_foreground))

    def forward(self, queries: Tensor, visual: Tensor) -> Tensor:
        projected = F.layer_norm(self.project(queries), (self.visual_channels,))
        logits = torch.bmm(projected, visual.flatten(2)) / math.sqrt(self.visual_channels)
        return (logits + self.bias(projected)).reshape(visual.shape[0], -1, *visual.shape[2:])


@dataclass
class StageAOutput:
    """Full-resolution logits, plus the coarse maps deep supervision needs."""

    logits: Tensor
    scales: list[Tensor]


class StageA(nn.Module):
    """Segment every requested structure name from one intensity volume.

    Name queries attend over the flattened bottleneck - the query says *what*,
    the keys say *where* (they alone carry the positional encoding), the values
    carry the unmodified visual content - and the aligned queries are turned into
    masks at three decoder scales by :class:`MaskHead`. Inference uses the
    full-resolution map; training also supervises the coarse two.
    """

    def __init__(
        self,
        vocab_size: int,
        resolution: int,
        *,
        encoder_channels: Sequence[int] = (32, 64, 128, 256),
        token_dim: int = 256,
        num_heads: int = 4,
        bottleneck: int | None = None,
        prior_foreground: float = 0.0016,
        deep_supervision: Sequence[float] = (0.1, 0.3, 0.6),
    ) -> None:
        super().__init__()
        widths = [int(w) for w in encoder_channels]
        grid = bottleneck_for(resolution, widths, bottleneck)
        self.config = dict(
            vocab_size=vocab_size, resolution=resolution, encoder_channels=tuple(widths),
            token_dim=token_dim, num_heads=num_heads, bottleneck=grid,
            prior_foreground=prior_foreground, deep_supervision=tuple(deep_supervision),
        )
        if widths[-1] != token_dim:
            raise ValueError(
                f"the last encoder width ({widths[-1]}) must equal token_dim ({token_dim}); "
                "the prompt decoder attends over the bottleneck directly"
            )
        self.deep_supervision = tuple(float(w) for w in deep_supervision)
        self.encoder = Encoder(1, widths, "relu", coords=False)
        self.decoder = Decoder(widths, "relu")
        self.prompt = NamePrompt(vocab_size, token_dim)
        self.pos = PosEnc3D((grid,) * 3, token_dim)
        self.attention = nn.MultiheadAttention(token_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(token_dim)
        self.heads = nn.ModuleList(
            MaskHead(token_dim, width, prior_foreground) for width in widths[-2::-1]
        )

    def forward(self, image: Tensor, name_ids: Tensor, deep_supervision: bool = True) -> StageAOutput:
        """``[B, 1, D, H, W]`` and ``[B, P]`` names -> ``[B, P, D, H, W]`` logits."""
        features = self.encoder(image)
        values = features[-1].flatten(2).transpose(1, 2)
        keys = values + self.pos(features[-1].shape[2:])
        queries = self.prompt(name_ids)
        attended, _ = self.attention(queries, keys, values, need_weights=False)
        queries = self.norm(attended + queries)

        stages = self.decoder(features)
        logits = self.heads[-1](queries, stages[-1])
        scales = (
            [head(queries, stage) for head, stage in zip(self.heads[:-1], stages[:-1])] + [logits]
            if deep_supervision
            else [logits]
        )
        return StageAOutput(logits=logits, scales=scales)

    @torch.no_grad()
    def masks_for(self, image: Tensor, name_ids: Tensor, threshold: float = 0.5) -> Tensor:
        """Binary masks for the named structures, in the order they were asked for."""
        logits = self(image, name_ids, deep_supervision=False).logits
        return (torch.sigmoid(logits.float()) >= threshold).float()


# ---------------------------------------------------------------------------
# Stage B: relational segmenter
# ---------------------------------------------------------------------------
class Evidence(nn.Module):
    """One clause -> one spatial evidence map at the bottleneck.

    The relation token is first fused with its structure token by a small
    cross-attention block: the relation queries ``[relation, structure]``, then a
    residual MLP. The fused clause is then grounded with **visual locations as
    queries** and the clause's three tokens ``{fused, relation, structure}`` as
    keys and values, so each location decides for itself how much of the
    relation, the anchor's appearance and their combination it needs.

    This is the only attention over space in Stage B, and it is affordable
    exactly because the bottleneck is 512 tokens.
    """

    def __init__(self, visual_channels: int, dim: int, heads: int, grid: Sequence[int], act: str) -> None:
        super().__init__()
        self.dim = dim
        self.clause_attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.clause_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.norm = nn.LayerNorm(dim)

        self.to_query = nn.Linear(visual_channels, dim)
        self.pos = PosEnc3D(grid, dim)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.attention_norm = nn.LayerNorm(dim)
        self.project = nn.Sequential(
            conv(visual_channels + dim, visual_channels, 1), *norm_act(visual_channels, act),
            conv(visual_channels, visual_channels), *norm_act(visual_channels, act),
        )

    def forward(self, visual: Tensor, relation: Tensor, structure: Tensor) -> tuple[Tensor, Tensor]:
        """``-> (evidence map [B, C, ...], fused clause token [B, dim])``."""
        memory = torch.stack([relation, structure], dim=1)
        attended, _ = self.clause_attention(relation.unsqueeze(1), memory, memory, need_weights=False)
        clause = self.clause_norm(attended.squeeze(1) + relation)
        clause = self.norm(clause + self.mlp(clause))

        tokens = torch.stack([clause, relation, structure], dim=1)
        queries = self.to_query(visual.flatten(2).transpose(1, 2)) + self.pos(visual.shape[2:])
        grounded, _ = self.attention(queries, tokens, tokens, need_weights=False)
        grounded = self.attention_norm(grounded + queries)
        grid = grounded.transpose(1, 2).reshape(visual.shape[0], self.dim, *visual.shape[2:])
        return self.project(torch.cat([visual, grid], dim=1)), clause


class Intersection(nn.Module):
    """Pointwise fusion of ``[H_1, ..., H_A, H_1 * ... * H_A]``, then a refinement.

    The product term is the point: it is high only where *every* relation is
    satisfied at once, which is the definition of the target, and no sum of the
    maps can express that. The maps are kept alongside it so softer combinations
    stay available. Each map is squashed to ``[0, 1]`` first, or the product of
    unbounded activations has no usable gradient.
    """

    def __init__(self, channels: int, n_clauses: int, hidden: int, act: str) -> None:
        super().__init__()
        self.fuse = nn.Sequential(
            conv((n_clauses + 1) * channels, hidden, 1), *norm_act(hidden, act),
            conv(hidden, hidden, 1), *norm_act(hidden, act),
            conv(hidden, channels, 1),
        )
        self.refine = nn.Sequential(conv(channels, channels), *norm_act(channels, act))

    def forward(self, maps: Sequence[Tensor]) -> Tensor:
        product = torch.sigmoid(maps[0])
        for other in maps[1:]:
            product = product * torch.sigmoid(other)
        return self.refine(self.fuse(torch.cat([*maps, product], dim=1)))


@dataclass
class StageBOutput:
    logits: Tensor
    evidence: list[Tensor]


class StageB(nn.Module):
    """Segment the structure described only by its relations to named anchors.

    Inputs are the ordered anchor masks, the clause indices, and the binary
    occupancy of the scene. It never receives the label volume, the target mask,
    the target name or the target's position - the point of the experiment is
    that it has to derive them.

    Occupancy enters on the decoder side only. It says *what* there is (some
    structure is here), and the encoder must not see it: grounding queries that
    could look at the target's own voxels would let the model ignore the prompt
    and pick "a blob that is not an anchor". The anchors are subtracted from it
    for the same reason the prompt names them - they are the given, not the
    answer.
    """

    def __init__(
        self,
        vocab_size: int,
        resolution: int,
        n_anchors: int = 3,
        *,
        encoder_channels: Sequence[int] = (32, 64, 128, 256),
        token_dim: int = 256,
        num_heads: int = 4,
        intersection_hidden: int | None = None,
        bottleneck: int | None = None,
        prior_foreground: float = 0.0016,
    ) -> None:
        super().__init__()
        widths = [int(w) for w in encoder_channels]
        grid = bottleneck_for(resolution, widths, bottleneck)
        hidden = int(intersection_hidden or token_dim)
        self.config = dict(
            vocab_size=vocab_size, resolution=resolution, n_anchors=n_anchors,
            encoder_channels=tuple(widths), token_dim=token_dim, num_heads=num_heads,
            intersection_hidden=hidden, bottleneck=grid, prior_foreground=prior_foreground,
        )
        self.n_anchors = n_anchors
        self.encoder = Encoder(n_anchors, widths, "leaky_relu", coords=True)
        self.prompt = RelationPrompt(vocab_size, n_anchors, token_dim)
        self.structure = StructureEncoder(widths[-1], vocab_size, n_anchors, token_dim)
        # One evidence branch, applied to every clause: sharing the parameters is
        # what stops a clause being grounded by a slot-specific shortcut. All that
        # distinguishes branch 2 from branch 1 is its tokens.
        self.evidence = Evidence(widths[-1], token_dim, num_heads, (grid,) * 3, "leaky_relu")
        self.intersection = Intersection(widths[-1], n_anchors, hidden, "leaky_relu")
        self.decoder = Decoder(
            widths, "leaky_relu", coords=True, refine=True,
            context_dim=n_anchors * token_dim, occupancy=True,
        )
        self.head = nn.Conv3d(widths[0], 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, prior_bias(prior_foreground))

    def forward(self, anchors: Tensor, direction_ids: Tensor, name_ids: Tensor, occupancy: Tensor) -> StageBOutput:
        """``[B, A, D, H, W]`` anchors + ``[B, A]`` clause ids + occupancy -> logits."""
        if anchors.shape[1] != self.n_anchors:
            raise ValueError(f"expected {self.n_anchors} anchor channels, got {anchors.shape[1]}")
        anchors = anchors.to(torch.float32)
        occupancy = occupancy.to(torch.float32) * (1 - anchors.amax(dim=1, keepdim=True))

        features = self.encoder(anchors)
        relations = self.prompt(direction_ids, name_ids)
        structures = self.structure(anchors, features[-1], name_ids)

        maps, clauses = [], []
        for slot in range(self.n_anchors):
            evidence, clause = self.evidence(features[-1], relations[:, slot], structures[:, slot])
            maps.append(evidence)
            clauses.append(clause)
        features[-1] = features[-1] + self.intersection(maps)
        stages = self.decoder(features, context=torch.cat(clauses, dim=-1), occupancy=occupancy)
        return StageBOutput(logits=self.head(stages[-1]), evidence=maps)
