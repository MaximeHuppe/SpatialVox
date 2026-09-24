"""Both networks, and the blocks they share.

**Stage A** (:class:`StageA`) is a promptable segmenter: an intensity volume and
a set of structure names in, one mask per name out. It is trained beforehand on
every name that may be an anchor, then frozen. Nothing below trains it.

**Stage B** (:class:`StageB`) is the relational model documented in
``documentation/SpatialVox.md``. Its inputs are the MRI and three
clauses, each a direction and an anchor name, and it segments a structure the
prompt never names::

    image   [B, 1, 128, 128, 128]
    clauses [B, 3] x {direction, name}
            ->  target logits [B, 1, 128, 128, 128]
            ->  null logit             one number: the clauses name nothing
            ->  centroid [B, 3]        from a heatmap, not from the mask

**Names stop at Stage A.** They buy three soft masks and are then gone: no name
embedding, no pair embedding, no slot embedding downstream. The direction ids
are consumed by :mod:`src.mapper` and by nothing else - the direction is already
the shape of ``F_i``, and a token would let the triple of anchor names stand in
for the target. The carver sees ``B(I)`` beside maps computed from detached
masks, and no coordinate grid.

Stage A's feature pyramid stays out too. Those features were trained to light up
*named* structures; ``B`` is a separate encoder trained without class ids, so a
structure Stage A has never seen is still a boundary in the image. That is the
whole lesion claim: the mask is painted from the MRI, not chosen from a list of
things Stage A already knows how to draw.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.geometry import volume_center_world
from src.instance import run_instance
from src.mapper import PositionalMapper3D

EPS = 1e-6

#: Floor under ``log(where_raw)`` and ``log(where_mass)``, and the scale both are
#: divided by. The measured range of ``where_mass`` on ``data/mri`` spans 1e-21
#: to 2e-2, so the raw number is indistinguishable from zero to a convolution
#: whose other nine input channels live in ``[0, 1]``. Dividing by ``-log(floor)``
#: maps the clamped log onto ``[-1, 0]``: the same one number, monotonically
#: reparameterised, still saying "the conjunction has no mass".
LOG_FLOOR = 1e-9


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


def prior_bias(fraction: float) -> float:
    """``log(p / (1 - p))``: the head bias that makes sigmoid output ``p`` at init.

    A zero-initialised head predicts 0.5 everywhere, but one structure covers
    well under 1% of a volume, so training then begins by pushing hundreds of
    thousands of background logits down before Dice carries usable gradient.
    """
    return math.log(fraction / (1.0 - fraction))


def grid_world_axes(
    grid: Sequence[int], full: Sequence[int], spacing: Sequence[float], device, dtype
) -> list[Tensor]:
    """World ``(x, y, z)`` axes of a grid ``full / grid`` times coarser, in world units.

    A cell of a grid ``f`` times coarser covers input indices ``f*i .. f*i+f-1``,
    so it sits at ``f*i + (f-1)/2`` of the full grid. Used by the soft-argmax, so
    a centroid read off the carver's 64^3 working grid is in the same millimetres
    as the label volume's centroid and the two are directly comparable.
    """
    axes = []
    for axis, (size, full_size) in enumerate(zip(grid[::-1], full[::-1])):  # (z,y,x) -> (x,y,z)
        factor = full_size / size
        index = torch.arange(size, device=device, dtype=dtype)
        axes.append((index * factor + (factor - 1.0) / 2.0) * float(spacing[axis]))
    return axes


class Encoder(nn.Module):
    """Stem plus one stride-2 stage per further width."""

    def __init__(self, in_channels: int, widths: Sequence[int], act: str) -> None:
        super().__init__()
        self.widths = list(widths)
        inputs = [in_channels] + self.widths[:-1]
        self.stages = nn.ModuleList(
            nn.Sequential(
                ConvBlock(channels, width, act, stride=1 if level == 0 else 2),
                ResBlock(width, act),
            )
            for level, (channels, width) in enumerate(zip(inputs, self.widths))
        )

    def forward(self, x: Tensor) -> list[Tensor]:
        """``[B, C, D, H, W]`` -> one feature map per scale, finest first."""
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


class Decoder(nn.Module):
    """Upsample and concatenate the skip, once per scale.

    Returns every stage's features, coarse to fine, so Stage A can supervise all
    of them.
    """

    def __init__(self, widths: Sequence[int], act: str) -> None:
        super().__init__()
        widths = list(widths)
        outputs = widths[-2::-1]  # coarse to fine: w_{D-1} ... w_0
        inputs = widths[:0:-1]  # w_D ... w_1
        self.fuse = nn.ModuleList(ConvBlock(i + s, s, act) for i, s in zip(inputs, outputs))

    def forward(self, features: Sequence[Tensor]) -> list[Tensor]:
        x, stages = features[-1], []
        for level, skip in enumerate(features[-2::-1]):
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=True)
            x = self.fuse[level](torch.cat([x, skip], dim=1))
            stages.append(x)
        return stages


# ---------------------------------------------------------------------------
# Stage A: promptable segmenter
# ---------------------------------------------------------------------------
class NamePrompt(nn.Module):
    """A structure name as a learned embedding, projected to the token width.

    The vocabulary is closed, so an embedding table is the exact and
    deterministic representation - there is no open-vocabulary text to
    generalise over. This is the **only** name embedding in the project, and it
    lives on the far side of the freeze.
    """

    def __init__(self, vocab_size: int, dim: int) -> None:
        super().__init__()
        self.table = nn.Embedding(vocab_size, dim)
        self.projection = nn.Linear(dim, dim)
        nn.init.trunc_normal_(self.table.weight, std=0.02)

    def forward(self, name_ids: Tensor) -> Tensor:
        return self.projection(self.table(name_ids))


class MaskHead(nn.Module):
    """Per-prompt mask logits: a scaled dot product plus a per-query bias.

    Visual features are only *read*, never modulated by the prompt set, and
    queries never interact. So the logits for one name do not depend on which
    other names were requested - which is what lets Stage A be trained on the
    whole vocabulary and queried for just the three anchors a clause names.
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
    masks at each decoder scale by :class:`MaskHead`. Inference uses the
    full-resolution map; training also supervises the coarse ones.
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
        self.encoder = Encoder(1, widths, "relu")
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
    def probability(self, image: Tensor, name_ids: Tensor) -> Tensor:
        """Soft masks for the named structures, in the order they were asked for.

        The *probability*, not a threshold. A cut at 0.5 makes the centroid the
        mapper reads jump, and can delete a dim but real anchor in one step.
        """
        return torch.sigmoid(self(image, name_ids, deep_supervision=False).logits.float())


# ---------------------------------------------------------------------------
# Stage B: boundary encoder, carver, null head
# ---------------------------------------------------------------------------
class BoundaryEncoder(nn.Module):
    """``B(I)``: generic boundary features from the MRI, and nothing else.

    No prompt, no name, no direction, no coordinate grid and no label reaches
    this module - ``forward`` takes one argument and it is the image. That is the
    point: its pretraining objectives carry no class id, so a structure Stage A
    has never been given a mask for is still a boundary here.

    Small beside Stage A: three stages at 128/64/32 and a symmetric path back,
    16-32 channels. Stage A's pyramid is not a substitute, because those features
    were trained to light up *named* structures.

    The residual pair is used at every stride-2 stage but **not** at full
    resolution. A 16-to-16 3x3x3 convolution on a 128^3 volume is by a wide
    margin the most expensive operation in Stage B: measured on this corpus, the
    two extra ones cost 350 ms of a 530 ms training step, two thirds of ``B``, for
    6% more parameters. Depth is cheaper one octave down, and the up path reads
    the finest skip again anyway.
    """

    def __init__(self, widths: Sequence[int] = (16, 32, 32), act: str = "leaky_relu") -> None:
        super().__init__()
        widths = [int(w) for w in widths]
        if len(widths) < 2:
            raise ValueError(f"the boundary encoder needs at least two widths, got {widths}")
        self.widths, self.out_channels = widths, widths[0]
        self.down = nn.ModuleList(
            nn.Sequential(
                ConvBlock(inp, width, act, stride=1 if level == 0 else 2),
                *([] if level == 0 else [ResBlock(width, act)]),
            )
            for level, (inp, width) in enumerate(zip([1] + widths[:-1], widths))
        )
        self.up = nn.ModuleList(
            ConvBlock(coarse + skip, skip, act)
            for coarse, skip in zip(widths[:0:-1], widths[-2::-1])
        )

    def forward(self, image: Tensor) -> Tensor:
        """``[B, 1, D, H, W]`` -> ``[B, out_channels, D, H, W]``."""
        skips = []
        x = image
        for stage in self.down:
            x = stage(x)
            skips.append(x)
        for level, skip in enumerate(skips[-2::-1]):
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=True)
            x = self.up[level](torch.cat([x, skip], dim=1))
        return x


class BoundaryPretrainer(nn.Module):
    """``B`` plus its pretext heads - the module ``scripts/train.py boundary`` trains.

    A separate model rather than a mode of :class:`StageB`, so the heads that
    read the label volume are not reachable from the relational forward at all.
    The trained ``encoder`` weights are then loaded into ``StageB.boundary`` and
    given a lower learning rate.
    """

    def __init__(
        self, widths: Sequence[int] = (16, 32, 32), mask_fraction: float = 0.5, patch: int = 16
    ) -> None:
        super().__init__()
        self.config = dict(
            widths=tuple(int(w) for w in widths),
            mask_fraction=float(mask_fraction),
            patch=int(patch),
        )
        self.mask_fraction, self.patch = float(mask_fraction), int(patch)
        self.encoder = BoundaryEncoder(widths)
        channels = self.encoder.out_channels
        self.reconstruct = nn.Conv3d(channels, 1, 1)
        self.boundary = nn.Conv3d(channels, 1, 1)
        self.edge = nn.Conv3d(channels, 1, 1)

    def blank(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """Blank a fraction of ``patch``-sized cubes. Returns ``(blanked, holes)``.

        Whole cubes, not scattered voxels: a voxel-wise mask is filled in by its
        own neighbours and teaches nothing about structure.
        """
        grid = [max(s // self.patch, 1) for s in image.shape[2:]]
        coarse = (torch.rand(image.shape[0], 1, *grid, device=image.device) < self.mask_fraction).float()
        holes = F.interpolate(coarse, size=image.shape[2:], mode="nearest")
        return image * (1 - holes), holes

    @staticmethod
    def edges(image: Tensor) -> Tensor:
        """``|grad I|`` by central differences - the class-agnostic edge target."""
        gradients = []
        for axis in range(2, 5):
            gradients.append(0.5 * (image.roll(-1, axis) - image.roll(1, axis)))
        return torch.sqrt(sum(g**2 for g in gradients) + EPS)

    def forward(self, image: Tensor) -> dict[str, Tensor]:
        features = self.encoder(image)
        return {
            "reconstruct": self.reconstruct(features),
            "boundary": self.boundary(features),
            "edge": self.edge(features),
        }


class NullHead(nn.Module):
    """``valid = MLP(where_mass, mass_0, mass_1, mass_2)``. Four numbers, no image.

    It cannot decide "empty" by looking at tissue, which is the point: the
    feasible field's mass is the reason an impossible prompt loses. The inputs
    are taken in ``log10``, which is a monotone reparameterisation of the same
    four numbers and not extra information - measured on ``data/mri`` they span
    1e-21 to 2e-2, and a linear layer cannot resolve that range.

    Its ceiling is a property of those inputs, not of its width:
    ``scripts/gate_mapper.py`` measures ``where_mass`` alone at AUC 0.848 for
    "names one structure" against "names none", because the mapper cannot see
    which regions hold tissue and a roomy but empty conjunction looks exactly
    like a valid one.
    """

    def __init__(self, n_anchors: int = 3, hidden: int = 32) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1 + n_anchors, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, where_mass: Tensor, masses: Tensor) -> Tensor:
        """``[B, 1]`` and ``[B, A]`` -> ``[B]`` logits."""
        features = torch.cat([where_mass, masses], dim=-1).clamp_min(1e-24).log10()
        return self.mlp(features.float()).squeeze(-1)


class Carver(nn.Module):
    """The WHAT: ``B(I)`` and the geometric maps in, target logits and a heatmap out.

    ``residual = two 16-channel blocks, stride-2 stem, then 1x1 up to 128^3``.
    The stem is what keeps 3x3x3 convolutions affordable on a 128^3 volume.

    ``full_resolution_skip`` carries ``B(I)`` past that stem, so the final 1x1
    sees the boundary features at the resolution they were computed at. It
    defaults on, and it is a departure from the literal reading of §4 recorded in
    ``documentation/SpatialVox.md`` (Deviations): a stride-2 stem otherwise destroys exactly
    the full-resolution boundary detail the method claims the mask is drawn from,
    which would leave a trilinear upsample of a 2.5 mm grid as the only path to
    the output. It is a flag so the claim stays measurable.

    The heatmap is a *separate* 1x1 on the working grid, not a reading of the
    mask: it survives the prompt being empty, which the mask does not.

    The mask head is applied in two halves that are *exactly* the 1x1 above
    (``_update_ideas/2026-09-22-null-head-decides-emptiness.md``). A 1x1
    convolution and a trilinear upsample are both linear, and the upsample's
    weights sum to one, so ``head(cat[up(f), B]) = up(W_f f + b) + W_B B``: the
    feature half runs on the working grid and a single channel is upsampled,
    instead of upsampling ``width`` channels and concatenating them with ``B(I)``
    into a ``(width + 16)``-channel tensor at full resolution. Same parameters,
    same checkpoints, same function.
    """

    def __init__(
        self,
        in_channels: int,
        boundary_channels: int,
        *,
        width: int = 16,
        blocks: int = 2,
        act: str = "leaky_relu",
        full_resolution_skip: bool = True,
        prior_foreground: float = 0.0016,
    ) -> None:
        super().__init__()
        self.full_resolution_skip = bool(full_resolution_skip) and boundary_channels > 0
        self.stem = ConvBlock(in_channels, width, act, stride=2)
        self.blocks = nn.Sequential(*[ResBlock(width, act) for _ in range(int(blocks))])
        self.head = nn.Conv3d(width + (boundary_channels if self.full_resolution_skip else 0), 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, prior_bias(prior_foreground))
        self.heatmap = nn.Conv3d(width, 1, 1)

    def forward(self, x: Tensor, boundary: Tensor | None) -> tuple[Tensor, Tensor]:
        """``-> (logits [B, 1, D, H, W], heatmap logits [B, 1, D/2, H/2, W/2])``."""
        features = self.blocks(self.stem(x))
        width = features.shape[1]
        coarse = F.conv3d(features, self.head.weight[:, :width], self.head.bias)
        logits = F.interpolate(coarse, size=x.shape[2:], mode="trilinear", align_corners=True)
        if self.full_resolution_skip:
            logits = logits + F.conv3d(boundary, self.head.weight[:, width:])
        return logits, self.heatmap(features)


def soft_argmax(
    heatmap: Tensor, full: Sequence[int], spacing: Sequence[float]
) -> Tensor:
    """``[B, 1, d, h, w]`` logits -> ``[B, 3]`` world ``(x, y, z)``.

    The expectation under a softmax over voxels, computed from the three axis
    marginals of that distribution rather than a dense coordinate grid. The
    result is continuous, so reading it off the carver's coarse working grid
    costs resolution in the *weights*, not in the coordinate.
    """
    weights = torch.softmax(heatmap.flatten(1).float(), dim=-1).reshape(heatmap.shape)
    axes = grid_world_axes(heatmap.shape[2:], full, spacing, heatmap.device, torch.float32)
    return torch.stack(
        [
            (weights.sum(dim=(1, 2, 3)) * axes[0]).sum(-1),  # x
            (weights.sum(dim=(1, 2, 4)) * axes[1]).sum(-1),  # y
            (weights.sum(dim=(1, 3, 4)) * axes[2]).sum(-1),  # z
        ],
        dim=-1,
    )


@dataclass
class StageBOutput:
    """Everything one relational forward produces.

    ``logits`` and ``valid`` are the answers; the rest is what the losses and the
    report need, and is carried rather than recomputed because every piece of it
    is a pure function of inputs the caller no longer holds.

    Under ``answer_mode=instance``, ``logits`` is the logit of the winning
    proposal mask (or a flat background field when the score is null).
    """

    logits: Tensor  # [B, 1, D, H, W]
    valid: Tensor  # [B] one logit: do the clauses name a structure
    centroid: Tensor  # [B, 3] world (x, y, z)
    where_raw: Tensor  # [B, 1, D, H, W] the product, never renormalised
    where_mass: Tensor  # [B, 1]
    fields: Tensor  # [B, A, D, H, W]
    anchors: Tensor  # [B, A, D, H, W] detached soft masks
    masses: Tensor  # [B, A]
    anchor_centroids: Tensor  # [B, A, 3]


class StageB(nn.Module):
    """Segment the structure the three clauses describe and never name.

    One forward: the three names go to the frozen Stage A and produce detached
    soft masks; :class:`~src.mapper.PositionalMapper3D` turns those masks and the
    three direction ids into the fields and their product; then either

    * **instance** (default on this branch): class-agnostic seed-flood proposals
      scored by ``where_raw`` at each centroid; or
    * **carver**: the B2 dense residual path (kept for parent comparison).

    The segmenter is a submodule so the checkpoint is self-contained and the
    signature can admit nothing else - but it is frozen at construction, it is
    excluded from :meth:`trainable_parameters`, and :meth:`train` keeps it in
    ``eval``. The relational loss does not flow back into it.
    """

    ANSWER_MODES = ("instance", "carver")

    def __init__(
        self,
        segmenter: dict,
        *,
        spacing: Sequence[float] = (1.0, 1.0, 1.0),
        n_anchors: int = 3,
        tau: float = 0.5,
        min_mass: float = 1e-6,
        answer_mode: str = "instance",
        boundary_widths: Sequence[int] = (16, 32, 32),
        carver_width: int = 16,
        carver_blocks: int = 2,
        full_resolution_skip: bool = True,
        use_image: bool = True,
        carver_sees_anchors: bool = False,
        additive_prior: bool = False,
        alpha: float = 0.35,
        background_logit: float = -10.0,
        prior_foreground: float = 0.0016,
        dilate_radius: int = 4,
        region_threshold: float = 0.5,
        max_seeds: int = 16,
        intensity_tol: float = 1.0,
        tol_mode: str = "std",
        score_null: float = 0.5,
    ) -> None:
        super().__init__()
        if answer_mode not in self.ANSWER_MODES:
            raise ValueError(f"answer_mode must be one of {self.ANSWER_MODES}, got {answer_mode!r}")
        self.config = dict(
            segmenter=dict(segmenter), spacing=tuple(float(v) for v in spacing),
            n_anchors=int(n_anchors), tau=float(tau), min_mass=float(min_mass),
            answer_mode=str(answer_mode),
            boundary_widths=tuple(int(w) for w in boundary_widths),
            carver_width=int(carver_width), carver_blocks=int(carver_blocks),
            full_resolution_skip=bool(full_resolution_skip), use_image=bool(use_image),
            carver_sees_anchors=bool(carver_sees_anchors),
            additive_prior=bool(additive_prior), alpha=float(alpha),
            background_logit=float(background_logit), prior_foreground=float(prior_foreground),
            dilate_radius=int(dilate_radius), region_threshold=float(region_threshold),
            max_seeds=int(max_seeds), intensity_tol=float(intensity_tol),
            tol_mode=str(tol_mode), score_null=float(score_null),
        )
        self.n_anchors = int(n_anchors)
        self.spacing = tuple(float(v) for v in spacing)
        self.answer_mode = str(answer_mode)
        self.use_image = bool(use_image)
        self.carver_sees_anchors = bool(carver_sees_anchors)
        self.additive_prior = bool(additive_prior)
        self.background_logit = float(background_logit)
        self.dilate_radius = int(dilate_radius)
        self.region_threshold = float(region_threshold)
        self.max_seeds = int(max_seeds)
        self.intensity_tol = float(intensity_tol)
        self.tol_mode = str(tol_mode)
        self.score_null = float(score_null)

        self.segmenter = StageA(**segmenter)
        self.segmenter.requires_grad_(False).eval()
        resolution = int(self.segmenter.config["resolution"])
        self.resolution = resolution
        self.center = tuple(volume_center_world((resolution,) * 3, self.spacing).tolist())

        self.mapper = PositionalMapper3D(tau=tau, min_mass=min_mass)
        # Instance v1 floods on intensity; BoundaryEncoder is kept when use_image
        # so a pretrained B can load for a later affinity flood without a rebuild.
        self.boundary = BoundaryEncoder(boundary_widths) if self.use_image else None
        if self.answer_mode == "carver":
            boundary_channels = self.boundary.out_channels if self.boundary is not None else 0
            geometry = (2 if self.carver_sees_anchors else 1) * self.n_anchors + 3
            self.carver = Carver(
                boundary_channels + geometry, boundary_channels,
                width=carver_width, blocks=carver_blocks,
                full_resolution_skip=full_resolution_skip, prior_foreground=prior_foreground,
            )
            self.null = NullHead(self.n_anchors)
            self.alpha = nn.Parameter(torch.tensor(float(alpha))) if self.additive_prior else None
        else:
            self.carver = None
            self.null = None
            self.alpha = None
        self.boundary_lr_scale = 1.0

    # -- the freeze -------------------------------------------------------
    def train(self, mode: bool = True) -> "StageB":
        super().train(mode)
        self.segmenter.eval()  # frozen; never a training-mode submodule
        return self

    def trainable_parameters(self):
        """Every parameter except the frozen segmenter's - what the optimiser gets."""
        frozen = {id(p) for p in self.segmenter.parameters()}
        return [p for p in self.parameters() if id(p) not in frozen]

    def parameter_groups(self, lr: float) -> list[dict]:
        """Two groups when ``B`` was pretrained: the answer path at ``lr``, ``B`` below it."""
        if self.boundary is None or self.boundary_lr_scale == 1.0:
            return [{"params": self.trainable_parameters(), "lr": lr}]
        boundary = list(self.boundary.parameters())
        held = {id(p) for p in boundary}
        return [
            {"params": [p for p in self.trainable_parameters() if id(p) not in held], "lr": lr},
            {"params": boundary, "lr": lr * self.boundary_lr_scale},
        ]

    def load_segmenter(self, model: StageA) -> "StageB":
        """Copy a trained Stage A's weights in, then re-freeze."""
        if dict(model.config) != dict(self.config["segmenter"]):
            raise ValueError(
                f"segmenter architecture {dict(model.config)} does not match the one this "
                f"Stage B was built for, {dict(self.config['segmenter'])}"
            )
        self.segmenter.load_state_dict(model.state_dict())
        self.segmenter.requires_grad_(False).eval()
        return self

    @classmethod
    def from_segmenter(cls, model: StageA, **kwargs) -> "StageB":
        """Build around a trained Stage A and copy its weights in."""
        return cls(dict(model.config), **kwargs).load_segmenter(model)

    # -- the forward ------------------------------------------------------
    def anchor_probability(self, image: Tensor, name_ids: Tensor) -> Tensor:
        """``A_i = stop_gradient(sigmoid(anchor_logits_i))``, for those three names only."""
        return self.segmenter.probability(image, name_ids).detach()

    def forward(
        self,
        image: Tensor,
        direction_ids: Tensor,
        name_ids: Tensor,
        *,
        anchors: Tensor | None = None,
        boundary_image: Tensor | None = None,
    ) -> StageBOutput:
        """``[B, 1, D, H, W]`` + ``[B, A]`` directions + ``[B, A]`` names -> :class:`StageBOutput`."""
        if name_ids.shape[1] != self.n_anchors or direction_ids.shape != name_ids.shape:
            raise ValueError(
                f"expected [B, {self.n_anchors}] direction and name ids, got "
                f"{tuple(direction_ids.shape)} and {tuple(name_ids.shape)}"
            )
        if anchors is None:
            anchors = self.anchor_probability(image, name_ids)
        anchors = anchors.detach().float()

        with torch.no_grad():
            field = self.mapper(anchors, direction_ids, self.spacing, self.center)
        where = field.where_raw

        if self.answer_mode == "instance":
            return self._forward_instance(image, field, anchors, boundary_image)
        return self._forward_carver(image, field, anchors, where, boundary_image)

    def _forward_instance(
        self,
        image: Tensor,
        field,
        anchors: Tensor,
        boundary_image: Tensor | None,
    ) -> StageBOutput:
        """Seed-flood + rule score. No relational mask gradients into G."""
        source = image if boundary_image is None else boundary_image
        batch = source.shape[0]
        logits, centroids, valid = [], [], []
        for i in range(batch):
            result = run_instance(
                source[i],
                field.where_raw[i],
                self.spacing,
                dilate_radius=self.dilate_radius,
                region_threshold=self.region_threshold,
                max_seeds=self.max_seeds,
                intensity_tol=self.intensity_tol,
                tol_mode=self.tol_mode,
                score_null=self.score_null,
            )
            # Finite background so BCE against a disagreeing target stays finite.
            mask_logits = torch.where(
                result.mask > 0.5,
                torch.full_like(result.mask, 10.0),
                torch.full_like(result.mask, self.background_logit),
            )
            # Exclude predicted anchors from the answer body.
            mask_logits = mask_logits.masked_fill(
                anchors[i].amax(dim=0) > 0.5, self.background_logit
            )
            logits.append(mask_logits)
            if result.winner >= 0:
                centroids.append(result.centroids[result.winner])
                valid.append(torch.tensor(8.0, device=source.device))
            else:
                centroids.append(torch.zeros(3, device=source.device))
                # where_mass alone as a soft emptiness signal (threshold null).
                mass = float(field.where_mass[i, 0])
                valid.append(torch.tensor(
                    math.log(max(mass, 1e-12) / max(1.0 - mass, 1e-12)),
                    device=source.device,
                ))
        return StageBOutput(
            logits=torch.stack(logits).unsqueeze(1),
            valid=torch.stack(valid),
            centroid=torch.stack(centroids),
            where_raw=field.where_raw,
            where_mass=field.where_mass,
            fields=field.fields,
            anchors=anchors,
            masses=field.masses,
            anchor_centroids=field.centroids,
        )

    def _forward_carver(
        self,
        image: Tensor,
        field,
        anchors: Tensor,
        where: Tensor,
        boundary_image: Tensor | None,
    ) -> StageBOutput:
        """B2 dense carver path (parent baseline)."""
        log_where = where.clamp_min(LOG_FLOOR).log() / -math.log(LOG_FLOOR)
        log_mass = (
            field.where_mass.clamp_min(LOG_FLOOR).log() / -math.log(LOG_FLOOR)
        ).reshape(-1, 1, 1, 1, 1).expand_as(where)

        boundary = None
        if self.boundary is not None:
            source = image if boundary_image is None else boundary_image
            boundary = self.boundary(source.to(torch.float32))
        parts = ([anchors] if self.carver_sees_anchors else []) + [
            field.fields, where, log_where, log_mass
        ]
        dtype = boundary.dtype if boundary is not None else torch.float32
        logits, heatmap = self.carver(
            torch.cat(([boundary] if boundary is not None else []) + [p.to(dtype) for p in parts], dim=1),
            boundary,
        )

        if self.alpha is not None:
            logits = logits + self.alpha * torch.logit(where.clamp(1e-4, 1 - 1e-4))
        logits = logits.masked_fill(anchors.amax(dim=1, keepdim=True) > 0.5, self.background_logit)

        return StageBOutput(
            logits=logits,
            valid=self.null(field.where_mass, field.masses),
            centroid=soft_argmax(heatmap, (self.resolution,) * 3, self.spacing),
            where_raw=where,
            where_mass=field.where_mass,
            fields=field.fields,
            anchors=anchors,
            masses=field.masses,
            anchor_centroids=field.centroids,
        )
