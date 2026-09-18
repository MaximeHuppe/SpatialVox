# 04 — Stage A: the promptable structure segmenter

`src/models.py: StageA`

Intensity volume in, one binary mask per requested structure name out. On
synthetic data it segments primitives; on MRI it is the anatomy segmenter that
supplies Stage B's anchors. It is also the only place in the project that looks
at image intensities.

```
image [B, 1, V, V, V]                      names [B, P]
  │                                          │
  ▼                                          ▼
Encoder: V → V/2 → … → bottleneck        NamePrompt → queries [B, P, dim]
  │                                           │
  ├────── bottleneck + 3D pos.enc. ───────────┤
  │                       (keys) / (values)   │
  │                    cross-attention ───────┘
  │                                           ▼
Decoder: bottleneck → V (skips)          aligned queries [B, P, dim]
  │   features at V/4, V/2, V                 │
  └────────── scaled dot product ─────────────┴──► logits at each scale
                                                  (0.1 / 0.3 / 0.6)
```

## The encoder

`ConvBlock(stride) + ResBlock` per scale, `InstanceNorm3d(affine=False)`, ReLU,
no transposed convolutions. The stem is stride 1; every later stage halves each
spatial dimension. Widths double from `base_channels` and stop at
`max_channels`; the bottleneck width must equal `dim`, because the prompt
decoder attends over it directly.

Stage A sees an image, so it takes no coordinate channels: the intensities
already say where things are, and the prompt decoder below supplies the spatial
grounding. (Stage B, which sees only masks, does need them — see
[05](05_stage_b.md).)

## The prompt decoder

Name queries attend over the flattened bottleneck. The asymmetry is the point and
is what makes the queries *spatially* aligned:

- **queries** carry no positional encoding — they say *what*;
- **keys** are the bottleneck plus a learned decomposed 3D positional encoding —
  they say *where*;
- **values** are the bottleneck unmodified — the visual content.

`aligned = LayerNorm(MHA(Q, K, V) + Q)`. The positional encoding is factorised
over the three axes, so it costs `(D + H + W) · C` parameters instead of
`D · H · W · C`.

This is the only cross-attention in Stage A, and it is affordable because the
bottleneck is 512 tokens — attention at full resolution would be 262,144 queries
at 64³ and eight times that at 128³.

## Deep supervision

The decoder produces features at three scales (16³, 32³, 64³ for a 64³ input),
and a mask head is applied to each. The loss is their weighted sum, `0.1 / 0.3 /
0.6` coarse to fine, with the targets max-pooled down to each scale — max, not
average, because at a quarter resolution a torus is about one voxel thick and
averaging can delete it, which would supervise the coarse head towards an empty
mask for a structure that is there.

Inference uses the full-resolution map alone. At a depth other than three the
configured weights cannot apply, and a doubling ramp normalised to sum to one
stands in, keeping the same shape: coarse scales matter least.

## The mask head

`aligned queries → project → scaled dot product against the full-resolution
decoder features`, plus a per-query bias:

```
logits[p] = ⟨project(query_p), features⟩ / sqrt(C) + bias(query_p)
```

Two consequences follow from what this head does *not* do. The queries never
interact with each other, and the visual features are never modulated by the
prompt set. So **the logits for one name do not depend on which other names were
requested** (`tests/test_models.py` checks this exactly). That is what lets Stage
A be trained on the whole vocabulary and then queried for just the three anchors
a relational prompt happens to name — no retraining, no separate inference path.

It is also what makes `prompts_per_item` safe: sampling a subset of names per
training item is a pure cost saving, not a change of objective. With a hundred
anatomical labels at 128³ you cannot afford every name in every item, and you do
not have to.

## The head bias starts at the foreground prior

`prior_bias(p) = log(p / (1 − p))`, with `p = model.prior_foreground`.

A zero-initialised head predicts 0.5 for every voxel. One structure covers well
under 1% of a volume, so training then opens by pushing a quarter of a million
background logits down before the Dice term carries any usable gradient. Starting
the bias at the base rate skips that plateau — measured on this architecture it
is worth roughly an order of magnitude in early convergence. The bias stays
learnable and the head weight is still zero-initialised, so nothing else changes.

One side effect worth knowing: an untrained model outputs a *constant* volume, so
any test of input sensitivity has to give the head a real weight first.

## Absent structures

A mask is `labels == id`, which is empty when the structure is not in that scene —
normal on real data, never on the synthetic corpus. Dice and IoU score two empty
masks as 1.0 and one empty against one non-empty as 0.0, so "this structure is
not here" is a correct answer the model can learn rather than a degenerate case.
