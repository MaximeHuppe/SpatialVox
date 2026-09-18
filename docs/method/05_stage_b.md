# 05 — Stage B: the relational target segmenter

`src/models.py: StageB`

Three ordered anchor masks, three clause tokens and one anonymous occupancy map
in; the target mask out. It never receives the label volume, the target mask, the
target's name or the target's position — inferring them is the task.

```
anchors [B, 3, V, V, V] ──► Encoder (+ world x,y,z at every scale) ──► features
                                                  │
 direction ids, name ids ──► RelationPrompt ──► R [B, 3, dim]
 anchors + bottleneck    ──► StructureEncoder ─► S [B, 3, dim]
                                                  │
     per clause i:  C_i = fuse(R_i, S_i)          │   ── shared branch
                    H_i = attend(bottleneck | C_i, R_i, S_i)
                                                  │
     Intersection([H_1, H_2, H_3, H_1·H_2·H_3]) ──┤
                                                  ▼
 occupancy (labels>0, anchors removed) ──► Decoder (skips, FiLM) ──► logits [B,1,V,V,V]
```

## What goes in, and what deliberately does not

| stream | reaches | why |
|---|---|---|
| anchor masks, ordered | encoder | the WHERE signal: the only geometry the prompt refers to |
| clause tokens | grounding branches, decoder FiLM | the relations themselves |
| occupancy `labels > 0`, anchors subtracted | decoder only | the WHAT signal: "some structure is here" |
| label volume, target mask, target name, target centroid | **nothing** | they are the answer |

Two placements carry most of the design.

**Occupancy never reaches the encoder.** If grounding queries could see the
target's own voxels, the model could ignore the prompt entirely and learn "the
blob that is not an anchor" — and on a corpus where that heuristic usually works,
it would score well. Keeping occupancy on the decoder side means the *where* is
decided from the relations, and occupancy only sharpens the boundary of a region
already chosen.

**The anchors are subtracted from occupancy.** They are the given, not the
answer; leaving them in would hand the decoder a free copy of the conditioning it
is supposed to be reasoning about. What remains is a single binary channel in
which the target is one unmarked structure among several — it says "something is
here", never "this one".

## Structure tokens: geometry read off the masks

`StructureEncoder` builds one token per anchor channel:

```
token_i = project(masked_pool(bottleneck, mask_i))   # what it looks like
        + geometry_mlp(mask_geometry(mask_i))        # where it is, how big
        + name_embedding(a_i)                        # what it is called
        + slot_embedding(i)                          # which clause it answers
```

`mask_geometry` returns eight numbers per channel: normalised centroid `(x, y, z)`
in `[-1, 1]`, bounding-box extent as a fraction of each axis, the cube root of
the occupied volume fraction (a linear size, which keeps the feature O(0.1)
rather than O(0.001)), and a presence flag.

Every one of those is measured **from the mask channel itself**, never read out
of the manifest. That is what makes ground-truth and predicted anchors
interchangeable: when Stage A hands over a slightly wrong mask, the token
describes *that* mask, so the oracle-versus-predicted gap measures segmentation
error rather than a change of interface. The presence flag exists so an empty
predicted channel reads as "no structure at all" rather than "a tiny structure at
the origin" — and `masked_pool` falls back to globally pooled features instead of
producing a NaN.

## Grounding one clause at a time

`Evidence`, applied per clause:

1. **Fuse** what was asked for with what the channel contains: the relation
   token queries `[R_i, S_i]` through a small cross-attention block, then a
   residual MLP — `C_i = LayerNorm(fused + MLP(fused))`.
2. **Ground** it: cross-attention with **visual locations as queries** and the
   clause's three tokens `{C_i, R_i, S_i}` as keys and values, the queries
   carrying a learned decomposed 3D positional encoding. Each location decides
   for itself how much of the relation, the anchor's appearance and their
   combination it needs.
3. A small convolutional head turns the attended grid into the evidence map
   `H_i`.

This is the only attention over space in Stage B, and it happens at the
bottleneck alone — 512 tokens. Full-resolution attention would be 262,144 queries
at 64³, which is why the decoder is conditioned by FiLM instead.

**The branch is shared across clauses.** One `Evidence` module is applied three
times. That is not only a parameter saving: with per-slot parameters a clause
could be grounded by a slot-specific shortcut, whereas with shared weights the
*only* thing distinguishing branch 2 from branch 1 is its tokens — which carry
the direction, the name, the pair and the slot embedding. The correspondence has
to be learned through the tokens or not at all.

## The intersection is explicit

```
Intersection([H_1, H_2, H_3, σ(H_1)·σ(H_2)·σ(H_3)])
```

The product term is the point of the module. It is high only where *every*
relation is satisfied at once, which is the definition of the target, and no sum
of the three maps can express that. The individual maps are kept alongside it so
softer combinations stay available. Each map is squashed through a sigmoid first:
the product of three unbounded activations is numerically brutal and its gradient
vanishes or explodes.

Three 1×1 convolutions mix the stack, then a 3×3 convolution refines it
spatially. The result is added to the bottleneck, so the decoder receives visual
features that have been *conditioned* rather than replaced.

## Decoder

Per stage: concatenate the world coordinates for that scale, trilinear upsample,
concatenate the encoder skip, concatenate the max-pooled occupancy through a 1×1
convolution, apply FiLM from the concatenated clause tokens, refine. FiLM goes on
every stage except the finest — at 64³ that is 16³ and 32³, which is where the
reference architecture puts it. FiLM is
`y = (1 + γ(context))·x + β(context)` with both projections zero-initialised, so
the module starts as the identity and the decoder is never handed a randomly
scrambled feature map at step 0.

Occupancy is max-pooled rather than averaged because a thin structure disappears
under averaging at a quarter resolution.

The head is a 1×1 convolution with zero weight and the foreground-prior bias, for
the reason given in [04](04_stage_a.md).

Every module above is the one from `exp/realistic-appearance`, parameter for
parameter: `tests/test_reference_parity.py` ports a checkpoint from that branch
into these classes and checks that every output tensor is bit-identical.

## Sizing

`model.encoder_channels` lists one width per scale, finest first, with a stride-2
stage between each pair — so its length sets the depth and the decoder mirrors it:

```yaml
encoder_channels: [32, 64, 128, 256]   # decoder: 128 -> 64 -> 32
bottleneck: 8                          # checked against resolution / 2^(len - 1)
```

The bottleneck is derived and then *checked* against the configured value, so
changing the resolution without changing the widths is an error that says what
to write rather than a model that silently attends over 4,096 tokens. Embedding
tables are sized from the vocabulary. See [08](08_scaling.md).
