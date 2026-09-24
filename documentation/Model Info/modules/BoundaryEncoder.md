---
tags:
  - phase-b
---

### Relations
- `StageB.boundary`, always built. There is no prompt-only carver
- pretrained by [[BoundaryPretrainer]], then given 0.1× the carver's learning rate
- feeds [[Carver]] as keys and values only. It is not a stem channel
- documented in [[SpatialVox#11. Step 7 — Boundary encoder B(I), the WHAT]]

`B(I)` — the WHAT. Generic boundary features from the MRI, and **nothing else**. `forward` takes one argument and it is the image: no prompt, no name, no direction, no coordinate grid, no label.

That is the whole point. Its pretraining objectives carry no class id, so a structure Stage A has never been given a mask for is still a boundary here. **Stage A's feature pyramid is not a substitute** — those features were trained to light up *named* structures, which is the opposite property.

```mermaid
flowchart LR
    A["image [B,1,128³]"] --> B["**BoundaryEncoder**<br/>128³ → 64³ → 32³ → back"] --> C["boundary features [B,16,128³]"]
```

---

### Params (`__init__`)

`BoundaryEncoder(widths=(16, 32, 32), act="leaky_relu")`

| param | source | shipped | role |
| --- | --- | --- | --- |
| `widths` | `model.stage_b.boundary_widths` | `[16, 32, 32]` | one per scale; `widths[0]` is also `out_channels` |
| `act` | fixed | `"leaky_relu"` | Stage B convention |

| group | attribute | what | # params |
| --- | --- | --- | --- |
| down | `down.0` | `ConvBlock(1 → 16)` @128³, **stride 1, no ResBlock** | 432 |
| | `down.1` | `ConvBlock(16 → 32, s2) + ResBlock(32)` @64³ | 69,120 |
| | `down.2` | `ConvBlock(32 → 32, s2) + ResBlock(32)` @32³ | 82,944 |
| up | `up.0` | `ConvBlock(32+32 → 32)` @64³ | 55,296 |
| | `up.1` | `ConvBlock(32+16 → 16)` @128³ | 20,736 |
| **total** | | | **228,528** |

**87.3% of Stage B's trainable parameters** (228,528 of 261,748) — `B` is the large half, the carver the small one. Small beside the frozen Stage A's 17.0M.

#### No residual block at full resolution — measured

The finest stage is a bare `ConvBlock`. A 16→16 3×3×3 convolution on a 128³ volume is by a wide margin the most expensive operation in Stage B: the residual pair there cost **350 ms of a 530 ms training step** — two thirds of `B` — for 6% more parameters. Depth is cheaper one octave down, and the up path reads the finest skip again anyway. Recorded in [[SpatialVox#20.4 Architectural additions]].

---

### Source Code

```python
class BoundaryEncoder(nn.Module):
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
        skips = []
        x = image
        for stage in self.down:
            x = stage(x)
            skips.append(x)
        for level, skip in enumerate(skips[-2::-1]):
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=True)
            x = self.up[level](torch.cat([x, skip], dim=1))
        return x
```

---

### Forward

`[B, 1, D, H, W] → [B, 16, D, H, W]`

| step | shape | note |
| --- | --- | --- |
| input | `[B,1,128³]` | z-scored over the brain (`data.normalize`) |
| `down.0` | `[B,16,128³]` | full resolution, one conv |
| `down.1` | `[B,32,64³]` | stride 2 + residual |
| `down.2` | `[B,32,32³]` | stride 2 + residual |
| `up.0` | `[B,32,64³]` | trilinear ↑, `cat` with `down.1` |
| `up.1` | `[B,16,128³]` | trilinear ↑, `cat` with `down.0` |

Output is at **full resolution**, which matters: [[Carver]] carries it past its own stride-2 stem so the final 1×1 sees boundaries at the resolution they were computed at.

#### The signature is the guarantee

```python
def forward(self, image: Tensor) -> Tensor:
```

One argument. `tests/test_models.py::test_the_boundary_encoder_sees_the_image_and_nothing_else` asserts the parameter set is exactly `{"image"}` — a structural test, because the failure it guards against is somebody *adding* an argument.

---

### Invariants

1. **The image, and only the image.** No prompt, name, direction, coordinate grid or label.
2. **Full-resolution output.** `K` and `V` are read at that resolution.
3. **Trained without class ids.** The label-adjacency map used in pretraining carries no class channel and no target indicator, and is a *pretraining target*, never an inference input.
4. **Not Stage A's pyramid.** Those features carry named-structure semantics; these must not.

---

### What it does and does not buy — measured

`B(I)` is the difference between the model reading the image and redrawing a spatial prior. The prompt-only ablation (`B04 prompt-only-seed1` (archived)) removes it and keeps everything else:

| | supervised classes | held-out classes |
| --- | --- | --- |
| full model | **0.7812** | 0.0053 |
| `B(I)` removed | 0.5232 | **0.1171** |
| *prompt-blind floor* | *0.3067* | *0.1097* |

Removing it costs **0.258 Dice** — so the mask is not a shape redrawn from the field. Image replacement says the same from the other side: feeding `B` a *different subject's* MRI while keeping this subject's anchors and fields costs 40% of the Dice (0.7922 → 0.4747) while the centroid holds (1.72 → 5.47 mm).

**But the two rows cross.** On classes never supervised as a target, removing `B(I)` *improves* transfer 22×. A carver with `B(I)` can learn what each of its supervised classes looks like, so it does, and then has nothing to say about a new one; a carver without it can only put a blob where the field points, so it stays class-agnostic. The image is what makes the supervised number good and the transfer number bad.

Whether that is the image's fault or the corpus's is what the synthetic series measures ([[Result_tracker]]). A single global threshold recovers `labels > 0` at IoU **0.0796** on `data/mri`. On the easy synthetic corpus (IoU ≈ 0.994) transfer is perfect and meaningless, 0.98 (`B07 synthetic-stage-b` (archived)). On the hard one (0.27) it holds at 0.71 against a 0.15 floor, but the held-out shapes are twins of trained ones (`B08 hard-stage-b` (archived)). On the MRI-measured, family-split corpus (0.21) it is above floor but unstable, 0.37 / 0.46 against 0.25 / 0.16 (`B09 mri-stage-b` (archived)). The working hypothesis, that `B(I)` generalises when the image carries a class-agnostic boundary, is not settled yet: every run is single-seed, and a same-config replicate differs by 0.09 on the held-out curves.
