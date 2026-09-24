---
tags:
  - phase-a
---

### Relations
none — basic block
- documented in [[SpatialVox#8. Step 4 — Stage A, the frozen segmenter]] (8.3)

Closed-vocabulary name → token. 23 names on `data/mri`; a lookup is exact. Open-vocabulary text is out of scope (closed compiler).

```mermaid
flowchart LR
    A["name_ids [B,P] int64"] --> B["**NamePrompt**<br/>Embedding → Linear"] --> C["[B,P,256]"]
    C --> D["Stage A: attention queries"]
```

**One instance now.** Older Stage B variants carried separate name tables past the freeze. The current architecture forbids that: **names stop at Stage A**. This is the only name embedding in the project, and it lives on the far side of the freeze. See [[MODEL PHASE B]].

---

### Params (`__init__`)

`NamePrompt(vocab_size, dim)`

| param | source | runtime | role |
| --- | --- | --- | --- |
| `vocab_size` `V` | `len(corpus.vocab)` | `23` | rows of `table`; **not** the sequence length `P` |
| `dim` `d` | `token_dim` | `256` | table and projection width; must equal the last encoder width |

| attribute                | type           | runtime shape | init                               | # params   |
| ------------------------ | -------------- | ------------- | ---------------------------------- | ---------- |
| `self.table`             | `nn.Embedding` | `(23, 256)`   | `trunc_normal_(std=0.02)`          | 5,888      |
| `self.projection.weight` | `nn.Linear`    | `(256, 256)`  | Linear default (`kaiming_uniform`) | 65,536     |
| `self.projection.bias`   |                | `(256,)`      | Linear default                     | 256        |
| **total**                |                |               |                                    | **71,680** |

A vocabulary-sized table is legal **here and nowhere else**. Anchors are consumed by name, and a held-out class still appears as an anchor, so its row is trained — which is precisely why the project may claim "never supervised as a relational target" and may **not** claim "zero-shot on an unseen structure". No module downstream of Stage A may have a parameter sized by the vocabulary; `tests/test_models.py::test_names_reach_stage_a_and_stop_there` holds every Stage B output bit-identical under an arbitrary renaming.

The square projection is mathematically redundant with the table. It was kept for compatibility with the reference Stage A checkpoint.

---

### Source Code

```python
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
```

---

### Forward

`[B, P] int64` → `[B, P, dim]`

`P` is the number of names in *this* call, not a constructor argument. The table always has `V = 23` rows; `P` is how many of them are indexed.

| caller             | construction          | `P`  | `name_ids` comes from | consumed as           |
| ------------------ | --------------------- | ---- | --------------------- | --------------------- |
| [[MODEL PHASE A]] training | `NamePrompt(23, 256)` | `23` | `batch["prompt_ids"]` | attention **queries** |
| [[MODEL PHASE B]] via the frozen segmenter | the same weights | `3` | `anchors - 1` | attention **queries**, then the masks are detached |

**Stage A:**

```python
queries = self.prompt(name_ids)                              # [B, 23, 256]
attended, _ = self.attention(queries, keys, values, ...)     # queries vs bottleneck
```

**Inside Stage B** — the same call, three names instead of twenty-three, under `torch.no_grad()`:

```python
anchors = stop_gradient(sigmoid(self.segmenter(image, name_ids).logits))   # [B,3,128³]
```

The token never leaves Stage A. What crosses into the relational path is a mask.

#### 0. Inputs

**`B`**: batch size (`2` in measured tables)
**`P`**: names requested this forward (`23` when training Stage A, `3` when Stage B queries it)
**`V`**: vocabulary size (`23`) — table rows, independent of `P`

`name_ids` is **vocabulary index** = `label - 1`. Label volumes store `vocab index + 1` (0 = background). Range is `0…22`, never `23`, never a label id.

| variable | code shape | runtime A / B | dtype | range | meaning |
| --- | --- | --- | --- | --- | --- |
| `name_ids` | `[B, P]` | `(2, 23)` / `(2, 3)` | `int64` | `0…22` | row of `table` |

#### 1. Table lookup

```python
self.table(name_ids)    # [B, P] → [B, P, dim]
```

| | Stage A | Stage B |
| --- | --- | --- |
| shape | `(2, 23, 256)` | `(2, 3, 256)` |
| dtype | **`float32`** | **`float32`** |

Embedding lookup is **not** autocast. Row `k` is `table.weight[k]`. Duplicate ids in the same batch return the same row (Stage A typically asks every class once; Stage B asks three, possibly repeating a class across examples not slots).

#### 2. Projection

```python
return self.projection(...)    # [B, P, dim] → [B, P, dim]
```

$$
t_p = W\, E[\text{name}_p] + b, \qquad E \in \mathbb{R}^{V \times d},\; W \in \mathbb{R}^{d \times d}
$$

| | Stage A | Stage B |
| --- | --- | --- |
| shape | `(2, 23, 256)` | `(2, 3, 256)` |
| dtype | **`bfloat16`** | **`bfloat16`** under autocast; Stage A's `norm` after attention is `float32` |

Shipped: \(W\) is square (`d = 256`).

#### 3. Output

| variable | code shape | runtime A / B | dtype | meaning |
| --- | --- | --- | --- | --- |
| return | `[B, P, dim]` | `(2, 23, 256)` / `(2, 3, 256)` | `bfloat16` (measured at the Linear) | one token per requested name, **same order as `name_ids`** |

No pooling, no slot embedding, no LayerNorm — `StageA.norm` after attention is the caller's.

---

### Indexing invariant

Output slot `i` is name `name_ids[i]`. Downstream alignment is the caller's job:

| | slot `i` is |
| --- | --- |
| Stage A | the `i`-th requested structure; `MaskHead` writes logits channel `i` |
| Stage B | the same structure as anchor mask channel `i`, `direction_ids[i]`, and field `F_i` |

Never reorder `name_ids` without the matching masks / directions.

---

### Gradient

`nn.Embedding` is sparse: only rows present in this batch get gradient.

| instance | lookups / example | P(a given row updates) |
| --- | --- | --- |
| Stage A (`P = 23`) | all 23 | ~100% |
| Stage B (`P = 3`) | 3 of 23 | ~13% |

`trunc_normal_(std=0.02)` on `table` (not the Linear) is for that sparsity — PyTorch's `N(0,1)` default is too large when a row sees few steps.

`projection` is dense: every step updates all of \(W, b\).
