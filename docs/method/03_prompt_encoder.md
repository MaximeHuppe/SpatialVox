# 03 — Prompt encoder

`src/vocab.py`, `src/models.py: NamePrompt, RelationPrompt`

## The prompt is a rendering, not the source

The primary representation is an ordered clause list:

```python
[{"direction": "superior", "anchor": "cube"},
 {"direction": "medial",   "anchor": "torus"},
 {"direction": "anterior", "anchor": "sphere"}]
```

The sentence is generated from it by `Vocabulary.render`, and `Vocabulary.parse`
takes it back:

```
segment the structure that is superior to the cube, medial to the torus,
and anterior to the sphere.
```

`parse(render(clauses)) == clauses` for every valid clause list, and the parser
is built from the closed vocabularies at call time, so a synonym, an unknown name
or a target name cannot be parsed. Both the text and structured paths converge on
`Vocabulary.clause_ids`, the **single** place in the project where language
becomes integers. A prompt cannot mean one thing on disk and another in the
network, because there is only one conversion.

`Vocabulary.validate` enforces the rest: known tokens, pairwise-distinct
directions, pairwise-distinct anchors.

## Three tables, one token per clause

```
token_i = direction_embedding(d_i)
        + name_embedding(a_i)
        + pair_embedding(d_i, a_i)
        + slot_embedding(i)
```

then a `LayerNorm`. Output `[B, A, dim]` — one token per clause, **never pooled**.

Each term earns its place:

- **direction** — what the relation is.
- **name** — which structure it is to. Shared with Stage A
  (`NamePrompt`), so a structure name means the same thing to both networks.
- **pair** — `6 × |vocab|` rows, the one term that lets a relation be more than
  the sum of its parts. "lateral to the thalamus" need not behave like "lateral"
  plus "thalamus"; without this table the model can only ever represent
  additive combinations.
- **slot** — which clause position this is. Needed because the evidence branch is
  shared across clauses (see [05](05_stage_b.md)), so the slot has to live in the
  token rather than in the parameters.

The clauses stay separate all the way into the grounding branches. Pooling them
into one vector before grounding would destroy the correspondence between clause
*i* and mask channel *i*, which is exactly what the task is about — and it would
make the conjunction inexpressible, since a mean of three relations is not the
region where all three hold.

## Why an embedding table and not a language model

The vocabulary is closed. Every prompt is generated from a finite grammar over a
known list of structures, so an embedding table is the *exact* representation:
there is no open-vocabulary text to generalise over, and a learned table cannot
hallucinate a synonym. It also costs nothing to grow — a hundred anatomical names
is a hundred rows.

If free-text prompts become a requirement, `NamePrompt` is the swap point: a
frozen sentence encoder over the structure name, projected to `dim`, changes that
module and nothing else. The `pair` table would then need a different treatment
(it is indexed by name id), which is the real cost of the change and the reason
it is not there already.

## Structure tokens are the other half

The prompt tokens say *what was asked for*. Their counterparts — what the anchor
masks actually contain — are the structure tokens, built in
`src/models.py: StructureEncoder` and described in [05](05_stage_b.md). Clause
*i* is fused with structure token *i*, and that pairing is the correspondence the
whole model is built around.
