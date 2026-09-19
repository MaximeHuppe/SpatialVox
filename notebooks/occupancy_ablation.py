# %% [markdown]
# # Why `anchors-only` and `none` scored the same
#
# `runs/occ_oracle_anchors-only` (Dice 0.2212) and `runs/occ_oracle_none`
# (Dice 0.2294) differ by 0.008. They should not differ at all: **the two arms
# feed Stage B a bitwise-identical tensor.**
#
# `StageB.forward` masks the occupancy channel with the complement of the anchor
# union (`src/models.py:726`):
#
# ```python
# occupancy = occupancy.to(torch.float32) * (1 - anchors.amax(dim=1, keepdim=True))
# ```
#
# and `occupancy_mode="anchors-only"` sets occupancy *to* that same anchor union
# (`src/engine.py:300`). Binary masks, so `u * (1 - u) == 0` exactly. The decoder
# receives zeros either way.
#
# This script proves that, measures what the 0.008 gap really is, and then asks
# the question the ablation was meant to answer — which it turns out not to have
# answered, because the `all` arm is a silhouette leak.
#
# Run as a script (`.venv/bin/python notebooks/occupancy_ablation.py`) or open as
# a notebook — the `# %%` markers make it one. Needs `pip install plotly`.

# %%
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data import Corpus, ExampleDataset, collate, loader
from src.engine import dice_iou, load_model, masks_from, occupancy_from, resolve_device

CORPUS = "data/synthetic"
RUNS = {
    "all": "runs/occ_oracle_all",
    "anchors-only": "runs/occ_oracle_anchors-only",
    "none": "runs/occ_oracle_none",
}
SPLIT = "val"
SCENE_INDEX = 0        # which example of the split to visualise

# Categorical slots 1-3 of the validated palette; fixed order, never cycled.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, SURFACE = "#1a1a19", "#6b6b68", "#fcfcfb"
SEQUENTIAL = "Blues"                                        # one hue, light -> dark
DIVERGING = [[0.0, BLUE], [0.5, "#f0f0ef"], [1.0, ORANGE]]  # two hues, neutral midpoint

cfg = load_config()
corpus = Corpus.load(CORPUS)
device = resolve_device(cfg.train.device)
vocab = corpus.vocab
torch.manual_seed(cfg.train.seed)

# %% [markdown]
# ## 1. What Stage B received in these runs
#
# > These three checkpoints were trained **before** `model.stage_b_image` existed,
# > so they carry `image=False` and the analysis below is their post-mortem. §8
# > covers what changed as a result. The checkpoints still load unaltered.
#
# A common misreading is that Stage B saw the intensity volume plus an occupancy
# map. It did not. For these runs `StageB.forward` took **three binary anchor
# masks** (encoder), **the clause ids**, and **one binary occupancy channel**
# (decoder only). The image never reached it — in `mode: oracle` no Stage A is
# even constructed, and `StageBTask.source` reads `batch["labels"]`, not
# `batch["image"]`.
#
# So in the `none` arm the network was given three blobs in an otherwise empty
# 64³ volume and a sentence. Nothing else — which is the whole story.

# %%
import inspect

from src.models import StageB

reference = load_model(RUNS["none"] + "/best.pt", device).eval()
first_conv = reference.encoder.stages[0][0][0]
print("StageB.forward parameters:", list(inspect.signature(StageB.forward).parameters)[1:])
print(f"encoder input channels   : {first_conv.in_channels} "
      f"= {reference.n_anchors} anchor masks + 3 coordinate channels "
      "— no image channel, no occupancy channel")
print("occupancy enters at      : the decoder only, after the anchor subtraction")
print("image flag on this run   :", reference.config.get("image", False), "(see section 8)")

dataset = ExampleDataset(corpus, SPLIT, normalize_mode=cfg.data.normalize)
batch = collate([dataset[i] for i in range(4)])
batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
anchors = masks_from(batch["labels"], batch["anchors"])
print(f"\nanchors tensor {tuple(anchors.shape)}, values {torch.unique(anchors).tolist()}"
      "  <- binary, which is what makes u * (1 - u) exactly zero")

# %% [markdown]
# ## 2. The mechanism: `anchors-only` is an alias of `none`
#
# Occupancy is counted before and after the subtraction `StageB.forward` applies.
# `anchors-only` survives with exactly zero voxels.

# %%
def occupancy_for(batch, anchors, mode):
    """Occupancy as `StageBTask` would build it, for an oracle source."""
    full = (batch["labels"] > 0).float().unsqueeze(1) if mode == "all" else None
    return occupancy_from(batch, mode, anchors=anchors, full=full)


def after_subtraction(occupancy, anchors):
    """The tensor the decoder actually sees — `src/models.py:726`."""
    return occupancy.float() * (1 - anchors.float().amax(dim=1, keepdim=True))


rows = []
post = {}
for mode in ("all", "anchors-only", "none"):
    occupancy = occupancy_for(batch, anchors, mode)
    post[mode] = after_subtraction(occupancy, anchors)
    rows.append((mode, int(occupancy.sum()), int(post[mode].sum())))

print(f"{'occupancy_mode':<16}{'voxels in':>12}{'voxels at decoder':>20}")
for mode, before, after in rows:
    print(f"{mode:<16}{before:>12}{after:>20}")

identical = torch.equal(post["anchors-only"], post["none"])
print(f"\nanchors-only and none, bitwise identical at the decoder: {identical}")
assert identical, "the premise of this analysis"

# %% [markdown]
# ## 3. So the 0.008 gap is the noise floor, measured for free
#
# Same seed, same config, identical input — the two runs should have been
# bit-identical. They were not, because nothing in the codebase pins
# non-determinism: `src/engine.py:549` sets `torch.manual_seed` and
# `torch.cuda.manual_seed_all` and stops there. No `cudnn.deterministic`, no
# `use_deterministic_algorithms`, and cuDNN autotuning over bf16 3D convolutions
# is free to pick different kernels — the two arms even allocate the
# pre-subtraction occupancy differently (`anchors.amax` vs `torch.zeros`).
#
# That accident is useful: it is a duplicate-run replicate, so the spread between
# these two curves is this setup's *non-determinism* noise. Because the two runs
# shared a seed, it is a **lower bound** on seed-to-seed spread, not an estimate
# of it — which only strengthens the case for the ≥3 seeds per arm that
# `docs/experiments_plan.md` §4 asks for.

# %%
curves = {
    name: [json.loads(line)["val"]["dice"] for line in open(Path(path) / "metrics.jsonl")]
    for name, path in RUNS.items()
}
duplicate = np.array(curves["anchors-only"]) - np.array(curves["none"])
print(f"per-epoch |difference|   mean {np.abs(duplicate).mean():.4f}   max {np.abs(duplicate).max():.4f}")
print(f"correlation of the two curves          {np.corrcoef(curves['anchors-only'], curves['none'])[0, 1]:.4f}")
print(f"best-Dice gap that prompted this       {abs(0.22117388 - 0.22942082):.4f}")
print("\n=> non-determinism alone moves val Dice by ~0.01 typical, ~0.03 worst case.")
print("   That is a LOWER BOUND on seed-to-seed spread, not a measurement of it:")
print("   these two runs shared a seed. Real seed variation can only be larger.")

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

figure = go.Figure()
for name, color in zip(("all", "anchors-only", "none"), (BLUE, ORANGE, AQUA)):
    figure.add_trace(go.Scatter(
        y=curves[name], mode="lines", name=name,
        line=dict(color=color, width=2),
        hovertemplate=f"{name}<br>epoch %{{x}}<br>val dice %{{y:.4f}}<extra></extra>",
    ))
    # Direct label: identity is never colour-alone, and it carries the relief
    # rule for the low-contrast slot.
    figure.add_annotation(
        x=len(curves[name]) - 1, y=curves[name][-1], text=f" {name}",
        showarrow=False, xanchor="left", font=dict(color=color, size=12),
    )
figure.update_layout(
    title="Validation Dice per epoch<br><sub>anchors-only and none are the same experiment "
          "— the spread between them is run-to-run noise</sub>",
    xaxis_title="epoch", yaxis_title="val Dice",
    plot_bgcolor=SURFACE, paper_bgcolor=SURFACE, font=dict(color=INK),
    xaxis=dict(gridcolor="#e8e8e6", zeroline=False, range=[0, len(curves["none"]) + 4]),
    yaxis=dict(gridcolor="#e8e8e6", zeroline=False, rangemode="tozero"),
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    margin=dict(l=60, r=60, t=70, b=50), height=420, showlegend=True,
)
figure.show()

# %% [markdown]
# ## 4. The arm that did move is a silhouette leak
#
# `all` reached Dice 0.9925 with **Hausdorff 0.66 — sub-voxel**. That is the
# signature of copying an outline, not of inferring a region. And after the
# anchor subtraction, `all` occupancy is exactly *target + 6 distractors*: the
# target's own silhouette is handed to the decoder.
#
# The test is eval-only, on the existing checkpoint. Re-score every checkpoint
# under every occupancy, including one the training never used:
# **`distractors-only` = all structures − anchors − target**, which keeps the
# candidate blobs but withholds the answer's outline.

# %%
@torch.no_grad()
def score(model, batches, kind):
    """Mean Dice over a split with occupancy built `kind`-wise."""
    scores = []
    for raw in batches:
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in raw.items()}
        anchor_masks = masks_from(b["labels"], b["anchors"])
        target = masks_from(b["labels"], b["target"].unsqueeze(1))
        full = (b["labels"] > 0).float().unsqueeze(1)
        if kind == "all":
            occupancy = full
        elif kind == "distractors-only":
            occupancy = (full - target).clamp(0, 1)     # every structure except the answer
        elif kind in ("none", "anchors-only"):
            occupancy = occupancy_from(b, kind, anchors=anchor_masks)
        else:
            raise ValueError(kind)
        logits = model(anchor_masks, b["direction_ids"], b["anchors"] - 1, occupancy).logits
        scores += dice_iou(logits, target)[0].flatten().tolist()
    return float(np.mean(scores))


batches = list(loader(dataset, batch_size=cfg.train.batch_size, shuffle=False, workers=cfg.train.workers))
evaluations = ("all", "distractors-only", "anchors-only", "none")
matrix = {}
for name, path in RUNS.items():
    model = load_model(Path(path) / "best.pt", device).eval()
    matrix[name] = {kind: score(model, batches, kind) for kind in evaluations}
    del model

header = "".join(f"{k:>18}" for k in evaluations)
print(f"Dice on {SPLIT} (n={len(dataset)}), rows = checkpoint, columns = occupancy at eval")
print(f"{'trained with':<16}{header}")
for name in RUNS:
    print(f"{name:<16}" + "".join(f"{matrix[name][k]:>18.4f}" for k in evaluations))

leak = matrix["all"]["all"] - matrix["all"]["distractors-only"]
print(f"\nall-checkpoint: {matrix['all']['all']:.4f} with the target in occupancy, "
      f"{matrix['all']['distractors-only']:.4f} without it  (drop {leak:.4f})")
print("  exactly zero, not merely degraded: a model *selecting* among the candidate")
print("  blobs would pick a wrong one and still score ~0.1-0.2. It emits nothing,")
print("  so it has learned output <= occupancy and cannot mark a voxel absent from it.")
print(f"\nnone-checkpoint: {matrix['none']['none']:.4f} with an empty occupancy, "
      f"{matrix['none']['all']:.4f} handed the full one")
print("  unchanged: it never built an occupancy pathway at all.")

# %%
labels = [f"{k}" for k in evaluations]
values = [matrix["all"][k] for k in evaluations]
figure = go.Figure(go.Bar(
    x=labels, y=values,
    marker=dict(color=[BLUE if v > 0.5 else MUTED for v in values],
                line=dict(color=SURFACE, width=2)),
    text=[f"{v:.4f}" for v in values], textposition="outside",   # direct labels
    hovertemplate="%{x}<br>dice %{y:.4f}<extra></extra>",
))
figure.update_layout(
    title="The <b>all</b> checkpoint, re-scored under each occupancy<br>"
          "<sub>remove the target's voxels from occupancy and it predicts nothing — "
          "the 0.99 was the answer's outline, supplied as input</sub>",
    xaxis_title="occupancy at evaluation", yaxis_title=f"val Dice",
    plot_bgcolor=SURFACE, paper_bgcolor=SURFACE, font=dict(color=INK),
    xaxis=dict(gridcolor="#e8e8e6"), yaxis=dict(gridcolor="#e8e8e6", range=[0, 1.1]),
    margin=dict(l=60, r=30, t=90, b=50), height=420, showlegend=False,
)
figure.show()

# %% [markdown]
# ## 5. Where each model puts its evidence
#
# Stage B localises in `Evidence` → `Intersection`. Each clause produces one
# spatial evidence map at the 8³ bottleneck; `Intersection.forward` then takes
# the **pointwise product of their sigmoids** — high only where every relation
# holds at once, which is the definition of the target.
#
# Those maps are the model's "where", so we read them out for the same scene
# under two checkpoints: `all` (which has the outline) and `none` (which does
# not). `Evidence` is a single shared module applied once per slot, so a hook on
# it fires three times per forward, in slot order.

# %%
example = collate([dataset[SCENE_INDEX]])
example = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in example.items()}
print(example["prompt"][0])
print(f"target (never an input): {example['target_name'][0]}")

example_anchors = masks_from(example["labels"], example["anchors"])
example_target = masks_from(example["labels"], example["target"].unsqueeze(1))
example_full = (example["labels"] > 0).float().unsqueeze(1)


@torch.no_grad()
def probe(path, occupancy):
    """Run one model, returning its evidence maps, conjunction, attention and output."""
    model = load_model(Path(path) / "best.pt", device).eval()
    captured = []
    handle = model.evidence.attention.register_forward_pre_hook(
        lambda module, args: captured.append(tuple(a.detach() for a in args))
    )
    output = model(example_anchors, example["direction_ids"], example["anchors"] - 1, occupancy)
    handle.remove()

    # `Evidence.forward` asks for need_weights=False, so replay the captured
    # inputs to get the weights themselves. Queries are the 512 bottleneck
    # locations; keys/values are the clause's three tokens.
    attention = [
        model.evidence.attention(*args, need_weights=True, average_attn_weights=True)[1]
        for args in captured
    ]
    maps = [torch.sigmoid(m).mean(dim=1, keepdim=True) for m in output.evidence]
    conjunction = torch.stack([torch.sigmoid(m) for m in output.evidence]).prod(0).mean(dim=1, keepdim=True)
    del model
    return dict(
        maps=[upsample(m) for m in maps],
        conjunction=upsample(conjunction),
        probability=torch.sigmoid(output.logits.float())[0, 0].cpu().numpy(),
        attention=torch.stack(attention),   # [slots, B, 512, 3]
    )


def upsample(volume):
    """8³ bottleneck map -> the 64³ grid, for display next to the masks."""
    resized = F.interpolate(volume.float(), size=example["labels"].shape[1:],
                            mode="trilinear", align_corners=False)
    return resized[0, 0].cpu().numpy()


probes = {
    "all": probe(RUNS["all"], example_full),
    "none": probe(RUNS["none"], torch.zeros_like(example_full)),
}

# %%
def mip(volume):
    """Maximum-intensity projection down z — an axial view of a 64³ volume."""
    return np.asarray(volume).max(axis=0)


def normalise(plane):
    span = plane.max() - plane.min()
    return (plane - plane.min()) / span if span > 0 else np.zeros_like(plane)


directions = example["directions"][0]
anchor_names = example["anchor_names"][0]

# Columns 1-3 are the three clauses; column 4 is what the model has to work
# from, column 5 what it produced. The reference row shows the matching truth,
# so the target sits directly above the two predictions.
panels = [
    [mip(example_anchors[0, i].cpu().numpy()) for i in range(3)]
    + [mip(example_full[0, 0].cpu().numpy()), mip(example_target[0, 0].cpu().numpy())],
    [normalise(mip(m)) for m in probes["all"]["maps"]]
    + [normalise(mip(probes["all"]["conjunction"])), mip(probes["all"]["probability"])],
    [normalise(mip(m)) for m in probes["none"]["maps"]]
    + [normalise(mip(probes["none"]["conjunction"])), mip(probes["none"]["probability"])],
]
# Every panel is titled: shared column headers would sit only above the
# reference row and mislabel the two model rows.
titles = (
    [f"anchor {i + 1} · {n}" for i, n in enumerate(anchor_names)]
    + ["occupancy (all)", "target — never an input"]
    + [f"{name} · {what}"
       for name in ("all", "none")
       for what in ("clause 1", "clause 2", "clause 3", "conjunction", "probability")]
)
rows = ["scene (reference)", "checkpoint: all", "checkpoint: none"]

figure = make_subplots(rows=3, cols=5, subplot_titles=titles,
                       horizontal_spacing=0.012, vertical_spacing=0.07)
for r, row in enumerate(panels, start=1):
    for c, plane in enumerate(row, start=1):
        index = (r - 1) * 5 + c
        figure.add_trace(
            go.Heatmap(z=plane, colorscale=SEQUENTIAL, showscale=False, zmin=0, zmax=1,
                       hovertemplate="%{z:.3f}<extra></extra>"),
            row=r, col=c,
        )
        figure.update_xaxes(visible=False, row=r, col=c)
        figure.update_yaxes(visible=False, scaleanchor="x" if index == 1 else f"x{index}",
                            row=r, col=c)

for annotation in figure.layout.annotations:
    annotation.font = dict(size=11, color=INK)
for r, name in enumerate(rows):
    figure.add_annotation(text=f"<b>{name}</b>", xref="paper", yref="paper",
                          x=-0.012, y=1 - (r + 0.5) / 3, showarrow=False,
                          xanchor="right", font=dict(size=12, color=INK), textangle=-90)

clauses = " · ".join(f"clause {i + 1}: {d} to the {n}"
                     for i, (d, n) in enumerate(zip(directions, anchor_names)))
figure.update_layout(
    title="Evidence maps, axial MIP — same scene, same prompt, two checkpoints<br>"
          f"<sub>{clauses}<br>each evidence panel is scaled to its own range, so read position, "
          "not brightness; the probability panels are absolute 0-1</sub>",
    plot_bgcolor=SURFACE, paper_bgcolor=SURFACE, font=dict(color=INK),
    margin=dict(l=130, r=30, t=130, b=30), height=760, showlegend=False,
)
figure.show()

# %% [markdown]
# Row 1 is the scene. Row 2 is the `all` checkpoint, which had the target's
# outline in its occupancy channel. Row 3 is `none`, which had an empty one.
#
# The telling comparison is *within* row 2. Its clause and conjunction maps are
# coarse, smooth regions — they live at the 8³ bottleneck, and they look much
# like row 3's. Yet its probability panel is a crisp rectangle matching the
# target voxel for voxel. That sharpness is not in the relational machinery; it
# can only have come from the occupancy channel at the decoder.
#
# So the two models localise about equally well and to about the same precision.
# What `all` has on top is the answer's outline to snap to.

# %%
mass = {}
for name, data in probes.items():
    probability = data["probability"]
    target = example_target[0, 0].cpu().numpy()
    inside = probability[target > 0.5].mean()
    outside = probability[(example_full[0, 0].cpu().numpy() > 0.5) & (target < 0.5)].mean()
    mass[name] = (inside, outside, probability.sum())
    print(f"{name:<6} mean p inside target {inside:.4f}   on other structures {outside:.4f}   "
          f"total mass {probability.sum():>9.1f}")
print(f"\ntarget voxels: {int(example_target.sum())}")

# %% [markdown]
# ## 6. Attention inside a clause
#
# The only attention over space in Stage B has the 512 bottleneck locations as
# *queries* and a clause's three tokens as keys and values, so the weights say
# how much each location leans on `{fused clause, relation, structure}` — not
# where in the volume it looks. Reported as such.

# %%
tokens = ["fused clause", "relation", "structure"]
print(f"{'checkpoint':<12}{'clause':<8}" + "".join(f"{t:>16}" for t in tokens))
for name, data in probes.items():
    weights = data["attention"]            # [slots, B, 512, 3]
    for slot in range(weights.shape[0]):
        share = weights[slot, 0].mean(0).cpu().numpy()
        print(f"{name:<12}{slot + 1:<8}" + "".join(f"{v:>16.4f}" for v in share))

# %%
figure = make_subplots(rows=1, cols=2, subplot_titles=[f"checkpoint: {n}" for n in probes],
                       shared_yaxes=True, horizontal_spacing=0.08)
for column, (name, data) in enumerate(probes.items(), start=1):
    weights = data["attention"][:, 0].mean(1).cpu().numpy()      # [slots, 3]
    for t, (token, color) in enumerate(zip(tokens, (BLUE, ORANGE, AQUA))):
        figure.add_trace(
            go.Bar(x=[f"clause {s + 1}" for s in range(weights.shape[0])],
                   y=weights[:, t], name=token, marker=dict(color=color, line=dict(color=SURFACE, width=2)),
                   text=[f"{v:.2f}" for v in weights[:, t]], textposition="outside",
                   showlegend=(column == 1),
                   hovertemplate=f"{token}<br>%{{x}}<br>%{{y:.4f}}<extra></extra>"),
            row=1, col=column,
        )
figure.update_layout(
    title="Mean attention weight per clause token, averaged over the 512 bottleneck locations",
    barmode="group", plot_bgcolor=SURFACE, paper_bgcolor=SURFACE, font=dict(color=INK),
    yaxis=dict(gridcolor="#e8e8e6", title="attention weight", range=[0, 1]),
    legend=dict(orientation="h", yanchor="bottom", y=1.12, xanchor="left", x=0),
    margin=dict(l=60, r=30, t=130, b=50), height=420,
)
figure.show()

# %% [markdown]
# ## 7. What this means for the ablation
#
# | arm | occupancy at the decoder | what it measures |
# |---|---|---|
# | `all` | target + 6 distractors | the answer's outline was supplied |
# | `anchors-only` | **empty** | identical to `none` |
# | `none` | empty | relations alone |
#
# Two results sharpen this beyond "the arms are duplicates":
#
# * `all` scores **exactly 0.0000** once the target's voxels leave the occupancy
#   channel. Not degraded — zero. It has learned `output ⊆ occupancy` and cannot
#   mark a voxel that occupancy does not already contain, so its 0.9925 is not
#   interpretable as relational capacity at all.
# * `none` scores the same (0.2282 vs 0.2163) whether it is handed an empty
#   occupancy or the full one. It never built a pathway for that channel. So the
#   0.22 arms are not "relational reasoning under a harder condition" either —
#   and `distractors-only` **training** may well land at 0.22 rather than
#   somewhere in the middle. That is the experiment, not a foregone conclusion.
#
# Because `forward` always subtracts the anchors, occupancy can only ever carry
# *non-anchor* structures. `anchors-only` is therefore not an arm, and a
# hypothetical `non-anchors` would be an alias of `all`. The only real knob is
# **whether the target's own voxels are in the occupancy channel** — which is
# why the three-arm ablation has two distinct conditions, and they are the two
# extremes.
#
# ## 8. What was changed as a result
#
# The deeper problem is not that one arm was a duplicate. It is that with an
# empty occupancy channel **Stage B could not see that any structure existed** —
# three blobs in an empty volume and a sentence. It was not reasoning badly; it
# was guessing a region, which is all the inputs allowed.
#
# So Stage B now takes the intensity volume, at the decoder, area-pooled to each
# scale (`model.stage_b_image`, on by default). The encoder still sees only the
# anchors, so localisation stays provably relational; the image carries "what is
# there" in place of an occupancy map built from ground-truth labels, which was
# oracle information no deployment has. Same `none` arm, same everything else:
#
# | arm | image | epochs | val Dice |
# |---|---|---|---|
# | `none` | off | 30 | 0.2294 |
# | `none` | **on** | 3 | **0.8912** |
# | `distractors-only` | **on** | 2 | **0.9197** |
#
# Those two image rows are short smoke runs at different epoch counts, not a
# head-to-head — they establish that the path trains, not how the arms rank.
# Ranking the arms is the ≥3-seed re-run in `docs/experiments_plan.md` §3.1.
#
# **And read them with this caveat.** `synthetic.appearance` puts background at
# `[0.12, 0.04]` and structures at `[0.45, 0.75]`, which do not overlap:
# background p99.9 is 0.260, structure p0.1 is 0.275, so one threshold recovers
# `labels > 0` at IoU 0.9998. On this corpus the image is therefore nearly as
# informative as `occupancy_mode: all` — the same candidate set, un-thresholded.
# What changes is that it comes from an acquired volume rather than ground-truth
# labels. 0.89 approaches the `all` ceiling of 0.99; it does not beat it, and the
# synthetic task did not get harder. On real MRI no threshold separates
# structures, which is what Stage A is for.
#
# The selection is nevertheless real, and these three controls are why:
#
# | check | Dice |
# |---|---|
# | real image | 0.8905 |
# | emit *every* non-anchor structure (no selection at all) | 0.2570 |
# | target erased from the image | 0.0019 |
# | held-out target class `triangular_prism`, never a target in training | 0.8356 |
#
# Plus the counterfactuals: `flip_direction` alone costs 0.57 Dice on val and
# 0.49 on test. A model emitting "the non-anchor blobs" would score 0.257, and
# one keyed to the target's class could not transfer to a class it never saw as
# a target. The boundary is easy here; choosing *which* structure is not, and
# that part is being done from the relations.
#
# `occupancy_mode` survives as an independent axis on top of it, now with four
# arms: `all` (the leak), `distractors-only` (new — the honest middle, an oracle
# **diagnostic** that `StageBTask` refuses under `mode: predicted`, because
# excluding a target you have not found is not something inference can do),
# `anchors-only` and `none` (kept, and documented as the aliases they are). The
# shipped default is now `none`.
#
# Everything measured above still stands as the post-mortem of the three runs in
# `runs/occ_oracle_*`, which were trained without the image and remain loadable.
# They are not comparable with anything trained after this change.
#
# And per §3: with non-determinism alone moving Dice by up to 0.03, no arm here
# is readable without the ≥3 seeds that `docs/experiments_plan.md` §4 asks for.

# %% [markdown]
# ## 9. The prompt-blind baseline — what any Dice here has to beat
#
# `select_anchors` ranks candidates by centroid distance, so a prompt's three
# anchors are the *nearest* structures to its target. That hands a model a
# shortcut that needs no prompt at all: "of the seven non-anchor structures, take
# the one closest to the anchors". Everything below is prompt-blind except the
# last row.
#
# This is the number `CLAUDE.md` §5 requires alongside any reported Dice. It is
# cheap, it needs no checkpoint, and it is what makes 0.89 readable.

# %%
from src.geometry import AmbiguousDirection, classify, centroids_world, volume_center_world
from src.data import load_nifti


def overlap(a, b):
    a, b = a > 0, b > 0
    return 2 * (a & b).sum() / max(a.sum() + b.sum(), 1)


def baselines(split):
    """Prompt-blind proximity, and the oracle that does read the relations."""
    records = ExampleDataset(corpus, split).records
    nearest, oracle, unique, near_miss = [], [], [], []
    for record in records:
        labels = load_nifti(corpus.root / "scenes" / record["scene"] / "labels.nii.gz", np.int16)
        centroids = centroids_world(labels, len(corpus.vocab), corpus.spacing)
        middle = volume_center_world(labels.shape, corpus.spacing)
        target, anchors_ = record["target"], list(record["anchors"])
        candidates = [int(v) for v in np.unique(labels) if v and int(v) not in anchors_]

        centre = np.mean([centroids[a] for a in anchors_], axis=0)
        pick = min(candidates, key=lambda l: np.linalg.norm(centroids[l] - centre))
        nearest.append(overlap(labels == pick, labels == target))

        def satisfied(label):
            total = 0
            for anchor, direction in zip(anchors_, record["directions"]):
                try:
                    total += classify(centroids[label], centroids[anchor], middle) == direction
                except AmbiguousDirection:
                    pass
            return total

        scores = [satisfied(l) for l in candidates]
        solutions = [l for l, s in zip(candidates, scores) if s == 3]
        unique.append(len(solutions) == 1)
        near_miss.append(sum(1 for s in scores if s == 2))
        oracle.append(overlap(labels == solutions[0], labels == target) if len(solutions) == 1 else np.nan)
    return dict(n=len(records), nearest=np.mean(nearest), oracle=np.nanmean(oracle),
                unique=np.mean(unique), near_miss=np.mean(near_miss))


for split in ("val", "test"):
    b = baselines(split)
    print(f"{split} (n={b['n']}, 7 candidates per example)")
    print(f"  prompt-blind: nearest candidate to the anchor centroid   {b['nearest']:.4f}")
    print(f"  oracle: the one satisfying all three stated relations    {b['oracle']:.4f}")
    print(f"  conjunction unique in {100 * b['unique']:.1f}% of examples; "
          f"{b['near_miss']:.2f} near-misses (2 of 3) on average\n")

# %% [markdown]
# So the readable range is **0.674 → 1.000**, not 0 → 1. Against it:
#
# | | val | test (held-out class) |
# |---|---|---|
# | prompt-blind proximity | 0.674 | 0.674 |
# | Stage B, image, `occupancy: none` | 0.891 | 0.836 |
# | oracle relational solver | 1.000 | 1.000 |
#
# The task is well posed — the conjunction picks out exactly one structure in
# 97.8% of examples — and the model is clearly above the shortcut. But two things
# stop that gap from being called relational reasoning:
#
# 1. **`permute_both` drops ~0.22 on every checkpoint**, old and new. It
#    preserves every relation and must not move. Until it doesn't, an unknown
#    fraction of the gap up to ~0.22 Dice is slot position, not relation content.
# 2. **The gap is smaller on the held-out class** (0.836 vs 0.891) — exactly
#    where relational reasoning has to carry the most weight.
#
# Fixing the control is the highest-value next step: without it, no arm in this
# ablation can be attributed to the thing the project is about.
