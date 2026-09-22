#!/usr/bin/env python
# %% [markdown]
# # Looking inside Stage B
#
# Three things, in the order the data meets them:
#
# 1. **`A_i`** — what the three anchor masks actually are. Soft, detached, and the
#    only thing a name ever buys.
# 2. **The mapper's output** — `F_0, F_1, F_2` and their product `where_raw`. No
#    parameters produced these; they are `classify` written as a soft pyramid.
# 3. **Feature maps at every resolution** — `B(I)` at 128³/64³/32³ and the carver's
#    working grid, as heatmaps.
#
# A script and a notebook at once (`# %%` cells). Headless-safe: every figure is
# written to a self-contained HTML file under `vizualization/out/`, and the
# numbers that matter are printed as well, so it is useful over ssh.
#
# ```
# .venv/bin/python vizualization/inspect_stage_b.py                     # MRI
# .venv/bin/python vizualization/inspect_stage_b.py --config configs/synthetic-hard.yaml \
#     --checkpoint runs/hard-stage-b/best.pt --split val
# ```

# %%
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if "__file__" in dir() else Path.cwd()
sys.path.insert(0, str(ROOT))

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from src.config import load_config
from src.data import Corpus, ExampleDataset, anchor_cache_dir, collate
from src.engine import load_model, mask_centroid_world, resolve_device, roll_anchors
from src.geometry import DIRECTIONS, OPPOSITE

OUT = ROOT / "vizualization" / "out"
OUT.mkdir(parents=True, exist_ok=True)

#: One palette for the whole file, so a colour means the same thing everywhere.
ANCHOR_COLOURS = ("#e64a19", "#1e88e5", "#43a047")   # slot 0, 1, 2
TARGET_COLOUR = "#ffd600"
#: where_raw was white, which is invisible over bright anatomy — on data/mri the
#: panel read as bare greyscale. Amber survives both a bright brain and a dark
#: synthetic background.
WHERE_COLOUR = "#ff6f00"


def parse(argv=None):
    p = argparse.ArgumentParser(description="Look inside one Stage B forward.")
    p.add_argument("--config", type=Path, default=None, help="default configs/config.yaml")
    p.add_argument("--checkpoint", type=Path, default=ROOT / "runs/relational-seed1/best.pt")
    p.add_argument("--split", default="val")
    p.add_argument("--index", type=int, default=0, help="which example of the split")
    p.add_argument("--classes", default="train", help="targets.<name> to draw the example from")
    p.add_argument("--tag", default="", help="suffix for the output filenames, so two corpora can coexist")
    return p.parse_args(argv if argv is not None else ([] if "ipykernel" in sys.modules else None))


args = parse()
cfg = load_config(args.config)
corpus = Corpus.load(cfg.data.root)
device = resolve_device(cfg.train.device)
print(f"corpus {corpus.root} · {len(corpus.vocab)} structures at {corpus.shape} · {corpus.spacing} per voxel")

# %% [markdown]
# ## One example, one forward
#
# Everything below comes from a **single** `StageB.forward`. The model is loaded
# from its checkpoint, so the architecture is the checkpoint's, not the YAML's.

# %%
model = load_model(args.checkpoint, device)
classes = list(cfg.targets[args.classes]) if args.classes else None
cache = anchor_cache_dir(corpus.root, Path(cfg.train.stage_b.phase_a_checkpoint))
dataset = ExampleDataset(
    corpus, args.split, targets=classes,
    anchor_cache=cache if (cache / "meta.json").is_file() else None,
    normalize_mode=cfg.data.normalize,
)
batch = collate([dataset[args.index]])
batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

print(f"\nprompt: {batch['prompt'][0]}")
print(f"target: {batch['target_name'][0]}")
for slot, (name, direction) in enumerate(zip(batch["anchor_names"][0], batch["directions"][0])):
    print(f"  slot {slot}: the target is {direction:<9} of {name}")

# `anchors` holds LABEL values (vocabulary index + 1, §9), which is what
# `vocab.name()` is indexed by; Stage A wants the index itself.
name_ids = batch["name_ids"] if "name_ids" in batch else batch["anchors"] - 1
with torch.no_grad():
    out = model(batch["image"], batch["direction_ids"], name_ids,
                anchors=batch.get("anchor_probability"))

image = batch["image"][0, 0].float().cpu().numpy()
labels = batch["labels"][0].cpu().numpy()
target_mask = (labels == int(batch["target"][0])).astype(np.float32)
anchors = out.anchors[0].float().cpu().numpy()          # [3, D, H, W]
fields = out.fields[0].float().cpu().numpy()            # [3, D, H, W]
where = out.where_raw[0, 0].float().cpu().numpy()       # [D, H, W]
spacing = corpus.spacing

# the slice we cut everything through: the target's own centroid
centroid, _ = mask_centroid_world(torch.from_numpy(target_mask)[None, None], spacing)
cz, cy, cx = (int(round(float(centroid[0, i]) / spacing[i])) for i in (2, 1, 0))
cz, cy, cx = (int(np.clip(v, 0, n - 1)) for v, n in zip((cz, cy, cx), labels.shape))
print(f"\nslicing through the target centroid at (z,y,x) = ({cz}, {cy}, {cx})")


# %%
def grey(slice2d: np.ndarray) -> go.Heatmap:
    """The anatomy underneath, always the same greyscale and scaling."""
    lo, hi = np.percentile(slice2d, (1, 99))
    return go.Heatmap(
        z=slice2d, colorscale="gray", zmin=lo, zmax=hi,
        showscale=False, hoverinfo="skip",
    )


#: Every overlay drawn since the last save(), and how much of the panel it covers.
#: A panel that renders as bare greyscale is the one failure this file can have
#: while still exiting 0, so it is reported rather than left to the eye.
_layers: list[tuple[str, float]] = []


def overlay(slice2d: np.ndarray, colour: str, name: str, threshold: float = 0.05) -> go.Heatmap:
    """A soft map on top, transparent where it is ~zero so the image shows through."""
    masked = np.where(slice2d > threshold, slice2d, np.nan)
    _layers.append((name, float((slice2d > threshold).mean())))
    return go.Heatmap(
        z=masked, colorscale=[[0, "rgba(0,0,0,0)"], [1, colour]],
        zmin=0, zmax=1, showscale=False, name=name,
        hovertemplate=f"{name}: %{{z:.3f}}<extra></extra>",
    )


def contour(slice2d: np.ndarray, colour: str, name: str) -> go.Contour:
    return go.Contour(
        z=slice2d, contours=dict(start=0.5, end=0.5, size=0, coloring="none"),
        line=dict(color=colour, width=2), showscale=False, name=name, hoverinfo="skip",
    )


def save(fig: go.Figure, stem: str, *, empty_is_the_point: bool = False) -> Path:
    """Write the figure, and say so if any overlay came out with nothing in it.

    ``empty_is_the_point`` for the counterfactual figure, where a prediction that
    vanishes under a broken prompt is the result rather than a rendering fault.
    """
    path = OUT / f"{stem}{'_' + args.tag if args.tag else ''}.html"
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)
    print(f"  -> {path.relative_to(ROOT)}")
    empty = [name for name, covered in _layers if covered == 0.0]
    if empty and empty_is_the_point:
        print(f"     (empty, as it should be: {', '.join(empty)} — the model predicts"
              " nothing once the prompt is broken)")
    elif empty:
        print(f"     !! nothing to see in: {', '.join(empty)} — that panel is bare greyscale")
    _layers.clear()
    return path


# %% [markdown]
# ## 1 · What is `A_i`?
#
# `A_i = stop_gradient(sigmoid(stage_a(image, name_ids)_i))`.
#
# Three things to see:
#
# - it is **soft**, not a threshold — the colour ramp is the probability, and a
#   cut at 0.5 would make the centroid the mapper reads jump;
# - it is **detached** — no gradient flows back into Stage A;
# - it is the **only** thing a name buys. Nothing downstream of here sees a name.

# %%
# Each anchor is sliced through ITS OWN soft centroid, not the target's. The
# three structures a prompt names are rarely coplanar — on data/mri the target's
# plane misses two of the three entirely, and those panels render as bare
# greyscale. A figure whose point is "what is A_i" must not be empty.
slot_z = [
    int(np.clip(round(float(out.anchor_centroids[0, i, 2]) / spacing[2]), 0, labels.shape[0] - 1))
    for i in range(anchors.shape[0])
]
titles = [f"slot {i}: {n}<br><sub>{d} → the target · axial z={z}</sub>"
          for i, (n, d, z) in enumerate(zip(batch["anchor_names"][0], batch["directions"][0], slot_z))]
fig = make_subplots(rows=1, cols=3, subplot_titles=titles, horizontal_spacing=0.04)
for slot, z in enumerate(slot_z):
    fig.add_trace(grey(image[z]), row=1, col=slot + 1)
    fig.add_trace(overlay(anchors[slot][z], ANCHOR_COLOURS[slot], f"A_{slot}"), row=1, col=slot + 1)
    fig.add_trace(contour(target_mask[z], TARGET_COLOUR, "target"), row=1, col=slot + 1)
    # The soft centroid is the ONLY thing the mapper takes from this mask, so it
    # is drawn. When Stage A splits a structure in two the cross lands in empty
    # space between the fragments — the failure is otherwise invisible here.
    c = out.anchor_centroids[0, slot].cpu().numpy()
    fig.add_trace(go.Scatter(
        x=[c[0] / spacing[0]], y=[c[1] / spacing[1]], mode="markers",
        marker=dict(symbol="x-thin", size=13, line=dict(color="#ffffff", width=3)),
        name=f"centroid {slot}", hovertemplate="soft centroid<extra></extra>",
    ), row=1, col=slot + 1)
fig.update_layout(
    title="1 · A_i — the three detached soft anchor masks, each through its own centroid"
          "<br><sub>yellow outline = the target where it reaches that plane; the model is never shown it</sub>",
    height=430, width=1250, template="plotly_dark", margin=dict(t=110),
)
fig.update_yaxes(autorange="reversed", scaleanchor="x", constrain="domain")
fig.update_xaxes(constrain="domain")
save(fig, "1_anchors")

print("\nwhat the mapper reads off each mask (it consumes the CENTROID, not the mask):")
print(f"  {'slot':<5}{'structure':<24}{'mass':>11}{'max p':>8}{'soft centroid (x,y,z)':>30}")
for slot in range(anchors.shape[0]):
    mass = float(out.masses[0, slot])
    c = out.anchor_centroids[0, slot].cpu().numpy()
    print(f"  {slot:<5}{batch['anchor_names'][0][slot]:<24}{mass:>11.2e}{anchors[slot].max():>8.3f}"
          f"{f'({c[0]:6.1f},{c[1]:6.1f},{c[2]:6.1f})':>30}")
print(f"  min_mass = {model.mapper.min_mass:.1e}; a slot below it writes F_i = 0 and zeroes where_raw")

# A split mask keeps a plausible mass and a max of 1.0 while its centroid drifts
# to the gap between the fragments. Counting components is what catches it.
try:
    from scipy.ndimage import label as _cc
    print("\n  mask integrity (the mapper's centroid is only meaningful on ONE blob):")
    for slot in range(anchors.shape[0]):
        lab = int(batch["anchors"][0, slot])
        truth, predicted = _cc(labels == lab)[1], _cc(anchors[slot] > 0.5)[1]
        flag = "  <- SPLIT, centroid falls between fragments" if predicted > truth else ""
        print(f"    slot {slot} {batch['anchor_names'][0][slot]:<22}"
              f"truth {truth} component(s), Stage A {predicted}{flag}")
except ImportError:
    pass

# %% [markdown]
# ### 1b · What the mapper actually sees
#
# Not the masks — **four points**. The mapper reduces each `A_i` to one soft
# centroid and places a pyramid there; nothing else about the mask survives.
# Drawing those points, with an arrow per direction word, is the most faithful
# picture of the task there is, and unlike a slice it cannot hide an anchor that
# lives in another plane.
#
# The target centroid is drawn for reference only — the model is never given it.

# %%
anchor_centroids = out.anchor_centroids[0].cpu().numpy()          # [3, 3] world (x,y,z)
target_centroid = centroid[0].cpu().numpy()
extent = [n * sp for n, sp in zip(labels.shape[::-1], spacing)]   # (x, y, z)

fig = go.Figure()
for slot in range(3):
    c = anchor_centroids[slot]
    fig.add_trace(go.Scatter3d(
        x=[c[0]], y=[c[1]], z=[c[2]], mode="markers+text",
        marker=dict(size=9, color=ANCHOR_COLOURS[slot]),
        text=[f" {batch['anchor_names'][0][slot]}"], textposition="middle right",
        name=f"slot {slot}: {batch['directions'][0][slot]}",
    ))
    # anchor -> target: the relation the clause asserts
    fig.add_trace(go.Scatter3d(
        x=[c[0], target_centroid[0]], y=[c[1], target_centroid[1]], z=[c[2], target_centroid[2]],
        mode="lines", line=dict(color=ANCHOR_COLOURS[slot], width=4, dash="dot"),
        hovertemplate=f"target is {batch['directions'][0][slot]} of "
                      f"{batch['anchor_names'][0][slot]}<extra></extra>",
        showlegend=False,
    ))
fig.add_trace(go.Scatter3d(
    x=[target_centroid[0]], y=[target_centroid[1]], z=[target_centroid[2]],
    mode="markers+text", marker=dict(size=11, color=TARGET_COLOUR, symbol="diamond"),
    text=[f" {batch['target_name'][0]} (truth)"], textposition="middle right", name="target",
))
predicted_centroid = out.centroid[0].cpu().numpy()
fig.add_trace(go.Scatter3d(
    x=[predicted_centroid[0]], y=[predicted_centroid[1]], z=[predicted_centroid[2]],
    mode="markers", marker=dict(size=9, color="#00e5ff", symbol="cross"), name="predicted centroid",
))
fig.update_layout(
    title="1b · the four points the mapper reduces everything to"
          "<br><sub>dotted line = the relation a clause asserts · the model never sees the yellow diamond</sub>",
    height=640, width=900, template="plotly_dark", margin=dict(t=100),
    scene=dict(xaxis_title="x (world)", yaxis_title="y (world)", zaxis_title="z (world)",
               xaxis=dict(range=[0, extent[0]]), yaxis=dict(range=[0, extent[1]]),
               zaxis=dict(range=[0, extent[2]]), aspectmode="data"),
)
save(fig, "1b_centroids")

# %% [markdown]
# ## 2 · The mapper's output
#
# `F_i = sigmoid(margin_i / tau)` — `classify` as a soft 45° pyramid — and
# `where_raw = F_0 · F_1 · F_2`.
#
# **Zero parameters produced these.** The product is *never* divided by its own
# maximum: a sigmoid is never exactly zero, so an impossible conjunction still has
# a tiny peak, and renormalising would turn it into a confident answer.

# %%
fig = make_subplots(
    rows=1, cols=4, horizontal_spacing=0.03,
    subplot_titles=[f"F_{i}: {d}<br><sub>of {n}</sub>" for i, (n, d) in
                    enumerate(zip(batch["anchor_names"][0], batch["directions"][0]))]
                   + ["<b>where_raw = F₀·F₁·F₂</b><br><sub>never renormalised</sub>"],
)
for slot in range(3):
    fig.add_trace(grey(image[cz]), row=1, col=slot + 1)
    fig.add_trace(overlay(fields[slot][cz], ANCHOR_COLOURS[slot], f"F_{slot}", 0.02),
                  row=1, col=slot + 1)
    fig.add_trace(contour(anchors[slot][cz], "#ffffff", "anchor"), row=1, col=slot + 1)
fig.add_trace(grey(image[cz]), row=1, col=4)
fig.add_trace(overlay(where[cz], WHERE_COLOUR, "where_raw", 0.02), row=1, col=4)
fig.add_trace(contour(target_mask[cz], TARGET_COLOUR, "target"), row=1, col=4)
fig.update_layout(
    title=f"2 · the mapper — three pyramids and their product  ·  axial z={cz}"
          f"<br><sub>white outline = the anchor each field is placed on · yellow = the target</sub>",
    height=420, width=1600, template="plotly_dark", margin=dict(t=110),
)
fig.update_yaxes(autorange="reversed", scaleanchor="x", constrain="domain")
fig.update_xaxes(constrain="domain")
save(fig, "2_mapper")

at_target = float(where[cz, cy, cx])
inside = float((target_mask * (where > 0.05)).sum() / max(target_mask.sum(), 1))
print(f"\n  where_raw at the target's centroid : {at_target:.4f}   (the gate asks: > 0.5?)")
print(f"  where_mass                         : {float(out.where_mass[0, 0]):.3e}")
print(f"  volume fraction of where_raw > .05 : {float((where > 0.05).mean()):.2%}")
print(f"  target voxels inside that region   : {inside:.1%}")
# The field CONTAINS the target without POINTING at it. Measure that on this very
# example rather than quoting a corpus-wide constant: the centre of mass of
# where_raw is not the target's centroid, because the conjunction of three cones
# is an elongated wedge with the target somewhere near its apex.
field_centre, _ = mask_centroid_world(torch.from_numpy(where)[None, None], spacing)
offset = float(torch.linalg.vector_norm(field_centre[0] - centroid[0]))
print(f"\n  the field's own centre of mass is {offset:.1f} world units from the target's")
print("  centroid: where_raw CONTAINS the target, it does not POINT at it. The")
print("  conjunction of three cones is an elongated wedge, not a blob on the answer.")

# %% [markdown]
# ### The same thing in three planes
#
# One axial slice can mislead — a pyramid is a 3D object. Here is `where_raw`
# through all three planes of the target's centroid.

# %%
planes = [("axial z", where[cz], image[cz], target_mask[cz]),
          ("coronal y", where[:, cy], image[:, cy], target_mask[:, cy]),
          ("sagittal x", where[:, :, cx], image[:, :, cx], target_mask[:, :, cx])]
fig = make_subplots(rows=1, cols=3, subplot_titles=[p[0] for p in planes], horizontal_spacing=0.04)
for col, (_, w, im, tm) in enumerate(planes, start=1):
    fig.add_trace(grey(im), row=1, col=col)
    fig.add_trace(overlay(w, WHERE_COLOUR, "where_raw", 0.02), row=1, col=col)
    fig.add_trace(contour(tm, TARGET_COLOUR, "target"), row=1, col=col)
fig.update_layout(title="2b · where_raw through the target's centroid, three planes",
                  height=430, width=1250, template="plotly_dark", margin=dict(t=90))
fig.update_yaxes(autorange="reversed", scaleanchor="x", constrain="domain")
fig.update_xaxes(constrain="domain")
save(fig, "2b_mapper_planes")

# %% [markdown]
# ## 3 · Feature maps at every resolution
#
# `B(I)` is a small U-net: 128³ → 64³ → 32³ and back. The carver then takes a
# stride-2 stem, so its own working grid is 64³.
#
# Captured with forward hooks — nothing in `src/` is modified. For each stage the
# heatmap is the **mean over channels** (what the stage responds to overall), and
# beside it the single **highest-variance channel** (what it most discriminates).

# %%
captured: dict[str, torch.Tensor] = {}


def grab(name):
    def hook(_module, _inputs, output):
        captured[name] = output.detach().float()
    return hook


handles = []
if model.boundary is not None:
    for i, stage in enumerate(model.boundary.down):
        handles.append(stage.register_forward_hook(grab(f"B down {i}")))
    for i, stage in enumerate(model.boundary.up):
        handles.append(stage.register_forward_hook(grab(f"B up {i}")))
handles.append(model.carver.stem.register_forward_hook(grab("carver stem")))
handles.append(model.carver.blocks.register_forward_hook(grab("carver blocks")))

with torch.no_grad():
    out = model(batch["image"], batch["direction_ids"], name_ids,
                anchors=batch.get("anchor_probability"))
for h in handles:
    h.remove()

print("\ncaptured:")
for name, t in captured.items():
    print(f"  {name:<16} {tuple(t.shape[1:])}")


# %%
def at_depth(volume: np.ndarray, fraction: float) -> np.ndarray:
    """The slice at the same *relative* depth, whatever the resolution."""
    return volume[int(np.clip(round(fraction * (volume.shape[0] - 1)), 0, volume.shape[0] - 1))]


depth = cz / max(labels.shape[0] - 1, 1)
names = list(captured)
fig = make_subplots(
    rows=2, cols=len(names), horizontal_spacing=0.015, vertical_spacing=0.10,
    subplot_titles=([f"{n}<br><sub>{tuple(captured[n].shape[1:])}</sub>" for n in names]
                    + ["" for _ in names]),
)
# Auto-scaling every panel independently would make a bright panel mean nothing —
# the comparison this figure exists for is ACROSS resolutions, so each row gets
# one shared range and one colourbar.
planes = {}
for name in names:
    feature = captured[name][0].cpu().numpy()                     # [C, d, h, w]
    busiest = int(feature.reshape(feature.shape[0], -1).var(1).argmax())
    planes[name] = (at_depth(feature.mean(0), depth), at_depth(feature[busiest], depth), busiest)
mean_lo = min(float(v[0].min()) for v in planes.values())
mean_hi = max(float(v[0].max()) for v in planes.values())
chan_lo = min(float(v[1].min()) for v in planes.values())
chan_hi = max(float(v[1].max()) for v in planes.values())

for col, name in enumerate(names, start=1):
    mean_plane, chan_plane, busiest = planes[name]
    last = col == len(names)
    fig.add_trace(
        go.Heatmap(z=mean_plane, colorscale="Viridis", zmin=mean_lo, zmax=mean_hi,
                   showscale=last, colorbar=dict(len=0.4, y=0.79, x=1.005, thickness=12,
                                                 title=dict(text="mean", side="right")),
                   hovertemplate="mean %{z:.3f}<extra></extra>"),
        row=1, col=col,
    )
    fig.add_trace(
        go.Heatmap(z=chan_plane, colorscale="Inferno", zmin=chan_lo, zmax=chan_hi,
                   showscale=last, colorbar=dict(len=0.4, y=0.21, x=1.005, thickness=12,
                                                 title=dict(text="channel", side="right")),
                   hovertemplate=f"ch {busiest}: %{{z:.3f}}<extra></extra>"),
        row=2, col=col,
    )
    # The channel index belongs in the title, not only the tooltip.
    fig.layout.annotations[len(names) + col - 1].update(
        text=f"ch {busiest}<br><sub>highest variance</sub>") if len(fig.layout.annotations) > len(names) else None
fig.update_layout(
    title="3 · feature maps by resolution  ·  top: mean over channels · bottom: highest-variance channel"
          f"<br><sub>same relative depth in every panel (z ≈ {depth:.0%} through the volume) · "
          "one shared scale per row, so brightness is comparable across resolutions</sub>",
    height=720, width=280 * len(names), template="plotly_dark", margin=dict(t=120),
)
fig.update_yaxes(autorange="reversed", scaleanchor="x", constrain="domain")
fig.update_xaxes(constrain="domain")
save(fig, "3_features")

# %% [markdown]
# ### What the carver finally produces
#
# The mask, and — separately — the heatmap whose soft-argmax is the reported
# centroid. The heatmap is **not** a reading of the mask: it survives the prompt
# being empty, which the mask does not.

# %%
probability = torch.sigmoid(out.logits[0, 0].float()).cpu().numpy()
with torch.no_grad():
    # The same 1x1 the model uses, re-applied to the captured features. Verified
    # against the model: soft_argmax of this lands 0.00e+00 from out.centroid.
    #
    # The LOGITS are plotted, not their softmax. A softmax over 64^3 voxels is
    # very nearly a delta — it rendered as two lit pixels on a flat field. Since
    # log(softmax(x)) = x - logsumexp(x), an additive constant, the logits and
    # the log-probability are the same picture under per-panel scaling, and this
    # one is readable.
    heatmap = model.carver.heatmap(captured["carver blocks"])[0, 0].float().cpu().numpy()

fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.04,
                    subplot_titles=["predicted probability", "thresholded vs truth",
                                    "heatmap logits<br><sub>soft-argmax integrates their softmax</sub>"])
fig.add_trace(grey(image[cz]), row=1, col=1)
fig.add_trace(overlay(probability[cz], "#00e5ff", "p", 0.02), row=1, col=1)
fig.add_trace(grey(image[cz]), row=1, col=2)
fig.add_trace(overlay((probability[cz] >= 0.5).astype(float), "#00e5ff", "predicted", 0.5), row=1, col=2)
fig.add_trace(contour(target_mask[cz], TARGET_COLOUR, "truth"), row=1, col=2)
fig.add_trace(go.Heatmap(z=at_depth(heatmap, depth), colorscale="Magma", showscale=True,
                         colorbar=dict(len=0.85, thickness=12, x=1.005),
                         hovertemplate="logit %{z:.2f}<extra></extra>"), row=1, col=3)
fig.update_layout(
    title="the two outputs  ·  cyan = predicted, yellow = truth",
    height=430, width=1250, template="plotly_dark", margin=dict(t=90),
)
fig.update_yaxes(autorange="reversed", scaleanchor="x", constrain="domain")
fig.update_xaxes(constrain="domain")
save(fig, "4_outputs")

# %% [markdown]
# ## 5 · Does it actually read the prompt?
#
# The figures above show a model that works. None of them show it *reading the
# clauses* — a spatial prior would look identical. That is what the
# counterfactuals of CLAUDE.md §6 are for, and they are the one thing here that
# is evidence rather than illustration:
#
# - **`permute_clauses`** — same three anchors, directions rotated. Must drop.
# - **`flip_direction`** — one clause replaced by its opposite. Must drop.
# - **`permute_both`** — slots rotated *as a unit*, so every relation is
#   preserved. Must **not** move. `where_raw` is a product and therefore exactly
#   permutation-invariant, so this control is weak by construction — the only
#   order dependence left is the carver's `cat`.
#
# One example, so these are illustrations of the probes, not the measurement.
# `scripts/evaluate.py` runs them over the population.

# %%
def dice_of(logits) -> tuple[float, np.ndarray]:
    p = torch.sigmoid(logits[0, 0].float()).cpu().numpy() >= float(cfg.train.threshold)
    return float(2 * (p * target_mask).sum() / max(p.sum() + target_mask.sum(), 1)), p


probes = {}
with torch.no_grad():
    probes["as prompted"] = out
    # directions rotated, anchors held: every clause now names the wrong side
    probes["permute_clauses"] = model(batch["image"], batch["direction_ids"].roll(1, 1),
                                      name_ids, anchors=batch.get("anchor_probability"))
    # one clause replaced by its opposite
    flipped = batch["direction_ids"].clone()
    flipped[0, 0] = DIRECTIONS.index(OPPOSITE[batch["directions"][0][0]])
    probes[f"flip_direction<br><sub>{batch['directions'][0][0]} → "
           f"{OPPOSITE[batch['directions'][0][0]]}</sub>"] = model(
        batch["image"], flipped, name_ids, anchors=batch.get("anchor_probability"))
    # slots rotated as a unit - every relation preserved, so this must NOT move
    rolled = roll_anchors(batch, 1)
    probes["permute_both<br><sub>control: must not move</sub>"] = model(
        batch["image"], batch["direction_ids"].roll(1, 1),
        rolled["name_ids"], anchors=rolled.get("anchor_probability"))

base_dice, _ = dice_of(out.logits)
fig = make_subplots(rows=1, cols=len(probes), horizontal_spacing=0.02,
                    subplot_titles=list(probes))
for col, (name, probe) in enumerate(probes.items(), start=1):
    d, mask = dice_of(probe.logits)
    fig.add_trace(grey(image[cz]), row=1, col=col)
    fig.add_trace(overlay(mask[cz].astype(float), "#00e5ff", name.split("<")[0], 0.5), row=1, col=col)
    fig.add_trace(contour(target_mask[cz], TARGET_COLOUR, "truth"), row=1, col=col)
    fig.layout.annotations[col - 1].update(
        text=f"{name}<br><sub>Dice {d:.3f}  ({d - base_dice:+.3f})</sub>")
fig.update_layout(
    title="5 · counterfactuals — the probes that separate reading the prompt from a spatial prior"
          "<br><sub>cyan = prediction · yellow = truth · one example, illustrative</sub>",
    height=450, width=380 * len(probes), template="plotly_dark", margin=dict(t=120),
)
fig.update_yaxes(autorange="reversed", scaleanchor="x", constrain="domain")
fig.update_xaxes(constrain="domain")
save(fig, "5_counterfactuals", empty_is_the_point=True)

print("\ncounterfactuals on this one example (illustrative — evaluate.py does the population):")
for name, probe in probes.items():
    d, _ = dice_of(probe.logits)
    plain = name.split("<")[0]
    verdict = "" if plain == "as prompted" else (
        "  (control: should stay put)" if plain.startswith("permute_both") else "  (should drop)")
    print(f"  {plain:<18}Dice {d:.4f}   {d - base_dice:+.4f}{verdict}")

# %%
predicted = probability >= float(cfg.train.threshold)
overlap = 2 * (predicted * target_mask).sum() / max(predicted.sum() + target_mask.sum(), 1)
error = float((out.centroid[0].cpu() - centroid[0]).norm())
print(f"\n  Dice on this example      : {overlap:.4f}   <- ILLUSTRATIVE ONLY: one example,")
print("                                       one seed. Not a result. A reportable Dice")
print("                                       carries the prompt-blind floor, the")
print("                                       counterfactuals and the gate (CLAUDE.md §6).")
print(f"  predicted / true voxels   : {int(predicted.sum())} / {int(target_mask.sum())}")
print(f"  centroid error            : {error:.2f} world units")
print(f"  null head says valid      : {float(out.valid[0]):+.3f} (logit; > 0 = the clauses name something)")
print(f"\nall figures written to {OUT.relative_to(ROOT)}/")
