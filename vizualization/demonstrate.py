#!/usr/bin/env python
# %% [markdown]
# # Given a prompt, does the model find the right thing?
#
# The claim has two halves, and this file is built to show them in order:
#
# 1. **it locates the right region** — the prompt narrows the volume, and the
#    model's own centroid lands on the structure the clauses describe;
# 2. **it then segments the right structure** — and the mask it paints is that
#    structure, not a blob in roughly the right place.
#
# The figures are ordered as the evidence, not as the architecture. Internals —
# `A_i`, the fields, the feature pyramid — are in `inspect_stage_b.py`.
#
# ```
# .venv/bin/python vizualization/demonstrate.py                       # MRI
# .venv/bin/python vizualization/demonstrate.py --config configs/synthetic-hard.yaml \
#     --checkpoint runs/hard-stage-b/best.pt --tag hard
# ```

# %%
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
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

TRUTH = "#ffd600"        # ground truth, everywhere, always an outline
PREDICTION = "#00e5ff"   # what the model painted
REGION = "#ff6f00"       # where the prompt points (amber: readable on brain and on noise)
ANCHOR_COLOURS = ("#e64a19", "#1e88e5", "#43a047")
#: distinct hues for the composite panel, one per prompt
WHEEL = ("#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
         "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990")


def parse(argv=None):
    p = argparse.ArgumentParser(description="Show that the prompt locates and segments the target.")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--checkpoint", type=Path, default=ROOT / "runs/relational-seed1/best.pt")
    p.add_argument("--split", default="val")
    p.add_argument("--tag", default="", help="suffix for output filenames")
    p.add_argument("--population", type=int, default=60, help="examples for the scatter")
    return p.parse_args(argv if argv is not None else ([] if "ipykernel" in sys.modules else None))


args = parse()
cfg = load_config(args.config)
corpus = Corpus.load(cfg.data.root)
device = resolve_device(cfg.train.device)
model = load_model(args.checkpoint, device)
spacing = corpus.spacing
SUPERVISED = list(cfg.targets["train"])
SPLIT_OF = {name: split for split in ("train", "val", "test") for name in cfg.targets[split]}

cache = anchor_cache_dir(corpus.root, Path(cfg.train.stage_b.phase_a_checkpoint))
cache = cache if (cache / "meta.json").is_file() else None
print(f"corpus {corpus.root} · {corpus.shape} at {spacing} · checkpoint {args.checkpoint.name}")


# %%
def predict(dataset, index, *, direction_ids=None, rolled=None):
    """One forward, plus everything needed to judge it. No label ever enters the model."""
    batch = collate([dataset[index]])
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    source = rolled if rolled is not None else batch
    with torch.no_grad():
        out = model(batch["image"],
                    batch["direction_ids"] if direction_ids is None else direction_ids,
                    source["name_ids"] if "name_ids" in source else source["anchors"] - 1,
                    anchors=source.get("anchor_probability"))

    labels = batch["labels"][0].cpu().numpy()
    truth = (labels == int(batch["target"][0])).astype(np.float32)
    probability = torch.sigmoid(out.logits[0, 0].float()).cpu().numpy()
    mask = probability >= float(cfg.train.threshold)
    true_centre, _ = mask_centroid_world(torch.from_numpy(truth)[None, None], spacing)
    region = out.where_raw[0, 0].float().cpu().numpy()
    z = int(np.clip(round(float(true_centre[0, 2]) / spacing[2]), 0, labels.shape[0] - 1))
    return dict(
        batch=batch, out=out, image=batch["image"][0, 0].float().cpu().numpy(),
        truth=truth, mask=mask, region=region, z=z,
        name=batch["target_name"][0], prompt=batch["prompt"][0],
        directions=list(batch["directions"][0]), anchor_names=list(batch["anchor_names"][0]),
        dice=float(2 * (mask * truth).sum() / max(mask.sum() + truth.sum(), 1)),
        error=float((out.centroid[0].cpu() - true_centre[0]).norm()),
        true_centre=true_centre[0].cpu().numpy(),
        predicted_centre=out.centroid[0].cpu().numpy(),
        inside=float((truth * (region > 0.05)).sum() / max(truth.sum(), 1)),
    )


def grey(plane):
    lo, hi = np.percentile(plane, (1, 99))
    return go.Heatmap(z=plane, colorscale="gray", zmin=lo, zmax=hi, showscale=False, hoverinfo="skip")


def paint(plane, colour, name, threshold=0.05, opacity=1.0):
    return go.Heatmap(
        z=np.where(plane > threshold, plane, np.nan),
        colorscale=[[0, "rgba(0,0,0,0)"], [1, colour]], zmin=0, zmax=1,
        showscale=False, name=name, opacity=opacity,
        hovertemplate=f"{name}: %{{z:.3f}}<extra></extra>",
    )


def outline(plane, colour, name):
    return go.Contour(z=plane, contours=dict(start=0.5, end=0.5, size=0, coloring="none"),
                      line=dict(color=colour, width=2), showscale=False, name=name, hoverinfo="skip")


def marker(point, colour, symbol, name, size=12):
    return go.Scatter(x=[point[0] / spacing[0]], y=[point[1] / spacing[1]], mode="markers",
                      marker=dict(symbol=symbol, size=size, color=colour,
                                  line=dict(color="#ffffff", width=1.5)),
                      name=name, hovertemplate=f"{name}<extra></extra>", showlegend=False)


def save(fig, stem):
    path = OUT / f"{stem}{'_' + args.tag if args.tag else ''}.html"
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)
    print(f"  -> {path.relative_to(ROOT)}")
    return path


def square(fig):
    fig.update_yaxes(autorange="reversed", scaleanchor="x", constrain="domain",
                     showticklabels=False)
    fig.update_xaxes(constrain="domain", showticklabels=False)
    return fig


# %% [markdown]
# ## 1 · One scene, many prompts, a different structure each time
#
# The scene is **chosen by count, not by score**: whichever subject in the split
# supports the most distinct supervised targets, then the best median among those.
# Every prompt that scene supports is shown — nothing is dropped for looking bad.
#
# This is the claim in one figure. The image never changes. The three anchor
# *names* barely change. Only the direction words do, and the model lands on a
# different structure each time — including left/right pairs, where the two
# candidates are mirror images and only the prompt says which.

# %%
dataset = ExampleDataset(corpus, args.split, targets=SUPERVISED,
                         anchor_cache=cache, normalize_mode=cfg.data.normalize)
records = corpus.records(args.split, targets=SUPERVISED)
by_scene = defaultdict(dict)
for index, record in enumerate(records):
    by_scene[record["scene"]].setdefault(record["target"], index)

ranked = sorted(by_scene.items(), key=lambda kv: -len(kv[1]))[:4]
scored = []
for scene, targets in ranked:
    runs = [predict(dataset, i) for i in targets.values()]
    scored.append((len(runs), float(np.median([r["dice"] for r in runs])), scene, runs))
scored.sort(key=lambda s: (-s[0], -s[1]))
count, median_dice, scene, runs = scored[0]
# Order as mirror pairs where the vocabulary has them: Left-Thalamus beside
# Right-Thalamus. Adjacent columns are then the SAME structure on opposite sides
# of the same brain, and the only thing that differs is the direction words.
def pairing(run):
    name = run["name"]
    for prefix in ("Left-", "Right-"):
        if name.startswith(prefix):
            return (name[len(prefix):], prefix != "Left-")
    return (name, False)


runs.sort(key=pairing)
print(f"\nhero scene {scene}: {count} distinct supervised targets, median Dice {median_dice:.3f}")

columns = len(runs)
fig = make_subplots(
    rows=2, cols=columns, horizontal_spacing=0.006, vertical_spacing=0.09,
    subplot_titles=[f"<b>{r['name']}</b><br><sub>{' · '.join(r['directions'])}"
                    f"<br>{r['inside']:.0%} of the target inside</sub>" for r in runs]
                   + [f"<sub>Dice {r['dice']:.2f} · centroid {r['error']:.1f} mm</sub>" for r in runs],
)
for col, run in enumerate(runs, start=1):
    z = run["z"]
    # row 1 - where the prompt points. This is the MAPPER's wedge: parameter-free,
    # a function of the anchors and the direction words alone. It narrows the
    # volume; it does not pick the structure.
    fig.add_trace(grey(run["image"][z]), row=1, col=col)
    fig.add_trace(paint(run["region"][z], REGION, "where the prompt points", 0.02, 0.75), row=1, col=col)
    fig.add_trace(outline(run["region"][z], REGION, "region edge"), row=1, col=col)
    fig.add_trace(outline(run["truth"][z], TRUTH, "truth"), row=1, col=col)
    # the model's own answer to "where is it", carried onto the region panel so
    # the two steps read as one movement: the prompt narrows, the model commits.
    fig.add_trace(marker(run["predicted_centre"], PREDICTION, "x", "predicted centroid"), row=1, col=col)
    # row 2 - what the model painted, against the truth it was never shown.
    fig.add_trace(grey(run["image"][z]), row=2, col=col)
    fig.add_trace(paint(run["mask"][z].astype(float), PREDICTION, "prediction", 0.5, 0.75), row=2, col=col)
    fig.add_trace(outline(run["truth"][z], TRUTH, "truth"), row=2, col=col)
    fig.add_trace(marker(run["predicted_centre"], PREDICTION, "x", "predicted centroid"), row=2, col=col)

fig.update_layout(
    title=f"1 · one subject ({scene}), {columns} prompts, {columns} structures"
          "<br><sub>top: where the prompt points (the mapper's wedge — no parameters) · "
          "bottom: what the model segmented, cyan, against the truth outline it never saw"
          "<br>every supervised target this scene supports is shown, none omitted · "
          "a low containment % is normal — the wedge is a weak bias, never a crop</sub>",
    height=780, width=max(250 * columns, 900), template="plotly_dark",
    margin=dict(t=150), showlegend=False,
)
square(fig)
save(fig, "1_one_scene_many_prompts")

print(f"  {'target':<20}{'prompt (directions)':<34}{'Dice':>7}{'centroid':>10}{'in region':>11}")
for run in sorted(runs, key=lambda r: -r["dice"]):
    print(f"  {run['name']:<20}{' · '.join(run['directions']):<34}"
          f"{run['dice']:>7.3f}{run['error']:>8.2f}mm{run['inside']:>10.0%}")
dices = np.array([r["dice"] for r in runs]); errors = np.array([r["error"] for r in runs])
print(f"  median Dice {np.median(dices):.3f} · median centroid error {np.median(errors):.2f} mm "
      f"· worst {errors.max():.2f} mm")


# %% [markdown]
# ## 2 · The same claim, one prompt at a time
#
# Four panels, left to right, in the order the model does the work.
#
# A caution the figure is built around: **`where_raw` is not a model output.**
# It is the mapper's wedge — parameter-free, a function of the three anchor
# centroids and the direction words alone. It *narrows* the volume; any model
# with the same prompt gets the same wedge. So the wedge is labelled "where the
# prompt points", and the thing that earns the word *locates* is panel 3: the
# model's own soft-argmax centroid against the true one, in millimetres.

# %%
# The median run, not the best one. Figure 1 earns its caption by showing every
# target including the weak ones; picking the top scorer for the figure a reader
# actually screenshots would give that back.
showcase = sorted(runs, key=lambda r: r["dice"])[len(runs) // 2]
z = showcase["z"]
titles = [
    "the three named anchors<br><sub>projected through the volume</sub>",
    f"where the prompt points<br><sub>{showcase['inside']:.0%} of the target inside</sub>",
    f"where the model says it is<br><sub>{showcase['error']:.2f} mm from truth</sub>",
    f"what it segmented<br><sub>Dice {showcase['dice']:.3f}</sub>",
]
fig = make_subplots(rows=1, cols=4, horizontal_spacing=0.012, subplot_titles=titles)
# The three named structures are rarely coplanar with the target - at this depth
# two of them typically do not appear at all. They are projected through the
# volume (max over z) so the prompt's three anchors are all visible at once, and
# their centroids, which is the only part of them the mapper reads, are marked.
anchor_probability = showcase["out"].anchors[0].float().cpu().numpy()
fig.add_trace(grey(showcase["image"][z]), row=1, col=1)
for slot in range(3):
    fig.add_trace(paint(anchor_probability[slot].max(axis=0), ANCHOR_COLOURS[slot],
                        showcase["anchor_names"][slot], 0.5, 0.8), row=1, col=1)
    fig.add_trace(marker(showcase["out"].anchor_centroids[0, slot].cpu().numpy(),
                         ANCHOR_COLOURS[slot], "circle", showcase["anchor_names"][slot], 11),
                  row=1, col=1)
fig.add_trace(outline(showcase["truth"][z], TRUTH, "truth"), row=1, col=1)

fig.add_trace(grey(showcase["image"][z]), row=1, col=2)
fig.add_trace(paint(showcase["region"][z], REGION, "where_raw", 0.02, 0.75), row=1, col=2)
fig.add_trace(outline(showcase["truth"][z], TRUTH, "truth"), row=1, col=2)

fig.add_trace(grey(showcase["image"][z]), row=1, col=3)
fig.add_trace(paint(showcase["region"][z], REGION, "where_raw", 0.02, 0.25), row=1, col=3)
fig.add_trace(outline(showcase["truth"][z], TRUTH, "truth"), row=1, col=3)
fig.add_trace(marker(showcase["true_centre"], TRUTH, "circle", "true centroid", 14), row=1, col=3)
fig.add_trace(marker(showcase["predicted_centre"], PREDICTION, "x", "predicted centroid", 14), row=1, col=3)

fig.add_trace(grey(showcase["image"][z]), row=1, col=4)
fig.add_trace(paint(showcase["mask"][z].astype(float), PREDICTION, "prediction", 0.5, 0.75), row=1, col=4)
fig.add_trace(outline(showcase["truth"][z], TRUTH, "truth"), row=1, col=4)

fig.update_layout(
    title=f"2 · \"{showcase['prompt']}\""
          f"<br><sub>→ <b>{showcase['name']}</b>, a name the prompt never uses · "
          f"the MEDIAN of this scene's {len(runs)} prompts, not the best · "
          "yellow = truth, shown to you and never to the model</sub>",
    height=470, width=1500, template="plotly_dark", margin=dict(t=140), showlegend=False,
)
square(fig)
save(fig, "2_one_prompt_step_by_step")


# %% [markdown]
# ## 3 · Is that cherry-picked?
#
# One scene is an anecdote. This is every example the split gives, up to
# `--population`, as **located** (centroid error, mm) against **segmented**
# (Dice) — the two halves of the claim on one pair of axes.
#
# Coloured by what the target class *is to Stage B*, which is the only way these
# numbers may be read (CLAUDE.md §6):
#
# - **`targets.train`** — supervised as relational targets. The claim is about these.
# - **`targets.val` / `targets.test`** — never supervised as a target. Scored every
#   epoch, never used to pick a checkpoint. On `data/mri` these collapse, and
#   showing that is the point: the figure is not an advertisement.

# %%
all_targets = list(corpus.vocab.names)
population = ExampleDataset(corpus, args.split, targets=all_targets,
                            anchor_cache=cache, normalize_mode=cfg.data.normalize)
# Stratify by target class rather than striding the manifest. A stride follows
# the manifest's class mix, which left targets.test with a single example - too
# few for the median to mean anything, in the population the claim is weakest on.
rows = corpus.records(args.split, targets=all_targets)
per_class = defaultdict(list)
for index, record in enumerate(rows):
    per_class[corpus.vocab.name(record["target"])].append(index)
quota = max(1, args.population // max(len(per_class), 1))
# Spread across each class's rows, never the first N. The manifest is ordered by
# scene, so indices[:quota] samples one or two subjects and calls it a class -
# it reported targets.test at 0.001 where the true population value is 0.201.
def spread(indices, k):
    if len(indices) <= k:
        return list(indices)
    return [indices[int(round(j * (len(indices) - 1) / (k - 1)))] for j in range(k)] if k > 1         else [indices[len(indices) // 2]]


chosen = [i for indices in per_class.values() for i in spread(indices, quota)]
sampled = [predict(population, i) for i in sorted(chosen)]
print(f"\npopulation: {len(sampled)} examples, up to {quota} per target class "
      f"({len(per_class)} classes present in {args.split})")

fig = go.Figure()
# The vocabulary is wider than the three target splits: some classes are only
# ever anchors. They are never supervised as a target either, so they belong on
# the plot - dropping them silently would flatter it.
LABELS = {"train": "targets.train (supervised)", "val": "targets.val (held out)",
          "test": "targets.test (held out)", None: "never a target in any split"}
for split, colour in (("train", "#43a047"), ("val", "#ff6f00"),
                      ("test", "#e64a19"), (None, "#8e8e8e")):
    group = [r for r in sampled if SPLIT_OF.get(r["name"]) == split]
    if not group:
        continue
    dice = np.array([r["dice"] for r in group]); error = np.array([r["error"] for r in group])
    fig.add_trace(go.Scatter(
        x=error, y=dice, mode="markers",
        name=f"{LABELS[split]}  (n={len(group)}, median Dice {np.median(dice):.3f})",
        marker=dict(size=9, color=colour, opacity=0.75, line=dict(color="#111", width=1)),
        text=[r["name"] for r in group],
        hovertemplate="%{text}<br>centroid %{x:.2f} mm · Dice %{y:.3f}<extra></extra>",
    ))
    print(f"  {LABELS[split]:<28} n={len(group):<4} median Dice {np.median(dice):.3f}   "
          f"median centroid error {np.median(error):.2f} mm   "
          f"located within 5 mm: {(error <= 5).mean():.0%}")
fig.add_vline(x=5, line=dict(color="#888", dash="dot"),
              annotation_text="5 mm", annotation_position="top")
fig.update_layout(
    title="3 · located vs segmented, every sampled example"
          "<br><sub>x: how far the model's centroid is from the truth · y: Dice of the mask it painted"
          "<br>a single seed — non-determinism alone moves val Dice ~0.01 typical / ~0.03 worst on this project</sub>",
    xaxis_title="centroid error (mm) — LOCATED", yaxis_title="Dice — SEGMENTED",
    height=620, width=980, template="plotly_dark", margin=dict(t=140),
    legend=dict(yanchor="bottom", y=0.02, xanchor="right", x=0.98),
)
save(fig, "3_located_vs_segmented")


# %% [markdown]
# ## 4 · The control: is it the prompt, or a prior?
#
# Everything above is consistent with a model that ignores the clauses and puts a
# plausible structure in a plausible place. The counterfactuals are what rules
# that out — same image, same anchors, clauses broken.

# %%
index = next(i for i in range(len(dataset)) if dataset[i]["target_name"] == showcase["name"])
base = predict(dataset, index)
batch = base["batch"]
probes = {"as prompted": base}
probes["clauses rotated"] = predict(dataset, index, direction_ids=batch["direction_ids"].roll(1, 1))
flipped = batch["direction_ids"].clone()
flipped[0, 0] = DIRECTIONS.index(OPPOSITE[base["directions"][0]])
probes[f"one clause flipped<br><sub>{base['directions'][0]} → {OPPOSITE[base['directions'][0]]}</sub>"] = \
    predict(dataset, index, direction_ids=flipped)
rolled = roll_anchors(batch, 1)
probes["slots rolled together<br><sub>control — relations preserved</sub>"] = \
    predict(dataset, index, direction_ids=batch["direction_ids"].roll(1, 1), rolled=rolled)

fig = make_subplots(rows=1, cols=len(probes), horizontal_spacing=0.012,
                    subplot_titles=[f"{k}<br><sub>Dice {v['dice']:.3f} ({v['dice'] - base['dice']:+.3f})</sub>"
                                    for k, v in probes.items()])
for col, run in enumerate(probes.values(), start=1):
    fig.add_trace(grey(base["image"][base["z"]]), row=1, col=col)
    fig.add_trace(paint(run["mask"][base["z"]].astype(float), PREDICTION, "prediction", 0.5, 0.75),
                  row=1, col=col)
    fig.add_trace(outline(base["truth"][base["z"]], TRUTH, "truth"), row=1, col=col)
fig.update_layout(
    title=f"4 · break the prompt, keep everything else  ·  target {base['name']}"
          "<br><sub>the first three must drop; the fourth preserves every relation and must not move"
          "<br>one example — scripts/evaluate.py runs these over the population</sub>",
    height=460, width=380 * len(probes), template="plotly_dark", margin=dict(t=140), showlegend=False,
)
square(fig)
save(fig, "4_control_counterfactuals")

print("\ncontrol (one example):")
for name, run in probes.items():
    plain = name.split("<")[0]
    print(f"  {plain:<26}Dice {run['dice']:.4f}   {run['dice'] - base['dice']:+.4f}")
print(f"\nall figures written to {OUT.relative_to(ROOT)}/")
