# %% [markdown]
# # One example, end to end
#
# Load a volume, generate a relational prompt for it, segment the anchors with
# Stage A, run Stage B, score it, and look at the result in 3D.
#
# Run it as a script (`.venv/bin/python notebooks/demo.py`) or open it as a
# notebook — the `# %%` markers make it one in VS Code and Jupyter.
#
# Needs two checkpoints and `pip install plotly`.

# %%
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data import Corpus, build_examples, load_nifti, normalize
from src.engine import (
    StageBTask, dice_iou, hausdorff, load_model, load_stage_a, resolve_device,
    resolve_phase_a_checkpoint, resolve_stage_b_mode, surface,
)
from src.geometry import DIRECTIONS

CORPUS = "data/mri"                    # a corpus directory
STAGE_B = "runs/phase-b/current/best.pt"  # segments the target from the relations
# Stage A path and oracle-vs-predicted follow configs/config.yaml
# (train.stage_b.mode / phase_a_checkpoint). Override with --set there.
SCENE = None                       # a scene id, or None for the first test scene
TARGET = None                      # a structure name, or None for the first feasible one

cfg = load_config()
corpus = Corpus.load(CORPUS)
device = resolve_device(cfg.train.device)
vocab = corpus.vocab

# %% [markdown]
# ## 1. Load a volume
#
# A scene is one intensity volume and one integer label volume. Nothing else is
# stored — every mask below is `labels == id`.

# %%
scene = SCENE or corpus.scene_ids("test")[0]
directory = corpus.root / "scenes" / scene
image = load_nifti(directory / "image.nii.gz", np.float32)
labels = load_nifti(directory / "labels.nii.gz", np.int16)
print(f"{scene}: image {image.shape} in [{image.min():.2f}, {image.max():.2f}], "
      f"{len(np.unique(labels)) - 1} structures")

# %% [markdown]
# ## 2. Generate a prompt
#
# The prompt is *derived from the geometry*, not stored: rank the other
# structures by distance, keep the nearest with pairwise-distinct directions,
# render the sentence. The target is never named in it.

# %%
examples = build_examples(scene, labels, vocab, corpus.spacing, corpus.n_anchors)
example = next(e for e in examples if TARGET in (None, vocab.name(e["target"])))

print(example["prompt"])
print(f"\ntarget (hidden from the model): {vocab.name(example['target'])}")
for slot, (anchor, direction) in enumerate(zip(example["anchors"], example["directions"]), 1):
    print(f"  clause {slot}: {direction:<9} to the {vocab.name(anchor)}")

# %% [markdown]
# ## 3. Load both stages
#
# Each checkpoint carries its own architecture, so nothing has to be described
# here. Stage A turns the image into the three named anchor masks; Stage B turns
# those plus the prompt into the target.

# %%
stage_b = load_model(STAGE_B, device)
mode = resolve_stage_b_mode(cfg.train.stage_b)
segmenter_path = resolve_phase_a_checkpoint(cfg.train.stage_b)
stage_a = load_stage_a(segmenter_path, device) if segmenter_path else None
task = StageBTask(
    stage_b, vocab,
    mode=mode,
    segmenter=stage_a,
    threshold=cfg.train.threshold,
    occupancy_mode=str(cfg.train.stage_b.occupancy_mode),
)
print(f"mode {task.mode}: anchors {'predicted by Stage A' if stage_a else 'ground truth'}")
print(f"occupancy: {task.occupancy_mode} from {'segmenter' if stage_a else 'gt'}")

# Anchors, the clause indices, and a label volume. Occupancy comes from
# `train.stage_b.occupancy_mode` of either the segmenter or the ground truth;
# the label volume is also the supervision target.
batch = {
    "image": torch.from_numpy(normalize(image, cfg.data.normalize))[None, None].to(device),
    "labels": torch.from_numpy(labels.astype(np.int64))[None].to(device),
    "target": torch.tensor([example["target"]], device=device),
    "anchors": torch.tensor([example["anchors"]], device=device),
    "direction_ids": torch.tensor([[DIRECTIONS.index(d) for d in example["directions"]]], device=device),
    "target_name": [vocab.name(example["target"])],
    "anchor_names": [[vocab.name(a) for a in example["anchors"]]],
    "directions": [example["directions"]],
}

# %% [markdown]
# ## 4. Inference

# %%
with torch.no_grad():
    prediction = task(batch)
probability = torch.sigmoid(prediction.logits.float())
predicted = (probability >= cfg.train.threshold)[0, 0].cpu().numpy()
target = prediction.target[0, 0].cpu().numpy()
anchors = task.anchors(batch)[0].cpu().numpy()  # what Stage A actually supplied

# %% [markdown]
# ## 5. Evaluate
#
# The anchors' own Dice is reported next to the target's: a poor target score is
# only interpretable once you know whether Stage A found the anchors.
#
# Stage A emits one *independent* mask per name — that is what lets Stage B ask
# for just the three anchors a prompt names — so nothing forces its outputs to be
# a partition. One name can cover two structures, and two names can claim the
# same voxels. The "spills onto" line below catches exactly that, which is the
# usual reason a predicted anchor looks like two copies of the same shape.

# %%
dice, iou = (float(v) for v in dice_iou(probability, prediction.target, cfg.train.threshold))
distance = hausdorff(
    torch.from_numpy(predicted).float(), torch.from_numpy(target).float(),
    corpus.spacing, cfg.evaluation.hausdorff_percentile,
)
# Empty against non-empty is undefined, not zero: Hausdorff says nan and it is
# counted rather than averaged in.
print(f"target      dice {dice:.4f}   iou {iou:.4f}   hd95 {distance:.2f}")
if task.anchor_scores:
    anchor_dice = sum(task.anchor_scores) / len(task.anchor_scores)
    print(f"anchors     dice {anchor_dice:.4f}  (Stage A vs ground truth)")
print(f"voxels      predicted {int(predicted.sum())}   target {int(target.sum())}")

for slot, label in enumerate(example["anchors"]):
    spill = {
        vocab.name(other): int((anchors[slot] > 0.5)[labels == other].sum())
        for other in np.unique(labels)
        if other not in (0, label) and (anchors[slot] > 0.5)[labels == other].sum() > 20
    }
    if spill:
        print(f"  anchor {slot + 1} ({vocab.name(label)}) spills onto {spill}")

# %% [markdown]
# ## 6. Visualise
#
# Slices hide the point of this task, which is *where a structure sits relative
# to the others*. So: one interactive 3D scene, every structure drawn as its
# surface voxels. Rotate it and read the prompt off the picture — the prediction
# should sit superior to one anchor, medial to another, and so on. The grey
# distractors are the structures the model had to rule out.

# %%
import plotly.graph_objects as go


def points(mask):
    """Surface voxels of a mask as world ``(x, y, z)`` — a few hundred points."""
    voxels = surface(torch.as_tensor(np.ascontiguousarray(mask), dtype=torch.float32)).numpy()
    return dict(x=voxels[:, 2], y=voxels[:, 1], z=voxels[:, 0])  # array is (z, y, x)


def trace(mask, name, color, size=3, opacity=1.0):
    return go.Scatter3d(
        **points(mask), name=name, mode="markers",
        marker=dict(size=size, color=color, opacity=opacity),
    )


source = "Stage A" if stage_a else "ground truth"
named = set(example["anchors"]) | {example["target"]}
distractors = (labels > 0) & ~np.isin(labels, list(named))

figure = go.Figure([
    trace(distractors, "other structures", "lightgrey", size=2, opacity=0.25),
    *[
        trace(anchors[slot], f"anchor {slot + 1} ({source}): {direction} to the {vocab.name(anchor)}", color)
        for slot, (anchor, direction, color) in enumerate(
            zip(example["anchors"], example["directions"], ["#4C78A8", "#72B7B2", "#54A24B"])
        )
    ],
    trace(target, f"target: {vocab.name(example['target'])}", "#333333", size=4, opacity=0.35),
    trace(predicted, "prediction", "#E45756", size=4),
])
subtitle = f"dice {dice:.3f} · iou {iou:.3f} · hd95 {distance:.2f}"
if task.anchor_scores:
    subtitle += f" · anchors {sum(task.anchor_scores) / len(task.anchor_scores):.3f}"
figure.update_layout(
    title=f"{example['prompt']}<br><sub>{subtitle}</sub>",
    scene=dict(  # RAS: x right/lateral, y anterior, z superior
        xaxis_title="x (lateral)", yaxis_title="y (anterior)", zaxis_title="z (superior)",
        aspectmode="data",
    ),
    legend=dict(yanchor="top", y=0.95, xanchor="left", x=0),
    margin=dict(l=0, r=0, t=70, b=0), height=700,
)
figure.show()
