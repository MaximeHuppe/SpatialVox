#!/usr/bin/env python
"""Precompute the frozen Stage A's soft masks, once per scene.

    .venv/bin/python scripts/cache_anchors.py --segmenter runs/phase-a/current/best.pt

Stage A is frozen and Stage B does not rotate, so ``sigmoid(anchor_logits)`` for a
scene is a **constant**. Recomputing it every time an example of that scene comes
round costs about a fifth of a training step and 7 GB of the peak, for an answer
that cannot change. This writes it down.

The cache is keyed by the SHA-256 of the checkpoint file, so a different Stage A
writes a different directory and a stale one can never be picked up silently.

Each scene is stored as the bounding box of ``p > threshold`` per structure, in
float16. Measured on ``data/mri``: 3.3 MB of crop per scene at ``1e-3``, so 0.7 GB
for 200 subjects, against 19 GB for the dense volumes. The truncated tail is what
sits below ``1e-3`` of probability, which changes an anchor mass by under 0.01%
and a centroid by under a hundredth of a voxel - three orders below Stage A's own
0.84 mm median centroid error. ``tests/test_data.py`` pins that the cache
reproduces the segmenter it was built from.

Stage A is run in float32 here even though training runs it under bf16 autocast.
Measured: bf16 moves a probability by up to 0.04 and flips ~40 voxels per batch
across the 0.5 the anchor exclusion uses, *purely by changing the batch size*.
Caching the exact value replaces one sample of that noise with the answer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.data import Corpus, anchor_cache_dir, load_nifti, normalize
from src.engine import load_stage_a, resolve_device


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="data/mri")
    parser.add_argument("--segmenter", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=1e-3)
    parser.add_argument("--normalize", default="zscore-brain")
    parser.add_argument("--force", action="store_true", help="rewrite scenes already cached")
    args = parser.parse_args()

    corpus = Corpus.load(args.root)
    device = resolve_device("auto")
    model = load_stage_a(args.segmenter, device)
    out = anchor_cache_dir(corpus.root, args.segmenter)
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(
        json.dumps(
            {
                "segmenter": str(args.segmenter),
                "sha256": hashlib.sha256(Path(args.segmenter).read_bytes()).hexdigest(),
                "threshold": float(args.threshold),
                "normalize": args.normalize,
                "vocab": list(corpus.vocab.names),
                "shape": list(corpus.shape),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    scenes = sorted({row["scene"] for split in ("train", "val", "test")
                     for row in corpus.records(split, targets=corpus.vocab.names)})
    names = torch.arange(len(corpus.vocab), device=device).unsqueeze(0)
    started, written, total = time.perf_counter(), 0, 0
    for index, scene_id in enumerate(scenes):
        path = out / f"{scene_id}.npz"
        if path.exists() and not args.force:
            continue
        image = normalize(
            load_nifti(corpus.root / "scenes" / scene_id / "image.nii.gz"), args.normalize
        )
        # float32, deliberately. The live path runs Stage A under bf16 autocast,
        # where changing the batch size alone moves a probability by up to 0.04
        # and flips a handful of voxels across the 0.5 the anchor exclusion uses.
        # The cache is the exact value instead of one sample of that, so it is
        # the more faithful of the two and does not depend on the kernel that
        # wrote it.
        with torch.no_grad():
            probability = model.probability(torch.as_tensor(image, device=device)[None, None], names)[0]
        boxes, chunks = [], []
        for channel in range(probability.shape[0]):
            occupied = torch.nonzero(probability[channel] > args.threshold)
            if occupied.numel() == 0:
                boxes.append([0, 0, 0, 0, 0, 0])
                chunks.append(np.zeros(0, dtype=np.float16))
                continue
            low, high = occupied.amin(0), occupied.amax(0) + 1
            boxes.append([*low.tolist(), *high.tolist()])
            crop = probability[channel, low[0]:high[0], low[1]:high[1], low[2]:high[2]]
            chunks.append(crop.to(torch.float16).cpu().numpy().reshape(-1))
        offsets = np.cumsum([0] + [c.size for c in chunks])
        np.savez(
            path,
            bbox=np.array(boxes, dtype=np.int16),
            data=np.concatenate(chunks) if chunks else np.zeros(0, np.float16),
            offset=offsets.astype(np.int64),
        )
        written += 1
        total += path.stat().st_size
        print(f"\r{index + 1}/{len(scenes)}  {scene_id}  {total / 1e6:.0f} MB", end="", flush=True)
    elapsed = time.perf_counter() - started
    print(f"\n{written} scenes written to {out} in {elapsed:.0f}s")
    print("scripts/train.py b picks this up automatically for the same checkpoint")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
