#!/usr/bin/env python
"""Train Stage A or Stage B.

    scripts/train.py a                                   # the structure segmenter
    scripts/train.py b                                   # the relational model, oracle anchors
    scripts/train.py b --segmenter runs/stage_a/best.pt  # end to end, predicted anchors
    scripts/train.py b --overfit 1 --set train.stage_b.epochs=200

``--overfit N`` restricts training and validation to the first N scenes, and turns
augmentation off: it is the bug catcher, and memorising one scene while its pose
changes every epoch would measure something else. If Stage B cannot memorise one
scene, channel order, world coordinates, prompt indices or the decoder are wrong,
and nothing measured on the full corpus will mean anything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, parse_overrides
from src.data import Corpus, ExampleDataset, SceneDataset, loader
from src.engine import StageATask, StageBTask, Trainer, format_table, load_model
from src.models import StageA, StageB


def loss_weights(cfg) -> dict[str, float]:
    """``train.loss`` -> the keyword arguments :func:`segmentation_loss` takes."""
    loss = cfg.train.loss
    if loss["name"] != "dice_bce":
        raise ValueError(f"train.loss.name must be 'dice_bce', got {loss['name']!r}")
    return {"lambda_dice": float(loss["lambda_dice"]), "lambda_bce": float(loss["lambda_bce"])}


def build(stage: str, cfg, corpus: Corpus, overfit: int | None):
    """Datasets, model and task for one stage."""
    resolution = min(corpus.shape)
    common = dict(
        encoder_channels=tuple(cfg.model.encoder_channels),
        bottleneck=cfg.model.bottleneck,
        token_dim=cfg.model.token_dim,
        num_heads=cfg.model.num_heads,
        prior_foreground=cfg.model.prior_foreground,
    )
    stage_cfg = cfg.train[f"stage_{stage}"]
    # --overfit is a wiring test, not a generalisation test: memorising one scene
    # while its pose changes every epoch measures something else entirely.
    augment = bool(stage_cfg.augment) and not overfit
    if stage == "a":
        # Stage A learns every structure in every split: the target-class split
        # constrains what Stage B may be supervised on, not what anatomy exists.
        datasets = {
            split: SceneDataset(
                corpus,
                split,
                prompts_per_item=stage_cfg.prompts_per_item,
                augment=augment and split == "train",
                limit=overfit,
                normalize_mode=cfg.data.normalize,
            )
            for split in ("train", "val")
        }
        model = StageA(
            len(corpus.vocab), resolution,
            deep_supervision=tuple(cfg.model.deep_supervision), **common,
        )
        return datasets, model, StageATask(model, corpus.vocab, loss_weights=loss_weights(cfg))

    scenes = {split: corpus.scene_ids(split)[:overfit] if overfit else None for split in ("train", "val")}
    datasets = {
        split: ExampleDataset(
            corpus, split, scenes=scenes[split],
            augment=augment and split == "train",
            normalize_mode=cfg.data.normalize,
        )
        for split in ("train", "val")
    }
    if overfit:  # validate on what we are trying to memorise
        datasets["val"] = ExampleDataset(
            corpus, "train", scenes=scenes["train"], normalize_mode=cfg.data.normalize
        )
    model = StageB(
        len(corpus.vocab), resolution, corpus.n_anchors,
        intersection_hidden=cfg.model.intersection_hidden, **common,
    )
    task = StageBTask(
        model, corpus.vocab, threshold=cfg.train.threshold, loss_weights=loss_weights(cfg)
    )
    return datasets, model, task


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["a", "b"])
    parser.add_argument("--segmenter", type=Path, help="Stage A checkpoint; makes Stage B use predicted anchors")
    parser.add_argument("--overfit", type=int, metavar="N", help="train on the first N scenes only")
    parser.add_argument("--out", type=Path, help="run directory (default runs/stage_<stage>)")
    parser.add_argument("--set", dest="overrides", action="append", metavar="KEY=VALUE")
    args = parser.parse_args()

    if args.segmenter is not None and args.stage != "b":
        parser.error("--segmenter only applies to stage b")

    cfg = load_config(overrides=parse_overrides(args.overrides))
    corpus = Corpus.load(cfg.data.root)
    stage_cfg = cfg.train[f"stage_{args.stage}"]
    datasets, model, task = build(args.stage, cfg, corpus, args.overfit)
    if args.segmenter is not None:
        task.segmenter = load_model(args.segmenter)

    out_dir = args.out or Path("runs") / f"stage_{args.stage}{'_predicted' if args.segmenter else ''}"
    train_cfg = {
        key: value
        for key, value in cfg.train.to_dict().items()
        if not key.startswith("stage_")
    }
    trainer = Trainer(
        task,
        loader(datasets["train"], batch_size=cfg.train.batch_size, shuffle=True,
               workers=cfg.train.workers, seed=cfg.train.seed),
        loader(datasets["val"], batch_size=cfg.train.batch_size, shuffle=False, workers=cfg.train.workers),
        train_cfg,
        out_dir,
        stage=stage_cfg.to_dict(),
        evaluation=cfg.evaluation.to_dict(),
        logging=cfg.logging.to_dict(),
        spacing=corpus.spacing,
    )
    if args.segmenter is not None:
        task.segmenter.to(trainer.device).eval()

    print(f"corpus {corpus.root} | {len(corpus.vocab)} structures at {corpus.shape}")
    print(f"train {len(datasets['train'])} | val {len(datasets['val'])} -> {out_dir}")
    trainer.fit()

    task.model = trainer.model = load_model(out_dir / "best.pt", trainer.device)
    summary = trainer.evaluate(with_hausdorff=trainer.wants_hausdorff and args.stage == "b")
    print(format_table(summary, f"best val ({args.stage})"))
    if "anchor_dice" in summary:
        print(f"  predicted anchors scored Dice {summary['anchor_dice']:.4f} against the ground truth")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
