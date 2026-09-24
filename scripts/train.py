#!/usr/bin/env python
"""Train a stage of the relational pipeline.

    scripts/train.py a                  # the promptable segmenter, then frozen
    scripts/train.py boundary           # pretrain B(I) with no class ids
    scripts/train.py b                  # the relational model
    scripts/train.py b --segmenter runs/phase-a/current/best.pt
    scripts/train.py b --twin           # trained image-free twin (refine frozen, B off)
    scripts/train.py b --overfit 1 --set train.stage_b.epochs=200

The order is the pipeline's: ``a`` is trained on every name that may be an
anchor and is then frozen inside ``b``; ``boundary`` is optional and its
checkpoint goes in ``train.stage_b.boundary_checkpoint``.

``--overfit N`` restricts training and validation to the first N scenes and
turns the direction flip off: it is the bug catcher, and a model that cannot
memorise one scene has something wrong with its channel order, its world
coordinates or its prompt indices, so nothing measured on the full corpus will
mean anything.

**Checkpoint selection.** ``best.pt`` is chosen on ``targets.train`` - the eight
classes Stage B is supervised on. ``targets.val`` (caudate, putamen) and
``targets.test`` (hippocampus) are scored every epoch and never selected on;
they are the transfer curves, and choosing a checkpoint on them would be
choosing on the number being reported.

**Twin.** ``--twin`` trains the mandatory image-use comparator: ``refine`` / q /
K / V stay at zero init and ``B`` is not computed. Report every Dice as full
minus twin, never as an absolute.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, parse_overrides, require_bool
from src.data import Corpus, ExampleDataset, SceneDataset, anchor_cache_dir, loader
from src.engine import (
    BoundaryTask, StageATask, StageBTask, Trainer, format_table, load_model,
    load_stage_a, resolve_anchor_source, resolve_segmenter,
)
from src.models import BoundaryPretrainer, StageA, StageB


def loss_weights(cfg) -> dict[str, float]:
    """``train.loss`` -> the keyword arguments :func:`segmentation_loss` takes."""
    loss = cfg.train.loss
    if loss["name"] != "dice_bce":
        raise ValueError(f"train.loss.name must be 'dice_bce', got {loss['name']!r}")
    return {"lambda_dice": float(loss["lambda_dice"]), "lambda_bce": float(loss["lambda_bce"])}


def assert_targets_agree(cfg, corpus: Corpus) -> dict[str, list[str]]:
    """One target split: config wins, meta must match when present."""
    resolved = {split: list(cfg.targets[split]) for split in cfg.targets}
    meta = corpus.meta.get("targets")
    if meta:
        meta_norm = {split: list(names) for split, names in meta.items()}
        if meta_norm != resolved:
            raise ValueError(
                "corpus meta.targets disagrees with config targets. "
                f"meta={meta_norm} config={resolved}. Rebuild manifests or fix the config."
            )
    return resolved


def resolve_retarget_whitelist(stage_cfg, trained: Sequence[str]) -> list[str] | None:
    """``train`` / ``true`` -> trained names; ``null`` / ``false`` -> unrestricted."""
    raw = stage_cfg.get("retarget_only_to", "train")
    if raw in (None, "null", False):
        return None
    if raw in (True, "train", "true"):
        return list(trained)
    if isinstance(raw, (list, tuple)):
        return [str(n) for n in raw]
    raise ValueError(
        f"train.stage_b.retarget_only_to must be 'train', null, or a name list, got {raw!r}"
    )


def stage_a(cfg, corpus: Corpus, overfit: int | None):
    stage_cfg = cfg.train.stage_a
    augment = bool(stage_cfg.augment) and not overfit
    datasets = {
        split: SceneDataset(
            corpus, split,
            prompts_per_item=stage_cfg.prompts_per_item,
            augment=augment and split == "train",
            limit=overfit,
            normalize_mode=cfg.data.normalize,
        )
        # Stage A learns every structure in every split: the target-class split
        # constrains what Stage B may be supervised on, not what anatomy exists.
        for split in ("train", "val")
    }
    model = StageA(
        len(corpus.vocab), min(corpus.shape),
        encoder_channels=tuple(cfg.model.stage_a.encoder_channels),
        bottleneck=cfg.model.stage_a.bottleneck,
        token_dim=cfg.model.stage_a.token_dim,
        num_heads=cfg.model.stage_a.num_heads,
        prior_foreground=cfg.model.stage_a.prior_foreground,
        deep_supervision=tuple(cfg.model.stage_a.deep_supervision),
    )
    return datasets, StageATask(model, corpus.vocab, loss_weights=loss_weights(cfg)), {}


def boundary(cfg, corpus: Corpus, overfit: int | None):
    """``B`` on its own, with the three class-agnostic pretext objectives."""
    stage_cfg = cfg.train.boundary
    datasets = {
        split: SceneDataset(
            corpus, split, prompts_per_item=1, augment=False, limit=overfit,
            normalize_mode=cfg.data.normalize,
        )
        for split in ("train", "val")
    }
    model = BoundaryPretrainer(
        widths=tuple(cfg.model.stage_b.boundary_widths),
        mask_fraction=float(stage_cfg.mask_fraction),
        patch=int(stage_cfg.patch),
    )
    return datasets, BoundaryTask(model, corpus.vocab, loss_weights=stage_cfg.loss.to_dict()), {}


def stage_b(cfg, corpus: Corpus, overfit: int | None, segmenter: StageA,
            twin: bool, anchors: Path | None):
    stage_cfg, model_cfg = cfg.train.stage_b, cfg.model.stage_b
    targets = assert_targets_agree(cfg, corpus)
    scenes = {split: corpus.scene_ids(split)[:overfit] if overfit else None for split in ("train", "val")}
    val_examples = cfg.train.get("val_examples")
    val_examples = None if val_examples in (None, 0) else int(val_examples)
    flip = 0.0 if overfit else float(stage_cfg.flip_probability)
    retarget = None if overfit else resolve_retarget_whitelist(stage_cfg, targets["train"])

    datasets = {
        # Explicit targets from config (asserted against meta) so a fold cannot
        # silently train on a stale meta.targets list.
        "train": ExampleDataset(
            corpus, "train", targets=targets["train"], scenes=scenes["train"],
            flip_probability=flip, retarget_only_to=retarget,
            anchor_cache=anchors, normalize_mode=cfg.data.normalize,
            leave_out=(list(targets["train"])
                       if require_bool(stage_cfg.get("leave_one_out", False), "leave_one_out")
                       else None),
        ),
        # The selection curve: held-out *subjects*, trained *classes*.
        "val": ExampleDataset(
            corpus, "val", scenes=scenes["val"], targets=targets["train"],
            sample=val_examples, seed=int(cfg.train.seed), anchor_cache=anchors,
            normalize_mode=cfg.data.normalize,
        ),
    }
    extra = {}
    # Named after the split they are drawn from, not after the class list: both
    # are VAL-SPLIT subjects restricted to a held-out class set. The test split
    # is untouched until `scripts/evaluate.py --split test`, and a column called
    # `test_*` that was not the test split is exactly the reporting error
    # `CLAUDE.md` exists to prevent.
    for name, classes in (("val:targets.val", targets["val"]),
                          ("val:targets.test", targets["test"])):
        try:
            extra[name] = ExampleDataset(
                corpus, "val", scenes=scenes["val"], targets=list(classes),
                sample=val_examples, seed=int(cfg.train.seed), anchor_cache=anchors,
                normalize_mode=cfg.data.normalize,
            )
        except ValueError:  # that split holds none of those classes
            pass
    if overfit:  # validate on what we are trying to memorise
        datasets["val"] = ExampleDataset(
            corpus, "train", targets=targets["train"], scenes=scenes["train"],
            anchor_cache=anchors, normalize_mode=cfg.data.normalize,
        )
        extra = {}

    attention = model_cfg.carver.get("attention_resolution", 64)
    if attention in (None, "null"):
        attention = None
    else:
        attention = int(attention)

    model = StageB.from_segmenter(
        segmenter,
        spacing=corpus.spacing,
        n_anchors=corpus.n_anchors,
        tau=float(model_cfg.mapper.tau),
        min_mass=float(model_cfg.mapper.min_mass),
        boundary_widths=tuple(model_cfg.boundary_widths),
        carver_width=int(model_cfg.carver.width),
        carver_blocks=int(model_cfg.carver.blocks),
        carver_sees_anchors=require_bool(
            model_cfg.get("carver_sees_anchors", False), "carver_sees_anchors"
        ),
        attention_resolution=attention,
        additive_prior=require_bool(model_cfg.additive_prior, "additive_prior"),
        alpha=float(model_cfg.alpha),
        background_logit=float(model_cfg.background_logit),
        prior_foreground=float(model_cfg.prior_foreground),
        prompt_only=bool(twin),
    )
    checkpoint = stage_cfg.get("boundary_checkpoint")
    if checkpoint not in (None, "", "null") and not twin:
        pretrained = load_model(checkpoint)
        model.boundary.load_state_dict(pretrained.encoder.state_dict())
        model.boundary_lr_scale = float(stage_cfg.boundary_lr_scale)
        print(f"B(I) initialised from {checkpoint}, lr scale {model.boundary_lr_scale}")

    ignore_labels: list[int] = []
    if require_bool(stage_cfg.get("never_supervise_heldout", False), "never_supervise_heldout"):
        held = list(targets["val"]) + list(targets["test"])
        ignore_labels = [corpus.vocab.label(name) for name in held]
        print(f"never_supervise_heldout: ignoring labels {held}")

    task = StageBTask(
        model, corpus.vocab,
        spacing=corpus.spacing,
        anchor_source=resolve_anchor_source(stage_cfg),
        loss_weights=stage_cfg.loss.to_dict(),
        far_epsilon=float(stage_cfg.far.epsilon),
        far_dilation=int(stage_cfg.far.dilation),
        field_centroid_on=str(stage_cfg.field_centroid_on),
        mask_on=str(stage_cfg.mask_on),
        ignore_labels=ignore_labels,
    )
    return datasets, task, extra


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["a", "boundary", "b"])
    parser.add_argument("--segmenter", type=Path, help="Stage A checkpoint (overrides train.stage_b.phase_a_checkpoint)")
    parser.add_argument(
        "--twin", action="store_true",
        help="trained image-free twin: freeze refine at zero, do not run B(I)",
    )
    parser.add_argument("--overfit", type=int, metavar="N", help="train on the first N scenes only")
    parser.add_argument("--out", type=Path, help="run directory (default runs/<stage>)")
    parser.add_argument("--config", type=Path, help="config file (default configs/config.yaml)")
    parser.add_argument("--set", dest="overrides", action="append", metavar="KEY=VALUE")
    args = parser.parse_args()

    if args.segmenter is not None and args.stage != "b":
        parser.error("--segmenter only applies to stage b")
    if args.twin and args.stage != "b":
        parser.error("--twin only applies to stage b")

    cfg = load_config(args.config, overrides=parse_overrides(args.overrides))
    corpus = Corpus.load(cfg.data.root)
    stage_key = {"a": "stage_a", "boundary": "boundary", "b": "stage_b"}[args.stage]
    stage_cfg = cfg.train[stage_key]

    anchors = None
    if args.stage == "a":
        datasets, task, extra = stage_a(cfg, corpus, args.overfit)
    elif args.stage == "boundary":
        datasets, task, extra = boundary(cfg, corpus, args.overfit)
    else:
        path = resolve_segmenter(cfg.train.stage_b, args.segmenter)
        segmenter = load_stage_a(path)
        # Stage A is frozen and nothing rotates, so its output per scene is a
        # constant. Use the precomputed one when it exists for *this* checkpoint.
        anchors = anchor_cache_dir(corpus.root, path)
        anchors = anchors if (anchors / "meta.json").is_file() else None
        print(
            f"anchor masks: {anchors or 'live from ' + str(path)}"
            + ("" if anchors else "  (run scripts/cache_anchors.py for a ~20% faster step)")
        )
        datasets, task, extra = stage_b(cfg, corpus, args.overfit, segmenter, args.twin, anchors)

    suffix = "-twin" if args.twin else ""
    out_dir = args.out or Path("runs") / f"{stage_key}{suffix}"
    train_cfg = {
        key: value for key, value in cfg.train.to_dict().items()
        if key not in ("stage_a", "stage_b", "boundary")
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
    trainer.extra_loaders = {
        name: loader(dataset, batch_size=cfg.train.batch_size, shuffle=False, workers=cfg.train.workers)
        for name, dataset in extra.items()
    }
    probe_examples = cfg.train.get("probe_examples")
    if args.stage == "b" and probe_examples and not args.overfit:
        # Five passes over the selection set every epoch is most of the
        # validation budget, and the per-epoch probe is a trend line - the drops
        # that get reported come from `scripts/evaluate.py` over the whole split.
        trainer.probe_loader = loader(
            ExampleDataset(
                corpus, "val", targets=list(cfg.targets.train),
                sample=int(probe_examples), seed=int(cfg.train.seed),
                anchor_cache=anchors, normalize_mode=cfg.data.normalize,
            ),
            batch_size=cfg.train.batch_size, shuffle=False, workers=cfg.train.workers,
        )

    print(f"corpus {corpus.root} | {len(corpus.vocab)} structures at {corpus.shape}"
          f" | {corpus.spacing} mm/voxel")
    print(f"train {len(datasets['train'])} | val {len(datasets['val'])}"
          + "".join(f" | {name} {len(d)}" for name, d in extra.items())
          + f" -> {out_dir}")
    if args.stage == "b":
        mode = "twin (image-free)" if args.twin else "full (geometry + B)"
        print(f"anchors: {task.anchor_source} | carver: {mode}"
              f" | sees_anchors={task.model.carver_sees_anchors}"
              f" | tau {task.model.mapper.tau} | flip {stage_cfg.flip_probability}")
        print("selecting best.pt on val-split subjects with TRAINED classes;"
              " the two held-out class curves are reported, never selected on")
    trainer.fit()

    task.model = trainer.model = load_model(out_dir / "best.pt", trainer.device)
    summary = trainer.evaluate(with_hausdorff=trainer.wants_hausdorff and args.stage == "b")
    print(format_table(summary, f"best val ({args.stage})"))
    if "anchor_dice" in summary:
        print(f"  predicted anchors scored Dice {summary['anchor_dice']:.4f} against the ground truth")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
