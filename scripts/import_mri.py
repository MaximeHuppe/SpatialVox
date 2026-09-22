#!/usr/bin/env python
"""Build an MRI corpus: ``scenes/<id>/{image,labels}.nii.gz`` plus manifests.

    scripts/import_mri.py                         # 200 HCP subjects -> data/mri
    scripts/import_mri.py --n 12 --smoke          # a handful, into data/mri_smoke
    scripts/import_mri.py --source data/raw
    scripts/import_mri.py --download --ids 100206,100307

Source layouts (auto-detected per subject):

    {id}/T1w/T1w_acpc_dc_restore_1.25.nii.gz + wmparc.nii.gz   # HCP FreeSurfer
    {id}/t1.nii.gz + mask.nii.gz                               # collapsed dense ids

Each scene is reoriented to RAS and centre-cropped (or padded) to
``data.resolution`` at native spacing — the T1 is not zoomed.
and written as::

    <output>/
      meta.json  vocab.json  train.jsonl val.jsonl test.jsonl
      scenes/<subject_id>/image.nii.gz
      scenes/<subject_id>/labels.nii.gz

Train with ``scripts/train.py a --set data.root=data/mri --set data.normalize=zscore``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.config import load_config, parse_overrides
from src.data import build_examples, load_nifti, write_corpus, write_scene
from src.mri import (
    download_subject,
    label_names,
    list_subjects,
    load_subject,
    remap_source_labels,
    split_subjects,
    subject_has_examples,
    volume_cfg,
)

SMOKE = {"mri.n_subjects": 12, "mri.output": "data/mri_smoke"}


def _parse_ids(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, help="HCP tree or processed subject folders")
    parser.add_argument("--output", type=Path, help="corpus directory (default mri.output)")
    parser.add_argument("--n", type=int, dest="n_subjects", help="how many subjects to keep")
    parser.add_argument("--download", action="store_true", help="fetch subjects with aws s3 into --source")
    parser.add_argument("--ids", help="comma-separated HCP ids (with --download, or a keep-list)")
    parser.add_argument("--smoke", action="store_true", help="12 subjects into data/mri_smoke")
    parser.add_argument("--set", dest="overrides", action="append", metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(overrides={**(SMOKE if args.smoke else {}), **parse_overrides(args.overrides)})
    mri = cfg.mri
    source = Path(args.source or mri.source)
    output = Path(args.output or mri.output)
    n_subjects = int(args.n_subjects if args.n_subjects is not None else mri.n_subjects)
    image_names, label_files, brainmask_names = volume_cfg(mri)
    structures = mri.structures.to_dict()
    requested = _parse_ids(args.ids)

    if args.download:
        aws = mri.aws.to_dict()
        ids = requested or list_subjects(source, image_names=image_names, label_names_=label_files)[:n_subjects]
        if not ids:
            raise FileNotFoundError(
                "nothing to download: pass --ids 100206,100307 or point --source at ids to complete"
            )
        source.mkdir(parents=True, exist_ok=True)
        for subject_id in ids:
            print(f"download {subject_id}")
            download_subject(
                subject_id, source, uri=str(aws["uri"]), profile=aws.get("profile") or None
            )

    subject_ids = list_subjects(source, image_names=image_names, label_names_=label_files)
    if requested:
        allowed = set(requested)
        subject_ids = [sid for sid in subject_ids if sid in allowed]
    if not subject_ids:
        raise FileNotFoundError(
            f"no T1+label subjects under {source}. Point --source at HCP "
            f"({mri.source}) or a folder of t1.nii.gz + mask.nii.gz, "
            f"or pass --download --ids ..."
        )

    output = output.resolve()
    print(f"writing scenes to {output}/scenes/<id>/ as they load")
    scheme: str | None = None
    spacing = None
    vocab = None
    kept: list[str] = []
    skipped = 0
    for subject_id in subject_ids:
        if len(kept) >= n_subjects:
            break
        image, labels, subject_spacing, subject_scheme = load_subject(
            source,
            subject_id,
            resolution=int(cfg.data.resolution),
            image_names=image_names,
            label_names_=label_files,
            brainmask_names=brainmask_names,
            apply_brainmask=bool(mri.apply_brainmask),
        )
        if scheme is None:
            scheme = subject_scheme
        elif subject_scheme != scheme:
            print(f"skip {subject_id}: label scheme {subject_scheme} != {scheme}")
            skipped += 1
            continue
        names = label_names(structures, scheme)
        remapped, vocab = remap_source_labels(labels, names)
        if not subject_has_examples(remapped, vocab, subject_spacing, cfg.data.n_anchors):
            skipped += 1
            continue
        write_scene(output, subject_id, image, remapped, subject_spacing)
        spacing = subject_spacing
        kept.append(subject_id)
        print(f"\rwrote {output / 'scenes' / subject_id}  {len(kept)}/{n_subjects}  skipped {skipped}", end="", flush=True)
    print()
    if len(kept) < n_subjects:
        print(f"only {len(kept)} usable subjects (asked {n_subjects})")
    if not kept or scheme is None or spacing is None or vocab is None:
        raise RuntimeError(f"no usable subjects under {source}")

    splits = split_subjects(kept, mri.subject_fractions.to_dict(), seed=int(mri.split_seed))
    manifests: dict[str, list] = {}
    stats: dict[str, int] = {}
    shape = None
    for split, ids in splits.items():
        manifests[split] = []
        for scene_id in ids:
            labels = load_nifti(output / "scenes" / scene_id / "labels.nii.gz", np.int16)
            shape = labels.shape
            manifests[split] += build_examples(
                scene_id, labels, vocab, spacing, cfg.data.n_anchors,
                shuffle_clauses=bool(cfg.data.get("shuffle_clauses", True)),
                triples=int(cfg.data.get("triples", 300)),
                locality=int(cfg.data.get("locality", 8)),
                stats=stats,
            )
    write_corpus(
        output,
        vocab,
        manifests,
        shape=shape,
        spacing=spacing,
        n_anchors=cfg.data.n_anchors,
        shuffle_clauses=bool(cfg.data.get("shuffle_clauses", True)),
        targets=cfg.targets.to_dict(),
        extra={
            "source": "mri",
            "label_scheme": scheme,
            "n_subjects": len(kept),
            "skipped": skipped,
            "origin": str(source),
        },
    )
    if stats.get("dropped_ambiguous"):
        print(f"dropped {stats['dropped_ambiguous']} examples whose clauses matched more than one structure")
    if stats.get("pool_fallbacks"):
        print(
            f"WARNING: {stats['pool_fallbacks']} examples fell back to the deterministic"
            " nearest anchor set (the pool window held no feasible triple)"
        )
    print(
        f"corpus written to {output}  "
        + "  ".join(f"{split}={len(ids)}" for split, ids in splits.items())
    )
    print(f"train with: scripts/train.py a --set data.root={output} --set data.normalize=zscore")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
