"""HCP / FreeSurfer import: RAS cube, id remap, corpus layout."""

from __future__ import annotations

import numpy as np
import nibabel as nib
import pytest

from src.config import load_config
from src.data import Corpus, ExampleDataset, import_corpus
from src.mri import (
    crop_or_pad,
    detect_scheme,
    label_names,
    pad_to_cube,
    prepare_volume,
    remap_source_labels,
    resize,
    split_subjects,
)
from conftest import NAMES


def _write_nifti(path, data, affine):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(np.ascontiguousarray(data), affine)
    nib.save(image, str(path))
    return path


def test_label_names_follow_yaml_order_and_scheme():
    cfg = load_config()
    structures = cfg.mri.structures.to_dict()
    fs = label_names(structures, "fs")
    dense = label_names(structures, "dense")
    assert list(fs.values()) == list(structures)
    assert fs[17] == "Left-Hippocampus" and dense[14] == "Left-Hippocampus"
    assert fs[16] == "Brain-Stem" and 24 not in fs  # CSF is dropped


def test_mri_targets_are_mirror_closed_and_inside_the_vocabulary():
    cfg = load_config()
    structures = set(cfg.mri.structures)
    listed = {name for names in cfg.targets.to_dict().values() for name in names}
    assert listed <= structures
    assert not (listed & set(NAMES))
    for fold, names in cfg.targets.to_dict().items():
        types: dict[str, set[str]] = {}
        for name in names:
            if name.startswith("Left-"):
                types.setdefault(name[5:], set()).add("L")
            elif name.startswith("Right-"):
                types.setdefault(name[6:], set()).add("R")
            else:
                pytest.fail(f"{fold} target {name} is not a hemisphere pair")
        assert all(sides == {"L", "R"} for sides in types.values()), fold


def test_remap_drops_unlisted_freesurfer_ids():
    names = {10: "Left-Thalamus", 49: "Right-Thalamus", 16: "Brain-Stem"}
    labels = np.zeros((4, 4, 4), dtype=np.int32)
    labels[1, 1, 1] = 10
    labels[1, 1, 2] = 49
    labels[2, 2, 2] = 3  # cortex, not in the list
    labels[3, 3, 3] = 1003  # aparc gyrus — must not clip onto the last listed id
    remapped, vocab = remap_source_labels(labels, names)
    assert vocab.names == ("Left-Thalamus", "Right-Thalamus", "Brain-Stem")
    assert remapped[1, 1, 1] == 1
    assert remapped[1, 1, 2] == 2
    assert remapped[2, 2, 2] == 0
    assert remapped[3, 3, 3] == 0


def test_pad_to_cube_keeps_the_centre():
    volume = np.zeros((3, 5, 3), dtype=np.int16)
    volume[1, 2, 1] = 7
    padded = pad_to_cube(volume)
    assert padded.shape == (5, 5, 5)
    assert padded[2, 2, 2] == 7


def test_crop_or_pad_does_not_interpolate():
    volume = np.arange(5 * 5 * 5, dtype=np.int16).reshape(5, 5, 5)
    cropped = crop_or_pad(volume, 3)
    assert cropped.shape == (3, 3, 3)
    assert cropped[0, 0, 0] == volume[1, 1, 1]
    assert cropped[1, 1, 1] == volume[2, 2, 2]


def test_resize_nearest_preserves_a_label():
    volume = np.zeros((4, 4, 4), dtype=np.int16)
    volume[1:3, 1:3, 1:3] = 5
    out = resize(volume, (8, 8, 8), order=0)
    assert out.shape == (8, 8, 8)
    assert 5 in out and out.dtype == volume.dtype


def test_split_subjects_is_reproducible_and_covers_everyone():
    ids = [f"{i:03d}" for i in range(10)]
    first = split_subjects(ids, {"train": 0.8, "val": 0.1, "test": 0.1}, seed=7)
    second = split_subjects(ids, {"train": 0.8, "val": 0.1, "test": 0.1}, seed=7)
    assert first == second
    assert set().union(*first.values()) == set(ids)
    assert len(set(first["train"]) & set(first["val"])) == 0


def test_prepare_volume_reorients_las_to_ras_and_returns_zyx(tmp_path):
    """HCP T1 is LAS; a marker at index 0 (anatomical right) must land at high x."""
    affine = np.diag([-1.0, 1.0, 1.0, 1.0])
    image = np.zeros((6, 8, 6), dtype=np.float32)
    labels = np.zeros((6, 8, 6), dtype=np.int16)
    image[0, 4, 3] = 1.0
    labels[0, 4, 3] = 17  # Left-Hippocampus id, used only as a marker
    _write_nifti(tmp_path / "t1.nii.gz", image, affine)
    labels_path = _write_nifti(tmp_path / "aseg.nii.gz", labels, affine)
    assert nib.aff2axcodes(affine) == ("L", "A", "S")
    assert detect_scheme(labels_path) == "fs"

    out_image, out_labels, spacing = prepare_volume(
        tmp_path / "t1.nii.gz",
        tmp_path / "aseg.nii.gz",
        resolution=8,
        apply_brainmask=False,
    )
    assert out_image.shape == out_labels.shape == (8, 8, 8)  # (z, y, x)
    assert spacing == (1.0, 1.0, 1.0)  # native zooms; the T1 is not resampled
    # After RAS, anatomical right is high x, which is the last array axis.
    x_of_marker = int(np.argmax(out_image.max(axis=(0, 1))))
    assert x_of_marker > 4


def test_importing_prepared_mri_writes_the_corpus_layout(tmp_path):
    cfg = load_config()
    names = label_names(cfg.mri.structures.to_dict(), "fs")
    shape = (16, 16, 16)
    rng = np.random.default_rng(0)
    scenes = {}
    for index, sid in enumerate(("sub_a", "sub_b")):
        image = rng.random(shape).astype(np.float32)
        labels = np.zeros(shape, dtype=np.int32)
        for i, source_id in enumerate(names):
            labels[2 + i % 12, 2 + (2 * i) % 12, 2 + (3 * i) % 12] = source_id
        scenes[sid] = (image, labels)

    root = import_corpus(
        tmp_path / "mri",
        scenes,
        names,
        {"train": ["sub_a"], "val": ["sub_b"], "test": ["sub_b"]},
        spacing=(2.0, 2.0, 2.0),
        targets=cfg.targets.to_dict(),
        extra={"source": "mri"},
    )
    corpus = Corpus.load(root)
    assert (root / "scenes" / "sub_a" / "image.nii.gz").is_file()
    assert (root / "scenes" / "sub_a" / "labels.nii.gz").is_file()
    assert (root / "train.jsonl").is_file()
    assert corpus.vocab.names[0] == "Left-Lateral-Ventricle"
    assert corpus.meta["source"] == "mri"
    item = ExampleDataset(corpus, "train")[0]
    assert item["target_name"] in cfg.targets.train
    assert item["target_name"] not in item["prompt"]
