"""Synthetic generation, the corpus on disk, and what the datasets hand the model."""

from __future__ import annotations

import numpy as np
import pytest

from src.data import (
    Corpus,
    ExampleDataset,
    ROTATIONS,
    SceneDataset,
    build_examples,
    check_scene,
    collate,
    import_corpus,
    load_nifti,
    rotate,
    save_nifti,
)
from src.geometry import centroids_world, classify, volume_center_world
from src.synthetic import SHAPE_NAMES, half_extent, draw_params, voxelize


# -- synthetic geometry ------------------------------------------------------
def test_every_primitive_voxelises_inside_its_own_bounding_box():
    rng = np.random.default_rng(0)
    shape, spacing = (32, 32, 32), (1.0, 1.0, 1.0)
    for name in SHAPE_NAMES:
        params = draw_params(name, rng, scale=0.5)
        center = np.array([16.0, 16.0, 16.0])
        mask = voxelize(name, params, center, shape, spacing)
        assert mask.any(), name
        half = half_extent(name, params)
        indices = np.nonzero(mask)  # (z, y, x)
        for axis, world_axis in enumerate((2, 1, 0)):
            span = (indices[axis].min(), indices[axis].max())
            assert center[world_axis] - half[world_axis] - 1 <= span[0]
            assert span[1] <= center[world_axis] + half[world_axis] + 1


def test_a_generated_scene_holds_one_instance_of_every_structure(scene):
    _, labels = scene
    check_scene(labels, len(SHAPE_NAMES), margin=1)
    assert sorted(np.unique(labels)) == list(range(len(SHAPE_NAMES) + 1))


def test_structures_never_overlap(scene):
    """One label volume cannot represent overlap, so count voxels instead."""
    _, labels = scene
    total = sum(int((labels == label).sum()) for label in range(1, len(SHAPE_NAMES) + 1))
    assert total == int((labels != 0).sum())


def test_intensity_does_not_identify_a_structure(scene):
    """Structure means share one range, so brightness must not be a giveaway."""
    image, labels = scene
    means = [float(image[labels == label].mean()) for label in range(1, len(SHAPE_NAMES) + 1)]
    assert max(means) - min(means) < 0.35  # inside the configured [0.45, 0.75] band
    assert float(image[labels == 0].mean()) < min(means)  # background is still darker


def test_a_scene_is_reproducible_from_its_seed():
    from src.synthetic import generate_scene

    settings = ((24, 24, 24), (1.0, 1.0, 1.0), 1, {"background": [0.1, 0.02], "structure": [0.4, 0.8], "noise": 0.02, "blur": 0.5})
    first = generate_scene(11, *settings)
    second = generate_scene(11, *settings)
    assert np.array_equal(first[1], second[1]) and np.allclose(first[0], second[0])


# -- examples and manifests --------------------------------------------------
def test_an_example_never_names_its_own_target(scene, vocab):
    _, labels = scene
    for example in build_examples("s", labels, vocab, (1.0, 1.0, 1.0), 3):
        assert example["target"] not in example["anchors"]
        assert vocab.name(example["target"]) not in example["prompt"]
        assert len(set(example["directions"])) == 3


def test_stored_directions_match_the_geometry_they_describe(scene, vocab):
    _, labels = scene
    centroids = centroids_world(labels, len(vocab), (1.0, 1.0, 1.0))
    center = volume_center_world(labels.shape, (1.0, 1.0, 1.0))
    for example in build_examples("s", labels, vocab, (1.0, 1.0, 1.0), 3):
        for anchor, direction in zip(example["anchors"], example["directions"]):
            assert classify(centroids[example["target"]], centroids[anchor], center) == direction


def test_prompt_clause_order_is_the_channel_order(scene, vocab):
    _, labels = scene
    for example in build_examples("s", labels, vocab, (1.0, 1.0, 1.0), 3):
        clauses = vocab.parse(example["prompt"])
        assert [c["anchor"] for c in clauses] == [vocab.name(a) for a in example["anchors"]]
        assert [c["direction"] for c in clauses] == example["directions"]


def test_check_scene_rejects_a_structure_on_the_border():
    labels = np.zeros((8, 8, 8), dtype=np.uint8)
    labels[0, 0, 0] = 1
    with pytest.raises(ValueError, match="border"):
        check_scene(labels, 1, margin=1)


def test_nifti_survives_a_round_trip(tmp_path):
    volume = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    save_nifti(tmp_path / "v.nii.gz", volume, (1.0, 2.0, 3.0), np.float32)
    assert np.allclose(load_nifti(tmp_path / "v.nii.gz"), volume)


def test_corpus_splits_by_target_class(corpus):
    for split in ("train", "val", "test"):
        allowed = set(corpus.meta["targets"][split])
        assert {corpus.vocab.name(r["target"]) for r in corpus.records(split)} <= allowed
    # The three target sets are disjoint, which is what makes val and test transfer.
    sets = [set(corpus.meta["targets"][s]) for s in ("train", "val", "test")]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])


# -- rotation augmentation ---------------------------------------------------
def test_there_are_exactly_twenty_four_axis_aligned_rotations():
    assert len(ROTATIONS) == 24
    probe = np.arange(27).reshape(3, 3, 3)
    assert len({rotate(probe, r).tobytes() for r in ROTATIONS}) == 24


def test_rotation_preserves_every_structure(scene):
    _, labels = scene
    for sequence in ROTATIONS:
        rotated = rotate(labels, sequence)
        assert np.array_equal(np.unique(rotated), np.unique(labels))
        assert rotated.shape == labels.shape


def test_augmented_clauses_describe_the_rotated_volume(corpus):
    """The point of the rewrite: prompt and tensor can never disagree."""
    dataset = ExampleDataset(corpus, "train", augment=True)
    spacing = corpus.spacing
    for epoch in range(3):
        dataset.set_epoch(epoch)
        for index in range(len(dataset)):
            item = dataset[index]
            labels = item["labels"].numpy()
            centroids = centroids_world(labels, len(corpus.vocab), spacing)
            center = volume_center_world(labels.shape, spacing)
            target = int(item["target"])
            for anchor, direction in zip(item["anchors"].tolist(), item["directions"]):
                assert classify(centroids[target], centroids[anchor], center) == direction
            clauses = corpus.vocab.parse(item["prompt"])
            assert [c["direction"] for c in clauses] == item["directions"]
            assert [c["anchor"] for c in clauses] == item["anchor_names"]


def test_augmentation_actually_moves_the_volume(corpus):
    dataset = ExampleDataset(corpus, "train", augment=True)
    poses = set()
    for epoch in range(8):
        dataset.set_epoch(epoch)
        poses.add(dataset[0]["labels"].numpy().tobytes())
    assert len(poses) > 1


# -- datasets ----------------------------------------------------------------
def test_stage_a_items_carry_labels_not_masks(corpus):
    item = SceneDataset(corpus, "train")[0]
    assert item["image"].shape == (1, *corpus.shape)
    assert item["labels"].shape == corpus.shape
    assert item["prompt_ids"].tolist() == list(range(len(corpus.vocab)))


def test_stage_a_can_subsample_the_vocabulary(corpus):
    item = SceneDataset(corpus, "train", prompts_per_item=3)[0]
    assert item["prompt_ids"].shape == (3,)
    assert len(set(item["prompt_ids"].tolist())) == 3


def test_stage_b_items_expose_only_what_the_model_may_see(corpus):
    item = ExampleDataset(corpus, "train")[0]
    assert set(item) == {
        "image", "labels", "target", "anchors", "direction_ids",
        "example_id", "scene", "prompt", "target_name", "anchor_names", "directions",
    }
    assert item["anchors"].shape == (corpus.n_anchors,)
    assert item["target"] not in item["anchors"]


def test_collate_keeps_one_metadata_entry_per_sample(corpus):
    dataset = ExampleDataset(corpus, "train")
    batch = collate([dataset[0], dataset[1]])
    assert batch["anchors"].shape == (2, corpus.n_anchors)
    assert len(batch["anchor_names"]) == 2 and len(batch["anchor_names"][0]) == corpus.n_anchors


#: Anatomical names on FreeSurfer-style, non-contiguous source label ids.
ANATOMY = {
    17: "Left-Hippocampus", 53: "Right-Hippocampus", 16: "Brain-Stem", 12: "Left-Putamen",
    51: "Right-Putamen", 10: "Left-Thalamus", 49: "Right-Thalamus", 11: "Left-Caudate",
    50: "Right-Caudate", 8: "Left-Cerebellum-Cortex",
}


def test_importing_real_volumes_produces_the_same_corpus_shape(tmp_path):
    """The MRI entry point: arbitrary label ids and real names in, a corpus out."""
    from src.synthetic import generate_scene

    appearance = {"background": [0.12, 0.04], "structure": [0.45, 0.75], "noise": 0.03, "blur": 0.6}
    source_ids = np.array(list(ANATOMY))  # synthetic label i+1 stands in for source id i
    scenes = {}
    for index, seed in enumerate((3, 4)):
        image, labels = generate_scene(seed, (24, 24, 24), (1.0, 1.0, 1.0), 1, appearance)
        index_of = np.clip(labels.astype(np.int64) - 1, 0, None)
        scenes[f"subject_{index}"] = (image, source_ids[index_of] * (labels > 0))

    root = import_corpus(
        tmp_path / "mri", scenes, ANATOMY, {"train": ["subject_0"], "val": ["subject_1"]}
    )
    corpus = Corpus.load(root)
    assert corpus.vocab.names[0] == "Left-Hippocampus"
    assert (corpus.root / "scenes" / "subject_0" / "labels.nii.gz").is_file()

    item = ExampleDataset(corpus, "train")[0]
    assert set(item["labels"].unique().tolist()) == set(range(len(ANATOMY) + 1))
    assert item["target_name"] in ANATOMY.values()
    assert corpus.vocab.parse(item["prompt"]) == [
        {"direction": d, "anchor": a} for d, a in zip(item["directions"], item["anchor_names"])
    ]
