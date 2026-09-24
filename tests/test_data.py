"""The corpus on disk, the flip, and what the datasets hand the model."""

from __future__ import annotations

import numpy as np
import pytest

import torch

from src.data import (
    AnchorCache,
    Corpus,
    ExampleDataset,
    ROTATIONS,
    SceneDataset,
    anchor_cache_dir,
    build_examples,
    collate,
    import_corpus,
    load_nifti,
    normalize,
    relational_triple_key,
    rotate,
    save_nifti,
    stabilize_relational_manifests,
)
from src.geometry import (
    OPPOSITE, centroids_world, classify, solutions_for, volume_center_world,
)


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
    from conftest import make_scene

    source_ids = np.array(list(ANATOMY))  # fixture label i+1 stands in for source id i
    scenes = {}
    for index, seed in enumerate((3, 4)):
        image, labels = make_scene(seed, shift=1)
        index_of = np.clip(labels.astype(np.int64) - 1, 0, len(source_ids) - 1)
        scenes[f"subject_{index}"] = (image, source_ids[index_of] * (labels > 0))

    root = import_corpus(
        tmp_path / "mri", scenes, ANATOMY, {"train": ["subject_0"], "val": ["subject_1"]}
    )
    corpus = Corpus.load(root)
    assert corpus.vocab.names[0] == "Left-Hippocampus"
    assert (corpus.root / "scenes" / "subject_0" / "labels.nii.gz").is_file()

    item = ExampleDataset(corpus, "train")[0]
    assert set(item["labels"].unique().tolist()) <= set(range(len(ANATOMY) + 1))
    assert item["target_name"] in ANATOMY.values()
    assert corpus.vocab.parse(item["prompt"]) == [
        {"direction": d, "anchor": a} for d, a in zip(item["directions"], item["anchor_names"])
    ]


# ---------------------------------------------------------------------------
# Brain-masked normalisation
# ---------------------------------------------------------------------------
def test_zscore_brain_ignores_the_masked_out_background():
    """`zscore` measures over the whole volume, and `mri.apply_brainmask` makes
    66.5% of an HCP volume exact zero. Measured on data/mri, that puts tissue at
    +1.35 sigma compressed into a 0.52 spread, with an offset that drifts per
    subject with head size. `zscore-brain` restores mean 0, std 1 on tissue.
    """
    rng = np.random.default_rng(0)
    volume = np.zeros((16, 16, 16), dtype=np.float32)
    brain = (slice(4, 12),) * 3
    volume[brain] = rng.normal(800.0, 50.0, size=(8, 8, 8))

    whole = normalize(volume, "zscore")
    masked = normalize(volume, "zscore-brain")
    assert abs(float(masked[brain].mean())) < 0.05
    assert abs(float(masked[brain].std()) - 1.0) < 0.05
    # The old mode leaves tissue far from 0 and compresses its spread.
    assert float(whole[brain].mean()) > 0.5
    assert float(whole[brain].std()) < float(masked[brain].std())


def test_zscore_brain_survives_an_all_background_volume():
    assert not np.isnan(normalize(np.zeros((4, 4, 4), np.float32), "zscore-brain")).any()


def test_an_unknown_normalize_mode_names_the_ones_that_exist():
    with pytest.raises(ValueError, match="zscore-brain"):
        normalize(np.zeros((2, 2, 2), np.float32), "quantile")


# ---------------------------------------------------------------------------
# The flip: §5's three outcomes, measured rather than assumed
# ---------------------------------------------------------------------------
def outcome(corpus, record, labels):
    """What the corpus rule says about a record's clauses, independently of the dataset."""
    centroids = centroids_world(labels, len(corpus.vocab), corpus.spacing)
    center = volume_center_world(labels.shape, corpus.spacing)
    present = [int(v) for v in np.unique(labels) if v != 0]
    return solutions_for(record["anchors"], record["directions"], centroids, present, center)


def test_a_flip_is_retargeted_emptied_or_dropped_and_never_assumed_empty(corpus):
    """§5: "A flip is not assumed to be empty." Each item must agree with the rule."""
    dataset = ExampleDataset(corpus, "train", flip_probability=1.0, normalize_mode="none")
    seen = {"retargeted": 0, "empty": 0, "dropped": 0}
    for index in range(len(dataset)):
        item = dataset[index]
        labels = load_nifti(corpus.root / "scenes" / item["scene"] / "labels.nii.gz", np.int16)
        record = {"anchors": item["anchors"].tolist(), "directions": item["directions"]}
        if int(item["keep"]) == 0:
            seen["dropped"] += 1
            continue
        solutions = outcome(corpus, record, labels)
        if int(item["valid"]) == 1:
            seen["retargeted"] += 1
            assert solutions == [int(item["target"])]
            assert item["target_name"] == corpus.vocab.name(int(item["target"]))
        else:
            seen["empty"] += 1
            assert solutions == []
            assert int(item["target"]) == 0
            assert item["target_name"] == ExampleDataset.NONE
    assert seen["empty"] > 0 and seen["dropped"] > 0, seen


def test_retarget_whitelist_drops_flips_onto_held_out_classes(corpus):
    """Default transfer hygiene: a flip may only name a trained-class target."""
    trained = list(corpus.meta["targets"]["train"])
    held = [n for n in corpus.vocab.names if n not in trained]
    dataset = ExampleDataset(
        corpus, "train", flip_probability=1.0, retarget_only_to=trained, normalize_mode="none",
    )
    for index in range(len(dataset)):
        item = dataset[index]
        if int(item["keep"]) == 1 and int(item["valid"]) == 1:
            assert item["target_name"] in trained
            assert item["target_name"] not in held


def test_a_record_with_target_zero_is_an_empty_prompt_however_it_got_there(corpus):
    """`scripts/evaluate.py` writes an empty-prompt population straight into
    `records`; `valid` has to follow the target, not a separate flag."""
    dataset = ExampleDataset(corpus, "train", normalize_mode="none")
    dataset.records = [{**dataset.records[0], "target": 0}]
    item = dataset[0]
    assert int(item["valid"]) == 0 and int(item["keep"]) == 1
    assert item["target_name"] == ExampleDataset.NONE


def test_an_unflipped_item_is_valid_kept_and_its_manifest_target(corpus):
    dataset = ExampleDataset(corpus, "train", flip_probability=0.0, normalize_mode="none")
    item = dataset[0]
    assert int(item["valid"]) == 1 and int(item["keep"]) == 1
    assert int(item["target"]) == dataset.records[0]["target"]
    assert item["directions"] == dataset.records[0]["directions"]


def test_the_flip_is_reproducible_from_the_epoch_and_the_index(corpus):
    dataset = ExampleDataset(corpus, "train", flip_probability=0.5, normalize_mode="none")
    first = [dataset[i]["directions"] for i in range(6)]
    assert [dataset[i]["directions"] for i in range(6)] == first
    dataset.set_epoch(1)
    assert [dataset[i]["directions"] for i in range(6)] != first


def test_a_stage_b_item_carries_no_mask_and_no_target_geometry(corpus):
    """The dataset hands over the label volume; the task derives the target from it.

    `labels` is there because the *losses* need a target and the metrics need
    something to score against - `tests/test_engine.py` pins that it never
    reaches the model.
    """
    item = ExampleDataset(corpus, "train", normalize_mode="none")[0]
    assert set(item) == {
        "image", "labels", "target", "anchors", "direction_ids", "valid", "keep",
        "example_id", "scene", "prompt", "target_name", "anchor_names", "directions",
    }
    assert item["image"].shape == (1, *corpus.shape)


# ---------------------------------------------------------------------------
# The precomputed anchors
# ---------------------------------------------------------------------------
def test_the_anchor_cache_reproduces_the_masks_it_was_built_from(corpus, tmp_path):
    """The cache is an optimisation, so it has to be numerically the same thing."""
    rng = np.random.default_rng(0)
    dense = np.zeros((len(corpus.vocab), *corpus.shape), dtype=np.float32)
    for channel in range(len(corpus.vocab)):
        z, y, x = rng.integers(4, 20, 3)
        dense[channel, z:z + 4, y:y + 4, x:x + 4] = rng.random((4, 4, 4)).astype(np.float32)
    dense[dense < 1e-3] = 0.0  # the cache's own truncation, applied to the reference

    boxes, chunks = [], []
    for channel in range(len(corpus.vocab)):
        occupied = np.nonzero(dense[channel] > 1e-3)
        low = [int(a.min()) for a in occupied]
        high = [int(a.max()) + 1 for a in occupied]
        boxes.append([*low, *high])
        chunks.append(dense[channel][tuple(slice(l, h) for l, h in zip(low, high))].astype(np.float16).reshape(-1))
    directory = tmp_path / "anchors"
    directory.mkdir()
    np.savez(
        directory / "train_0.npz",
        bbox=np.array(boxes, dtype=np.int16),
        data=np.concatenate(chunks),
        offset=np.cumsum([0] + [c.size for c in chunks]).astype(np.int64),
    )
    cache = AnchorCache(directory, corpus.shape)
    taken = cache.take("train_0", [2, 5, 1])
    assert np.allclose(taken, dense[[1, 4, 0]].astype(np.float16), atol=1e-3)


def test_the_anchor_cache_directory_is_keyed_by_the_checkpoint(tmp_path):
    """A different Stage A writes a different directory - no silent staleness."""
    first, second = tmp_path / "a.pt", tmp_path / "b.pt"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    assert anchor_cache_dir(tmp_path, first) != anchor_cache_dir(tmp_path, second)
    assert anchor_cache_dir(tmp_path, first) == anchor_cache_dir(tmp_path, first)


def test_the_dataset_emits_cached_anchors_in_slot_order(corpus, tmp_path):
    directory = tmp_path / "anchors"
    directory.mkdir()
    size = len(corpus.vocab)
    dense = np.zeros((size, *corpus.shape), dtype=np.float32)
    for channel in range(size):
        dense[channel, channel, channel, channel] = 1.0  # a unique voxel per structure
    boxes = [[c, c, c, c + 1, c + 1, c + 1] for c in range(size)]
    for scene_id in corpus.scene_ids("train"):
        np.savez(
            directory / f"{scene_id}.npz",
            bbox=np.array(boxes, dtype=np.int16),
            data=np.ones(size, dtype=np.float16),
            offset=np.arange(size + 1, dtype=np.int64),
        )
    item = ExampleDataset(corpus, "train", anchor_cache=directory, normalize_mode="none")[0]
    masks = item["anchor_probability"]
    assert masks.shape == (corpus.n_anchors, *corpus.shape)
    for slot, label in enumerate(item["anchors"].tolist()):
        channel = label - 1
        assert float(masks[slot, channel, channel, channel]) == 1.0
        assert float(masks[slot].sum()) == 1.0


# -- cross-subject triple stability ------------------------------------------
def test_relational_triple_key_ignores_clause_order(vocab):
    """Shuffling slots must not change the identity of a triple."""
    a = {"anchors": [1, 2, 3], "directions": ["superior", "medial", "anterior"]}
    b = {"anchors": [3, 1, 2], "directions": ["anterior", "superior", "medial"]}
    assert relational_triple_key(a, vocab) == relational_triple_key(b, vocab)


def test_relational_triple_key_keeps_pairing(vocab):
    """Swapping which anchor has which direction is a different triple."""
    a = {"anchors": [1, 2, 3], "directions": ["superior", "medial", "anterior"]}
    b = {"anchors": [2, 1, 3], "directions": ["superior", "medial", "anterior"]}
    assert relational_triple_key(a, vocab) != relational_triple_key(b, vocab)


def test_stabilize_drops_triples_that_name_different_targets_across_subjects(vocab):
    """The MRI leak: same sentence → thalamus on A, caudate on B → drop both."""
    shared = {
        "anchors": [1, 2, 3],
        "directions": ["superior", "medial", "anterior"],
        "prompt": "x",
    }
    manifests = {
        "train": [
            {**shared, "scene": "s0", "id": "s0__a", "target": vocab.label("alpha")},
            {**shared, "scene": "s1", "id": "s1__g", "target": vocab.label("gamma")},
            {
                "scene": "s0", "id": "s0__stable", "target": vocab.label("beta"),
                "anchors": [4, 5, 6], "directions": ["inferior", "lateral", "posterior"],
                "prompt": "y",
            },
            {
                "scene": "s1", "id": "s1__stable", "target": vocab.label("beta"),
                "anchors": [4, 5, 6], "directions": ["inferior", "lateral", "posterior"],
                "prompt": "y",
            },
        ],
        "val": [],
        "test": [],
    }
    filtered, stats = stabilize_relational_manifests(manifests, vocab)
    kept_ids = {row["id"] for row in filtered["train"]}
    assert kept_ids == {"s0__stable", "s1__stable"}
    assert stats["examples_dropped_unstable_triple"] == 2
    assert stats["triples_colliding"] == 1
    assert stats["triples_stable"] == 1


def test_stabilize_keeps_a_triple_reused_for_the_same_target(vocab):
    row = {
        "anchors": [1, 2, 3], "directions": ["superior", "medial", "anterior"],
        "target": vocab.label("alpha"), "prompt": "x",
    }
    manifests = {
        "train": [
            {**row, "scene": "s0", "id": "a"},
            {**row, "scene": "s1", "id": "b"},
        ],
    }
    filtered, stats = stabilize_relational_manifests(manifests, vocab)
    assert len(filtered["train"]) == 2
    assert stats["examples_dropped_unstable_triple"] == 0
    assert stats["triples_colliding"] == 0


def test_stabilize_drops_cross_split_train_vs_heldout_collision(vocab):
    """Train-only scope: train keeps a stable-within-train triple; val is kept too.

    The exposure stratum counts val rows whose words mean a different target than
    train taught - those used to be dropped by the global filter, which hid the
    recogniser shortcut.
    """
    shared = {
        "anchors": [1, 2, 3],
        "directions": ["superior", "medial", "anterior"],
        "prompt": "x",
    }
    manifests = {
        "train": [{**shared, "scene": "s0", "id": "train_hit", "target": vocab.label("alpha")}],
        "val": [{**shared, "scene": "s1", "id": "held_hit", "target": vocab.label("gamma")}],
    }
    filtered, stats = stabilize_relational_manifests(manifests, vocab)
    assert [row["id"] for row in filtered["train"]] == ["train_hit"]
    assert [row["id"] for row in filtered["val"]] == ["held_hit"]
    assert stats["examples_dropped_unstable_triple"] == 0
    assert stats["triples_colliding"] == 0
    assert stats["define_on"] == "train"
    assert stats["exposure_rows_kept"] == 1


def test_stabilize_define_on_all_still_drops_cross_split_collisions(vocab):
    """Legacy global pass: both sides of a train↔held-out collision go away."""
    shared = {
        "anchors": [1, 2, 3],
        "directions": ["superior", "medial", "anterior"],
        "prompt": "x",
    }
    manifests = {
        "train": [{**shared, "scene": "s0", "id": "train_hit", "target": vocab.label("alpha")}],
        "val": [{**shared, "scene": "s1", "id": "held_hit", "target": vocab.label("gamma")}],
    }
    filtered, stats = stabilize_relational_manifests(manifests, vocab, define_on="all")
    assert filtered["train"] == []
    assert filtered["val"] == []
    assert stats["examples_dropped_unstable_triple"] == 2
    assert stats["triples_colliding"] == 1
