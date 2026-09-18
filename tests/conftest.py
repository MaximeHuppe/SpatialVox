"""A tiny corpus, generated once per session, that every test can train on."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import Corpus, build_examples, write_corpus, write_scene
from src.synthetic import SHAPE_NAMES, generate_scene
from src.vocab import Vocabulary

RESOLUTION = 32  # small enough to generate and train on in seconds
SPACING = (1.0, 1.0, 1.0)
APPEARANCE = {"background": [0.12, 0.04], "structure": [0.45, 0.75], "noise": 0.03, "blur": 0.6}
TARGETS = {"train": list(SHAPE_NAMES[:7]), "val": list(SHAPE_NAMES[7:9]), "test": [SHAPE_NAMES[9]]}


@pytest.fixture(scope="session")
def scene():
    """One synthetic scene: ``(image, labels)`` at :data:`RESOLUTION`."""
    return generate_scene(7, (RESOLUTION,) * 3, SPACING, 1, APPEARANCE)


@pytest.fixture(scope="session")
def vocab() -> Vocabulary:
    return Vocabulary(SHAPE_NAMES)


@pytest.fixture(scope="session")
def corpus(tmp_path_factory, vocab) -> Corpus:
    root = tmp_path_factory.mktemp("corpus")
    manifests = {}
    # A target with no feasible anchor set is dropped, so each split needs more
    # than one scene for every one of its target classes to be represented.
    for offset, (split, count) in enumerate((("train", 3), ("val", 2), ("test", 2))):
        manifests[split] = []
        for index in range(count):
            scene_id = f"{split}_{index}"
            # An explicit seed, not hash(): hash() is salted per process, which
            # would make the corpus - and every test that trains on it - differ
            # from run to run.
            image, labels = generate_scene(
                1000 * offset + index, (RESOLUTION,) * 3, SPACING, 1, APPEARANCE
            )
            write_scene(root, scene_id, image, labels, SPACING)
            manifests[split] += build_examples(scene_id, labels, vocab, SPACING, 3)
    write_corpus(
        root, vocab, manifests,
        shape=(RESOLUTION,) * 3, spacing=SPACING, n_anchors=3, targets=TARGETS,
    )
    return Corpus.load(root)
