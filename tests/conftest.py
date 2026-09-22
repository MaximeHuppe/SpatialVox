"""A tiny corpus, built once per session, that every test can train on.

The volumes are written by hand rather than generated: eight axis-aligned boxes
at fixed positions in a 32^3 cube, plus an intensity image that has an edge at
every box face. That is all the tests need - a label volume with enough
structures for anchor-first generation to find triples whose conjunction is
unique, and an image whose boundaries line up with them.

Hand-written because the properties the tests depend on are then *visible*: the
positions below are chosen so the eight centroids differ on all three axes, so
no pair is equidistant from the mid-sagittal plane, and so no two structures
share a centroid. A random generator would have to be trusted to keep doing that.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import Corpus, build_examples, write_corpus, write_scene
from src.vocab import Vocabulary

RESOLUTION = 32
SPACING = (1.0, 1.0, 1.0)

#: ``name: (centre (z, y, x), half-extent)``. Deliberately asymmetric about the
#: mid-sagittal plane (x = 15.5) so medial/lateral is decidable for every pair.
#: No name is a substring of another - the corpus checks that a prompt never
#: contains its own target's name, and "eta" inside "theta" would fail it for
#: the wrong reason.
STRUCTURES: dict[str, tuple[tuple[int, int, int], int]] = {
    "alpha": ((7, 8, 6), 3),
    "beta": ((8, 20, 9), 2),
    "gamma": ((14, 9, 22), 3),
    "delta": ((15, 22, 19), 2),
    "sigma": ((21, 7, 11), 2),
    "omega": ((22, 19, 7), 3),
    "kappa": ((25, 12, 24), 2),
    "rho": ((18, 26, 26), 2),
}
NAMES = tuple(STRUCTURES)
TARGETS = {"train": list(NAMES[:4]), "val": list(NAMES[4:6]), "test": list(NAMES[6:])}


def make_scene(seed: int, shift: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """``(image, labels)``: boxes at the fixed positions, jittered by ``shift``."""
    rng = np.random.default_rng(seed)
    labels = np.zeros((RESOLUTION,) * 3, dtype=np.uint16)
    image = rng.normal(0.1, 0.02, labels.shape).astype(np.float32)
    for index, (name, (centre, half)) in enumerate(STRUCTURES.items()):
        offset = rng.integers(-shift, shift + 1, 3) if shift else np.zeros(3, int)
        low = [int(np.clip(c + o - half, 1, RESOLUTION - 2)) for c, o in zip(centre, offset)]
        high = [int(np.clip(l + 2 * half + 1, 2, RESOLUTION - 1)) for l in low]
        box = tuple(slice(l, h) for l, h in zip(low, high))
        labels[box] = index + 1
        image[box] = 0.5 + 0.03 * index + rng.normal(0, 0.01, labels[box].shape)
    return image, labels


@pytest.fixture(scope="session")
def scene():
    """One scene: ``(image, labels)`` at :data:`RESOLUTION`."""
    return make_scene(0)


@pytest.fixture(scope="session")
def vocab() -> Vocabulary:
    return Vocabulary(NAMES)


@pytest.fixture(scope="session")
def corpus(tmp_path_factory, vocab) -> Corpus:
    root = tmp_path_factory.mktemp("corpus")
    manifests = {}
    for offset, (split, count) in enumerate((("train", 3), ("val", 2), ("test", 2))):
        manifests[split] = []
        for index in range(count):
            scene_id = f"{split}_{index}"
            image, labels = make_scene(1000 * offset + index, shift=1)
            write_scene(root, scene_id, image, labels, SPACING)
            manifests[split] += build_examples(
                scene_id, labels, vocab, SPACING, 3, triples=40, locality=8
            )
    write_corpus(
        root, vocab, manifests,
        shape=(RESOLUTION,) * 3, spacing=SPACING, n_anchors=3, targets=TARGETS,
    )
    return Corpus.load(root)
