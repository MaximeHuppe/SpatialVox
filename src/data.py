"""The corpus on disk, and the two datasets read from it.

Layout, identical for synthetic and real data::

    <root>/
      meta.json                     shape, spacing, anchors per prompt, target split
      vocab.json                    ordered structure names; label id == index + 1
      scenes/<scene_id>/image.nii.gz    float intensity volume  (RAS)
      scenes/<scene_id>/labels.nii.gz   integer label volume    (RAS)
      train.jsonl val.jsonl test.jsonl  one relational example per line

A manifest line is small on purpose::

    {"scene": ..., "id": ..., "target": 7, "anchors": [3, 1, 9],
     "directions": ["superior", "medial", "anterior"], "prompt": "segment ..."}

Masks are never stored: every mask in this project is ``labels == id``, derived
on the accelerator from the one label volume the dataset returns. That keeps a
sample small enough to stay cheap at 128^3 with a large vocabulary, and it makes
it impossible for a stored mask to drift out of step with its label volume.

Augmentation is one axis-aligned rotation of the octahedral group per item. It
needs no "relation rewrite": the rotated volume is fed back through the same
:func:`src.geometry.select_anchors` that built the corpus, so the clauses are
re-derived rather than relabelled.
"""

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.geometry import (
    DIRECTIONS, anchor_first_examples, centroids_world, directions_for,
    select_anchors, solutions_for, volume_center_world,
)
from src.vocab import Vocabulary

# ---------------------------------------------------------------------------
# NIfTI I/O. In memory volumes are (z, y, x); NIfTI stores (x, y, z) with an
# affine that maps voxel indices to world millimetres, so every write transposes
# and every read transposes back.
# ---------------------------------------------------------------------------
def save_nifti(path: Path | str, volume: np.ndarray, spacing: Sequence[float], dtype=None) -> Path:
    """Write a ``(z, y, x)`` volume as a RAS ``.nii.gz``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.ascontiguousarray(volume if dtype is None else volume.astype(dtype, copy=False))
    affine = np.diag([*map(float, spacing), 1.0])
    image = nib.Nifti1Image(array.transpose(2, 1, 0), affine)
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    nib.save(image, path)
    return path


def load_nifti(path: Path | str, dtype=np.float32) -> np.ndarray:
    """Read a RAS ``.nii.gz`` back into a ``(z, y, x)`` array."""
    image = nib.load(str(path))
    payload = np.asanyarray(image.dataobj)
    if np.issubdtype(np.dtype(dtype), np.integer):
        payload = np.rint(payload)
    return np.ascontiguousarray(payload.transpose(2, 1, 0).astype(dtype, copy=False))


def normalize(image: np.ndarray, mode: str = "none") -> np.ndarray:
    """Intensity normalisation: ``none`` or a per-volume ``zscore``.

    ``none`` is the default because the synthetic generator already emits a
    bounded [0, 1] image and that is what the published runs were trained on.
    Real MRI has no such guarantee - set ``data.normalize: zscore`` for it, or
    replace this function with percentile clipping or a bias-field correction if
    a modality needs one.
    """
    image = image.astype(np.float32, copy=False)
    if mode == "none":
        return image
    if mode == "zscore":
        return (image - image.mean()) / (image.std() + 1e-6)
    raise ValueError(f"normalize must be 'none' or 'zscore', got {mode!r}")


# ---------------------------------------------------------------------------
# Building a corpus
# ---------------------------------------------------------------------------
def build_examples(
    scene_id: str,
    labels: np.ndarray,
    vocab: Vocabulary,
    spacing: Sequence[float],
    n_anchors: int = 3,
    *,
    pool: int | None = None,
    shuffle_clauses: bool = True,
    selection: str = "target-first",
    triples: int = 300,
    locality: int = 8,
    unique_only: bool = True,
    stats: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Every relational example a scene supports: one per structure, as target.

    A target is dropped - not repaired - when no set of ``n_anchors`` structures
    with pairwise-distinct directions exists for it.

    Anchor order is randomised, from a generator seeded by ``(scene_id, target)``
    so the manifest stays reproducible from the scene alone. Order must carry no
    information: the anchors are ranked by distance to choose them, and storing
    that ranking would make the slot index a proxy for proximity. The randomised
    order is then shared by the prompt clauses and the mask channels - clause
    ``i`` always describes channel ``i``.
    """
    present = [int(v) for v in np.unique(labels) if v != 0]
    centroids = centroids_world(labels, len(vocab), tuple(spacing))
    center = volume_center_world(labels.shape, tuple(spacing))
    seed = zlib.crc32(scene_id.encode("utf-8"))

    def record(index: int, target: int, anchors: Sequence[int], directions: Sequence[str]):
        clauses = [{"direction": d, "anchor": vocab.name(a)} for a, d in zip(anchors, directions)]
        return {
            "scene": scene_id,
            "id": f"{scene_id}__{vocab.name(target)}__{index}",
            "target": int(target),
            "anchors": [int(a) for a in anchors],
            "directions": list(directions),
            "prompt": vocab.render(clauses),
        }

    if selection == "anchor-first":
        rng = np.random.default_rng([seed, 0])
        found = anchor_first_examples(
            centroids, present, center, n_anchors,
            triples=triples, locality=locality, rng=rng, shuffle=shuffle_clauses,
        )
        counter: dict[int, int] = {}
        examples = []
        for anchors, directions, target in found:
            counter[target] = counter.get(target, 0) + 1
            examples.append(record(counter[target] - 1, target, anchors, directions))
        if stats is not None:
            stats["examples"] = stats.get("examples", 0) + len(examples)
        return examples

    if selection != "target-first":
        raise ValueError(f"selection must be 'target-first' or 'anchor-first', got {selection!r}")

    fallbacks: list[int] = []
    examples = []
    for target in present:
        rng = np.random.default_rng([seed, target])
        chosen = select_anchors(
            target, centroids, present, center, n_anchors,
            pool=pool, rng=rng, shuffle=shuffle_clauses, fallbacks=fallbacks,
        )
        if chosen is None:
            continue
        anchors = [label for label, _ in chosen]
        directions = [d for _, d in chosen]
        # A prompt that describes more than one structure teaches the model to
        # prefer one defensible reading over another. Drop it rather than
        # supervise on it - well-posedness by construction, not by luck.
        if unique_only and len(
            solutions_for(anchors, directions, centroids, present, center)
        ) != 1:
            if stats is not None:
                stats["dropped_ambiguous"] = stats.get("dropped_ambiguous", 0) + 1
            continue
        examples.append(record(0, target, anchors, directions))
    if stats is not None:
        stats["examples"] = stats.get("examples", 0) + len(examples)
        stats["pool_fallbacks"] = stats.get("pool_fallbacks", 0) + len(fallbacks)
    return examples


def check_scene(labels: np.ndarray, expected: int, margin: int) -> None:
    """Fail fast on a malformed scene: missing structures, or one touching the border."""
    present = {int(v) for v in np.unique(labels)} - {0}
    if len(present) != expected:
        raise ValueError(f"expected {expected} structures, found {sorted(present)}")
    if margin > 0:
        interior = np.zeros(labels.shape, dtype=bool)
        interior[margin:-margin, margin:-margin, margin:-margin] = True
        if np.any((labels != 0) & ~interior):
            raise ValueError(f"a structure is closer than {margin} voxels to the border")


def write_scene(root: Path | str, scene_id: str, image: np.ndarray, labels: np.ndarray, spacing) -> Path:
    """Write one scene's two volumes."""
    directory = Path(root) / "scenes" / scene_id
    save_nifti(directory / "image.nii.gz", image, spacing, np.float32)
    save_nifti(directory / "labels.nii.gz", labels, spacing, np.uint16)
    return directory


def write_corpus(
    root: Path | str,
    vocab: Vocabulary,
    manifests: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    shape: Sequence[int],
    spacing: Sequence[float],
    n_anchors: int,
    targets: Mapping[str, Sequence[str]],
    anchor_pool: int | None = None,
    selection: str = "target-first",
    shuffle_clauses: bool = True,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write ``vocab.json``, ``meta.json`` and one JSONL manifest per split."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    vocab.save(root / "vocab.json")
    counts = {}
    for split, records in manifests.items():
        rows = list(records)
        (root / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
        )
        counts[split] = len(rows)
    meta = {
        "shape": [int(v) for v in shape],
        "spacing": [float(v) for v in spacing],
        "n_anchors": int(n_anchors),
        "anchor_pool": int(n_anchors if anchor_pool is None else anchor_pool),
        "selection": str(selection),
        "shuffle_clauses": bool(shuffle_clauses),
        "targets": {split: list(names) for split, names in targets.items()},
        "examples": counts,
        **dict(extra or {}),
    }
    (root / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return root


def import_corpus(
    root: Path | str,
    scenes: Mapping[str, tuple[np.ndarray, np.ndarray]],
    label_names: Mapping[int, str],
    splits: Mapping[str, Sequence[str]],
    *,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    n_anchors: int = 3,
    anchor_pool: int | None = None,
    targets: Mapping[str, Sequence[str]] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Turn real volumes into a corpus of this project's shape.

    Args:
        scenes: ``{scene_id: (image, labels)}``; both ``(z, y, x)``, the label
            volume using whatever ids the source segmentation uses.
        label_names: ``{source label id: structure name}``. Ids not listed are
            dropped; the remaining names become the vocabulary, in the order
            given, and the written labels are remapped to ``index + 1``.
        splits: ``{split: [scene_id, ...]}``.
        targets: which structure names each split may supervise as a target;
            defaults to every name in every split.

    This is the only entry point real MRI needs - see
    ``docs/method/08_scaling.md``. ``scripts/import_mri.py`` is the HCP path.
    """
    vocab = Vocabulary(tuple(label_names.values()))
    remap = np.zeros(max(label_names) + 1, dtype=np.uint16)
    for source_id, name in label_names.items():
        remap[source_id] = vocab.label(name)

    shape: tuple[int, ...] | None = None
    manifests: dict[str, list[dict[str, Any]]] = {}
    for split, scene_ids in splits.items():
        manifests[split] = []
        for scene_id in scene_ids:
            image, labels = scenes[scene_id]
            source = labels.astype(np.int64, copy=False)
            mapped = np.zeros(source.shape, dtype=np.uint16)
            valid = (source >= 0) & (source < len(remap))
            mapped[valid] = remap[source[valid]]
            labels = mapped
            shape = shape or labels.shape
            write_scene(root, scene_id, image, labels, spacing)
            manifests[split] += build_examples(
                scene_id, labels, vocab, spacing, n_anchors, pool=anchor_pool
            )
    if shape is None:
        raise ValueError("no scenes to import")
    return write_corpus(
        root,
        vocab,
        manifests,
        shape=shape,
        spacing=spacing,
        n_anchors=n_anchors,
        targets=targets or {split: list(vocab.names) for split in splits},
        anchor_pool=anchor_pool,
        extra=extra,
    )


# ---------------------------------------------------------------------------
# Reading a corpus
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Corpus:
    """A corpus directory, its vocabulary and its geometry."""

    root: Path
    vocab: Vocabulary
    meta: dict[str, Any]

    @classmethod
    def load(cls, root: Path | str) -> "Corpus":
        root = Path(root)
        if not (root / "meta.json").is_file():
            raise FileNotFoundError(f"no corpus at {root}; run scripts/generate_data.py first")
        return cls(
            root=root,
            vocab=Vocabulary.load(root / "vocab.json"),
            meta=json.loads((root / "meta.json").read_text(encoding="utf-8")),
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.meta["shape"])  # type: ignore[return-value]

    @property
    def spacing(self) -> tuple[float, float, float]:
        return tuple(self.meta["spacing"])  # type: ignore[return-value]

    @property
    def n_anchors(self) -> int:
        return int(self.meta["n_anchors"])

    @property
    def selection(self) -> str:
        """``target-first`` (anchors nearest the target) or ``anchor-first``.

        Absent in corpora written before anchor-first existed, where it was
        always target-first.
        """
        return str(self.meta.get("selection", "target-first"))

    @property
    def shuffle_clauses(self) -> bool:
        """Whether slot order was randomised when the manifest was written."""
        return bool(self.meta.get("shuffle_clauses", True))

    @property
    def anchor_pool(self) -> int:
        """How many nearest candidates anchors were drawn from.

        Stored so augmentation, which re-derives the clauses every epoch, samples
        the same way the manifest was built. Absent in corpora written before the
        knob existed, where it was the deterministic nearest-feasible set.
        """
        return int(self.meta.get("anchor_pool", self.n_anchors))

    def records(self, split: str, targets: Sequence[str] | None = None) -> list[dict[str, Any]]:
        """Manifest rows of a split, filtered to the target classes it supervises."""
        path = self.root / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"missing manifest {path}")
        allowed = set(self.meta["targets"].get(split, self.vocab.names) if targets is None else targets)
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        kept = [row for row in rows if self.vocab.name(row["target"]) in allowed]
        if not kept:
            raise ValueError(f"no examples in {path} with targets {sorted(allowed)}")
        return kept

    def scene_ids(self, split: str) -> list[str]:
        """The distinct scenes of a split, in manifest order."""
        return list(dict.fromkeys(row["scene"] for row in self.records(split, targets=self.vocab.names)))


# ---------------------------------------------------------------------------
# Axis-aligned rotation augmentation
# ---------------------------------------------------------------------------
def _octahedral_rotations() -> tuple[tuple[tuple[tuple[int, int], int], ...], ...]:
    """The 24 proper rotations of a cube, as sequences of ``np.rot90`` calls."""
    generators = (((0, 1), 1), ((0, 2), 1), ((1, 2), 1))
    probe = np.arange(27).reshape(3, 3, 3)
    seen: dict[bytes, tuple] = {probe.tobytes(): ()}
    frontier: list[tuple] = [()]
    while frontier:
        grown = []
        for sequence in frontier:
            for generator in generators:
                candidate = sequence + (generator,)
                key = np.ascontiguousarray(rotate(probe, candidate)).tobytes()
                if key not in seen:
                    seen[key] = candidate
                    grown.append(candidate)
        frontier = grown
    return tuple(seen.values())


def rotate(volume: np.ndarray, sequence) -> np.ndarray:
    """Apply a rotation sequence to a ``(z, y, x)`` volume."""
    for axes, turns in sequence:
        volume = np.rot90(volume, turns, axes)
    return volume


ROTATIONS = _octahedral_rotations()


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class _SceneCache:
    """Decoded ``(image, labels)`` per scene, shared by that scene's examples."""

    def __init__(self, root: Path, enabled: bool, normalize_mode: str = "none") -> None:
        self.root, self.store = root, ({} if enabled else None)
        self.normalize_mode = normalize_mode

    def get(self, scene_id: str) -> tuple[np.ndarray, np.ndarray]:
        if self.store is not None and scene_id in self.store:
            return self.store[scene_id]
        directory = self.root / "scenes" / scene_id
        item = (
            normalize(load_nifti(directory / "image.nii.gz", np.float32), self.normalize_mode),
            load_nifti(directory / "labels.nii.gz", np.int16),
        )
        if self.store is not None:
            self.store[scene_id] = item
        return item


class SceneDataset(Dataset):
    """Stage A: one scene in, one mask per requested structure name out.

    Items carry the label volume rather than the masks; the training step builds
    ``labels == id`` on the accelerator. ``prompts_per_item`` samples a subset of
    the vocabulary per item, which is what keeps a large anatomical vocabulary
    affordable; evaluation always prompts every name.
    """

    def __init__(
        self,
        corpus: Corpus,
        split: str,
        *,
        prompts_per_item: int | None = None,
        augment: bool = False,
        limit: int | None = None,
        cache: bool = True,
        normalize_mode: str = "none",
    ) -> None:
        self.corpus, self.split = corpus, split
        self.scenes = corpus.scene_ids(split)[: limit or None]
        self.prompts_per_item = prompts_per_item
        self.augment = augment
        # Shared memory, so `set_epoch` reaches persistent dataloader workers,
        # which hold their own copy of this object and never re-read it.
        self.epoch = torch.zeros((), dtype=torch.long).share_memory_()
        self._cache = _SceneCache(corpus.root, cache, normalize_mode)

    def __len__(self) -> int:
        return len(self.scenes)

    def set_epoch(self, epoch: int) -> None:
        self.epoch.fill_(int(epoch))

    def __getitem__(self, index: int) -> dict[str, Any]:
        scene_id = self.scenes[index]
        image, labels = self._cache.get(scene_id)
        rng = np.random.default_rng([int(self.epoch), index])
        if self.augment:
            sequence = ROTATIONS[rng.integers(len(ROTATIONS))]
            image, labels = rotate(image, sequence), rotate(labels, sequence)
        size = len(self.corpus.vocab)
        prompt_ids = (
            np.arange(size)
            if self.prompts_per_item is None
            else rng.choice(size, size=min(self.prompts_per_item, size), replace=False)
        )
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0),
            "labels": torch.from_numpy(np.ascontiguousarray(labels)),
            "prompt_ids": torch.from_numpy(np.sort(prompt_ids)).long(),
            "scene": scene_id,
        }


class ExampleDataset(Dataset):
    """Stage B: three ordered anchors and a prompt in, one target mask out.

    Under augmentation the volume is rotated and the clauses are *re-derived*
    from the rotated geometry, so the prompt always describes the volume that is
    actually in the tensor. A rotation for which no feasible anchor set exists is
    skipped and the stored pose is used instead.
    """

    def __init__(
        self,
        corpus: Corpus,
        split: str,
        *,
        targets: Sequence[str] | None = None,
        scenes: Sequence[str] | None = None,
        augment: bool = False,
        limit: int | None = None,
        sample: int | None = None,
        seed: int = 0,
        cache: bool = True,
        normalize_mode: str = "none",
    ) -> None:
        self.corpus, self.split = corpus, split
        records = corpus.records(split, targets)
        if scenes is not None:
            records = [row for row in records if row["scene"] in set(scenes)]
        if sample is not None and 0 < sample < len(records):
            # Anchor-first generation emits hundreds of prompts per scene, which
            # would make validation cost more than training. Take a fixed random
            # subset rather than a prefix: `limit` would slice by manifest order,
            # which is scene order, so it would silently validate on the first
            # few subjects only. The seed is fixed, so the curve stays comparable
            # across epochs and runs.
            picked = np.random.default_rng(seed).permutation(len(records))[:sample]
            records = [records[int(i)] for i in sorted(picked)]
        self.records = records[: limit or None]
        self.augment = augment
        self.epoch = torch.zeros((), dtype=torch.long).share_memory_()
        self._cache = _SceneCache(corpus.root, cache, normalize_mode)

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch.fill_(int(epoch))

    def target_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.records:
            name = self.corpus.vocab.name(row["target"])
            counts[name] = counts.get(name, 0) + 1
        return dict(sorted(counts.items()))

    def _augment(self, image, labels, record, rng):
        """Rotate, then rebuild the clause list from the rotated geometry."""
        sequence = ROTATIONS[rng.integers(len(ROTATIONS))]
        rotated_labels = rotate(labels, sequence)
        vocab, spacing = self.corpus.vocab, self.corpus.spacing
        present = [int(v) for v in np.unique(rotated_labels) if v != 0]
        centroids = centroids_world(rotated_labels, len(vocab), spacing)
        center = volume_center_world(rotated_labels.shape, spacing)
        if self.corpus.selection == "anchor-first":
            # Keep the stored anchors and target; only re-derive the directions,
            # which is all the rotation changed. Re-*selecting* here would draw a
            # fresh triple nearest the target and quietly turn an anchor-first
            # corpus back into a target-first one, one epoch at a time.
            anchors = [int(a) for a in record["anchors"]]
            if any(a not in present for a in anchors):
                return image, labels, record
            directions = directions_for(record["target"], anchors, centroids, center)
            if directions is None:
                return image, labels, record
            if len(solutions_for(anchors, directions, centroids, present, center)) != 1:
                return image, labels, record  # the rotation broke uniqueness
            chosen = list(zip(anchors, directions))
        else:
            chosen = select_anchors(
                record["target"], centroids, present, center,
                self.corpus.n_anchors, pool=self.corpus.anchor_pool, rng=rng,
                shuffle=self.corpus.shuffle_clauses,
            )
        if chosen is None:  # no feasible anchor set in this pose: keep the stored one
            return image, labels, record
        clauses = [{"direction": d, "anchor": vocab.name(label)} for label, d in chosen]
        record = {
            **record,
            "anchors": [label for label, _ in chosen],
            "directions": [d for _, d in chosen],
            "prompt": vocab.render(clauses),
        }
        return rotate(image, sequence), rotated_labels, record

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image, labels = self._cache.get(record["scene"])
        if self.augment:
            image, labels, record = self._augment(
                image, labels, record, np.random.default_rng([int(self.epoch), index])
            )
        vocab = self.corpus.vocab
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0),
            "labels": torch.from_numpy(np.ascontiguousarray(labels)),
            "target": torch.tensor(record["target"], dtype=torch.long),
            "anchors": torch.tensor(record["anchors"], dtype=torch.long),
            "direction_ids": torch.tensor(
                [DIRECTIONS.index(d) for d in record["directions"]], dtype=torch.long
            ),
            "example_id": record["id"],
            "scene": record["scene"],
            "prompt": record["prompt"],
            "target_name": vocab.name(record["target"]),
            "anchor_names": [vocab.name(label) for label in record["anchors"]],
            "directions": list(record["directions"]),
        }


def collate(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Stack tensors; keep per-sample metadata as plain lists, one entry per sample."""
    return {
        key: torch.stack([item[key] for item in items])
        if isinstance(items[0][key], torch.Tensor)
        else [item[key] for item in items]
        for key in items[0]
    }


def loader(dataset: Dataset, *, batch_size: int, shuffle: bool, workers: int = 0, seed: int = 0) -> DataLoader:
    """A deterministic dataloader."""
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        generator=generator,
        collate_fn=collate,
        persistent_workers=workers > 0,
    )
