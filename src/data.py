"""The corpus on disk, and the two datasets read from it.

Layout::

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

Stage A augments with one axis-aligned rotation of the octahedral group per
item. **Stage B does not rotate**, and that is a change from earlier branches:
``documentation/SpatialVox.md`` (the direction flip) gives it exactly one
augmentation, the direction flip, and a rotated head is not a pose the boundary
encoder will ever be asked about. Dropping it also makes Stage A's output a
function of the scene alone, which is what lets it be computed once and cached.

The flip is the interesting one. One clause is replaced by its opposite with
probability ``train.stage_b.flip_probability``; the new clauses are then scored
against the corpus's own rule, and a flip is **not assumed to be empty**:

* exactly one structure -> that mask, that centroid, the null target ``valid``;
* none -> the empty mask, the null target ``invalid``, heatmap still on the field;
* more than one -> the example is dropped, and ``keep`` is how it is dropped.
"""

from __future__ import annotations

import hashlib
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
    DIRECTIONS, OPPOSITE, anchor_first_examples, centroids_world, solutions_for,
    volume_center_world,
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

    ``none`` is the default because a corpus may already be written with a
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
    if mode == "zscore-brain":
        brain = image[image != 0]
        if brain.size == 0:
            return image
        return (image - brain.mean()) / (brain.std() + 1e-6)
    raise ValueError(
        f"normalize must be 'none', 'zscore' or 'zscore-brain', got {mode!r}"
    )


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
    shuffle_clauses: bool = True,
    triples: int = 300,
    locality: int = 8,
    stats: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Every relational example a scene supports, anchor-first.

    The landmark triple is fixed first and the targets follow, so one anchor set
    serves several targets and only the direction words tell them apart. A triple
    whose conjunction is not unique is dropped, which makes well-posedness hold by
    construction rather than by luck.

    Anchor order is randomised, from a generator seeded by ``scene_id`` so the
    manifest stays reproducible from the scene alone. The randomised order is
    then shared by the prompt clauses and the mask channels - clause ``i`` always
    describes channel ``i`` (:func:`src.geometry.anchor_first_examples`).
    """
    present = [int(v) for v in np.unique(labels) if v != 0]
    centroids = centroids_world(labels, len(vocab), tuple(spacing))
    center = volume_center_world(labels.shape, tuple(spacing))
    rng = np.random.default_rng([zlib.crc32(scene_id.encode("utf-8")), 0])

    counter: dict[int, int] = {}
    examples = []
    for anchors, directions, target in anchor_first_examples(
        centroids, present, center, n_anchors,
        triples=triples, locality=locality, rng=rng, shuffle=shuffle_clauses,
    ):
        counter[target] = counter.get(target, 0) + 1
        clauses = [{"direction": d, "anchor": vocab.name(a)} for a, d in zip(anchors, directions)]
        examples.append(
            {
                "scene": scene_id,
                "id": f"{scene_id}__{vocab.name(target)}__{counter[target] - 1}",
                "target": int(target),
                "anchors": [int(a) for a in anchors],
                "directions": list(directions),
                "prompt": vocab.render(clauses),
            }
        )
    if stats is not None:
        stats["examples"] = stats.get("examples", 0) + len(examples)
    return examples


def relational_triple_key(row: Mapping[str, Any], vocab: Vocabulary) -> frozenset[tuple[str, str]]:
    """Order-invariant relational triple: unordered ``(anchor name, direction)`` pairs.

    Clause order is shuffled at generation, so identity is the *set* of pairs,
    not the slot sequence. Pairing is preserved: ``superior to X, medial to Y``
    is not the same triple as ``superior to Y, medial to X``.
    """
    anchors, directions = row["anchors"], row["directions"]
    if len(anchors) != len(directions):
        raise ValueError(
            f"anchors and directions must have the same length, got "
            f"{len(anchors)} and {len(directions)}"
        )
    return frozenset((vocab.name(int(a)), str(d)) for a, d in zip(anchors, directions))


def stabilize_relational_manifests(
    manifests: Mapping[str, Sequence[Mapping[str, Any]]],
    vocab: Vocabulary,
    *,
    define_on: str = "train",
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    """Drop triples that name different targets across subjects.

    Per-scene uniqueness (``solutions_for`` length 1) is not enough on real MRI:
    the same ``(anchor, direction)`` set can uniquely mean Left-Thalamus on one
    subject and Left-Caudate on another. Training then teaches "this sentence →
    paint a trained class", and held-out eval of the same sentence fails.

    ``define_on`` controls whose labels decide stability:

    * ``"train"`` (default): collect keys from the train split only, and drop
      unstable rows from **train** only. Val/test rows are kept intact so the
      stratum where the same words mean a held-out class stays measurable.
    * ``"all"``: the original global pass over every split (legacy; val/test
      labels then decide which training rows exist).

    Returns ``(filtered_manifests, stats)`` with counts of kept / dropped rows
    and how many distinct triples were stable vs colliding. Stats also report
    how many non-define rows share a triple with a train target (exposure
    stratum), without dropping them.
    """
    if define_on not in ("train", "all"):
        raise ValueError(f"define_on must be 'train' or 'all', got {define_on!r}")

    if define_on == "all":
        source_splits = list(manifests)
    else:
        if "train" not in manifests:
            raise ValueError("define_on='train' needs a 'train' split in manifests")
        source_splits = ["train"]

    by_key: dict[frozenset[tuple[str, str]], set[str]] = {}
    for split in source_splits:
        for row in manifests[split]:
            key = relational_triple_key(row, vocab)
            by_key.setdefault(key, set()).add(vocab.name(int(row["target"])))
    stable = {key for key, names in by_key.items() if len(names) == 1}

    # Train-target name per stable key (for the exposure stratum on other splits).
    train_target_of: dict[frozenset[tuple[str, str]], str] = {}
    for row in manifests.get("train", ()):
        key = relational_triple_key(row, vocab)
        if key in stable and key not in train_target_of:
            train_target_of[key] = vocab.name(int(row["target"]))

    filtered: dict[str, list[dict[str, Any]]] = {split: [] for split in manifests}
    kept = dropped = exposure = 0
    for split, rows in manifests.items():
        for row in rows:
            key = relational_triple_key(row, vocab)
            if define_on == "all" or split == "train":
                if key in stable:
                    filtered[split].append(dict(row))
                    kept += 1
                else:
                    dropped += 1
            else:
                # Hold-out splits: keep every row; count exposure to a train target.
                filtered[split].append(dict(row))
                kept += 1
                train_name = train_target_of.get(key)
                if train_name is not None and vocab.name(int(row["target"])) != train_name:
                    exposure += 1
    stats = {
        "examples_kept": kept,
        "examples_dropped_unstable_triple": dropped,
        "triples_stable": len(stable),
        "triples_colliding": len(by_key) - len(stable),
        "triples_total": len(by_key),
        "define_on": define_on,
        "exposure_rows_kept": exposure,
    }
    return filtered, stats


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
        "selection": "anchor-first",
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

    This is the only entry point real MRI needs; ``scripts/import_mri.py`` is
    the HCP path.
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
            manifests[split] += build_examples(scene_id, labels, vocab, spacing, n_anchors)
    if shape is None:
        raise ValueError("no scenes to import")
    manifests, stab = stabilize_relational_manifests(manifests, vocab)
    meta_extra = {
        **dict(extra or {}),
        "triple_stability": "train-unique-target",
        "triple_stability_stats": stab,
    }
    return write_corpus(
        root,
        vocab,
        manifests,
        shape=shape,
        spacing=spacing,
        n_anchors=n_anchors,
        targets=targets or {split: list(vocab.names) for split in splits},
        extra=meta_extra,
    )


# ---------------------------------------------------------------------------
# The frozen segmenter's soft masks, precomputed
# ---------------------------------------------------------------------------
def anchor_cache_dir(root: Path | str, segmenter: Path | str) -> Path:
    """Where ``scripts/cache_anchors.py`` writes one checkpoint's soft masks.

    Keyed by the SHA-256 of the checkpoint file, so a different Stage A writes a
    different directory. A stale cache cannot be picked up silently - it simply
    is not there for the checkpoint being used.
    """
    digest = hashlib.sha256(Path(segmenter).read_bytes()).hexdigest()[:12]
    return Path(root) / "anchors" / digest


class AnchorCache:
    """Read back one scene's soft masks, as bounding-box crops in float16.

    Stage A is frozen and Stage B does not rotate, so its output for a scene is a
    constant; this is that constant. ``take`` expands only the structures a
    prompt names, which is three of a vocabulary of twenty-three.
    """

    def __init__(self, directory: Path | str, shape: Sequence[int]) -> None:
        self.directory, self.shape = Path(directory), tuple(int(v) for v in shape)
        self.store: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def _scene(self, scene_id: str):
        if scene_id not in self.store:
            with np.load(self.directory / f"{scene_id}.npz") as archive:
                self.store[scene_id] = (archive["bbox"], archive["data"], archive["offset"])
        return self.store[scene_id]

    def take(self, scene_id: str, labels: Sequence[int]) -> np.ndarray:
        """``[K, D, H, W]`` float16 soft masks for those label ids, in order."""
        bbox, data, offset = self._scene(scene_id)
        out = np.zeros((len(labels), *self.shape), dtype=np.float16)
        for slot, label in enumerate(labels):
            channel = int(label) - 1  # label id == vocabulary index + 1
            z0, y0, x0, z1, y1, x1 = (int(v) for v in bbox[channel])
            if z1 <= z0:
                continue
            crop = data[offset[channel]:offset[channel + 1]]
            out[slot, z0:z1, y0:y1, x0:x1] = crop.reshape(z1 - z0, y1 - y0, x1 - x0)
        return out


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
            raise FileNotFoundError(f"no corpus at {root}; run scripts/import_mri.py first")
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
    """Stage B: an image and three clauses in, one target mask out.

    The prompt is the manifest's, except when the flip fires. Then one clause is
    replaced by its opposite and the result is re-scored against the same rule
    that built the corpus, using this scene's own geometry
    (:func:`src.geometry.solutions_for`) - so a flip is *measured*, never assumed
    to be empty:

    ==============================  ===========================================
    the new clauses name...         what the item carries
    ==============================  ===========================================
    exactly one structure           that target, ``valid = 1``, ``keep = 1``
    none                            ``target = 0``, ``valid = 0``, ``keep = 1``
    more than one                   ``keep = 0`` - in no loss and no metric
    ==============================  ===========================================

    ``keep = 0`` is how "dropped" is implemented. A ``Dataset`` has to return an
    item, so the example is emitted and then excluded from every term by weight;
    measured on ``data/mri``, a flip names two or more structures 33% of the
    time, names none 66% and retargets 1%.

    Flipping is training-only. A validation curve mixing retargeted and
    empty-mask prompts would move ``best.pt`` for reasons that have nothing to do
    with the model getting better, and an empty prediction against an empty
    target scores Dice 1.0.
    """

    #: What ``target_name`` says when the clauses name nothing. It is a group
    #: label for the metric breakdown, never a vocabulary entry.
    NONE = "<none>"

    def __init__(
        self,
        corpus: Corpus,
        split: str,
        *,
        targets: Sequence[str] | None = None,
        leave_out: Sequence[str] | None = None,
        scenes: Sequence[str] | None = None,
        flip_probability: float = 0.0,
        retarget_only_to: Sequence[str] | None = None,
        anchor_cache: Path | str | None = None,
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
        self.flip_probability = float(flip_probability)
        # When set, a flip that would name a single structure outside this set is
        # dropped (`keep = 0`) instead of becoming a supervised target. Default
        # in training is the trained-class list (S), so held-out classes stay
        # never-supervised-as-target under the flip. ``None`` is unrestricted.
        self.retarget_only_to = (
            None if retarget_only_to is None else {str(n) for n in retarget_only_to}
        )
        # Episodic leave-one-class-out. Each epoch one supervised class is
        # withheld from the loss, so the model is asked, DURING TRAINING, to
        # segment a class it is not being supervised on this epoch.
        #
        # The point is not regularisation. Measured on this project, a model
        # trained on every class at once learns to RECOGNISE which of them the
        # prompt is asking for and paint that class's remembered shape - held-out
        # volume comes out at 0.43-0.51 of truth and a quarter of predictions are
        # empty, while `where_raw` at the true centroid is 0.97, as good as for a
        # supervised class. Recognition is the cheaper route and nothing in the
        # loss forbids it. Rotating a class out makes that route fail while
        # training, which is the only pressure that reaches it.
        #
        # `keep = 0` is the existing per-example drop and it already reaches every
        # loss term and every metric, so nothing else has to change.
        self.leave_out = [str(n) for n in (leave_out or [])]
        self.epoch = torch.zeros((), dtype=torch.long).share_memory_()
        self._cache = _SceneCache(corpus.root, cache, normalize_mode)
        self._anchors = None if anchor_cache is None else AnchorCache(anchor_cache, corpus.shape)

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

    def flip(self, record, labels, rng) -> tuple[dict[str, Any], int]:
        """One clause to its opposite, re-scored. ``(record, keep)``.

        A retargeted flip carries the new target; an empty one carries
        ``target = 0``, which is the background and therefore *not a structure*.
        That is the single representation of "the clauses name nothing", and
        ``__getitem__`` derives ``valid`` from it rather than tracking a second
        flag that could disagree with it.

        When :attr:`retarget_only_to` is set, a flip that would name a single
        structure outside that whitelist is dropped instead of supervised.
        """
        directions = list(record["directions"])
        slot = int(rng.integers(len(directions)))
        directions[slot] = OPPOSITE[directions[slot]]
        if len(set(directions)) != len(directions):
            return record, 0  # two clauses naming one side is not a prompt
        vocab, spacing = self.corpus.vocab, self.corpus.spacing
        present = [int(v) for v in np.unique(labels) if v != 0]
        centroids = centroids_world(labels, len(vocab), spacing)
        center = volume_center_world(labels.shape, spacing)
        solutions = solutions_for(record["anchors"], directions, centroids, present, center)
        if len(solutions) > 1:
            return record, 0
        if len(solutions) == 1 and self.retarget_only_to is not None:
            if vocab.name(int(solutions[0])) not in self.retarget_only_to:
                return record, 0
        clauses = [
            {"direction": d, "anchor": vocab.name(a)}
            for a, d in zip(record["anchors"], directions)
        ]
        return {
            **record,
            "directions": directions,
            "target": int(solutions[0]) if solutions else 0,
            "prompt": vocab.render(clauses),
        }, 1

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image, labels = self._cache.get(record["scene"])
        keep = 1
        if self.leave_out:
            # One class per epoch, cycled deterministically so every class takes
            # its turn and a run is reproducible from its seed alone.
            withheld = self.leave_out[int(self.epoch) % len(self.leave_out)]
            if self.corpus.vocab.name(record["target"]) == withheld:
                keep = 0
        if self.flip_probability > 0:
            rng = np.random.default_rng([int(self.epoch), index])
            if float(rng.random()) < self.flip_probability:
                record, keep = self.flip(record, labels, rng)
        # Label 0 is the background and is never a structure, so `target = 0`
        # *is* "the clauses name nothing" - including for a population of empty
        # prompts written straight into `records` (`scripts/evaluate.py`).
        valid = int(record["target"] != 0)
        vocab = self.corpus.vocab
        item = {}
        if self._anchors is not None:
            # Exactly the three detached probabilities Stage A would have
            # produced. Nothing else about the cache reaches the model.
            item["anchor_probability"] = torch.from_numpy(
                self._anchors.take(record["scene"], record["anchors"])
            )
        return {
            **item,
            "image": torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0),
            "labels": torch.from_numpy(np.ascontiguousarray(labels)),
            "target": torch.tensor(record["target"], dtype=torch.long),
            "anchors": torch.tensor(record["anchors"], dtype=torch.long),
            "direction_ids": torch.tensor(
                [DIRECTIONS.index(d) for d in record["directions"]], dtype=torch.long
            ),
            "valid": torch.tensor(valid, dtype=torch.long),
            "keep": torch.tensor(keep, dtype=torch.long),
            "example_id": record["id"],
            "scene": record["scene"],
            "prompt": record["prompt"],
            "target_name": vocab.name(record["target"]) if valid else self.NONE,
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
