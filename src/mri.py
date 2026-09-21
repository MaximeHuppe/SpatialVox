"""Load HCP T1 + FreeSurfer labels into the project's ``(z, y, x)`` RAS cube.

Two on-disk layouts are accepted, auto-detected per subject:

* HCP FreeSurfer: ``{id}/T1w/T1w_acpc_dc_restore_1.25.nii.gz`` + ``wmparc.nii.gz``
* already-collapsed: ``{id}/t1.nii.gz`` + ``mask.nii.gz`` (VoxWhisper dense ids)

Labels are resampled onto the T1 (nearest-neighbour, wmparc is a finer grid),
reoriented to RAS, then centre-cropped or padded to ``data.resolution`` — native
voxels, no zoom. The crop is centred so the mid-sagittal plane stays the volume
centre. Ids not listed in ``mri.structures`` become background; ``import_corpus``
then remaps the rest to ``vocabulary index + 1``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy.ndimage import zoom

from src.data import build_examples
from src.vocab import Vocabulary

PathLike = Path | str

IMAGE_CANDIDATES = ("T1w_acpc_dc_restore_1.25.nii.gz", "t1.nii.gz")
LABEL_CANDIDATES = ("wmparc.nii.gz", "mask.nii.gz", "aseg.nii.gz")
BRAINMASK_CANDIDATES = ("brainmask_fs.nii.gz",)
HCP_FILES = (
    "T1w_acpc_dc_restore_1.25.nii.gz",
    "wmparc.nii.gz",
    "brainmask_fs.nii.gz",
)


def label_names(structures: Mapping[str, Mapping[str, int]], scheme: str) -> dict[int, str]:
    """``{source id: name}`` in YAML order, for ``import_corpus``."""
    if scheme not in ("fs", "dense"):
        raise ValueError(f"scheme must be 'fs' or 'dense', got {scheme!r}")
    names = {int(spec[scheme]): name for name, spec in structures.items()}
    if len(names) != len(structures):
        raise ValueError(f"duplicate {scheme} ids in mri.structures")
    return names


def find_volume(subject_dir: Path, filenames: Sequence[str]) -> Path | None:
    """First existing candidate, looking in ``T1w/`` then the subject folder."""
    for name in filenames:
        for candidate in (subject_dir / "T1w" / name, subject_dir / name):
            if candidate.is_file():
                return candidate
    return None


def detect_scheme(labels_path: Path) -> str:
    """Dense ids live in ``mask.nii.gz``; wmparc/aseg keep FreeSurfer ids."""
    return "dense" if labels_path.name == "mask.nii.gz" else "fs"


def list_subjects(source: PathLike, *, image_names=IMAGE_CANDIDATES, label_names_=LABEL_CANDIDATES) -> list[str]:
    """Subject ids under ``source`` that have both an image and a label volume."""
    root = Path(source)
    if not root.is_dir():
        return []
    found = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and find_volume(child, image_names) and find_volume(child, label_names_):
            found.append(child.name)
    return found


def _canonical(path: Path) -> nib.Nifti1Image:
    return nib.as_closest_canonical(nib.load(str(path)))


def _resample_to(source: nib.Nifti1Image, target: nib.Nifti1Image, order: int) -> nib.Nifti1Image:
    if source.shape[:3] == target.shape[:3] and np.allclose(source.affine, target.affine):
        return source
    return resample_from_to(source, target, order=order)


def pad_to_cube(volume: np.ndarray, value: float = 0) -> np.ndarray:
    """Equal pad so the original centre — the mid-sagittal plane on HCP — stays put."""
    return crop_or_pad(volume, int(max(volume.shape)), value=value)


def crop_or_pad(volume: np.ndarray, size: int, value: float = 0) -> np.ndarray:
    """Centre-crop or pad every axis to ``size``. Native voxels; no interpolation."""
    size = int(size)
    slices = []
    pads = []
    for length in volume.shape:
        length = int(length)
        if length >= size:
            start = (length - size) // 2
            slices.append(slice(start, start + size))
            pads.append((0, 0))
        else:
            extra = size - length
            slices.append(slice(None))
            pads.append((extra // 2, extra - extra // 2))
    cropped = volume[tuple(slices)]
    if any(pad != (0, 0) for pad in pads):
        return np.pad(cropped, pads, mode="constant", constant_values=value)
    return cropped


def resize(volume: np.ndarray, out_shape: Sequence[int], *, order: int) -> np.ndarray:
    """Isotropic zoom. ``order=0`` for labels, ``1`` for intensities."""
    out_shape = tuple(int(s) for s in out_shape)
    if volume.shape == out_shape:
        return volume
    factors = [out / inp for out, inp in zip(out_shape, volume.shape)]
    resampled = zoom(volume, factors, order=order, prefilter=order > 0)
    if order == 0:
        return np.rint(resampled).astype(volume.dtype, copy=False)
    return resampled.astype(np.float32, copy=False)


def prepare_volume(
    image_path: PathLike,
    labels_path: PathLike,
    *,
    resolution: int,
    brainmask_path: PathLike | None = None,
    apply_brainmask: bool = True,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """T1 + labels on disk -> RAS ``(z, y, x)`` cube and ``(x, y, z)`` spacing."""
    t1 = _canonical(Path(image_path))
    labels_img = _resample_to(_canonical(Path(labels_path)), t1, order=0)
    image = np.asanyarray(t1.dataobj).astype(np.float32, copy=False)
    labels = np.rint(np.asanyarray(labels_img.dataobj)).astype(np.int32, copy=False)
    if image.shape != labels.shape:
        raise ValueError(f"image {image.shape} and labels {labels.shape} differ after resampling")

    if apply_brainmask and brainmask_path is not None:
        mask_img = _resample_to(_canonical(Path(brainmask_path)), t1, order=0)
        brain = np.asanyarray(mask_img.dataobj) > 0
        if brain.shape == image.shape:
            image = image.copy()
            labels = labels.copy()
            image[~brain] = 0.0
            labels[~brain] = 0

    image = crop_or_pad(image, resolution)
    labels = crop_or_pad(labels, resolution)
    spacing = tuple(float(v) for v in t1.header.get_zooms()[:3])
    return (
        np.ascontiguousarray(image.transpose(2, 1, 0)),
        np.ascontiguousarray(labels.transpose(2, 1, 0)),
        spacing,
    )


def load_subject(
    source: PathLike,
    subject_id: str,
    *,
    resolution: int,
    image_names: Sequence[str] = IMAGE_CANDIDATES,
    label_names_: Sequence[str] = LABEL_CANDIDATES,
    brainmask_names: Sequence[str] = BRAINMASK_CANDIDATES,
    apply_brainmask: bool = True,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float], str]:
    """One subject -> ``(image, labels, spacing, scheme)``."""
    directory = Path(source) / subject_id
    image_path = find_volume(directory, image_names)
    labels_path = find_volume(directory, label_names_)
    if image_path is None or labels_path is None:
        raise FileNotFoundError(f"no T1/labels pair under {directory}")
    brainmask_path = find_volume(directory, brainmask_names)
    image, labels, spacing = prepare_volume(
        image_path,
        labels_path,
        resolution=resolution,
        brainmask_path=brainmask_path,
        apply_brainmask=apply_brainmask,
    )
    return image, labels, spacing, detect_scheme(labels_path)


def split_subjects(
    subject_ids: Sequence[str],
    fractions: Mapping[str, float],
    *,
    seed: int,
) -> dict[str, list[str]]:
    """Shuffle, then cut into ``train`` / ``val`` / ``test`` by fraction."""
    rng = np.random.default_rng(seed)
    order = list(subject_ids)
    rng.shuffle(order)
    n = len(order)
    n_train = int(round(n * float(fractions["train"])))
    n_val = int(round(n * float(fractions["val"])))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    return {
        "train": order[:n_train],
        "val": order[n_train : n_train + n_val],
        "test": order[n_train + n_val :],
    }


def remap_source_labels(
    labels: np.ndarray, names: Mapping[int, str]
) -> tuple[np.ndarray, Vocabulary]:
    """Drop unlisted ids and map the rest to ``vocabulary index + 1``."""
    vocab = Vocabulary(tuple(names.values()))
    remap = np.zeros(max(names) + 1, dtype=np.uint16)
    for source_id, name in names.items():
        remap[int(source_id)] = vocab.label(name)
    source = labels.astype(np.int64, copy=False)
    remapped = np.zeros(source.shape, dtype=np.uint16)
    valid = (source >= 0) & (source < len(remap))
    remapped[valid] = remap[source[valid]]
    return remapped, vocab


def subject_has_examples(
    labels: np.ndarray,
    vocab: Vocabulary,
    spacing: Sequence[float],
    n_anchors: int,
    *,
    pool: int | None = None,
) -> bool:
    """True when the remapped scene supports at least one relational example."""
    return bool(build_examples("probe", labels, vocab, spacing, n_anchors, pool=pool))


def download_subject(
    subject_id: str,
    dest: PathLike,
    *,
    uri: str,
    files: Sequence[str] = HCP_FILES,
    profile: str | None = None,
) -> Path:
    """Copy one HCP subject's T1 + wmparc + brainmask via ``aws s3 cp``."""
    dest_dir = Path(dest) / subject_id / "T1w"
    dest_dir.mkdir(parents=True, exist_ok=True)
    for filename in files:
        out = dest_dir / filename
        if out.is_file():
            continue
        src = f"{uri.rstrip('/')}/{subject_id}/T1w/{filename}"
        command = ["aws", "s3", "cp", src, str(out)]
        if profile:
            command.extend(["--profile", profile])
        try:
            subprocess.run(command, check=True)
        except FileNotFoundError as error:
            raise RuntimeError(
                "aws CLI not found; install it or point mri.source at a local HCP tree"
            ) from error
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"aws s3 cp failed for {src}. HCP openaccess needs a licensed AWS profile."
            ) from error
    return dest_dir.parent


def volume_cfg(cfg: Any) -> tuple[list[str], list[str], list[str]]:
    """Image / label / brainmask filename candidates from ``mri.volumes``."""
    volumes = cfg.volumes.to_dict() if hasattr(cfg, "volumes") else {}
    return (
        list(volumes.get("image") or IMAGE_CANDIDATES),
        list(volumes.get("labels") or LABEL_CANDIDATES),
        list(volumes.get("brainmask") or BRAINMASK_CANDIDATES),
    )
