"""Both networks still compute what ``exp/realistic-appearance`` computed.

The rewrite changed how the models are *built* - the depth follows from the
length of ``model.encoder_channels``, the tables from the vocabulary - but not
what they compute. This test proves it the only way worth trusting: it checks the
reference branch out, builds its models there in a subprocess, ports the weights
into these classes by name, and compares every output tensor.

It is what makes a run started after the rewrite comparable with one started
before. If it fails, a checkpoint from that branch no longer means the same
thing, and the change that broke it needs to be deliberate.

The test skips when the branch is absent (a shallow clone, a fork), so it never
blocks a checkout that simply does not have the history.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import pytest
import torch

from src.models import StageA, StageB

#: ``(pattern, replacement)`` pairs applied in order to a Stage A parameter name.
STAGE_A_RENAMES: tuple[tuple[str, str], ...] = (
    (r"^stem\.", "encoder.stages.0."),
    (r"^stage1\.", "encoder.stages.1."),
    (r"^stage2\.", "encoder.stages.2."),
    (r"^bottleneck\.", "encoder.stages.3."),
    (r"^prompt_encoder\.name_embedding\.", "prompt.table."),
    (r"^prompt_encoder\.projection\.", "prompt.projection."),
    (r"^prompt_decoder\.positional_encoding\.", "pos."),
    (r"^prompt_decoder\.attention\.", "attention."),
    (r"^prompt_decoder\.norm\.", "norm."),
    (r"^decode(\d)\.fuse\.conv\.weight$", lambda m: f"decoder.fuse.{int(m.group(1)) - 1}.0.weight"),
    (r"^fusion(\d)\.bias_head\.", lambda m: f"heads.{int(m.group(1)) - 1}.bias."),
    (r"^fusion(\d)\.", lambda m: f"heads.{int(m.group(1)) - 1}."),
    # Encoder stage internals, once the stage prefix above has been rewritten.
    (r"(encoder\.stages\.\d)\.transition\.conv\.weight$", r"\1.0.0.weight"),
    (r"(encoder\.stages\.\d)\.blocks\.0\.conv1\.weight$", r"\1.1.body.0.0.weight"),
    (r"(encoder\.stages\.\d)\.blocks\.0\.conv2\.weight$", r"\1.1.body.1.weight"),
)

STAGE_B_RENAMES: tuple[tuple[str, str], ...] = (
    (r"^encoder\.stem\.", "encoder.stages.0."),
    (r"^encoder\.stage1\.", "encoder.stages.1."),
    (r"^encoder\.stage2\.", "encoder.stages.2."),
    (r"^encoder\.bottleneck\.", "encoder.stages.3."),
    (r"^prompt_encoder\.direction_embedding\.", "prompt.direction."),
    (r"^prompt_encoder\.shape_encoder\.name_embedding\.", "prompt.name.table."),
    (r"^prompt_encoder\.shape_encoder\.projection\.", "prompt.name.projection."),
    (r"^prompt_encoder\.pair_embedding\.", "prompt.pair."),
    (r"^prompt_encoder\.slot_embedding\.", "prompt.slot."),
    (r"^prompt_encoder\.norm\.", "prompt.norm."),
    (r"^structure_encoder\.feature_projection\.", "structure.visual."),
    (r"^structure_encoder\.geometry_projection\.", "structure.geometry."),
    (r"^structure_encoder\.shape_embedding\.", "structure.name."),
    (r"^structure_encoder\.slot_embedding\.", "structure.slot."),
    (r"^structure_encoder\.norm\.", "structure.norm."),
    (r"^fusion\.clause_fusion\.0\.attention\.", "evidence.clause_attention."),
    (r"^fusion\.clause_fusion\.0\.attention_norm\.", "evidence.clause_norm."),
    (r"^fusion\.clause_fusion\.0\.mlp\.", "evidence.mlp."),
    (r"^fusion\.clause_fusion\.0\.norm\.", "evidence.norm."),
    (r"^fusion\.evidence_heads\.0\.query_projection\.", "evidence.to_query."),
    (r"^fusion\.evidence_heads\.0\.positional_encoding\.", "evidence.pos."),
    (r"^fusion\.evidence_heads\.0\.", "evidence."),
    (r"^decoder\.up(\d)\.fuse\.conv\.weight$", lambda m: f"decoder.fuse.{int(m.group(1)) - 1}.0.weight"),
    (r"^decoder\.occ_proj\.", "decoder.occupancy."),
    (r"^decoder\.head\.", "head."),
    (r"(encoder\.stages\.\d)\.transition\.conv\.weight$", r"\1.0.0.weight"),
    (r"(encoder\.stages\.\d)\.blocks\.0\.conv1\.weight$", r"\1.1.body.0.0.weight"),
    (r"(encoder\.stages\.\d)\.blocks\.0\.conv2\.weight$", r"\1.1.body.1.weight"),
)


def remap_state_dict(state: Mapping[str, Any], stage: str) -> dict[str, Any]:
    """Translate a pre-rewrite checkpoint's parameter names to the current ones.

    Args:
        state: the checkpoint's ``model_state_dict``.
        stage: ``"a"`` (``ShapeSegmenter``) or ``"b"`` (``RelationalVLM``).

    The architecture is unchanged, so this is a pure rename - every tensor keeps
    its shape and its values, and the result loads with ``strict=True``.
    """
    rules = {"a": STAGE_A_RENAMES, "b": STAGE_B_RENAMES}[stage]
    remapped = {}
    for key, value in state.items():
        for pattern, replacement in rules:
            key = re.sub(pattern, replacement, key)
        remapped[key] = value
    return remapped


def remap_state_dict(state: Mapping[str, Any], stage: str) -> dict[str, Any]:
    """Translate a reference checkpoint's parameter names to the current ones.

    The architecture is unchanged, so this is a pure rename - every tensor keeps
    its shape and its values, and the result loads with ``strict=True``.
    """
    rules = {"a": STAGE_A_RENAMES, "b": STAGE_B_RENAMES}[stage]
    remapped = {}
    for key, value in state.items():
        for pattern, replacement in rules:
            key = re.sub(pattern, replacement, key)
        remapped[key] = value
    return remapped


REFERENCE = "exp/realistic-appearance"
ROOT = Path(__file__).resolve().parents[1]
#: The reference's narrow "smoke" widths at its own 64^3 resolution. The widths
#: are irrelevant to the comparison and keep the dump small; the
#: resolution is not - the reference keys its FiLM stages to the literal sizes
#: 16 and 32, which are "every decoder stage but the finest" only at 64^3.
RESOLUTION, BOTTLENECK, DIM, HEADS = 64, 8, 64, 2
WIDTHS = dict(encoder_channels=(8, 16, 32, 64), token_dim=DIM, num_heads=HEADS, bottleneck=BOTTLENECK)

DUMP = '''
import sys, torch
sys.path.insert(0, ".")
from src.models.shape_segmenter import build_shape_segmenter
from src.models.relational_vlm import build_relational_vlm

torch.manual_seed(0)
a = build_shape_segmenter("smoke", input_resolution=%(res)d).eval()
b = build_relational_vlm("smoke", input_resolution=%(res)d).eval()
# Both mask heads are zero-initialised by design (they start at the foreground
# prior), so an untrained model emits a constant and would hide any upstream
# error. Give them a real weight, so the comparison exercises the whole network.
gen = torch.Generator().manual_seed(11)
for head in (a.fusion1, a.fusion2, a.fusion3):
    head.bias_head.weight.data = torch.randn(head.bias_head.weight.shape, generator=gen) * 0.05
b.decoder.head.weight.data = torch.randn(b.decoder.head.weight.shape, generator=gen) * 0.05

g = torch.Generator().manual_seed(7)
res = %(res)d
image = torch.randn(1, 1, res, res, res, generator=g)
names = torch.tensor([[0, 3, 7]])
anchors = (torch.rand(1, 3, res, res, res, generator=g) > 0.97).float()
directions = torch.tensor([[0, 2, 4]])
occupancy = (torch.rand(1, 1, res, res, res, generator=g) > 0.9).float()
with torch.no_grad():
    out_a = a(image, names, deep_supervision=True)
    out_b = b(anchors, directions, names, occupancy, return_evidence=True)
torch.save({"a_state": a.state_dict(), "b_state": b.state_dict(),
            "inputs": {"image": image, "names": names, "anchors": anchors,
                       "directions": directions, "occupancy": occupancy},
            "a_logits": out_a.logits, "a_scales": out_a.deep_supervision,
            "b_logits": out_b.logits, "b_evidence": out_b.evidence}, sys.argv[1])
'''


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    """The reference branch's own models, built by its own code."""
    if _git("rev-parse", "--verify", f"{REFERENCE}^{{commit}}").returncode != 0:
        pytest.skip(f"branch {REFERENCE} is not in this checkout")
    tree = tmp_path_factory.mktemp("reference")
    archive = subprocess.run(
        ["git", "archive", REFERENCE], cwd=ROOT, capture_output=True, check=True
    )
    subprocess.run(["tar", "-x", "-C", str(tree)], input=archive.stdout, check=True)

    script = tree / "_dump.py"
    script.write_text(DUMP % {"res": RESOLUTION}, encoding="utf-8")
    dump = tmp_path_factory.mktemp("dump") / "reference.pt"
    result = subprocess.run(
        [sys.executable, str(script), str(dump)], cwd=tree, capture_output=True, text=True
    )
    if result.returncode != 0:
        pytest.skip(f"could not build the reference models:\n{result.stderr[-2000:]}")
    return torch.load(dump, map_location="cpu", weights_only=False)


@pytest.mark.parametrize("stage", ["a", "b"])
def test_the_weights_port_over_by_name_alone(reference, stage):
    """A pure rename: every tensor keeps its shape, and nothing is left over."""
    model = StageA(10, RESOLUTION, **WIDTHS) if stage == "a" else StageB(10, RESOLUTION, 3, **WIDTHS)
    remapped = remap_state_dict(reference[f"{stage}_state"], stage)
    assert set(remapped) == set(model.state_dict())
    model.load_state_dict(remapped, strict=True)


def test_stage_a_reproduces_the_reference_exactly(reference):
    model = StageA(10, RESOLUTION, **WIDTHS).eval()
    model.load_state_dict(remap_state_dict(reference["a_state"], "a"), strict=True)
    with torch.no_grad():
        output = model(reference["inputs"]["image"], reference["inputs"]["names"])
    assert torch.equal(output.logits, reference["a_logits"])
    # ... including the two coarse deep-supervision maps, which are what the
    # loss actually sees at 1/4 and 1/2 resolution.
    assert len(output.scales) == len(reference["a_scales"])
    for got, want in zip(output.scales, reference["a_scales"]):
        assert torch.equal(got, want)


def test_stage_b_reproduces_the_reference_exactly(reference):
    model = StageB(10, RESOLUTION, 3, **WIDTHS).eval()
    model.load_state_dict(remap_state_dict(reference["b_state"], "b"), strict=True)
    inputs = reference["inputs"]
    with torch.no_grad():
        output = model(inputs["anchors"], inputs["directions"], inputs["names"], inputs["occupancy"])
    assert torch.equal(output.logits, reference["b_logits"])
    # The per-clause evidence maps too: the head alone could agree by accident,
    # but the grounding branches could not.
    for got, want in zip(output.evidence, reference["b_evidence"]):
        assert torch.equal(got, want)


def test_the_reference_output_is_not_trivially_constant(reference):
    """Guards the two tests above: a constant volume would agree with anything."""
    for key in ("a_logits", "b_logits"):
        values = reference[key]
        assert float(values.max() - values.min()) > 0.1, key
