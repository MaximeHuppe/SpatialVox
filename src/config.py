"""The one config file, loaded into a dotted-access mapping.

``configs/config.yaml`` holds every tunable in the project. Nothing in ``src/``
hard-codes a value that lives there. Scripts may override any leaf from the
command line with ``--set a.b.c=value``.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"


class Config(Mapping[str, Any]):
    """A read-only nested mapping with attribute access.

    ``cfg.model.stage_b.carver.width`` and ``cfg["model"]["stage_b"]`` are the
    same thing; nested dicts are wrapped on the way out.
    """

    def __init__(self, data: Mapping[str, Any]) -> None:
        self._data = dict(data)

    def __getitem__(self, key: str) -> Any:
        value = self._data[key]
        return Config(value) if isinstance(value, Mapping) else value

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as error:  # pragma: no cover - attribute protocol
            raise AttributeError(key) from error

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Config({self._data!r})"

    def to_dict(self) -> dict[str, Any]:
        """A plain, JSON-serialisable copy (for checkpoints and run metadata)."""
        return {
            key: Config(value).to_dict() if isinstance(value, Mapping) else value
            for key, value in self._data.items()
        }


def _set_path(data: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    for key in keys[:-1]:
        data = data.setdefault(key, {})
    data[keys[-1]] = value


def load_config(path: Path | str | None = None, overrides: Mapping[str, Any] | None = None) -> Config:
    """Read ``configs/config.yaml`` and apply ``{"a.b": value}`` overrides."""
    data = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text(encoding="utf-8"))
    for dotted, value in (overrides or {}).items():
        _set_path(data, dotted, value)
    return Config(data)


def parse_overrides(assignments: list[str] | None) -> dict[str, Any]:
    """Turn ``["train.epochs=5", "data.root=/tmp/x"]`` into a mapping.

    Values are parsed as Python literals when possible (so ``5`` is an int and
    ``[1,2]`` a list) and kept as strings otherwise.
    """
    parsed: dict[str, Any] = {}
    for assignment in assignments or []:
        if "=" not in assignment:
            raise ValueError(f"--set expects key=value, got {assignment!r}")
        key, raw = assignment.split("=", 1)
        try:
            parsed[key.strip()] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            parsed[key.strip()] = raw
    return parsed
