"""The structure vocabulary and the prompt language built on top of it.

A corpus owns its vocabulary: ``<root>/vocab.json`` is an ordered list of
structure names, and the label volume stores ``index + 1`` for the structure at
``index`` (0 is background). Ten synthetic primitives and eighty anatomical
labels are the same thing to every other module - only the length of this list
changes, and the embedding tables are sized from it.

The prompt is a deterministic rendering of the ordered clause list::

    [{"direction": d1, "anchor": a1}, ..., {"direction": dN, "anchor": aN}]

    segment the structure that is d1 to the a1, d2 to the a2, and d3 to the a3.

``parse(render(clauses)) == clauses`` for every valid clause list, and the
parser is built from the closed vocabularies, so an unknown name, a synonym or
a target name cannot appear in a prompt.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from src.geometry import DIRECTIONS

PROMPT_PREFIX = "segment the structure that is "


@dataclass(frozen=True)
class Vocabulary:
    """Ordered structure names. ``label id == index + 1``; 0 is background."""

    names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.names:
            raise ValueError("the vocabulary must name at least one structure")
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"structure names must be unique, got {self.names}")
        # ", " separates clauses in a rendered prompt, so a name may not contain it.
        bad = [name for name in self.names if ", " in name or not name.strip()]
        if bad:
            raise ValueError(f"structure names must be non-empty and free of ', ', got {bad}")

    def __len__(self) -> int:
        return len(self.names)

    def __contains__(self, name: object) -> bool:
        return name in self.names

    def index(self, name: str) -> int:
        """Zero-based index, which is what the embedding tables are indexed by."""
        try:
            return self.names.index(name)
        except ValueError as error:
            raise KeyError(f"unknown structure {name!r}; vocabulary is {self.names}") from error

    def label(self, name: str) -> int:
        """Integer label of a structure in the label volume."""
        return self.index(name) + 1

    def name(self, label: int) -> str:
        """Inverse of :meth:`label`."""
        if not 1 <= int(label) <= len(self.names):
            raise KeyError(f"label {label} is outside 1..{len(self.names)}")
        return self.names[int(label) - 1]

    def require(self, names: Sequence[str]) -> tuple[str, ...]:
        """Validate a sequence of names, returning it unchanged."""
        unknown = [name for name in names if name not in self.names]
        if unknown:
            raise KeyError(f"unknown structure(s) {unknown}; vocabulary is {self.names}")
        return tuple(names)

    # -- persistence -------------------------------------------------------
    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(list(self.names), indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | str) -> "Vocabulary":
        return cls(tuple(json.loads(Path(path).read_text(encoding="utf-8"))))

    # -- prompt language ---------------------------------------------------
    def render(self, clauses: Sequence[dict[str, str]]) -> str:
        """Clause list -> the canonical natural-language prompt."""
        self.validate(clauses)
        parts = [f"{clause['direction']} to the {clause['anchor']}" for clause in clauses]
        if len(parts) > 1:
            parts[-1] = "and " + parts[-1]
        return PROMPT_PREFIX + ", ".join(parts) + "."

    def parse(self, prompt: str) -> list[dict[str, str]]:
        """The canonical prompt -> its clause list. Inverse of :meth:`render`."""
        names = "|".join(sorted(map(re.escape, self.names), key=len, reverse=True))
        directions = "|".join(sorted(DIRECTIONS, key=len, reverse=True))
        body = prompt.strip()
        if not body.startswith(PROMPT_PREFIX) or not body.endswith("."):
            raise ValueError(f"prompt does not match the canonical template: {prompt!r}")
        pattern = re.compile(rf"^(?P<direction>{directions}) to the (?P<anchor>{names})$")
        parts = body[len(PROMPT_PREFIX) : -1].split(", ")
        clauses = []
        for index, part in enumerate(parts):
            last = index == len(parts) - 1
            if part.startswith("and ") != (last and len(parts) > 1):
                raise ValueError(f"misplaced 'and' in clause {part!r} of {prompt!r}")
            match = pattern.match(part.removeprefix("and "))
            if match is None:
                raise ValueError(f"unparseable clause {part!r} in prompt {prompt!r}")
            clauses.append(match.groupdict())
        self.validate(clauses)
        return clauses

    def validate(self, clauses: Sequence[dict[str, str]]) -> None:
        """Closed vocabularies, distinct directions, distinct anchors."""
        for clause in clauses:
            if set(clause) != {"direction", "anchor"}:
                raise ValueError(f"a clause has keys {{direction, anchor}}, got {sorted(clause)}")
            if clause["direction"] not in DIRECTIONS:
                raise ValueError(f"unknown direction {clause['direction']!r}")
        self.require([clause["anchor"] for clause in clauses])
        for field in ("direction", "anchor"):
            values = [clause[field] for clause in clauses]
            if len(set(values)) != len(values):
                raise ValueError(f"clause {field}s must be pairwise distinct, got {values}")

    def clause_ids(self, clauses: Sequence[dict[str, str]]) -> tuple[list[int], list[int]]:
        """Clauses -> ``(direction_ids, anchor_name_ids)``, the model's only text input.

        This is the single place the language side of the project becomes
        integers, so a prompt cannot mean one thing on disk and another in the
        network.
        """
        self.validate(clauses)
        return (
            [DIRECTIONS.index(clause["direction"]) for clause in clauses],
            [self.index(clause["anchor"]) for clause in clauses],
        )

    def clauses_from_ids(
        self, direction_ids: Sequence[int], anchor_name_ids: Sequence[int]
    ) -> list[dict[str, str]]:
        """Inverse of :meth:`clause_ids`, for logging and counterfactuals."""
        return [
            {"direction": DIRECTIONS[int(d)], "anchor": self.names[int(a)]}
            for d, a in zip(direction_ids, anchor_name_ids)
        ]
