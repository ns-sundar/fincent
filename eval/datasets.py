"""Load and validate Fincent intent eval datasets."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from src.core.schemas import Intent


DATASET_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class EvalCase:
    """One row from an intent eval dataset."""

    case_id: str
    dataset: str
    input: str
    expected_output: str
    retrieval_context: list[str]
    intent: Intent
    source: str
    handled_by_central: bool
    expected_agents: list[str]

    @property
    def display_name(self) -> str:
        """Human-readable case name for reports."""

        return f"{self.intent.value}: {self.input}"

    @property
    def pytest_id(self) -> str:
        """Stable, readable pytest/deepeval id."""

        slug = re.sub(r"[^a-zA-Z0-9]+", "_", self.input.lower()).strip("_")
        return f"{self.intent.value}__{slug[:80]}"


def _require_str(payload: dict[str, Any], key: str, *, context: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: expected non-empty string field '{key}'")
    return value


def _require_bool(payload: dict[str, Any], key: str, *, context: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{context}: expected boolean field '{key}'")
    return value


def _require_str_list(payload: dict[str, Any], key: str, *, context: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{context}: expected list[str] field '{key}'")
    return value


def _case_from_row(path: Path, index: int, row: Any) -> EvalCase:
    context = f"{path.name}[{index}]"
    if not isinstance(row, dict):
        raise ValueError(f"{context}: expected object")

    metadata = row.get("additional_metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"{context}: expected object field 'additional_metadata'")

    raw_intent = _require_str(metadata, "intent", context=context)
    try:
        intent = Intent(raw_intent)
    except ValueError as exc:
        raise ValueError(f"{context}: unknown intent '{raw_intent}'") from exc

    return EvalCase(
        case_id=f"{path.stem}:{index}",
        dataset=path.name,
        input=_require_str(row, "input", context=context),
        expected_output=_require_str(row, "expected_output", context=context),
        retrieval_context=_require_str_list(row, "retrieval_context", context=context),
        intent=intent,
        source=_require_str(metadata, "source", context=context),
        handled_by_central=_require_bool(metadata, "handled_by_central", context=context),
        expected_agents=_require_str_list(metadata, "expected_agents", context=context),
    )


def load_eval_cases(dataset_dir: Path = DATASET_DIR) -> list[EvalCase]:
    """Load every ``eval.*.json`` row from ``dataset_dir``."""

    cases: list[EvalCase] = []
    paths = sorted(dataset_dir.glob("eval.*.json"))
    if not paths:
        raise ValueError(f"No eval datasets found in {dataset_dir}")

    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"{path.name}: expected top-level JSON array")
        cases.extend(_case_from_row(path, index, row) for index, row in enumerate(payload, start=1))
    return cases


def case_ids(cases: Iterable[EvalCase]) -> list[str]:
    """Return stable pytest ids for eval cases."""

    return [case.pytest_id for case in cases]
