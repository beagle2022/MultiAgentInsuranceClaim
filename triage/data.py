"""Input loading. Never raises on bad rows: returns what it could parse plus load errors."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class BatchLoadError(Exception):
    """The file itself is unusable (missing, not JSON, not a list)."""


@dataclass
class ClaimRow:
    index: int
    raw: dict[str, Any]

    @property
    def claim_id(self) -> str:
        return str(self.raw.get("claim_id") or f"ROW-{self.index}")


@dataclass
class Batch:
    source: str
    rows: list[ClaimRow] = field(default_factory=list)
    load_errors: list[dict[str, Any]] = field(default_factory=list)


def load_batch(path: str | Path) -> Batch:
    p = Path(path)
    if not p.exists():
        raise BatchLoadError(f"Batch file not found: {p}")
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BatchLoadError(f"Batch file is not valid JSON ({exc.msg} at line {exc.lineno})") from exc
    if isinstance(payload, dict) and isinstance(payload.get("claims"), list):
        payload = payload["claims"]
    if not isinstance(payload, list):
        raise BatchLoadError("Batch must be a JSON array of claim objects (or {\"claims\": [...]})")

    batch = Batch(source=str(p))
    for i, item in enumerate(payload):
        if isinstance(item, dict):
            batch.rows.append(ClaimRow(i, item))
        else:
            batch.load_errors.append({"row_index": i, "code": "MALFORMED_ROW",
                                      "message": f"Row {i} is {type(item).__name__}, expected an object"})
    return batch


def load_policies(path: Path) -> dict[str, dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    return {r["policy_number"]: r for r in records}
