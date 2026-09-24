"""Memory.

Short-term:  `Thread` - the message list for one agent within one session (Foundry: a thread).
             `SessionStore` persists the supervisor's thread + last triage results so a session
             can be resumed across process restarts (`--session <id>`).
Long-term:   `LongTermMemory` - JSON-file store of triage outcomes keyed by policy number that
             survives across runs/sessions. The Briefing agent reads it ("this policyholder had a
             similar water-damage claim flagged 6 months ago") and the routine writes to it at the
             end of each cycle. Writes are atomic (temp file + os.replace; safe on Windows).
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from .tracing import TRACER


def _atomic_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- short-term
@dataclass
class Thread:
    agent: str
    id: str = field(default_factory=lambda: "thr_" + uuid.uuid4().hex[:10])
    messages: list[dict[str, Any]] = field(default_factory=list)

    def trimmed(self, max_messages: int = 40) -> list[dict[str, Any]]:
        """Bound context size. Cut only at a plain-text user turn so tool_use/tool_result pairs stay intact."""
        if len(self.messages) <= max_messages:
            return self.messages
        for i in range(len(self.messages) - max_messages, len(self.messages)):
            m = self.messages[i]
            if m["role"] == "user" and isinstance(m["content"], str):
                return self.messages[i:]
        return self.messages[-2:]


@dataclass
class Session:
    id: str
    supervisor_thread: Thread
    last_run: dict[str, Any] | None = None     # latest TriageResult as dict
    turns: int = 0


class SessionStore:
    def __init__(self, directory: Path) -> None:
        self.dir = directory

    def load_or_create(self, session_id: str | None) -> Session:
        if session_id and (p := self.dir / f"{session_id}.json").exists():
            raw = json.loads(p.read_text(encoding="utf-8"))
            TRACER.event("memory.session_resumed", session=session_id, turns=raw["turns"])
            return Session(raw["id"], Thread(**raw["supervisor_thread"]), raw.get("last_run"), raw["turns"])
        sid = session_id or datetime.now().strftime("s%Y%m%d-%H%M%S")
        return Session(sid, Thread("supervisor"))

    def save(self, s: Session) -> None:
        _atomic_write(self.dir / f"{s.id}.json",
                      {"id": s.id, "supervisor_thread": asdict(s.supervisor_thread),
                       "last_run": s.last_run, "turns": s.turns})


# --------------------------------------------------------------------------- long-term
class LongTermMemory:
    def __init__(self, path: Path, seed_path: Path | None = None) -> None:
        self.path = path
        self._lock = threading.Lock()
        if not path.exists():
            events = []
            if seed_path and seed_path.exists():
                events = json.loads(seed_path.read_text(encoding="utf-8")).get("events", [])
                TRACER.event("memory.seeded", events=len(events), source=seed_path.name)
            _atomic_write(path, {"version": 1, "events": events})
        self._data = self._read()

    def _read(self) -> dict[str, Any]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            # Corrupt store must not take the system down: quarantine it and start fresh.
            bad = self.path.with_suffix(".corrupt.json")
            os.replace(self.path, bad)
            TRACER.event("memory.store_corrupt", error=str(exc), quarantined=str(bad))
            return {"version": 1, "events": []}

    def history(self, policy_number: str, *, claim_type: str | None = None,
                exclude_claim_id: str | None = None, reference_date: date | None = None,
                within_days: int | None = None, only_validated: bool = False) -> list[dict[str, Any]]:
        out = []
        for e in self._data["events"]:
            if e["policy_number"] != policy_number or e["claim_id"] == exclude_claim_id:
                continue
            if only_validated and not e.get("validated", True):
                continue
            if claim_type and e["claim_type"] != claim_type:
                continue
            ev = dict(e)
            if reference_date and not e.get("loss_date"):
                continue
            if reference_date:
                delta = (reference_date - date.fromisoformat(e["loss_date"])).days
                if delta <= 0 or (within_days is not None and delta > within_days):
                    continue  # only strictly earlier losses count as "history"
                ev["days_before_reference"] = delta
                ev["months_before_reference"] = round(delta / 30.44, 1)
            out.append(ev)
        return sorted(out, key=lambda e: e.get("loss_date") or "")

    def record(self, event: dict[str, Any]) -> None:
        """Upsert by claim_id (a re-triage replaces the earlier outcome but keeps first_seen)."""
        with self._lock:
            events = self._data["events"]
            prev = next((e for e in events if e["claim_id"] == event["claim_id"]), None)
            if prev:
                event.setdefault("first_triaged_at", prev.get("first_triaged_at", prev.get("triaged_at")))
                events.remove(prev)
            events.append(event)
            _atomic_write(self.path, self._data)
        TRACER.event("memory.write", claim=event["claim_id"], rec=event.get("recommendation"))

    def get(self, claim_id: str) -> dict[str, Any] | None:
        return next((e for e in self._data["events"] if e["claim_id"] == claim_id), None)
