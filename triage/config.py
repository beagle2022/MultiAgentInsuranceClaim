"""Central configuration. Everything overridable via environment variables / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

try:  # optional convenience
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    # "claude" | "offline" | "foundry"
    backend: str = field(default_factory=lambda: os.getenv("TRIAGE_BACKEND", "claude"))
    specialist_model: str = field(default_factory=lambda: os.getenv("TRIAGE_MODEL", "claude-sonnet-5"))
    router_model: str = field(default_factory=lambda: os.getenv("TRIAGE_ROUTER_MODEL", "claude-haiku-4-5-20251001"))
    llm_timeout_s: float = field(default_factory=lambda: float(os.getenv("TRIAGE_LLM_TIMEOUT", "60")))
    llm_max_retries: int = field(default_factory=lambda: int(os.getenv("TRIAGE_LLM_RETRIES", "3")))
    max_tool_rounds: int = field(default_factory=lambda: int(os.getenv("TRIAGE_MAX_TOOL_ROUNDS", "10")))

    data_dir: Path = ROOT / "data"
    knowledge_dir: Path = ROOT / "knowledge"
    store_dir: Path = field(default_factory=lambda: Path(os.getenv("TRIAGE_STORE_DIR", str(ROOT / "memory_store"))))
    log_dir: Path = ROOT / "logs"

    # "Today" is pinned so the synthetic data behaves the same on every run / machine.
    as_of: date = field(default_factory=lambda: date.fromisoformat(os.getenv("TRIAGE_AS_OF", "2026-09-23")))

    @property
    def coverage_path(self) -> Path:
        return self.data_dir / "policy_coverage.json"

    @property
    def seed_memory_path(self) -> Path:
        return self.data_dir / "seed_memory.json"

    @property
    def long_term_path(self) -> Path:
        return self.store_dir / "long_term_memory.json"

    @property
    def sessions_dir(self) -> Path:
        return self.store_dir / "sessions"

    @property
    def checkpoints_dir(self) -> Path:
        return self.store_dir / "checkpoints"


SETTINGS = Settings()
