"""Backend-agnostic agent runtime primitives.

The shapes mirror Azure AI Foundry Agent Service so the Claude backend can be swapped for
`foundry_backend.FoundryBackend` without touching the routine or agent definitions:

    AgentSpec  ~ agents_client.create_agent(model, name, instructions, toolset)
    Tool       ~ FunctionTool entry (name + JSON schema + python callable)
    Thread     ~ agents_client.threads.create()      (see memory.py)
    run()      ~ messages.create + runs.create_and_process (auto function calling)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .memory import Thread


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    fn: Callable[..., Any] | None = None  # None => "output tool": its input IS the agent's structured result

    def spec(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}


@dataclass
class AgentSpec:
    name: str
    instructions: str
    tools: list[Tool]
    model: str
    output_tool: str | None = None
    force_output_tool: bool = False   # e.g. the router: must answer only via its structured tool
    max_tokens: int = 4096

    def tool(self, name: str) -> Tool | None:
        return next((t for t in self.tools if t.name == name), None)


@dataclass
class RunResult:
    agent: str
    text: str
    output: dict[str, Any] | None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    rounds: int = 0
    degraded: bool = False  # True when produced by the deterministic fallback, not the LLM


class AgentError(Exception):
    """An agent run could not complete (API failure after retries, timeout, loop limit, offline)."""


class Backend(Protocol):
    name: str

    def run(self, agent: AgentSpec, thread: Thread, user_input: str) -> RunResult: ...


def normalise_claim_id(raw: str) -> str:
    m = re.fullmatch(r"\s*[Cc]-?(\d+)\s*", raw or "")
    return f"C-{m.group(1)}" if m else (raw or "").strip()


def to_json(obj: Any) -> str:
    return json.dumps(obj, default=str, indent=None)
