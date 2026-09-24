"""LLM backends: Claude (Anthropic Messages API w/ tool use) and Offline (forces fallbacks)."""
from __future__ import annotations

import concurrent.futures as cf
import os
from typing import Any

from .config import Settings
from .memory import Thread
from .runtime import AgentError, AgentSpec, RunResult, to_json
from .tracing import TRACER

TOOL_TIMEOUT_S = float(os.getenv("TRIAGE_TOOL_TIMEOUT", "15"))
_POOL = cf.ThreadPoolExecutor(max_workers=4, thread_name_prefix="tool")


def _append_user(thread: Thread, content: str | list[dict[str, Any]]) -> None:
    """Keep strict user/assistant alternation (a run can end on a tool_result user turn)."""
    blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content
    if thread.messages and thread.messages[-1]["role"] == "user":
        prev = thread.messages[-1]["content"]
        prev_blocks = [{"type": "text", "text": prev}] if isinstance(prev, str) else prev
        thread.messages[-1]["content"] = prev_blocks + blocks
    else:
        thread.messages.append({"role": "user", "content": content})


def _block(b: Any) -> dict[str, Any]:
    if b.type == "text":
        return {"type": "text", "text": b.text}
    if b.type == "tool_use":
        return {"type": "tool_use", "id": b.id, "name": b.name, "input": b.input}
    return b.model_dump(exclude_none=True)  # future block types pass through


def execute_tool(agent: AgentSpec, name: str, args: dict[str, Any]) -> tuple[str, bool]:
    """Shared by all backends: schema check, timeout, and errors returned to the model (not raised)."""
    tool = agent.tool(name)
    with TRACER.span("tool.call", agent=agent.name, tool=name, args=args) as s:
        try:
            if tool is None or tool.fn is None:
                raise LookupError(f"Tool '{name}' is not registered to agent '{agent.name}'")
            missing = [k for k in tool.input_schema.get("required", []) if k not in args]
            if missing:
                raise ValueError(f"Missing required argument(s): {missing}")
            payload = to_json(_POOL.submit(tool.fn, **args).result(timeout=TOOL_TIMEOUT_S))
            s["result_chars"] = len(payload)
            return payload, False
        except cf.TimeoutError:
            err = f"Tool timed out after {TOOL_TIMEOUT_S}s"
        except Exception as exc:  # surfaced to the model, which can retry or adapt
            err = f"{type(exc).__name__}: {exc}"
        s["tool_error"] = err
        return err, True


class ClaudeBackend:
    name = "claude"

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self.settings = settings
        if client is None:
            import anthropic  # imported lazily so offline mode needs no SDK / key

            client = anthropic.Anthropic(timeout=settings.llm_timeout_s, max_retries=settings.llm_max_retries)
        self.client = client

    def _execute(self, agent: AgentSpec, call: dict[str, Any]) -> dict[str, Any]:
        content, is_error = execute_tool(agent, call["name"], call["input"])
        block = {"type": "tool_result", "tool_use_id": call["id"], "content": content}
        return block | {"is_error": True} if is_error else block

    # ------------------------------------------------------------------ agent loop
    def run(self, agent: AgentSpec, thread: Thread, user_input: str) -> RunResult:
        _append_user(thread, user_input)
        calls: list[dict[str, Any]] = []
        output: dict[str, Any] | None = None
        forced = {"type": "tool", "name": agent.output_tool}
        tool_choice: dict[str, Any] = forced if agent.force_output_tool else {"type": "auto"}
        nudged = False
        for rnd in range(1, self.settings.max_tool_rounds + 1):
            with TRACER.span("llm.call", agent=agent.name, model=agent.model, round=rnd) as s:
                try:
                    resp = self.client.messages.create(
                        model=agent.model, max_tokens=agent.max_tokens, system=agent.instructions,
                        tools=[t.spec() for t in agent.tools], tool_choice=tool_choice,
                        messages=thread.trimmed())
                except Exception as exc:  # SDK already retried transient errors w/ backoff
                    raise AgentError(f"{agent.name}: LLM call failed: {type(exc).__name__}: {exc}") from exc
                s.update(stop=resp.stop_reason, in_tok=resp.usage.input_tokens, out_tok=resp.usage.output_tokens)
            content = [_block(b) for b in resp.content]
            thread.messages.append({"role": "assistant", "content": content})
            text = "\n".join(b["text"] for b in content if b["type"] == "text").strip()
            uses = [b for b in content if b["type"] == "tool_use"]

            if not uses:
                if agent.output_tool and output is None and not nudged:
                    nudged = True  # model answered in prose; force the structured output tool once
                    _append_user(thread, f"Now submit your final result with the {agent.output_tool} tool.")
                    tool_choice = {"type": "tool", "name": agent.output_tool}
                    TRACER.event("agent.nudge_output", agent=agent.name)
                    continue
                return RunResult(agent.name, text, output, calls, rnd)

            results = []
            for u in uses:
                calls.append({"tool": u["name"], "input": u["input"]})
                if u["name"] == agent.output_tool:
                    output = u["input"]
                    results.append({"type": "tool_result", "tool_use_id": u["id"], "content": "Recorded."})
                else:
                    results.append(self._execute(agent, u))
            _append_user(thread, results)
            tool_choice = {"type": "auto"}
            if output is not None:
                return RunResult(agent.name, text, output, calls, rnd)
        raise AgentError(f"{agent.name}: exceeded {self.settings.max_tool_rounds} tool rounds")


class OfflineBackend:
    """No LLM. Every run raises, so the routine exercises its deterministic fallbacks end to end."""
    name = "offline"

    def run(self, agent: AgentSpec, thread: Thread, user_input: str) -> RunResult:
        raise AgentError("offline backend: LLM disabled")


def make_backend(settings: Settings):
    if settings.backend == "offline":
        return OfflineBackend()
    if settings.backend == "foundry":
        from .foundry_backend import FoundryBackend
        return FoundryBackend(settings)
    if not os.getenv("ANTHROPIC_API_KEY"):
        TRACER.event("backend.no_api_key", fallback="offline")
        return OfflineBackend()
    return ClaudeBackend(settings)


