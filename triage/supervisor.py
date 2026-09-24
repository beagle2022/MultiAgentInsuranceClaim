"""Supervisor: routes each handler message, runs the routine, answers follow-ups with context.

Routing is an LLM classification (forced structured tool call) whose *output* is dispatched by a
plain dict in code, so every decision is logged and the dispatch path is deterministic.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

from .agents import build_agents
from .backends import make_backend
from .config import Settings
from .data import load_policies
from .knowledge import KnowledgeBase, RuleEngine
from .memory import LongTermMemory, SessionStore, Thread
from .routine import AutoGate, GateHandler, TriageRoutine, run_agent_safely, summarise
from .runtime import Backend, RunResult, normalise_claim_id
from .skills import SkillContext
from .tracing import TRACER

_CLAIM_RE = re.compile(r"\b[Cc]-?\d{4}\b")
_POLICY_RE = re.compile(r"\bPOL-\d{4}\b", re.I)
_FILE_RE = re.compile(r"[\w.:/\\-]+\.json")


class Supervisor:
    def __init__(self, settings: Settings, backend: Backend | None = None, gate: GateHandler | None = None,
                 session_id: str | None = None) -> None:
        self.settings = settings
        self.backend = backend or make_backend(settings)
        kb = KnowledgeBase(settings.knowledge_dir)
        self.ctx = SkillContext(kb=kb, rules=RuleEngine(kb),
                                memory=LongTermMemory(settings.long_term_path, settings.seed_memory_path),
                                policies=load_policies(settings.coverage_path), as_of=settings.as_of)
        self.agents = build_agents(self.ctx, settings)
        self.sessions = SessionStore(settings.sessions_dir)
        self.session = self.sessions.load_or_create(session_id)
        self.ctx.last_run = self.session.last_run
        self.routine = TriageRoutine(self.backend, self.agents, self.ctx, gate or AutoGate(),
                                     settings.checkpoints_dir)
        self.dispatch: dict[str, Callable[[str, dict[str, Any]], str]] = {
            "triage_batch": self._triage, "explain_claim": self._answer,
            "policyholder_history": self._answer, "general_question": self._answer}
        TRACER.event("supervisor.ready", session=self.session.id, backend=self.backend.name,
                     agents=[a.name for a in vars(self.agents).values()])

    # ------------------------------------------------------------------ entry point
    def handle(self, message: str) -> str:
        TRACER.new_trace()
        self.session.turns += 1
        with TRACER.span("supervisor.turn", session=self.session.id, turn=self.session.turns) as s:
            route = self._route(message)
            s["intent"] = route["intent"]
            reply = self.dispatch[route["intent"]](message, route)
        self.sessions.save(self.session)
        return reply

    # ------------------------------------------------------------------ routing
    def _transcript(self, max_turns: int = 6) -> str:
        lines = []
        for m in self.session.supervisor_thread.messages:
            if m["role"] == "user" and isinstance(m["content"], str):
                lines.append(f"Handler: {m['content']}")
            elif m["role"] == "assistant":
                txt = " ".join(b["text"] for b in m["content"] if b.get("type") == "text")
                if txt:
                    lines.append(f"Assistant: {txt[:500]}")
        return "\n".join(lines[-2 * max_turns:]) or "(no prior turns)"

    def _route(self, message: str) -> dict[str, Any]:
        prompt = f"Conversation so far:\n{self._transcript()}\n\nLatest handler message: {message}"
        res = run_agent_safely(self.backend, self.agents.router, Thread("router"), prompt,
                               lambda: RunResult("router_fallback", "", self._regex_route(message)))
        route = res.output or self._regex_route(message)
        route["claim_ids"] = [normalise_claim_id(c) for c in route.get("claim_ids") or []]
        TRACER.event("supervisor.route", intent=route["intent"], claims=route["claim_ids"],
                     reason=route.get("reason"), degraded=res.degraded)
        return route

    @staticmethod
    def _regex_route(message: str) -> dict[str, Any]:
        claims = [normalise_claim_id(c) for c in _CLAIM_RE.findall(message)]
        pol = _POLICY_RE.search(message)
        file = _FILE_RE.search(message)
        low = message.lower()
        if re.search(r"\b(triage|process|run|batch)\b", low) and not claims:
            intent = "triage_batch"
        elif claims:
            intent = "explain_claim"
        elif pol or "history" in low:
            intent = "policyholder_history"
        else:
            intent = "general_question"
        return {"intent": intent, "claim_ids": claims, "policy_number": pol.group(0).upper() if pol else None,
                "batch_path": file.group(0) if file else None, "reason": "regex fallback router"}

    # ------------------------------------------------------------------ handlers
    def _triage(self, message: str, route: dict[str, Any]) -> str:
        path = route.get("batch_path") or str(self.settings.data_dir / "claims.json")
        if not Path(path).is_absolute() and not Path(path).exists():
            candidate = self.settings.data_dir / Path(path).name
            path = str(candidate) if candidate.exists() else path
        state = self.routine.run(path)
        self.session.last_run = state.to_dict()
        self.ctx.last_run = self.session.last_run
        reply = summarise(state)
        th = self.session.supervisor_thread
        th.messages.append({"role": "user", "content": message})
        th.messages.append({"role": "assistant", "content": [{"type": "text", "text": reply}]})
        return reply

    def _answer(self, message: str, route: dict[str, Any]) -> str:
        hint = ""
        if route.get("claim_ids"):
            hint = f"\n[router: claims {', '.join(route['claim_ids'])}]"
        elif route.get("policy_number"):
            hint = f"\n[router: policy {route['policy_number']}]"
        res = run_agent_safely(self.backend, self.agents.supervisor, self.session.supervisor_thread,
                               message + hint, lambda: RunResult("supervisor", self._template_answer(route), None))
        if res.degraded:  # keep the thread coherent for later turns
            th = self.session.supervisor_thread
            th.messages.append({"role": "user", "content": message})
            th.messages.append({"role": "assistant", "content": [{"type": "text", "text": res.text}]})
        return res.text

    def _template_answer(self, route: dict[str, Any]) -> str:
        if route["intent"] == "explain_claim" and route["claim_ids"]:
            out = []
            for cid in route["claim_ids"]:
                try:
                    r = self.ctx.triage_record(cid)
                except KeyError as exc:
                    out.append(str(exc))
                    continue
                if r["source"] == "long_term_memory":
                    out.append(f"{cid} (from an earlier session, {r.get('triaged_at')}): {r['recommendation']}. "
                               f"Flags: {', '.join(r.get('flags', [])) or 'none'}. {r.get('summary', '')}")
                    continue
                lines = [f"{cid}: final action {r.get('final_action')}."]
                if not r["validation"]["valid"]:
                    lines += [f"  - {e['code']}: {e['message']}" for e in r["validation"]["errors"]]
                for tr in r.get("assessment", {}).get("triggered_rules", []):
                    chunk = self.ctx.kb.get(tr["rule_id"])
                    lines.append(f"  - {chunk.title if chunk else tr['rule_id']}: {tr['evidence']}")
                if r.get("human_override"):
                    ho = r["human_override"]
                    lines.append(f"  - Human override {ho['from']} -> {ho['to']}: {ho['note']}")
                b = r.get("briefing") or {}
                if b.get("history_note"):
                    lines.append(f"  History: {b['history_note']}")
                out.append("\n".join(lines))
            return "\n\n".join(out)
        if route["intent"] == "policyholder_history" and route.get("policy_number"):
            ev = self.ctx.memory.history(route["policy_number"])
            if not ev:
                return f"No history for {route['policy_number']}."
            return "\n".join(f"- {e['claim_id']} {e['claim_type']} loss {e.get('loss_date')}: {e['recommendation']} "
                             f"({e.get('triaged_at')})" for e in ev)
        return ("(LLM unavailable - limited mode.) I can: triage a batch ('triage data/claims.json'), explain a "
                "claim ('why was C-2031 flagged?'), or show a policy's history ('history for POL-5521').")
