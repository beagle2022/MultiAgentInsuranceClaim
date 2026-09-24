"""The triage routine: an explicit, inspectable state machine.

    LOAD -> VALIDATE -> [GATE: validation] -> ASSESS -> GUARDRAIL -> [GATE: decision] -> BRIEF -> PERSIST -> DONE
                              |  \\-> BRIEF (no valid claims)            |
                              \\-> ABORTED                              \\-> ABORTED
    LOAD -> FAILED (unusable input)

Control flow is code, not prompt. LLM agents do the work *inside* a step; the routine decides what
step comes next, enforces allowed transitions, checkpoints state to disk after every step, and falls
back to deterministic logic when an agent fails (the step is then marked `degraded`).
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol

from .agents import AgentSet
from .data import BatchLoadError, load_batch
from .knowledge import ACTION_SEVERITY, RuleEngine
from .memory import Thread, _atomic_write
from .runtime import AgentError, AgentSpec, Backend, RunResult, to_json
from .skills import SkillContext
from .tracing import TRACER


class Step(str, Enum):
    LOAD = "load"
    VALIDATE = "validate"
    GATE_VALIDATION = "gate_validation"
    ASSESS = "assess"
    GUARDRAIL = "guardrail"
    GATE_DECISION = "gate_decision"
    BRIEF = "brief"
    PERSIST = "persist"
    DONE = "done"
    ABORTED = "aborted"
    FAILED = "failed"


TRANSITIONS: dict[Step, set[Step]] = {
    Step.LOAD: {Step.VALIDATE, Step.FAILED},
    Step.VALIDATE: {Step.GATE_VALIDATION},
    Step.GATE_VALIDATION: {Step.ASSESS, Step.BRIEF, Step.ABORTED},
    Step.ASSESS: {Step.GUARDRAIL},
    Step.GUARDRAIL: {Step.GATE_DECISION},
    Step.GATE_DECISION: {Step.BRIEF, Step.ABORTED},
    Step.BRIEF: {Step.PERSIST},
    Step.PERSIST: {Step.DONE},
}
TERMINAL = {Step.DONE, Step.ABORTED, Step.FAILED}


# --------------------------------------------------------------------------- HITL gates
class GateHandler(Protocol):
    def validation_gate(self, invalid: list[dict[str, Any]], valid_count: int) -> str: ...      # continue|abort
    def decision_gate(self, rows: list[dict[str, Any]]) -> dict[str, Any]: ...                  # see AutoGate


class AutoGate:
    """Non-interactive policy: continue past invalid rows, accept guarded recommendations."""
    by = "auto-policy"

    def validation_gate(self, invalid, valid_count):
        return "continue"  # invalid rows still get a documentation-request briefing

    def decision_gate(self, rows):
        return {"decision": "approve", "overrides": {}}


class CLIGate:
    by = "handler"

    def __init__(self, ask: Callable[[str], str] = input, out: Callable[[str], None] = print) -> None:
        self.ask, self.out = ask, out

    def validation_gate(self, invalid, valid_count):
        self.out(f"\n[CHECKPOINT 1/2] {len(invalid)} row(s) failed validation, {valid_count} valid:")
        for r in invalid:
            self.out(f"  - {r['key']:<16} " + "; ".join(e["code"] for e in r["errors"]))
        self.out("  Invalid rows will be briefed as 'request_more_documentation' and excluded from assessment.")
        ans = self.ask("  Continue? [Y]es / [a]bort: ").strip().lower()
        return "abort" if ans.startswith("a") else "continue"

    def decision_gate(self, rows):
        self.out("\n[CHECKPOINT 2/2] Recommended actions (after guardrail):")
        for r in rows:
            flag = " (guardrail raised)" if r["overridden"] else ""
            self.out(f"  - {r['claim_id']:<8} {r['final_action']:<27} rules={','.join(r['rules']) or '-'}{flag}")
        overrides: dict[str, Any] = {}
        while True:
            ans = self.ask("  [A]pprove all / [o]verride a claim / a[b]ort: ").strip().lower()
            if ans in ("", "a", "approve"):
                return {"decision": "approve", "overrides": overrides}
            if ans in ("b", "abort"):
                return {"decision": "abort", "overrides": {}}
            if ans.startswith("o"):
                cid = self.ask("    claim id: ").strip().upper()
                act = self.ask("    action [auto_approve|request_more_documentation|route_to_investigator]: ").strip()
                if cid not in {r["claim_id"] for r in rows} or act not in ACTION_SEVERITY:
                    self.out("    invalid claim id or action, ignored")
                    continue
                note = self.ask("    reason (required): ").strip()
                if not note:
                    self.out("    a reason is required for the audit trail, ignored")
                    continue
                overrides[cid] = {"action": act, "note": note}


# --------------------------------------------------------------------------- state
@dataclass
class TriageState:
    run_id: str
    batch_path: str
    step: Step = Step.LOAD
    claims: dict[str, dict[str, Any]] = field(default_factory=dict)
    steps: list[dict[str, Any]] = field(default_factory=list)
    intake_summary: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "batch_path": self.batch_path, "status": self.step.value,
                "claims": self.claims, "steps": self.steps, "intake_summary": self.intake_summary,
                "error": self.error, "completed_at": datetime.now().isoformat(timespec="seconds")}


def run_agent_safely(backend: Backend, agent: AgentSpec, thread: Thread, prompt: str,
                     fallback: Callable[[], RunResult]) -> RunResult:
    """Run an agent; on failure roll back the thread and return the deterministic fallback."""
    mark = len(thread.messages)
    try:
        with TRACER.span("agent.run", agent=agent.name, backend=backend.name) as s:
            res = backend.run(agent, thread, prompt)
            s.update(rounds=res.rounds, tools=[c["tool"] for c in res.tool_calls])
            return res
    except AgentError as exc:
        del thread.messages[mark:]
        TRACER.event("agent.fallback", agent=agent.name, reason=str(exc))
        res = fallback()
        res.degraded = True
        return res


# --------------------------------------------------------------------------- routine
class TriageRoutine:
    ASSESS_CHUNK = 8

    def __init__(self, backend: Backend, agents: AgentSet, ctx: SkillContext, gate: GateHandler,
                 checkpoint_dir: Path) -> None:
        self.backend, self.agents, self.ctx, self.gate = backend, agents, ctx, gate
        self.checkpoint_dir = checkpoint_dir
        self.handlers: dict[Step, Callable[[TriageState], Step]] = {
            Step.LOAD: self._load, Step.VALIDATE: self._validate, Step.GATE_VALIDATION: self._gate_validation,
            Step.ASSESS: self._assess, Step.GUARDRAIL: self._guardrail, Step.GATE_DECISION: self._gate_decision,
            Step.BRIEF: self._brief, Step.PERSIST: self._persist,
        }

    def run(self, batch_path: str) -> TriageState:
        st = TriageState(run_id="run_" + uuid.uuid4().hex[:8], batch_path=batch_path)
        with TRACER.span("routine.triage", run_id=st.run_id, batch=batch_path) as root:
            while st.step not in TERMINAL:
                current = st.step
                t0 = time.perf_counter()
                with TRACER.span(f"step.{current.value}", run_id=st.run_id) as s:
                    nxt = self.handlers[current](st)
                    s["next"] = nxt.value
                if nxt not in TRANSITIONS[current]:  # defensive: the graph is the contract
                    raise RuntimeError(f"Illegal transition {current.value} -> {nxt.value}")
                rec = st.steps[-1] if st.steps and st.steps[-1]["step"] == current.value else None
                if rec is None:
                    rec = {"step": current.value, "status": "ok", "note": ""}
                    st.steps.append(rec)
                rec["ms"] = round((time.perf_counter() - t0) * 1000)
                rec["next"] = nxt.value
                st.step = nxt
                self._checkpoint(st)
            root["status"] = st.step.value
        return st

    # ---- helpers
    def _note(self, st: TriageState, step: Step, status: str, note: str) -> None:
        st.steps.append({"step": step.value, "status": status, "note": note})

    def _checkpoint(self, st: TriageState) -> None:
        _atomic_write(self.checkpoint_dir / f"{st.run_id}.json", st.to_dict())

    # ---- LOAD
    def _load(self, st: TriageState) -> Step:
        try:
            self.ctx.batch = load_batch(st.batch_path)
            self.ctx._validation = None
        except BatchLoadError as exc:  # failure mode: malformed/missing input file
            st.error = str(exc)
            self._note(st, Step.LOAD, "failed", st.error)
            return Step.FAILED
        b = self.ctx.batch
        self._note(st, Step.LOAD, "ok" if not b.load_errors else "warning",
                   f"{len(b.rows)} rows parsed, {len(b.load_errors)} malformed")
        return Step.VALIDATE

    # ---- VALIDATE (Intake & Validation agent)
    def _validate(self, st: TriageState) -> Step:
        report = self.ctx.validate_batch()  # deterministic, authoritative
        seen: set[str] = set()
        for r in report["results"]:
            key = r["claim_id"] if r["claim_id"] not in seen else f"{r['claim_id']}@row{r['row_index']}"
            seen.add(key)
            raw = next((row.raw for row in self.ctx.batch.rows if row.index == r["row_index"]), {})
            st.claims[key] = {"key": key, "claim_id": r["claim_id"], "row_index": r["row_index"], "claim": raw,
                              "validation": {"valid": r["valid"], "errors": r["errors"]}, "degraded_steps": []}

        def fallback() -> RunResult:
            bad = [f"{r['claim_id']} ({', '.join(e['code'] for e in r['errors'])})"
                   for r in report["results"] if not r["valid"]]
            return RunResult(self.agents.intake.name,
                             "", {"summary": f"{report['valid']} of {report['total_rows']} rows valid. "
                                             f"Invalid: {'; '.join(bad) or 'none'}.", "patterns": []})

        res = run_agent_safely(self.backend, self.agents.intake, Thread("intake"),
                               f"Validate the claims batch loaded from {Path(st.batch_path).name}.", fallback)
        st.intake_summary = (res.output or {}).get("summary") or res.text
        self._note(st, Step.VALIDATE, "degraded" if res.degraded else "ok",
                   f"{report['valid']} valid / {report['invalid']} invalid")
        if res.degraded:
            for c in st.claims.values():
                c["degraded_steps"].append("validate")
        return Step.GATE_VALIDATION

    # ---- CHECKPOINT 1
    def _gate_validation(self, st: TriageState) -> Step:
        invalid = [c | {"errors": c["validation"]["errors"]} for c in st.claims.values() if not c["validation"]["valid"]]
        valid = len(st.claims) - len(invalid)
        if not invalid:
            self._note(st, Step.GATE_VALIDATION, "skipped", "no invalid rows")
            return Step.ASSESS
        decision = self.gate.validation_gate(invalid, valid)
        TRACER.event("hitl.validation_gate", decision=decision, by=self.gate.by, invalid=len(invalid))
        self._note(st, Step.GATE_VALIDATION, decision, f"{self.gate.by}: {decision}")
        if decision == "abort":
            return Step.ABORTED
        return Step.ASSESS if valid else Step.BRIEF

    # ---- ASSESS (Anomaly & Coverage agent, grounded in the KB)
    def _assess(self, st: TriageState) -> Step:
        ids = [c["claim_id"] for c in st.claims.values() if c["validation"]["valid"]]
        for cid in ids:
            st.claims[cid]["facts"] = self.ctx.coverage_facts(cid)
        degraded_any = False
        for i in range(0, len(ids), self.ASSESS_CHUNK):
            chunk = ids[i:i + self.ASSESS_CHUNK]
            res = run_agent_safely(self.backend, self.agents.coverage, Thread("coverage"),
                                   f"Assess these validated claims: {', '.join(chunk)}.",
                                   lambda chunk=chunk: RunResult(self.agents.coverage.name, "",
                                                                 {"assessments": [self._rule_assessment(c) for c in chunk]}))
            got = {a["claim_id"]: a for a in (res.output or {}).get("assessments", [])}
            for cid in chunk:
                a = got.get(cid)
                if a is None:  # agent skipped a claim -> per-claim fallback
                    a, res_deg = self._rule_assessment(cid), True
                    TRACER.event("assess.claim_missing", claim=cid)
                else:
                    res_deg = res.degraded
                a["source"] = "rule_engine_fallback" if res_deg else "coverage_agent"
                st.claims[cid]["assessment"] = a
                if res_deg:
                    degraded_any = True
                    st.claims[cid]["degraded_steps"].append("assess")
        self._note(st, Step.ASSESS, "degraded" if degraded_any else "ok", f"{len(ids)} claims assessed")
        return Step.GUARDRAIL

    def _rule_assessment(self, cid: str) -> dict[str, Any]:
        facts = self.ctx.coverage_facts(cid)
        hits = self.ctx.rules.evaluate(facts)
        return {"claim_id": cid,
                "coverage_status": "covered" if facts["peril_covered"] and facts["in_policy_period"] else "not_covered",
                "triggered_rules": [{"rule_id": h["rule_id"], "evidence": h["evidence"]} for h in hits],
                "fraud_indicators": [], "recommended_action": RuleEngine.decide(hits),
                "rationale": "Deterministic rule evaluation (LLM unavailable)."}

    # ---- GUARDRAIL (code re-checks the LLM's judgment; can only raise severity)
    def _guardrail(self, st: TriageState) -> Step:
        raised = 0
        for c in st.claims.values():
            if "assessment" not in c:
                continue
            hits = self.ctx.rules.evaluate(c["facts"])
            engine_action = RuleEngine.decide(hits)
            agent_action = c["assessment"]["recommended_action"]
            final = max(engine_action, agent_action, key=ACTION_SEVERITY.__getitem__)
            cited = {r["rule_id"] for r in c["assessment"].get("triggered_rules", [])}
            engine_ids = {h["rule_id"] for h in hits}
            known = {r["id"] for r in self.ctx.rules.rules} | {ch.doc_id for ch in self.ctx.kb.chunks}
            c["guardrail"] = {"engine_rules": sorted(engine_ids), "engine_action": engine_action,
                              "agent_action": agent_action, "final_action": final,
                              "overridden": final != agent_action,
                              "missed_rules": sorted(engine_ids - cited),
                              "unknown_rules_cited": sorted(cited - known)}
            if final != agent_action or c["guardrail"]["missed_rules"]:
                raised += final != agent_action
                TRACER.event("guardrail.check", claim=c["claim_id"], agent=agent_action, engine=engine_action,
                             final=final, missed=c["guardrail"]["missed_rules"])
            c["final_action"] = final
        self._note(st, Step.GUARDRAIL, "ok", f"{raised} recommendation(s) raised by guardrail")
        return Step.GATE_DECISION

    # ---- CHECKPOINT 2
    def _gate_decision(self, st: TriageState) -> Step:
        rows = [{"claim_id": c["claim_id"], "final_action": c["final_action"],
                 "rules": c["guardrail"]["engine_rules"], "overridden": c["guardrail"]["overridden"]}
                for c in st.claims.values() if "guardrail" in c]
        if not rows:
            self._note(st, Step.GATE_DECISION, "skipped", "nothing to decide")
            return Step.BRIEF
        d = self.gate.decision_gate(rows)
        TRACER.event("hitl.decision_gate", decision=d["decision"], by=self.gate.by, overrides=d["overrides"])
        if d["decision"] == "abort":
            self._note(st, Step.GATE_DECISION, "abort", f"{self.gate.by}: abort")
            return Step.ABORTED
        for cid, ov in d["overrides"].items():
            c = st.claims[cid]
            c["human_override"] = {"from": c["final_action"], "to": ov["action"], "note": ov["note"],
                                   "by": self.gate.by, "at": datetime.now().isoformat(timespec="seconds")}
            c["final_action"] = ov["action"]
        self._note(st, Step.GATE_DECISION, "approve", f"{self.gate.by}: approved, {len(d['overrides'])} override(s)")
        return Step.BRIEF

    # ---- BRIEF (Adjuster Briefing agent, uses long-term memory)
    def _brief(self, st: TriageState) -> Step:
        degraded = 0
        for c in st.claims.values():
            if not c["validation"]["valid"]:
                c["final_action"] = "request_more_documentation"
                issues = "; ".join(e["message"] for e in c["validation"]["errors"])
                c["briefing"] = {"claim_id": c["claim_id"], "summary": f"Not assessed - failed intake validation: {issues}.",
                                 "history_note": "Not evaluated.", "recommended_action": c["final_action"],
                                 "next_steps": ["Return to submitter to correct: " + issues], "confidence": "high"}
                continue
            payload = {k: c.get(k) for k in ("claim", "facts", "assessment", "guardrail", "human_override")}
            prompt = (f"Brief the adjuster on claim {c['claim_id']}. FINAL recommended action: {c['final_action']}.\n"
                      f"Context (JSON): {to_json(payload)}")
            res = run_agent_safely(self.backend, self.agents.briefing, Thread("briefing"), prompt,
                                   lambda c=c: RunResult(self.agents.briefing.name, "", self._template_brief(c)))
            b = res.output or self._template_brief(c)
            if b.get("recommended_action") != c["final_action"]:
                TRACER.event("briefing.action_mismatch", claim=c["claim_id"], briefed=b.get("recommended_action"),
                             final=c["final_action"])
                b["recommended_action"] = c["final_action"]
            c["briefing"] = b
            if res.degraded:
                degraded += 1
                c["degraded_steps"].append("brief")
        self._note(st, Step.BRIEF, "degraded" if degraded else "ok", f"{len(st.claims)} briefings, {degraded} templated")
        return Step.PERSIST

    def _template_brief(self, c: dict[str, Any]) -> dict[str, Any]:
        f, a = c["facts"], c["assessment"]
        hist = self.ctx.memory.history(f["policy_number"], claim_type=f["claim_type"], exclude_claim_id=c["claim_id"],
                                       reference_date=date.fromisoformat(c["claim"]["loss_date"]))
        prior = f["prior_same_peril_claims"] or hist
        note = "; ".join(f"similar {f['claim_type']} claim {p['claim_id']} about {p.get('months_before_reference', '?')} "
                         f"months before this loss ({p.get('recommendation', 'in current batch')})" for p in prior) \
            or "No prior history."
        rules = ", ".join(r["rule_id"] for r in a["triggered_rules"]) or "none"
        return {"claim_id": c["claim_id"],
                "summary": (f"{f['claim_type']} claim for {f['claim_amount']:,.0f} on {f['policy_number']} "
                            f"({f['pct_of_limit']}% of {f['policy_limit']:,.0f} limit). Coverage: {a['coverage_status']}. "
                            f"Rules triggered: {rules}. Description: {f['description']}"),
                "history_note": note[0].upper() + note[1:], "recommended_action": c["final_action"],
                "next_steps": [f"Review {r['rule_id']}: {r['evidence']}" for r in a["triggered_rules"]][:5]
                or ["Proceed with settlement."], "confidence": "medium"}

    # ---- PERSIST (long-term memory)
    def _persist(self, st: TriageState) -> Step:
        n = 0
        now = datetime.now().isoformat(timespec="seconds")
        for c in st.claims.values():
            raw = c["claim"]
            if "@row" in c["key"] or not raw.get("policy_number") or not raw.get("claim_id"):
                continue  # duplicates/unidentifiable rows are not policyholder history
            self.ctx.memory.record({
                "claim_id": c["claim_id"], "policy_number": raw["policy_number"], "claim_type": raw.get("claim_type"),
                "loss_date": raw.get("loss_date") or None, "claim_amount": raw.get("claim_amount"),
                "recommendation": c["final_action"], "validated": c["validation"]["valid"],
                "flags": c.get("guardrail", {}).get("engine_rules", [e["code"] for e in c["validation"]["errors"]]),
                "summary": c["briefing"]["summary"][:400], "triaged_at": now, "run_id": st.run_id})
            n += 1
        self._note(st, Step.PERSIST, "ok", f"{n} outcomes written to long-term memory")
        return Step.DONE


def summarise(st: TriageState) -> str:
    d = st.to_dict() if isinstance(st, TriageState) else st
    if d["status"] == "failed":
        return f"Triage failed at load: {d['error']}"
    lines = [f"Triage {d['run_id']} - status: {d['status']}", d.get("intake_summary", ""), "",
             f"{'Claim':<16}{'Action':<28}{'Rules':<22}Notes"]
    for c in d["claims"].values():
        rules = ",".join(c.get("guardrail", {}).get("engine_rules", [])) or \
            ",".join(e["code"] for e in c["validation"]["errors"])[:21]
        notes = []
        if c.get("guardrail", {}).get("overridden"):
            notes.append("guardrail-raised")
        if c.get("human_override"):
            notes.append(f"human:{c['human_override']['from']}->{c['human_override']['to']}")
        if c["degraded_steps"]:
            notes.append("degraded:" + "/".join(sorted(set(c["degraded_steps"]))))
        lines.append(f"{c['key']:<16}{c.get('final_action', '-'):<28}{rules:<22}{' '.join(notes)}")
    lines += ["", "Steps: " + " -> ".join(f"{s['step']}[{s['status']}]" for s in d["steps"])]
    return "\n".join(lines)


__all__ = ["TriageRoutine", "TriageState", "Step", "TRANSITIONS", "AutoGate", "CLIGate", "summarise",
           "run_agent_safely"]
