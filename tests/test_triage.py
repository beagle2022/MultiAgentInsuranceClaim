import json
from datetime import date

import pytest

from triage.backends import ClaudeBackend
from triage.memory import LongTermMemory, Thread
from triage.routine import TRANSITIONS, AutoGate, Step
from triage.runtime import AgentSpec, Tool
from triage.supervisor import Supervisor

from .fakes import FakeClient, resp, text, tool

EXPECTED = {  # deterministic outcome for data/claims.json (offline / rule-engine path)
    "C-2031": "route_to_investigator", "C-2032": "route_to_investigator", "C-2033": "route_to_investigator",
    "C-2034": "request_more_documentation", "C-2035": "request_more_documentation",
    "C-2036": "request_more_documentation", "C-2037": "auto_approve",
    "C-2037@row7": "request_more_documentation", "C-2038": "route_to_investigator",
    "C-2039": "route_to_investigator", "C-2040": "request_more_documentation",
}


def test_offline_end_to_end(settings):
    sup = Supervisor(settings, gate=AutoGate())
    sup.handle("triage data/claims.json")
    run = sup.session.last_run
    assert run["status"] == "done"
    assert {k: c["final_action"] for k, c in run["claims"].items()} == EXPECTED
    codes = {k: [e["code"] for e in c["validation"]["errors"]] for k, c in run["claims"].items()}
    assert "DATE_ORDER" in codes["C-2035"] and "INVALID_POLICY_FORMAT" in codes["C-2036"]
    assert codes["C-2037@row7"] == ["DUPLICATE_CLAIM_ID"] and "MISSING_FIELD" in codes["C-2034"]
    # memory-grounded briefing
    assert "C-1874" in run["claims"]["C-2031"]["briefing"]["history_note"]


def test_long_term_memory_survives_new_process(settings):
    Supervisor(settings, gate=AutoGate()).handle("triage data/claims.json")
    mem = LongTermMemory(settings.long_term_path)  # fresh instance = new run
    hist = mem.history("POL-5521", claim_type="water_damage", reference_date=date(2026, 8, 15), within_days=365)
    assert [e["claim_id"] for e in hist] == ["C-1874", "C-2032"]
    assert hist[-1]["months_before_reference"] == pytest.approx(6.1, abs=0.1)


def test_session_resume_multi_turn(settings):
    s1 = Supervisor(settings, gate=AutoGate(), session_id="t1")
    s1.handle("triage the batch")
    s2 = Supervisor(settings, gate=AutoGate(), session_id="t1")  # new process, same session
    assert s2.session.turns == 1
    assert "UW-06" in s2.handle("why was C2031 flagged?")


def test_malformed_and_unreadable_input(settings, tmp_path):
    sup = Supervisor(settings, gate=AutoGate())
    sup.handle("triage data/claims_malformed.json")
    claims = sup.session.last_run["claims"]
    assert claims["ROW-2"]["validation"]["errors"][0]["code"] == "MALFORMED_ROW"
    assert claims["C-3001"]["final_action"] == "auto_approve"
    out = sup.handle("triage data/claims_truncated.json")
    assert "not valid JSON" in out and sup.session.last_run["status"] == "failed"


def test_transition_graph_is_complete():
    reachable = {s for targets in TRANSITIONS.values() for s in targets}
    assert {Step.DONE, Step.ABORTED, Step.FAILED} <= reachable


# ------------------------------------------------------------------ real ClaudeBackend loop, fake API
def _script(agent, rnd, kw):
    if agent.startswith("You are the router"):
        return resp(tool("route_request", intent="triage_batch", reason="asked to triage"))
    if agent.startswith("You are the Intake"):
        return resp(tool("validate_claims_batch")) if rnd == 1 else \
            resp(tool("submit_validation_report", summary="7 valid, 4 invalid."))
    if agent.startswith("You are the Anomaly"):
        ids = kw["messages"][0]["content"].split(": ")[1].rstrip(".").split(", ")
        if rnd == 1:
            return resp(text("Fetching facts."), tool("get_coverage_facts", claim_ids=ids))
        if rnd == 2:
            return resp(tool("search_underwriting_knowledge", query="loss shortly after policy start"))
        # deliberately lenient on everything: guardrail must raise severity
        return resp(tool("submit_coverage_assessment", assessments=[
            {"claim_id": c, "coverage_status": "covered", "triggered_rules": [],
             "recommended_action": "auto_approve", "rationale": "looks fine"} for c in ids]))
    if agent.startswith("You are the Adjuster"):
        cid = kw["messages"][0]["content"].split("claim ")[1].split(".")[0]
        if rnd == 1:
            return resp(tool("get_policyholder_history", policy_number="POL-5521", claim_type="water_damage"))
        return resp(tool("submit_briefing", claim_id=cid, summary="s", history_note="h",
                         recommended_action="auto_approve", next_steps=["x"], confidence="low"))
    return resp(text("ok"))


def test_claude_loop_tools_and_guardrail(settings):
    fake = FakeClient(_script)
    sup = Supervisor(settings, backend=ClaudeBackend(settings, client=fake), gate=AutoGate())
    sup.handle("please triage today's claims")
    run = sup.session.last_run
    c33 = run["claims"]["C-2033"]
    assert c33["assessment"]["source"] == "coverage_agent"
    assert c33["guardrail"]["overridden"] and c33["final_action"] == "route_to_investigator"
    assert set(c33["guardrail"]["missed_rules"]) == {"UW-01", "UW-02", "UW-07"}
    assert run["claims"]["C-2037"]["final_action"] == "auto_approve"
    # briefing agent tried to downgrade; routine kept the authoritative action
    assert run["claims"]["C-2031"]["briefing"]["recommended_action"] == "route_to_investigator"
    # every request keeps strict role alternation
    for req in fake.requests:
        roles = [m["role"] for m in req["messages"]]
        assert all(a != b for a, b in zip(roles, roles[1:])), roles


def test_tool_exception_is_reported_to_model(settings):
    def boom():
        raise RuntimeError("downstream unavailable")

    seen = {}

    def script(agent, rnd, kw):
        if rnd == 1:
            return resp(tool("flaky"))
        seen["result"] = kw["messages"][-1]["content"][0]
        return resp(text("Tool failed; reporting partial result."))

    agent = AgentSpec("t", "You are a test agent.", [Tool("flaky", "d", {"type": "object", "properties": {}}, boom)], "m")
    res = ClaudeBackend(settings, client=FakeClient(script)).run(agent, Thread("t"), "go")
    assert seen["result"]["is_error"] and "downstream unavailable" in seen["result"]["content"]
    assert res.text.startswith("Tool failed")


def test_api_failure_degrades_gracefully(settings):
    def script(agent, rnd, kw):
        raise TimeoutError("simulated API timeout")

    sup = Supervisor(settings, backend=ClaudeBackend(settings, client=FakeClient(script)), gate=AutoGate())
    sup.handle("triage data/claims.json")
    run = sup.session.last_run
    assert run["status"] == "done"
    assert {k: c["final_action"] for k, c in run["claims"].items()} == EXPECTED
    assert "assess" in run["claims"]["C-2031"]["degraded_steps"]
    json.dumps(run)  # checkpoint/serialisable
