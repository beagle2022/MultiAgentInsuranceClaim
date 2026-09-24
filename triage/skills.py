"""Skills (function tools) with JSON schemas, bound to a per-run SkillContext.

Design rule: tools compute FACTS deterministically (validity, dates, % of limit, history);
agents retrieve RULES from the knowledge base and apply judgment; a code guardrail re-checks
the judgment. No tool returns a recommendation by itself.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from .data import Batch
from .knowledge import ACTIONS, KnowledgeBase, RuleEngine
from .memory import LongTermMemory
from .runtime import Tool, normalise_claim_id

REQUIRED_FIELDS = ["claim_id", "policy_number", "claim_type", "loss_date", "report_date", "claim_amount"]
POLICY_RE = re.compile(r"^POL-\d{4}$")


def _parse_date(v: Any) -> date | None:
    try:
        return date.fromisoformat(str(v))
    except (TypeError, ValueError):
        return None


@dataclass
class SkillContext:
    kb: KnowledgeBase
    rules: RuleEngine
    memory: LongTermMemory
    policies: dict[str, dict[str, Any]]
    as_of: date
    batch: Batch | None = None
    last_run: dict[str, Any] | None = None           # previous TriageResult (session memory)
    _validation: dict[str, Any] | None = field(default=None, repr=False)

    # ------------------------------------------------------------------ validation
    def validate_batch(self) -> dict[str, Any]:
        if self._validation is not None:
            return self._validation
        if self.batch is None:
            raise ValueError("No claims batch is loaded. Ask the handler to supply a batch file.")
        results, seen = [], {}
        for row in self.batch.rows:
            r, errs = row.raw, []
            for f in REQUIRED_FIELDS:
                if r.get(f) in (None, ""):
                    errs.append({"code": "MISSING_FIELD", "field": f, "message": f"'{f}' is missing or empty"})
            loss, rep = _parse_date(r.get("loss_date")), _parse_date(r.get("report_date"))
            for f, d in (("loss_date", loss), ("report_date", rep)):
                if r.get(f) not in (None, "") and d is None:
                    errs.append({"code": "INVALID_DATE", "field": f, "message": f"'{r.get(f)}' is not ISO YYYY-MM-DD"})
                elif d and d > self.as_of:
                    errs.append({"code": "FUTURE_DATE", "field": f, "message": f"{d} is after {self.as_of}"})
            if loss and rep and loss > rep:
                errs.append({"code": "DATE_ORDER", "field": "loss_date",
                             "message": f"loss_date {loss} is after report_date {rep}"})
            amt = r.get("claim_amount")
            if amt not in (None, ""):
                if isinstance(amt, bool) or not isinstance(amt, (int, float)):
                    errs.append({"code": "INVALID_AMOUNT", "field": "claim_amount", "message": f"'{amt}' is not numeric"})
                elif amt <= 0:
                    errs.append({"code": "INVALID_AMOUNT", "field": "claim_amount", "message": "must be greater than 0"})
            pol = r.get("policy_number")
            if pol:
                if not POLICY_RE.match(str(pol)):
                    errs.append({"code": "INVALID_POLICY_FORMAT", "field": "policy_number",
                                 "message": f"'{pol}' does not match POL-NNNN"})
                elif pol not in self.policies:
                    errs.append({"code": "UNKNOWN_POLICY", "field": "policy_number",
                                 "message": f"{pol} not found in coverage records"})
            cid = r.get("claim_id")
            if cid:
                if cid in seen:
                    errs.append({"code": "DUPLICATE_CLAIM_ID", "field": "claim_id",
                                 "message": f"duplicate of row {seen[cid]}"})
                else:
                    seen[cid] = row.index
            results.append({"row_index": row.index, "claim_id": row.claim_id, "valid": not errs, "errors": errs})
        for le in self.batch.load_errors:
            results.append({"row_index": le["row_index"], "claim_id": f"ROW-{le['row_index']}", "valid": False,
                            "errors": [{"code": le["code"], "field": None, "message": le["message"]}]})
        results.sort(key=lambda x: x["row_index"])
        self._validation = {"source": self.batch.source, "total_rows": len(results),
                            "valid": sum(r["valid"] for r in results),
                            "invalid": sum(not r["valid"] for r in results), "results": results}
        return self._validation

    def valid_claims(self) -> dict[str, dict[str, Any]]:
        ok = {r["row_index"] for r in self.validate_batch()["results"] if r["valid"]}
        return {row.raw["claim_id"]: row.raw for row in self.batch.rows if row.index in ok}

    # ------------------------------------------------------------------ coverage facts
    def coverage_facts(self, claim_id: str) -> dict[str, Any]:
        claim_id = normalise_claim_id(claim_id)
        claim = self.valid_claims().get(claim_id)
        if claim is None:
            raise KeyError(f"{claim_id} is not a validated claim in the current batch")
        pol = self.policies[claim["policy_number"]]
        loss, rep = date.fromisoformat(claim["loss_date"]), date.fromisoformat(claim["report_date"])
        start, end = date.fromisoformat(pol["active_from"]), date.fromisoformat(pol["active_to"])
        amount, limit = float(claim["claim_amount"]), float(pol["policy_limit"])

        prior = {e["claim_id"]: {**e, "origin": "long_term_memory"} for e in self.memory.history(
            claim["policy_number"], claim_type=claim["claim_type"], exclude_claim_id=claim_id,
            reference_date=loss, within_days=365, only_validated=True)}
        for other_id, other in self.valid_claims().items():  # same-batch earlier losses
            if other_id == claim_id or other["policy_number"] != claim["policy_number"] \
                    or other["claim_type"] != claim["claim_type"]:
                continue
            delta = (loss - date.fromisoformat(other["loss_date"])).days
            if 0 < delta <= 365:
                prior.setdefault(other_id, {"claim_id": other_id, "loss_date": other["loss_date"],
                                            "claim_amount": other["claim_amount"], "origin": "current_batch",
                                            "months_before_reference": round(delta / 30.44, 1)})
        return {
            "claim_id": claim_id, "policy_number": claim["policy_number"], "claim_type": claim["claim_type"],
            "description": claim.get("description", ""), "coverage_type": pol["coverage_type"],
            "covered_perils": pol["covered_perils"], "peril_covered": claim["claim_type"] in pol["covered_perils"],
            "policy_period": f"{start} to {end}", "in_policy_period": start <= loss <= end,
            "days_since_inception": (loss - start).days, "days_to_report": (rep - loss).days,
            "claim_amount": amount, "policy_limit": limit, "pct_of_limit": round(100 * amount / limit, 2),
            "exceeds_limit": amount > limit, "prior_same_peril_12m": len(prior),
            "prior_same_peril_claims": [{k: v for k, v in p.items() if k in
                                         ("claim_id", "loss_date", "claim_amount", "recommendation",
                                          "summary", "origin", "months_before_reference")} for p in prior.values()],
        }

    # ------------------------------------------------------------------ Q&A helpers
    def triage_record(self, claim_id: str) -> dict[str, Any]:
        claim_id = normalise_claim_id(claim_id)
        if self.last_run:
            rec = self.last_run.get("claims", {}).get(claim_id)
            if rec:
                return {"source": "current_session", **rec}
        stored = self.memory.get(claim_id)
        if stored:
            return {"source": "long_term_memory", **stored}
        raise KeyError(f"No triage record for {claim_id} in this session or in long-term memory")


# ============================================================================ tool builders
_ACTION = {"type": "string", "enum": ACTIONS}


def intake_tools(ctx: SkillContext) -> list[Tool]:
    return [
        Tool("validate_claims_batch",
             "Validate every row of the currently loaded claims batch. Checks required fields, ISO date "
             "format, loss_date <= report_date, future dates, numeric positive amounts, policy number format "
             "and existence, and duplicate claim IDs. Returns per-row results with error codes.",
             {"type": "object", "properties": {}, "additionalProperties": False},
             lambda: ctx.validate_batch()),
        Tool("submit_validation_report",
             "Submit your final intake report. Call exactly once, after validate_claims_batch.",
             {"type": "object", "required": ["summary"], "properties": {
                 "summary": {"type": "string", "description": "2-4 sentences for the claims handler."},
                 "patterns": {"type": "array", "items": {"type": "string"},
                              "description": "Notable cross-claim patterns (e.g. repeated policy, resubmissions)."}}}),
    ]


def knowledge_tool(ctx: SkillContext) -> Tool:
    return Tool("search_underwriting_knowledge",
                "Semantic search over the underwriting rules (UW-xx) and historical fraud indicator patterns "
                "(FI-xx). Returns the most relevant sections with IDs. Use it to find which rules apply.",
                {"type": "object", "required": ["query"], "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 8, "default": 4}}},
                lambda query, top_k=4: ctx.kb.search(query, top_k))


def history_tool(ctx: SkillContext) -> Tool:
    def run(policy_number: str, claim_type: str | None = None, reference_claim_id: str | None = None):
        ref = None
        if reference_claim_id:
            c = ctx.valid_claims().get(normalise_claim_id(reference_claim_id)) if ctx.batch else None
            ref = date.fromisoformat(c["loss_date"]) if c else None
        return ctx.memory.history(policy_number, claim_type=claim_type, reference_date=ref,
                                  exclude_claim_id=normalise_claim_id(reference_claim_id or ""))
    return Tool("get_policyholder_history",
                "Long-term memory lookup: prior triage outcomes for a policy across earlier sessions. Optionally "
                "filter by claim_type and give reference_claim_id to get 'months before this loss' values.",
                {"type": "object", "required": ["policy_number"], "properties": {
                    "policy_number": {"type": "string"}, "claim_type": {"type": "string"},
                    "reference_claim_id": {"type": "string"}}},
                run)


def policy_tool(ctx: SkillContext) -> Tool:
    def run(policy_number: str):
        if policy_number not in ctx.policies:
            raise KeyError(f"Unknown policy {policy_number}")
        return ctx.policies[policy_number]
    return Tool("lookup_policy_coverage", "Fetch the coverage record for a policy number.",
                {"type": "object", "required": ["policy_number"], "properties": {"policy_number": {"type": "string"}}},
                run)


def coverage_tools(ctx: SkillContext) -> list[Tool]:
    def facts(claim_ids: list[str]):
        out = {}
        for cid in claim_ids:
            try:
                out[cid] = ctx.coverage_facts(cid)
            except KeyError as exc:  # per-item error, don't fail the whole call
                out[cid] = {"error": str(exc)}
        return out
    return [
        Tool("get_coverage_facts",
             "Compute coverage facts for validated claims: peril covered?, in policy period?, days since "
             "inception, days to report, % of limit, exceeds limit?, and prior same-peril claims in the last "
             "12 months (from long-term memory and the current batch). Facts only - no decisions.",
             {"type": "object", "required": ["claim_ids"], "properties": {
                 "claim_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1}}},
             facts),
        policy_tool(ctx),
        knowledge_tool(ctx),
        Tool("submit_coverage_assessment",
             "Submit the final assessment for ALL claims you were given. Call exactly once.",
             {"type": "object", "required": ["assessments"], "properties": {"assessments": {
                 "type": "array", "items": {
                     "type": "object",
                     "required": ["claim_id", "coverage_status", "triggered_rules", "recommended_action", "rationale"],
                     "properties": {
                         "claim_id": {"type": "string"},
                         "coverage_status": {"type": "string", "enum": ["covered", "not_covered", "indeterminate"]},
                         "triggered_rules": {"type": "array", "items": {
                             "type": "object", "required": ["rule_id", "evidence"],
                             "properties": {"rule_id": {"type": "string"}, "evidence": {"type": "string"}}}},
                         "fraud_indicators": {"type": "array", "items": {"type": "string"},
                                              "description": "FI-xx ids retrieved from the knowledge base"},
                         "recommended_action": _ACTION,
                         "rationale": {"type": "string"}}}}}}),
    ]


def briefing_tools(ctx: SkillContext) -> list[Tool]:
    return [
        history_tool(ctx),
        Tool("submit_briefing", "Submit the adjuster briefing for this claim. Call exactly once.",
             {"type": "object",
              "required": ["claim_id", "summary", "history_note", "recommended_action", "next_steps", "confidence"],
              "properties": {
                  "claim_id": {"type": "string"},
                  "summary": {"type": "string", "description": "3-5 sentence plain-language summary for the adjuster."},
                  "history_note": {"type": "string",
                                   "description": "What prior claims/memory say about this policyholder, or 'No prior history.'"},
                  "recommended_action": _ACTION,
                  "next_steps": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
                  "confidence": {"type": "string", "enum": ["low", "medium", "high"]}}}),
    ]


def router_tools() -> list[Tool]:
    return [Tool("route_request", "Classify the handler's latest message and extract parameters.",
                 {"type": "object", "required": ["intent", "reason"], "properties": {
                     "intent": {"type": "string",
                                "enum": ["triage_batch", "explain_claim", "policyholder_history", "general_question"]},
                     "claim_ids": {"type": "array", "items": {"type": "string"}},
                     "policy_number": {"type": "string"},
                     "batch_path": {"type": "string", "description": "Only if the user named a file."},
                     "reason": {"type": "string", "description": "One short sentence justifying the route."}}})]


def supervisor_answer_tools(ctx: SkillContext) -> list[Tool]:
    return [
        Tool("get_claim_triage_record",
             "Fetch the triage record (validation, coverage assessment, guardrail result, briefing, human "
             "overrides) for a claim, from this session or long-term memory.",
             {"type": "object", "required": ["claim_id"], "properties": {"claim_id": {"type": "string"}}},
             lambda claim_id: ctx.triage_record(claim_id)),
        history_tool(ctx), policy_tool(ctx), knowledge_tool(ctx),
    ]
