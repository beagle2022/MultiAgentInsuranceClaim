# Multi-Agent Claims Triage Assistant

Supervisor + 3 specialist agents that triage a batch of synthetic property/auto claims:
**intake -> validate -> check coverage/anomalies -> brief the adjuster**, with two human checkpoints,
knowledge-grounded rule retrieval, short- and long-term memory, a code guardrail over LLM decisions,
and span-level tracing.

> **LLM backend:** Built and run on **Claude (Anthropic Messages API, tool use)** while Azure AI Foundry
> access is pending. The runtime is Foundry-shaped (agent / tool / thread / run), and
> `triage/foundry_backend.py` implements the same interface with the `azure-ai-agents` SDK
> (reference code, not yet executed - see [Foundry mapping](#foundry-mapping)). All data is synthetic.

---

## Quick start (Windows / PowerShell, Python 3.11+)

```powershell
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env      # add ANTHROPIC_API_KEY

python main.py --session demo                       # interactive chat, HITL prompts
python main.py --session demo --auto-gates --batch data/claims.json
python main.py --session demo --ask "why was C2031 flagged?"   # follow-up, new process, same session
python main.py --backend offline --session demo     # no LLM: deterministic fallbacks end to end
python -m pytest -q                                 # 8 tests, no API key needed
```
`--reset-memory` wipes `memory_store/` (long-term memory is re-seeded from `data/seed_memory.json`).
VS Code launch configs are in `.vscode/launch.json`.

---

## Architecture

```
 handler msg
     |
 [Supervisor]  router agent (forced structured tool: route_request) --> dict dispatch in code
     |                                     |
     | triage_batch                        | explain_claim / policyholder_history / general_question
     v                                     v
 [TriageRoutine state machine]         supervisor answer agent on the persistent session thread
  LOAD -> VALIDATE -> GATE#1 -> ASSESS -> GUARDRAIL -> GATE#2 -> BRIEF -> PERSIST -> DONE
            |                    |                               |
      Intake agent        Coverage agent                  Briefing agent
      validate_claims_    get_coverage_facts              get_policyholder_history
      batch               search_underwriting_knowledge   (long-term memory)
                          lookup_policy_coverage
```

| Agent | Role | Tools (skills) | Structured output |
|---|---|---|---|
| `supervisor_router` | Classify intent, resolve "that claim" from context | – | `route_request` (forced) |
| `supervisor` | Answer follow-ups, grounded in records | `get_claim_triage_record`, `get_policyholder_history`, `lookup_policy_coverage`, `search_underwriting_knowledge` | prose |
| `intake_validation` | Validate batch, summarise issues/patterns | `validate_claims_batch` | `submit_validation_report` |
| `anomaly_coverage` | Coverage + anomaly assessment, KB-grounded | `get_coverage_facts`, `search_underwriting_knowledge`, `lookup_policy_coverage` | `submit_coverage_assessment` |
| `adjuster_briefing` | Adjuster note + next steps, memory-aware | `get_policyholder_history` | `submit_briefing` |

All agents, instructions and tool schemas are defined in code (`triage/agents.py`, `triage/skills.py`).

### Key design decisions (the parts worth defending)

1. **Facts in code, rules in the KB, judgment in the LLM, final say in code + human.**
   Tools compute facts only (days since inception, % of limit, prior same-peril claims...). The Coverage
   agent must *retrieve* rules - no thresholds appear in any prompt. A guardrail then re-evaluates the same
   rules deterministically and can only **raise** severity; it also records `missed_rules` (rules the LLM
   didn't cite) - a built-in quality signal.
2. **One source of truth for rules.** Each rule in `knowledge/underwriting_rules.md` has prose (what the
   LLM retrieves) plus a machine-readable annotation in an HTML comment (what the guardrail evaluates).
   Change a threshold in one place; both paths follow.
3. **The routine is the orchestrator, not the prompt.** `TRANSITIONS` is an explicit graph; illegal
   transitions raise; state is checkpointed to `memory_store/checkpoints/<run_id>.json` after every step.
4. **Structured outputs via "output tools".** Each specialist finishes by calling a `submit_*` tool whose
   JSON schema (with enums for actions) is the contract. If the model answers in prose, the runtime forces
   the tool once (`tool_choice`).
5. **Graceful degradation over failure.** Every agent call has a deterministic fallback; the step is marked
   `degraded` and the claim carries `degraded_steps`, so the adjuster knows which parts were LLM-free.

### Routine and HITL checkpoints (`triage/routine.py`)

| Step | What happens | Next |
|---|---|---|
| LOAD | Parse file; bad rows become `MALFORMED_ROW`, unusable file -> FAILED | VALIDATE / FAILED |
| VALIDATE | Deterministic validation (authoritative) + Intake agent summary | GATE_VALIDATION |
| GATE #1 | Human sees invalid rows: continue or abort | ASSESS / BRIEF (none valid) / ABORTED |
| ASSESS | Coverage agent in chunks of 8, KB-grounded; per-claim fallback if skipped | GUARDRAIL |
| GUARDRAIL | Rule engine re-check, `final = max(agent, engine)` | GATE_DECISION |
| GATE #2 | Human approves / overrides a claim (reason mandatory, audited) / aborts | BRIEF / ABORTED |
| BRIEF | Briefing agent per claim with long-term memory; action locked to the approved one | PERSIST |
| PERSIST | Outcomes upserted to long-term memory | DONE |

`--auto-gates` swaps in `AutoGate` (non-interactive policy) for CI/demo.

### Memory (`triage/memory.py`)
* **Short-term (thread):** each agent run has a `Thread`; the supervisor thread spans the whole session and
  is persisted with the last triage result to `memory_store/sessions/<id>.json`, so
  `--session demo --ask "why was C2031 flagged?"` works from a new process. Threads are trimmed only at
  plain user turns so `tool_use`/`tool_result` pairs are never split.
* **Long-term:** `memory_store/long_term_memory.json`, keyed by policy, atomic writes, corrupt-file
  quarantine. Seeded with synthetic history (C-1874, a water-damage claim flagged in Sep 2025). Every run
  writes back, so on the next run C-2031's briefing cites both C-1874 (~11 months earlier) and C-2032
  (~6 months earlier). Only validated claims count as "prior history".

### Knowledge grounding (`triage/knowledge.py`)
Local vector store: markdown split by `##` section, TF-IDF vectors, cosine similarity - pure Python, no
extra deps. Exposed as `search_underwriting_knowledge`. On Foundry it is replaced by a hosted vector store
+ `FileSearchTool` (`FOUNDRY_HOSTED_FILE_SEARCH=1`); swapping to embeddings / Azure AI Search only changes
`KnowledgeBase`.

### Error handling
| Failure | Handling |
|---|---|
| Truncated / non-JSON / wrong-shape file | `BatchLoadError` -> routine ends in FAILED with a clear message |
| Bad rows (non-object, bad dates, text amounts) | Row-level validation errors, rest of batch proceeds |
| Tool exception or timeout (15s) | Returned to the model as `is_error` tool_result; model can adapt |
| LLM API error / timeout | SDK retries w/ backoff (`TRIAGE_LLM_RETRIES`), then `AgentError` -> thread rollback -> deterministic fallback |
| Runaway tool loop | `TRIAGE_MAX_TOOL_ROUNDS` cap |
| Agent skips a claim / downgrades action | Per-claim fallback; guardrail raises; briefing action locked |
| Corrupt memory file | Quarantined to `*.corrupt.json`, fresh store |

### Observability (`triage/tracing.py`)
Nested spans (`supervisor.turn > routine.triage > step.* > agent.run > llm.call / tool.call`) with
token counts, latencies, routing reasons, guardrail and HITL events - to console and
`logs/trace-YYYYMMDD.jsonl`. `TRIAGE_OTEL=1` also emits OpenTelemetry spans (console exporter; swap for
OTLP or `azure-monitor-opentelemetry` on Foundry).

---

## Expected outcome for `data/claims.json`

Sample data extended with synthetic cases for every validation and rule path.

| Claim | Action | Why |
|---|---|---|
| C-2031 | route_to_investigator | UW-06 repeat water damage (C-1874 from memory, C-2032 ~6 months earlier) |
| C-2032 | route_to_investigator | UW-06 (C-1874 ~5 months earlier) |
| C-2033 | route_to_investigator | UW-01 loss 3 days after inception, UW-02 100% of limit, UW-07; FI-01 jewelry |
| C-2034 | request_more_documentation | Missing loss date, amount 0 |
| C-2035 | request_more_documentation | Loss date after report date |
| C-2036 | request_more_documentation | Invalid policy number format |
| C-2037 | auto_approve | Small covered glass claim, no flags |
| C-2037 (row 7) | request_more_documentation | Duplicate claim ID |
| C-2038 | route_to_investigator | UW-02 98.7% of limit |
| C-2039 | route_to_investigator | UW-08 flood not covered under homeowners |
| C-2040 | request_more_documentation | UW-05 reported 66 days after loss |

`data/claims_malformed.json` and `data/claims_truncated.json` demo the error paths.

## Suggested demo flow
1. `python main.py --reset-memory --session demo` -> `triage this batch of claims` -> walk both checkpoints
   (override C-2037 with a reason to show the audit trail).
2. `why was C2031 flagged?` -> grounded answer with rule text + memory history.
3. `what about the theft one?` -> router resolves C-2033 from conversation context.
4. Quit, rerun with `--session demo` -> context preserved; triage again -> briefings now cite run 1.
5. `triage data/claims_truncated.json` -> graceful failure. `--backend offline` -> degraded mode.
6. Open `logs/trace-*.jsonl` and a checkpoint file to show traceability.

## Foundry mapping
| This build | Foundry Agent Service (`azure-ai-agents`) |
|---|---|
| `AgentSpec` | `agents_client.create_agent(model, name, instructions, tools, tool_resources)` |
| `Tool` (JSON schema + callable) | `FunctionToolDefinition(FunctionDefinition(...))`, executed on `requires_action` |
| `Thread` | `agents_client.threads.create()` |
| `ClaudeBackend.run` loop | `messages.create` -> `runs.create` -> poll -> `runs.submit_tool_outputs` |
| `KnowledgeBase.search` | `vector_stores.create_and_poll` + `FileSearchTool` (or Azure AI Search) |
| JSONL / OTel spans | Foundry tracing via Application Insights |

The routine, agents, skills, memory and guardrail are backend-independent; only `run()` changes.

## What was substituted, and why
* **LLM provider:** Claude instead of Foundry-hosted models, pending Foundry access. `foundry_backend.py`
  is written against azure-ai-agents 1.x but has not been run; verify SDK names before first use.
* **Vector store:** local TF-IDF instead of a hosted vector store, to keep the demo dependency-free.
* **"Today":** pinned to 2026-09-23 (`TRIAGE_AS_OF`) so date rules are reproducible.

## Limitations / next steps
Sequential briefing (parallelise with a bounded pool); embedding-based retrieval; evaluation set scoring
agent vs rule-engine agreement (`missed_rules` is the start of this); resumable runs from checkpoints;
role-based approval for downgrading overrides.
