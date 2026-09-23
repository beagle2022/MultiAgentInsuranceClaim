"""Claims Triage Assistant - CLI.

  python main.py                                  interactive chat (HITL checkpoints prompt you)
  python main.py --auto-gates --batch data/claims.json          one-shot triage, no prompts
  python main.py --session s1 --ask "why was C2031 flagged?"   follow-up in a saved session
  python main.py --backend offline ...            no LLM: deterministic fallbacks only
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import shutil
import sys

from triage.config import SETTINGS
from triage.routine import AutoGate, CLIGate
from triage.supervisor import Supervisor
from triage.tracing import TRACER


def main() -> int:
    ap = argparse.ArgumentParser(description="Multi-agent Claims Triage Assistant")
    ap.add_argument("--backend", choices=["claude", "offline", "foundry"], help="override TRIAGE_BACKEND")
    ap.add_argument("--session", help="session id to create/resume (short-term memory persisted to disk)")
    ap.add_argument("--batch", help="run one triage on this file and exit")
    ap.add_argument("--ask", help="send one message and exit")
    ap.add_argument("--auto-gates", action="store_true", help="auto-approve HITL checkpoints (non-interactive)")
    ap.add_argument("--reset-memory", action="store_true", help="wipe memory_store/ (re-seeds long-term memory)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles
    settings = dataclasses.replace(SETTINGS, backend=args.backend) if args.backend else SETTINGS
    TRACER.configure(settings.log_dir, logging.DEBUG if args.verbose else logging.INFO)
    if args.reset_memory and settings.store_dir.exists():
        shutil.rmtree(settings.store_dir)

    sup = Supervisor(settings, gate=AutoGate() if args.auto_gates else CLIGate(), session_id=args.session)
    print(f"\nSession {sup.session.id} | backend={sup.backend.name} | turns so far={sup.session.turns}")

    if args.batch:
        print(sup.handle(f"Triage the batch {args.batch}"))
        return 0
    if args.ask:
        print(sup.handle(args.ask))
        return 0

    print("Type a request (e.g. 'triage this batch of claims', 'why was C2031 flagged?'). /exit to quit.")
    while True:
        try:
            msg = input("\nhandler> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not msg:
            continue
        if msg in ("/exit", "/quit"):
            break
        try:
            print("\n" + sup.handle(msg))
        except Exception as exc:  # last-resort guard: keep the session alive
            TRACER.event("supervisor.unhandled", error=repr(exc))
            print(f"\nSomething went wrong handling that request ({type(exc).__name__}: {exc}). Session preserved.")
    print(f"Session saved as '{sup.session.id}'. Resume with --session {sup.session.id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
