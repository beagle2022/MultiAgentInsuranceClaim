"""Lightweight tracing: console + JSONL span log, with optional OpenTelemetry export.

Every agent run, tool call, routine step, routing decision and guardrail override goes
through `span()` / `event()`. If `opentelemetry-sdk` is installed and TRIAGE_OTEL=1, the same
spans are emitted to OTel (console exporter by default; point OTEL_EXPORTER_OTLP_ENDPOINT
at a collector, or swap in azure-monitor-opentelemetry for Foundry/App Insights).
"""
from __future__ import annotations

import contextvars
import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

log = logging.getLogger("triage")

_current_span: contextvars.ContextVar[str | None] = contextvars.ContextVar("span", default=None)
_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar("trace", default="-")


def _short(v: Any, n: int = 160) -> Any:
    if isinstance(v, (int, float, bool)) or v is None:
        return v
    s = v if isinstance(v, str) else json.dumps(v, default=str)
    return s if len(s) <= n else s[: n - 3] + "..."


class Tracer:
    def __init__(self) -> None:
        self._sink: Path | None = None
        self._otel = None
        if os.getenv("TRIAGE_OTEL") == "1":
            try:
                from opentelemetry import trace
                from opentelemetry.sdk.trace import TracerProvider
                from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

                provider = TracerProvider()
                provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
                trace.set_tracer_provider(provider)
                self._otel = trace.get_tracer("claims-triage")
            except ImportError:
                log.warning("TRIAGE_OTEL=1 but opentelemetry-sdk is not installed; using JSONL only")

    def configure(self, log_dir: Path, level: int = logging.INFO) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        self._sink = log_dir / f"trace-{time.strftime('%Y%m%d')}.jsonl"
        if not log.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s %(levelname)-5s %(message)s", "%H:%M:%S"))
            log.addHandler(h)
        log.setLevel(level)
        log.propagate = False

    def new_trace(self) -> str:
        tid = uuid.uuid4().hex[:12]
        _trace_id.set(tid)
        return tid

    def _write(self, record: dict[str, Any]) -> None:
        if self._sink:
            with self._sink.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")

    def event(self, name: str, **attrs: Any) -> None:
        rec = {"ts": time.time(), "trace": _trace_id.get(), "parent": _current_span.get(),
               "kind": "event", "name": name, **attrs}
        self._write(rec)
        log.info("%-24s %s", name, " ".join(f"{k}={_short(v)}" for k, v in attrs.items()))

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[dict[str, Any]]:
        span_id = uuid.uuid4().hex[:8]
        parent = _current_span.get()
        token = _current_span.set(span_id)
        result: dict[str, Any] = {}  # caller can attach outputs: `with span(...) as s: s["x"] = 1`
        start = time.perf_counter()
        status = "ok"
        otel_cm = self._otel.start_as_current_span(name) if self._otel else None
        otel_span = otel_cm.__enter__() if otel_cm else None
        log.debug("> %s %s", name, attrs)
        try:
            yield result
        except Exception as exc:
            status = f"error:{type(exc).__name__}"
            result.setdefault("error", str(exc))
            raise
        finally:
            ms = round((time.perf_counter() - start) * 1000, 1)
            rec = {"ts": time.time(), "trace": _trace_id.get(), "span": span_id, "parent": parent,
                   "kind": "span", "name": name, "status": status, "ms": ms, **attrs, **result}
            self._write(rec)
            if otel_span is not None:
                for k, v in {**attrs, **result, "status": status}.items():
                    otel_span.set_attribute(f"triage.{k}", _short(v, 1000) if not isinstance(v, (int, float, bool)) else v)
                otel_cm.__exit__(None, None, None)
            _current_span.reset(token)
            lvl = logging.INFO if status == "ok" else logging.WARNING
            log.log(lvl, "%-24s %s (%sms) %s", name, status, ms,
                    " ".join(f"{k}={_short(v)}" for k, v in result.items()))


TRACER = Tracer()
