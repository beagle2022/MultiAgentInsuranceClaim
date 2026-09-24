"""A scripted stand-in for anthropic.Anthropic so the real ClaudeBackend loop runs in tests."""
from __future__ import annotations

import itertools
from types import SimpleNamespace as NS

_ids = itertools.count()


def text(t):
    return NS(type="text", text=t)


def tool(name, **inp):
    return NS(type="tool_use", id=f"toolu_{next(_ids)}", name=name, input=inp)


def resp(*blocks):
    stop = "tool_use" if any(b.type == "tool_use" for b in blocks) else "end_turn"
    return NS(content=list(blocks), stop_reason=stop, usage=NS(input_tokens=10, output_tokens=5))


class FakeClient:
    """`script(agent_name, round_no, kwargs) -> response`; records every request."""

    def __init__(self, script):
        self.script, self.requests = script, []
        self.messages = self

    def create(self, **kw):
        self.requests.append(kw)
        agent = kw["system"].split("\n", 1)[0]
        rnd = sum(1 for r in self.requests if r["system"] == kw["system"]
                  and r["messages"][0] == kw["messages"][0])
        return self.script(agent, rnd, kw)
