"""Knowledge grounding: a small local vector store over markdown docs + a rule engine.

* `KnowledgeBase.search()` - TF-IDF vectors + cosine similarity over `## ` sections. This is
  what the Coverage agent calls through the `search_underwriting_knowledge` tool; the rules are
  NOT in any prompt. (Foundry equivalent: FileSearchTool over a vector store; see
  foundry_backend.py. Swapping in embeddings/Azure AI Search only changes this class.)
* `RuleEngine` - parses the machine-readable annotation embedded in each rule section and
  evaluates it against computed claim facts. Used as a guardrail on the LLM's decision and as
  the offline fallback. One source of truth: the same markdown file feeds both.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ACTION_SEVERITY = {"auto_approve": 0, "request_more_documentation": 1, "route_to_investigator": 2}
ACTIONS = list(ACTION_SEVERITY)

_TOKEN = re.compile(r"[a-z0-9_]+")
_RULE_ANN = re.compile(r"<!--\s*rule:\s*(\{.*?\})\s*-->", re.S)
_STOP = set("a an the of to in on for is are be by or and with within any must should it its this that as at "
            "from not no only other claim claims policy".split())


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


@dataclass
class Chunk:
    doc_id: str       # e.g. UW-06
    title: str
    text: str         # prose shown to the model (annotation stripped)
    source: str
    annotation: dict[str, Any] | None


class KnowledgeBase:
    def __init__(self, knowledge_dir: Path) -> None:
        self.chunks: list[Chunk] = []
        for path in sorted(knowledge_dir.glob("*.md")):
            self._ingest(path)
        if not self.chunks:
            raise RuntimeError(f"No knowledge documents found in {knowledge_dir}")
        self._build_index()

    def _ingest(self, path: Path) -> None:
        sections = re.split(r"^## ", path.read_text(encoding="utf-8"), flags=re.M)[1:]
        for sec in sections:
            title, _, body = sec.partition("\n")
            ann_match = _RULE_ANN.search(body)
            annotation = json.loads(ann_match.group(1)) if ann_match else None
            prose = _RULE_ANN.sub("", body).strip()
            doc_id = title.split()[0]
            self.chunks.append(Chunk(doc_id, title.strip(), prose, path.name, annotation))

    def _build_index(self) -> None:
        docs = [Counter(_tokens(c.title + " " + c.title + " " + c.text)) for c in self.chunks]
        n = len(docs)
        df = Counter(t for d in docs for t in d)
        self._idf = {t: math.log((1 + n) / (1 + c)) + 1 for t, c in df.items()}
        self._vecs = [self._vectorise(d) for d in docs]

    def _vectorise(self, tf: Counter) -> dict[str, float]:
        v = {t: (1 + math.log(c)) * self._idf.get(t, 0.0) for t, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    def search(self, query: str, top_k: int = 4) -> list[dict[str, Any]]:
        q = self._vectorise(Counter(_tokens(query)))
        scored = sorted(
            ((sum(w * vec.get(t, 0.0) for t, w in q.items()), c) for vec, c in zip(self._vecs, self.chunks)),
            key=lambda x: x[0], reverse=True)
        return [{"id": c.doc_id, "title": c.title, "source": c.source, "score": round(s, 3), "text": c.text}
                for s, c in scored[:top_k] if s > 0]

    def get(self, doc_id: str) -> Chunk | None:
        return next((c for c in self.chunks if c.doc_id == doc_id), None)

    @property
    def rules(self) -> list[dict[str, Any]]:
        return [c.annotation for c in self.chunks if c.annotation]


_OPS = {"<": lambda a, b: a < b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b,
        ">=": lambda a, b: a >= b, "==": lambda a, b: a == b}


class RuleEngine:
    def __init__(self, kb: KnowledgeBase) -> None:
        self.rules = kb.rules

    def evaluate(self, facts: dict[str, Any]) -> list[dict[str, Any]]:
        hits = []
        for rule in self.rules:
            evidence = []
            for metric, op, value in rule["when"]:
                actual = facts.get(metric)
                if actual is None or not _OPS[op](actual, value):
                    break
                evidence.append(f"{metric}={actual} ({op} {value})")
            else:
                hits.append({"rule_id": rule["id"], "action": rule["action"], "evidence": "; ".join(evidence)})
        return hits

    @staticmethod
    def decide(hits: list[dict[str, Any]]) -> str:
        return max((h["action"] for h in hits), key=ACTION_SEVERITY.__getitem__, default="auto_approve")
