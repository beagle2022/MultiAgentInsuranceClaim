"""Azure AI Foundry Agent Service backend (azure-ai-agents SDK) - same interface as ClaudeBackend.

STATUS: reference implementation, NOT exercised in this build (Claude was used as the LLM backend
while Foundry access is pending). It is written against azure-ai-agents 1.x; verify names against
the installed SDK version before the first run. Enable with TRIAGE_BACKEND=foundry and:

    PROJECT_ENDPOINT=https://<resource>.services.ai.azure.com/api/projects/<project>
    FOUNDRY_MODEL_DEPLOYMENT=<your model deployment name>
    FOUNDRY_HOSTED_FILE_SEARCH=1   # use a Foundry vector store + FileSearchTool for the KB

Mapping:  AgentSpec -> create_agent (function tools from our JSON schemas)
          Thread    -> threads.create (one Foundry thread per local Thread id)
          run()     -> messages.create + runs.create, polling; requires_action -> execute_tool ->
                       runs.submit_tool_outputs (manual loop so our tracing/timeouts still apply)
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

from azure.ai.agents import AgentsClient
from azure.ai.agents.models import (FilePurpose, FileSearchTool, FunctionDefinition, FunctionToolDefinition,
                                    ListSortOrder, MessageRole, RequiredFunctionToolCall, RunStatus,
                                    SubmitToolOutputsAction, ToolOutput)
from azure.identity import DefaultAzureCredential

from .backends import execute_tool
from .config import Settings
from .memory import Thread
from .runtime import AgentError, AgentSpec, RunResult
from .tracing import TRACER

KB_TOOL = "search_underwriting_knowledge"
_ACTIVE = {RunStatus.QUEUED, RunStatus.IN_PROGRESS, RunStatus.REQUIRES_ACTION}


class FoundryBackend:
    name = "foundry"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = AgentsClient(endpoint=os.environ["PROJECT_ENDPOINT"], credential=DefaultAzureCredential())
        self.model = os.environ.get("FOUNDRY_MODEL_DEPLOYMENT", "gpt-4o")
        self.hosted_kb = os.getenv("FOUNDRY_HOSTED_FILE_SEARCH") == "1"
        self._agents: dict[str, str] = {}
        self._threads: dict[str, str] = {}
        self._file_search: FileSearchTool | None = None

    # ---------------------------------------------------------------- provisioning (in code, not portal)
    def _kb_tool(self) -> FileSearchTool:
        if self._file_search is None:
            files = [self.client.files.upload_and_poll(file_path=str(p), purpose=FilePurpose.AGENTS)
                     for p in sorted(self.settings.knowledge_dir.glob("*.md"))]
            vs = self.client.vector_stores.create_and_poll(file_ids=[f.id for f in files], name="underwriting-kb")
            TRACER.event("foundry.vector_store", id=vs.id, files=len(files))
            self._file_search = FileSearchTool(vector_store_ids=[vs.id])
        return self._file_search

    def _agent_id(self, spec: AgentSpec) -> str:
        if spec.name in self._agents:
            return self._agents[spec.name]
        use_hosted = self.hosted_kb and spec.tool(KB_TOOL) is not None
        tools: list[Any] = [FunctionToolDefinition(function=FunctionDefinition(
            name=t.name, description=t.description, parameters=t.input_schema))
            for t in spec.tools if not (use_hosted and t.name == KB_TOOL)]
        instructions, resources = spec.instructions, None
        if use_hosted:
            fs = self._kb_tool()
            tools += fs.definitions
            resources = fs.resources
            instructions += f"\nUse the file_search tool wherever these instructions mention {KB_TOOL}."
        if spec.output_tool:
            instructions += f"\nYou MUST finish by calling {spec.output_tool}."
        agent = self.client.create_agent(model=self.model, name=spec.name, instructions=instructions,
                                         tools=tools, tool_resources=resources)
        TRACER.event("foundry.agent_created", agent=spec.name, id=agent.id, tools=len(tools))
        self._agents[spec.name] = agent.id
        return agent.id

    # ---------------------------------------------------------------- run
    def run(self, agent: AgentSpec, thread: Thread, user_input: str) -> RunResult:
        try:
            agent_id = self._agent_id(agent)
            tid = self._threads.get(thread.id) or self.client.threads.create().id
            self._threads[thread.id] = tid
            self.client.messages.create(thread_id=tid, role="user", content=user_input)
            run = self.client.runs.create(thread_id=tid, agent_id=agent_id)
        except Exception as exc:
            raise AgentError(f"{agent.name}: Foundry call failed: {exc}") from exc

        output, calls = None, []
        deadline = time.monotonic() + self.settings.llm_timeout_s * 3
        while run.status in _ACTIVE:
            if time.monotonic() > deadline:
                self.client.runs.cancel(thread_id=tid, run_id=run.id)
                raise AgentError(f"{agent.name}: run timed out")
            if run.status == RunStatus.REQUIRES_ACTION and isinstance(run.required_action, SubmitToolOutputsAction):
                outs = []
                for tc in run.required_action.submit_tool_outputs.tool_calls:
                    if not isinstance(tc, RequiredFunctionToolCall):
                        continue
                    args = json.loads(tc.function.arguments or "{}")
                    calls.append({"tool": tc.function.name, "input": args})
                    if tc.function.name == agent.output_tool:
                        output, content = args, "Recorded."
                    else:
                        content, _ = execute_tool(agent, tc.function.name, args)
                    outs.append(ToolOutput(tool_call_id=tc.id, output=content))
                run = self.client.runs.submit_tool_outputs(thread_id=tid, run_id=run.id, tool_outputs=outs)
                continue
            time.sleep(0.5)
            run = self.client.runs.get(thread_id=tid, run_id=run.id)

        if run.status != RunStatus.COMPLETED:
            raise AgentError(f"{agent.name}: run ended {run.status}: {getattr(run, 'last_error', None)}")
        text = ""
        for m in self.client.messages.list(thread_id=tid, order=ListSortOrder.DESCENDING):
            if m.role == MessageRole.AGENT:
                text = "\n".join(t.text.value for t in m.text_messages)
                break
        # mirror into the local thread so session persistence works the same as with Claude
        thread.messages.append({"role": "user", "content": user_input})
        thread.messages.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
        return RunResult(agent.name, text, output, calls, rounds=len(calls))

    def close(self) -> None:
        for aid in self._agents.values():
            self.client.delete_agent(aid)
