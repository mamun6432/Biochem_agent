"""The tool-use loop as a stream of events, independent of the LLM provider.

Developer : Abdullah Al Mamun

    {"type": "text", "text": ...}                                  agent prose
    {"type": "tool_call", "name": ..., "input": {...}}             before the tool runs
    {"type": "tool_result", "name": ..., "input": {...}, "output": ...}
    {"type": "note", "text": ...}                                  loop bookkeeping (nudges, limits)
    {"type": "done"}                                               agent stopped
"""
from __future__ import annotations

import json
from typing import Iterator

from .tools import AgentTools

NUDGE = ("Continue the search using the tools (run_experiment, read_ledger). "
         "When you are done, call finalize with the id of the single best experiment.")


def spec_from_call(inp: dict) -> dict:
    """The spec dict from run_experiment arguments. Lenient, because smaller local models
    often send the spec as a JSON string or flatten its fields into the call itself."""
    spec = inp.get("spec")
    if isinstance(spec, str):
        spec = json.loads(spec)  # JSONDecodeError goes back to the model as a tool error
    if spec is None:
        spec = {k: v for k, v in inp.items() if k != "reflection"}
    if not isinstance(spec, dict):
        raise ValueError("spec must be a JSON object")
    return spec


def _dispatch(tools: AgentTools):
    def run_experiment(**inp):
        return tools.run_experiment(spec_from_call(inp), inp.get("reflection", ""))

    return {"get_data_card": lambda **_: tools.get_data_card(),
            "run_experiment": run_experiment,
            "read_ledger": lambda **_: tools.read_ledger(),
            "finalize": lambda experiment_id, **_: tools.finalize(int(experiment_id))}


def _failed(out) -> bool:
    return isinstance(out, dict) and ("error" in out or out.get("status") == "rejected")


def agent_events(chat, tools: AgentTools, context: str, max_turns: int = 200,
                 nudges: int = 2, max_consecutive_failures: int = 8) -> Iterator[dict]:
    dispatch = _dispatch(tools)
    chat.user(f"Prior analysis results:\n{context}\n\nBegin the biomarker model search.")
    finalized, failures = False, 0

    for _ in range(max_turns):
        turn = chat.step()
        for text in turn.texts:
            yield {"type": "text", "text": text}
        if not turn.calls:
            if finalized or nudges <= 0:
                yield {"type": "done"}
                return
            nudges -= 1
            yield {"type": "note", "text": "Model stopped without calling finalize; nudging it to continue."}
            chat.user(NUDGE)
            continue
        results = []
        for c in turn.calls:
            yield {"type": "tool_call", "name": c.name, "input": c.input}
            fn = dispatch.get(c.name)
            if c.error:
                out = {"error": c.error}
            elif fn is None:
                out = {"error": f"unknown tool {c.name!r}; available: {', '.join(dispatch)}"}
            else:
                try:
                    out = fn(**c.input)
                except Exception as e:  # surface errors to the agent, don't crash
                    out = {"error": str(e)}
            if c.name == "finalize" and not (isinstance(out, dict) and "error" in out):
                finalized = True
            failures = failures + 1 if _failed(out) else 0
            yield {"type": "tool_result", "name": c.name, "input": c.input, "output": out}
            results.append((c, json.dumps(out, default=str)))
        chat.tool_results(results)

        if finalized:  # let the model summarise, then stop; the hold-out has been used
            for text in chat.step().texts:
                yield {"type": "text", "text": text}
            yield {"type": "done"}
            return
        if failures >= max_consecutive_failures:
            yield {"type": "note", "text": f"Stopped: {failures} tool calls in a row failed."}
            yield {"type": "done"}
            return

    yield {"type": "note", "text": f"Stopped after {max_turns} model turns."}
    yield {"type": "done"}
