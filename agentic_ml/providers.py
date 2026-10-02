"""LLM providers behind one small chat interface, so the agent loop is model-agnostic.

Developer : Abdullah Al Mamun
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
import uuid
from dataclasses import dataclass


TEXT_CALL_PREFIX = "text_"  # id marker for calls recovered from reply text


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict
    error: str | None = None  # set when the model's arguments could not be parsed


@dataclass
class Turn:
    texts: list[str]
    calls: list[ToolCall]


@dataclass(frozen=True)
class Preset:
    label: str
    kind: str                       # "anthropic" | "openai" (OpenAI-compatible)
    base_url: str | None = None
    key_env: str | None = None      # conventional env var name for the key
    needs_key: bool = True
    models: tuple[str, ...] = ()    # suggestions; users can type any model ID


PRESETS: dict[str, Preset] = {
    "ollama": Preset("Ollama (local)", "openai", "http://localhost:11434/v1", None, False,
                     ("gpt-oss:20b",)),
    "anthropic": Preset("Anthropic (Claude)", "anthropic", None, "ANTHROPIC_API_KEY", True,
                        ("claude-sonnet-4-6", "claude-sonnet-4-5", "claude-sonnet-5-5",
                         "claude-haiku-4-5", "claude-opus-5-5")),
    "openai": Preset("OpenAI", "openai", "https://api.openai.com/v1", "OPENAI_API_KEY"),
    "gemini": Preset("Google Gemini", "openai", "https://generativelanguage.googleapis.com/v1beta/openai/",
                     "GEMINI_API_KEY"),
    "openrouter": Preset("OpenRouter", "openai", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "custom": Preset("Other OpenAI-compatible", "openai", "http://localhost:8000/v1", None, False),
}


def ollama_models(base_url: str = PRESETS["ollama"].base_url, timeout: float = 2.0) -> list[str]:
    """Names of models pulled into a local Ollama server ([] if it isn't reachable)."""
    root = base_url.rstrip("/").removesuffix("/v1")
    try:
        with urllib.request.urlopen(f"{root}/api/tags", timeout=timeout) as r:
            return [m["name"] for m in json.load(r).get("models", [])]
    except Exception:
        return []


class AnthropicChat:
    def __init__(self, model: str, api_key: str | None, system: str, tools: list[dict],
                 max_tokens: int = 4000):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model, self.system, self.tools, self.max_tokens = model, system, tools, max_tokens
        self.messages: list[dict] = []

    def user(self, text: str):
        self.messages.append({"role": "user", "content": text})

    def step(self) -> Turn:
        resp = self.client.messages.create(model=self.model, max_tokens=self.max_tokens,
                                           system=self.system, messages=self.messages,
                                           **({"tools": self.tools} if self.tools else {}))
        self.messages.append({"role": "assistant", "content": resp.content})
        return Turn([b.text for b in resp.content if b.type == "text"],
                    [ToolCall(b.id, b.name, b.input) for b in resp.content if b.type == "tool_use"])

    def tool_results(self, results: list[tuple[ToolCall, str]]):
        self.messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": c.id, "content": out} for c, out in results]})


class OpenAICompatChat:
    def __init__(self, model: str, api_key: str | None, base_url: str, system: str, tools: list[dict]):
        from openai import OpenAI
        # local servers (Ollama, vLLM, LM Studio) accept any non-empty key
        self.client = OpenAI(api_key=api_key or "not-needed", base_url=base_url)
        self.model = model
        self.tools = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                        "parameters": t["input_schema"]}} for t in tools]
        self.messages: list[dict] = [{"role": "system", "content": system}]

    def user(self, text: str):
        self.messages.append({"role": "user", "content": text})

    def step(self) -> Turn:
        resp = self.client.chat.completions.create(model=self.model, messages=self.messages,
                                                   **({"tools": self.tools} if self.tools else {}))
        m = resp.choices[0].message
        calls, raw_calls = [], []
        for tc in m.tool_calls or []:
            tid = tc.id or f"call_{uuid.uuid4().hex[:12]}"
            raw = tc.function.arguments or "{}"
            try:
                args, err = (json.loads(raw) if isinstance(raw, str) else raw), None
            except json.JSONDecodeError as e:
                args, err = {}, f"tool arguments were not valid JSON ({e}); resend the call"
            calls.append(ToolCall(tid, tc.function.name, args if isinstance(args, dict) else {}, err))
            raw_calls.append({"id": tid, "type": "function",
                              "function": {"name": tc.function.name,
                                           "arguments": raw if isinstance(raw, str) else json.dumps(raw)}})
        entry = {"role": "assistant", "content": m.content or ""}
        if raw_calls:
            entry["tool_calls"] = raw_calls
        else:
            calls = self._calls_from_text(m.content or "")
        self.messages.append(entry)
        return Turn([m.content] if m.content and m.content.strip() else [], calls)

    def _calls_from_text(self, text: str) -> list[ToolCall]:
        """Small local models sometimes write the call as JSON in their reply instead of
        using tool_calls, e.g. ```json {"name": "finalize", "arguments": {...}}```."""
        names = {t["function"]["name"] for t in self.tools}
        blobs = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S) or [text.strip()]
        calls = []
        for blob in blobs:
            try:
                d = json.loads(blob)
            except json.JSONDecodeError:
                continue
            args = d.get("arguments", d.get("parameters")) if isinstance(d, dict) else None
            if d.get("name") in names and isinstance(args, dict):
                calls.append(ToolCall(f"{TEXT_CALL_PREFIX}{uuid.uuid4().hex[:12]}", d["name"], args))
        return calls

    def tool_results(self, results: list[tuple[ToolCall, str]]):
        for c, out in results:
            if c.id.startswith(TEXT_CALL_PREFIX):  # no tool_call to attach to; reply as the user
                self.messages.append({"role": "user", "content": f"Result of {c.name}: {out}"})
            else:
                self.messages.append({"role": "tool", "tool_call_id": c.id, "content": out})


def resolve_key(provider: str, api_key: str | None, allow_env: bool = True) -> str | None:
    """The user's key if given, else (when allowed) the provider's conventional env var."""
    if api_key:
        return api_key
    env = PRESETS[provider].key_env
    return os.environ.get(env) if (allow_env and env) else None


def make_chat(provider: str, model: str, system: str, tools: list[dict], api_key: str | None = None,
              base_url: str | None = None):
    p = PRESETS[provider]
    if p.needs_key and not api_key:
        raise ValueError(f"{p.label} needs an API key")
    if p.kind == "anthropic":
        return AnthropicChat(model, api_key, system, tools)
    return OpenAICompatChat(model, api_key, base_url or p.base_url, system, tools)
