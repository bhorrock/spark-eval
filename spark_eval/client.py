"""Minimal OpenAI-compatible client.

Talks only to the public API, so the same suites run against vLLM directly
or against a proxy such as Bifrost. Differences between the two runs point
at the proxy.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class ChatResult:
    content: str = ""
    reasoning: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict = field(default_factory=dict)
    ttft: float | None = None          # seconds to first streamed token
    latency: float = 0.0               # seconds, whole request
    token_times: list[float] = field(default_factory=list)
    error: str | None = None
    status: int | None = None

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d.pop("token_times")
        return d


def root_url(base_url: str) -> str:
    """http://host:8000/v1 -> http://host:8000"""
    return re.sub(r"/v1/?$", "", base_url.rstrip("/"))


def build_payload(model: str, messages: list[dict], tools=None, stream=False, **params) -> dict:
    payload: dict[str, Any] = {"model": model, "messages": messages, "stream": stream}
    if tools:
        payload["tools"] = tools
    if stream:
        payload["stream_options"] = {"include_usage": True}
    payload.update({k: v for k, v in params.items() if v is not None})
    return payload


class StreamAccumulator:
    """Rebuilds a message from chat.completion.chunk deltas."""

    def __init__(self, t0: float):
        self.t0 = t0
        self.res = ChatResult()
        self._calls: dict[int, dict] = {}

    def feed(self, line: str) -> None:
        if not line.startswith("data:"):
            return
        data = line[5:].strip()
        if not data or data == "[DONE]":
            return
        chunk = json.loads(data)
        if chunk.get("usage"):
            self.res.usage = chunk["usage"]
        for choice in chunk.get("choices", []):
            delta = choice.get("delta") or {}
            got_token = False
            if delta.get("content"):
                self.res.content += delta["content"]
                got_token = True
            r = delta.get("reasoning_content") or delta.get("reasoning")
            if r:
                self.res.reasoning += r
                got_token = True
            for tc in delta.get("tool_calls") or []:
                got_token = True
                slot = self._calls.setdefault(
                    tc.get("index", 0),
                    {"id": None, "type": "function", "function": {"name": "", "arguments": ""}},
                )
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["function"]["name"] += fn["name"]
                if fn.get("arguments"):
                    slot["function"]["arguments"] += fn["arguments"]
            if got_token:
                now = time.perf_counter()
                if self.res.ttft is None:
                    self.res.ttft = now - self.t0
                self.res.token_times.append(now)
            if choice.get("finish_reason"):
                self.res.finish_reason = choice["finish_reason"]

    def done(self) -> ChatResult:
        self.res.tool_calls = [self._calls[i] for i in sorted(self._calls)]
        self.res.latency = time.perf_counter() - self.t0
        return self.res


def parse_message(body: dict, t0: float) -> ChatResult:
    res = ChatResult(latency=time.perf_counter() - t0, usage=body.get("usage") or {})
    choice = (body.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    res.content = msg.get("content") or ""
    res.reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    res.tool_calls = msg.get("tool_calls") or []
    res.finish_reason = choice.get("finish_reason")
    return res


class Client:
    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.http = httpx.Client(headers=headers, timeout=timeout)

    def chat(self, messages: list[dict], tools=None, stream: bool = False, **params) -> ChatResult:
        payload = build_payload(self.model, messages, tools, stream, **params)
        url = f"{self.base_url}/chat/completions"
        t0 = time.perf_counter()
        try:
            if not stream:
                r = self.http.post(url, json=payload)
                if r.status_code != 200:
                    return ChatResult(error=r.text[:2000], status=r.status_code, latency=time.perf_counter() - t0)
                res = parse_message(r.json(), t0)
                res.status = 200
                return res
            with self.http.stream("POST", url, json=payload) as r:
                if r.status_code != 200:
                    r.read()
                    return ChatResult(error=r.text[:2000], status=r.status_code, latency=time.perf_counter() - t0)
                acc = StreamAccumulator(t0)
                for line in r.iter_lines():
                    acc.feed(line)
                res = acc.done()
                res.status = 200
                return res
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            return ChatResult(error=f"{type(e).__name__}: {e}", latency=time.perf_counter() - t0)

    def models(self) -> list[dict]:
        r = self.http.get(f"{self.base_url}/models")
        r.raise_for_status()
        return r.json().get("data", [])

    def health(self) -> bool:
        try:
            return self.http.get(f"{root_url(self.base_url)}/health", timeout=10).status_code == 200
        except httpx.HTTPError:
            return False

    def count_tokens(self, text: str) -> int | None:
        """vLLM's /tokenize endpoint. Returns None when the server lacks it."""
        try:
            r = self.http.post(f"{root_url(self.base_url)}/tokenize", json={"model": self.model, "prompt": text})
            if r.status_code == 200:
                return r.json().get("count")
        except httpx.HTTPError:
            pass
        return None

    def metrics(self) -> dict[str, float]:
        """Sum of each Prometheus series from vLLM's /metrics, labels dropped."""
        out: dict[str, float] = {}
        try:
            r = self.http.get(f"{root_url(self.base_url)}/metrics", timeout=10)
            if r.status_code != 200:
                return out
        except httpx.HTTPError:
            return out
        for line in r.text.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            m = re.match(r"^([a-zA-Z_:][\w:]*)(\{[^}]*\})?\s+([-+eE.\d]+|NaN|\+Inf)$", line)
            if m:
                try:
                    out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(3))
                except ValueError:
                    pass
        return out
