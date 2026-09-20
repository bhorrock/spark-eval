"""Generic runner for YAML case suites. Smoke, tools, long-context and quality
gates are all this runner with a different suite and pass rule."""
from __future__ import annotations

import copy
import zlib
from pathlib import Path

import yaml

from .. import checks, filler
from ..client import Client

PAD_HEADER = "Background notes. Ignore them unless a question refers to them.\n\n"


def load_suite(path: Path) -> dict:
    suite = yaml.safe_load(path.read_text())
    suite.setdefault("defaults", {})
    suite.setdefault("tools", {})
    return suite


def resolve_tools(suite: dict, names: list[str] | None) -> list[dict]:
    return [{"type": "function", "function": {"name": n, **suite["tools"][n]}} for n in names or []]


def padded_messages(case: dict, pad_tokens: int, client: Client, seed: int) -> list[dict]:
    """Insert filler (and any needles) ahead of the first user message."""
    messages = copy.deepcopy(case["messages"])
    needles = [(n["depth"], n["text"]) for n in case.get("needles") or []]
    if pad_tokens <= 0 and not needles:
        return messages
    pad = filler.build(pad_tokens, seed=seed, needles=needles, count_tokens=client.count_tokens)
    for m in messages:
        if m["role"] == "user":
            m["content"] = PAD_HEADER + pad + "\n\n---\n\n" + m["content"]
            break
    return messages


def run_suite(client: Client, suite_path: Path, contexts: list[int], repeat: int = 1,
              only: list[str] | None = None, log=print, stream_modes: list[bool] | None = None) -> dict:
    """`stream_modes` forces every case to run once per listed mode. vLLM parsers have
    separate streaming and non-streaming code paths, and a parser loaded as a plugin
    has usually had less testing on the streaming one."""
    suite = load_suite(suite_path)
    rows = []
    for ctx in contexts:
        for case in suite["cases"]:
            if only and case["id"] not in only:
                continue
            if ctx > 0 and case.get("pad") is False:
                continue                      # case opts out of padding; it ran at ctx 0
            if ctx == 0 and case.get("needles") and len(contexts) > 1:
                continue                      # a needle case without a haystack proves nothing
            tools = resolve_tools(suite, case.get("tools"))
            params = {**suite["defaults"], **(case.get("params") or {})}
            for stream, rep in [(m, r) for m in (stream_modes or [case.get("stream", False)]) for r in range(repeat)]:
                seed = zlib.crc32(f"{case['id']}:{ctx}".encode()) % 100000 + rep
                res = client.chat(padded_messages(case, ctx, client, seed), tools=tools,
                                  stream=stream, **params)
                fails = checks.evaluate(res, case.get("expect") or {})
                malformed = [] if res.error else checks.malformed_reasons(res, tools)
                status = "error" if res.error else "malformed" if malformed else "fail" if fails else "pass"
                task = next((m["content"] for m in reversed(case["messages"]) if m["role"] == "user"), "")
                rows.append({"id": case["id"], "task": task, "context": ctx, "stream": stream, "rep": rep, "status": status,
                             "failures": fails, "malformed": malformed,
                             "prompt_tokens": res.usage.get("prompt_tokens"),
                             "latency_s": round(res.latency, 2), "response": res.to_dict()})
                log(f"  [{status:9}] ctx={ctx:<7} {'stream' if stream else 'block '} {case['id']}" + (f"  {(malformed or fails)[0]}" if status != "pass" else ""))
    return {"suite": suite_path.name, "rows": rows, "by_context": summarize(rows)}


def summarize(rows: list[dict], key: str = "context") -> dict:
    out = {}
    for ctx in sorted({r[key] for r in rows}):
        sub = [r for r in rows if r[key] == ctx]
        n = len(sub)
        out[str(ctx)] = {
            "n": n,
            "pass_rate": round(sum(r["status"] == "pass" for r in sub) / n, 4),
            "malformed_rate": round(sum(r["status"] == "malformed" for r in sub) / n, 4),
            "error_rate": round(sum(r["status"] == "error" for r in sub) / n, 4),
        }
    return out
