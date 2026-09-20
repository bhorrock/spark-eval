"""Deterministic checks on a ChatResult. No judge model involved."""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path

import jsonschema

from .client import ChatResult

# Tool-call or reasoning markup that a working parser strips from `content`.
LEAK_PATTERNS = [
    r"<tool_call>", r"</tool_call>", r"<function=", r"<\|tool_call", r"<\|python_tag\|>",
    r"\[TOOL_CALLS\]", r"<tool_response>", r"<think>", r"</think>", r"<\|im_start\|>", r"<\|im_end\|>",
]
LEAK_RE = re.compile("|".join(LEAK_PATTERNS))
JSON_CALL_RE = re.compile(r'\{\s*"name"\s*:\s*"[\w.-]+"\s*,\s*"(arguments|parameters)"\s*:')


def malformed_reasons(res: ChatResult, tools: list[dict]) -> list[str]:
    """Reasons this response breaks the tool-calling protocol. Empty list means clean."""
    reasons = []
    by_name = {t["function"]["name"]: t["function"] for t in tools or []}
    m = LEAK_RE.search(res.content or "")
    if m:
        reasons.append(f"markup leaked into content: {m.group(0)}")
    if tools and not res.tool_calls and JSON_CALL_RE.search(res.content or ""):
        reasons.append("tool call written as JSON text in content")
    for tc in res.tool_calls:
        fn = tc.get("function") or {}
        name = fn.get("name") or ""
        if name not in by_name:
            reasons.append(f"unknown tool name: {name!r}")
            continue
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError as e:
            reasons.append(f"{name}: arguments are not JSON ({e.msg})")
            continue
        if not isinstance(args, dict):
            reasons.append(f"{name}: arguments are not an object")
            continue
        try:
            jsonschema.validate(args, by_name[name].get("parameters") or {"type": "object"})
        except jsonschema.ValidationError as e:
            reasons.append(f"{name}: schema violation: {e.message[:160]}")
    if res.tool_calls and res.finish_reason not in ("tool_calls", None):
        reasons.append(f"finish_reason is {res.finish_reason!r} with tool calls present")
    return reasons


def _args(tc: dict) -> dict:
    try:
        v = json.loads((tc.get("function") or {}).get("arguments") or "{}")
        return v if isinstance(v, dict) else {}
    except json.JSONDecodeError:
        return {}


def _norm(v):
    return v.strip().lower() if isinstance(v, str) else v


def _call_matches(tc: dict, want: dict) -> bool:
    if (tc.get("function") or {}).get("name") != want["name"]:
        return False
    got = _args(tc)
    for k, v in (want.get("args") or {}).items():
        if _norm(got.get(k)) != _norm(v):
            return False
    for k, pattern in (want.get("args_regex") or {}).items():
        if not re.search(pattern, str(got.get(k, "")), re.I):
            return False
    return True


def extract_code(text: str, lang: str | None = None) -> str | None:
    blocks = re.findall(r"```([\w+-]*)\n(.*?)```", text, re.S)
    if not blocks:
        return None
    aliases = {"python": {"python", "py"}, "node": {"javascript", "js", "node"}}.get(lang, {lang})
    for tag, body in blocks:
        if tag.lower() in aliases:
            return body
    return blocks[0][1]


def run_exec(spec: dict, content: str) -> str | None:
    """Run the model's code block plus a test snippet. Returns an error string or None."""
    lang = spec.get("lang", "python")
    code = extract_code(content, lang)
    if code is None:
        return "no code block in answer"
    suffix, cmd = {"python": (".py", ["python3"]), "node": (".js", ["node"])}[lang]
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / f"case{suffix}"
        path.write_text(code + "\n\n" + spec.get("test", ""))
        try:
            p = subprocess.run(cmd + [str(path)], capture_output=True, text=True, timeout=spec.get("timeout", 20))
        except subprocess.TimeoutExpired:
            return "test timed out"
        except FileNotFoundError:
            return f"{cmd[0]} is not installed"
    return None if p.returncode == 0 else f"test failed: {(p.stderr or p.stdout)[-400:]}"


def evaluate(res: ChatResult, expect: dict) -> list[str]:
    """Return failure messages for this case. Empty list means pass."""
    fails = []
    if res.error:
        return [f"request error ({res.status}): {res.error[:300]}"]
    content = res.content or ""
    for s in expect.get("contains") or []:
        if s.lower() not in content.lower():
            fails.append(f"missing text: {s!r}")
    for s in expect.get("not_contains") or []:
        if s.lower() in content.lower():
            fails.append(f"unwanted text: {s!r}")
    for pattern in expect.get("regex") or []:
        if not re.search(pattern, content, re.S | re.I):
            fails.append(f"no match for /{pattern}/")
    if expect.get("content_nonempty") and not content.strip():
        fails.append("content is empty")
    if expect.get("reasoning_present") and not (res.reasoning or "").strip():
        fails.append("no reasoning_content returned (check --reasoning-parser)")
    if "finish_reason" in expect and res.finish_reason != expect["finish_reason"]:
        fails.append(f"finish_reason {res.finish_reason!r}, wanted {expect['finish_reason']!r}")
    if "json_schema" in expect:
        try:
            jsonschema.validate(json.loads(content), expect["json_schema"])
        except json.JSONDecodeError:
            fails.append("content is not JSON")
        except jsonschema.ValidationError as e:
            fails.append(f"content JSON fails schema: {e.message[:160]}")
    if expect.get("no_tool_call") and res.tool_calls:
        names = [tc["function"]["name"] for tc in res.tool_calls]
        fails.append(f"called {names} when no call was needed")
    wanted = expect.get("tool_calls") or ([expect["tool_call"]] if "tool_call" in expect else [])
    if wanted:
        remaining = list(res.tool_calls)
        for want in wanted:
            hit = next((tc for tc in remaining if _call_matches(tc, want)), None)
            if hit is None:
                fails.append(f"no call matching {want}")
            else:
                remaining.remove(hit)
        if expect.get("exact_calls", True) and remaining and not fails:
            fails.append(f"{len(remaining)} extra call(s)")
    if "exec" in expect:
        err = run_exec(expect["exec"], content)
        if err:
            fails.append(err)
    return fails
