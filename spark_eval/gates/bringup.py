"""Gate 1. Is the server up, serving the right model, at the context length we need?"""
from __future__ import annotations

import re
from pathlib import Path

from ..client import Client

LOG_PATTERNS = {
    "kv_cache_tokens": r"GPU KV cache size:\s*([\d,]+)\s*tokens",
    "max_concurrency_at_max_len": r"Maximum concurrency for [\d,]+ tokens per request:\s*([\d.]+)x",
    "weights_gib": r"Model loading took\s*([\d.]+)\s*Gi?B",
    "kv_cache_gib": r"Available KV cache memory:\s*([\d.]+)\s*Gi?B",
}


def parse_server_log(path: Path) -> dict:
    text = path.read_text(errors="replace")
    out = {}
    for key, pattern in LOG_PATTERNS.items():
        hits = re.findall(pattern, text)
        if hits:
            out[key] = float(hits[-1].replace(",", ""))
    return out


def run(client: Client, min_context: int, server_log: Path | None, log=print) -> dict:
    fails, info = [], {}
    if not client.health():
        fails.append("/health did not return 200")
    try:
        served = {m["id"]: m for m in client.models()}
    except Exception as e:                                     # noqa: BLE001
        served, _ = {}, fails.append(f"/v1/models failed: {e}")
    if client.model not in served:
        fails.append(f"model {client.model!r} not served; server has {list(served)}")
    else:
        info["max_model_len"] = served[client.model].get("max_model_len")
        if info["max_model_len"] and info["max_model_len"] < min_context:
            fails.append(f"max_model_len {info['max_model_len']} is below the required {min_context}")
    res = client.chat([{"role": "user", "content": "Reply with the single word: ready"}], temperature=0, max_tokens=512)
    if res.error:
        fails.append(f"first completion failed: {res.error[:200]}")
    info["first_completion_s"] = round(res.latency, 2)
    if server_log:
        info.update(parse_server_log(server_log))
        conc = info.get("max_concurrency_at_max_len")
        if conc is not None and conc < 1:
            fails.append(f"KV cache holds {conc}x of one max-length request. 128K is nominal, not usable.")
    for f in fails:
        log(f"  [fail] {f}")
    return {"passed": not fails, "failures": fails, "info": info}
