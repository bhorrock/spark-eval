"""Gates 6 and 7. Latency and throughput under concurrency, then a soak.

Run these in exclusive mode. With another vLLM instance sharing the GPU the
numbers measure contention, not the candidate.
"""
from __future__ import annotations

import asyncio
import json
import statistics
import time

import httpx

from .. import filler
from ..client import Client, StreamAccumulator, build_payload

QUESTION = "\n\n---\n\nSummarise the notes above as a numbered list of distinct failure modes."


def pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return round(s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))], 4)


async def one_request(http: httpx.AsyncClient, client: Client, prompt: str, max_tokens: int) -> dict:
    payload = build_payload(client.model, [{"role": "user", "content": prompt}], stream=True,
                            temperature=0, max_tokens=max_tokens, ignore_eos=True)
    t0 = time.perf_counter()
    acc = StreamAccumulator(t0)
    try:
        async with http.stream("POST", f"{client.base_url}/chat/completions", json=payload) as r:
            if r.status_code != 200:
                await r.aread()
                return {"error": f"{r.status_code}: {r.text[:200]}"}
            async for line in r.aiter_lines():
                acc.feed(line)
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        return {"error": f"{type(e).__name__}: {e}"}
    res = acc.done()
    out_tokens = res.usage.get("completion_tokens") or len(res.token_times)
    decode_s = res.latency - (res.ttft or 0)
    return {"ttft": res.ttft, "latency": res.latency, "prompt_tokens": res.usage.get("prompt_tokens"),
            "completion_tokens": out_tokens,
            "decode_tps": out_tokens / decode_s if decode_s > 0 and out_tokens else None}


async def run_cell(client: Client, prompts: list[str], concurrency: int, max_tokens: int) -> dict:
    sem = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency + 4)
    async with httpx.AsyncClient(timeout=1800, limits=limits, headers=dict(client.http.headers)) as http:
        async def guarded(p):
            async with sem:
                return await one_request(http, client, p, max_tokens)
        t0 = time.perf_counter()
        rows = await asyncio.gather(*(guarded(p) for p in prompts))
        wall = time.perf_counter() - t0
    ok = [r for r in rows if "error" not in r]
    return {
        "requests": len(rows), "errors": len(rows) - len(ok),
        "error_samples": [r["error"] for r in rows if "error" in r][:3],
        "wall_s": round(wall, 2),
        "prompt_tokens_mean": round(statistics.mean(r["prompt_tokens"] for r in ok if r["prompt_tokens"]), 0) if ok and any(r["prompt_tokens"] for r in ok) else None,
        "ttft_p50": pct([r["ttft"] for r in ok if r["ttft"]], 50),
        "ttft_p95": pct([r["ttft"] for r in ok if r["ttft"]], 95),
        "decode_tps_per_req_p50": pct([r["decode_tps"] for r in ok if r["decode_tps"]], 50),
        "aggregate_output_tps": round(sum(r["completion_tokens"] for r in ok) / wall, 2) if ok else None,
    }


def metric_delta(before: dict, after: dict) -> dict:
    keys = [k for k in after if any(s in k for s in ("prefix_cache", "spec_decode", "preemption"))]
    return {k: round(after[k] - before.get(k, 0.0), 2) for k in keys if after[k] != before.get(k, 0.0)}


def prefix_cache_probe(client: Client, tokens: int, log=print) -> dict:
    """Send one long prompt twice. A working prefix cache cuts the second TTFT by a large factor."""
    prompt = filler.build(tokens, seed=424242, count_tokens=client.count_tokens) + QUESTION
    cold = client.chat([{"role": "user", "content": prompt}], stream=True, temperature=0, max_tokens=16)
    warm = client.chat([{"role": "user", "content": prompt}], stream=True, temperature=0, max_tokens=16)
    out = {"tokens": tokens, "ttft_cold": cold.ttft, "ttft_warm": warm.ttft,
           "speedup": round(cold.ttft / warm.ttft, 2) if cold.ttft and warm.ttft else None,
           "error": cold.error or warm.error}
    log(f"  prefix cache probe @{tokens}: cold {cold.ttft and round(cold.ttft, 2)}s, warm {warm.ttft and round(warm.ttft, 2)}s")
    return out


def run(client: Client, input_lens: list[int], concurrencies: list[int], max_tokens: int,
        requests_per_worker: int, probe_tokens: int, log=print) -> dict:
    cells = []
    seed = int(time.time()) % 100000      # fresh text every run, so the cache starts cold
    for n_in in input_lens:
        for c in concurrencies:
            n_req = c * requests_per_worker
            prompts = [filler.build(n_in, seed=seed + i, count_tokens=client.count_tokens) + QUESTION
                       for i in range(n_req)]
            seed += n_req
            before = client.metrics()
            cell = asyncio.run(run_cell(client, prompts, c, max_tokens))
            cell.update({"input_tokens": n_in, "concurrency": c, "server_metric_delta": metric_delta(before, client.metrics())})
            cells.append(cell)
            log(f"  in={n_in:<6} c={c:<2} ttft p50/p95 {cell['ttft_p50']}/{cell['ttft_p95']}s  "
                f"decode {cell['decode_tps_per_req_p50']} tok/s/req  aggregate {cell['aggregate_output_tps']} tok/s  "
                f"errors {cell['errors']}")
    probe = prefix_cache_probe(client, probe_tokens, log) if probe_tokens else None
    return {"passed": all(c["errors"] == 0 for c in cells), "cells": cells, "prefix_cache_probe": probe,
            "max_tokens": max_tokens}


def soak(client: Client, minutes: float, concurrency: int, input_lens: list[int], max_tokens: int, log=print) -> dict:
    """Mixed load for a fixed time. Looks for errors and for latency that drifts upward."""
    deadline = time.time() + minutes * 60
    windows, seed = [], int(time.time()) % 100000
    while time.time() < deadline:
        prompts = [filler.build(input_lens[i % len(input_lens)], seed=seed + i, count_tokens=None) + QUESTION
                   for i in range(concurrency * 2)]
        seed += len(prompts)
        cell = asyncio.run(run_cell(client, prompts, concurrency, max_tokens))
        cell["t"] = time.strftime("%H:%M:%S")
        windows.append(cell)
        log(f"  {cell['t']} ttft p95 {cell['ttft_p95']}s  aggregate {cell['aggregate_output_tps']} tok/s  errors {cell['errors']}")
    errors = sum(w["errors"] for w in windows)
    q = max(1, len(windows) // 4)
    first = statistics.mean(w["aggregate_output_tps"] or 0 for w in windows[:q])
    last = statistics.mean(w["aggregate_output_tps"] or 0 for w in windows[-q:])
    drift = round((last - first) / first, 4) if first else None
    return {"passed": errors == 0 and (drift is None or drift > -0.15), "errors": errors,
            "throughput_drift": drift, "windows": windows}
