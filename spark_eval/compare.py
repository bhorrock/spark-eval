"""Diff a candidate against the incumbent and apply promotion.yaml."""
from __future__ import annotations

import json
from pathlib import Path

GATES = ["bringup", "smoke", "tools", "longctx", "quality", "perf", "soak"]


def newest_gate_files(results_root: Path, label: str) -> dict[str, Path]:
    """Perf usually runs in a separate exclusive-mode session, so take the newest file per gate."""
    out = {}
    for gate in GATES:
        hits = sorted((results_root / label).glob(f"*/*/{gate}.json"), key=lambda p: p.parent.name)
        if hits:
            out[gate] = hits[-1]
    return out


def load(results_root: Path, label: str) -> dict:
    return {g: json.loads(p.read_text()) | {"_path": str(p.parent)} for g, p in newest_gate_files(results_root, label).items()}


def overall(gate: dict, key: str) -> float | None:
    rows = gate.get("rows") or []
    if not rows:
        return None
    status = {"pass_rate": "pass", "malformed_rate": "malformed", "error_rate": "error"}[key]
    return sum(r["status"] == status for r in rows) / len(rows)


def find_cell(perf: dict, want: dict) -> dict | None:
    return next((c for c in perf.get("cells", []) if all(c.get(k) == v for k, v in want.items())), None)


def compare(results_root: Path, candidate: str, incumbent: str, rules: dict) -> tuple[list[str], bool]:
    a, b = load(results_root, candidate), load(results_root, incumbent)
    lines, ok = [f"{'':34}{candidate[:22]:>24}{incumbent[:22]:>24}"], True
    for gate, res in a.items():
        if not res.get("passed", True):
            ok = False
            lines.append(f"  FAIL {candidate} did not pass its own {gate} gate: {'; '.join(res.get('failures', [])[:2])}")
    for gate in ("smoke", "tools", "longctx", "quality"):
        if gate not in a or gate not in b:
            lines.append(f"{gate:34}{'run missing on one side':>48}")
            ok = ok and gate == "smoke"
            continue
        for ctx in sorted(set(a[gate]["by_context"]) | set(b[gate]["by_context"]), key=int):
            ca, cb = a[gate]["by_context"].get(ctx, {}), b[gate]["by_context"].get(ctx, {})
            lines.append(f"{gate + ' pass @' + ctx:34}{ca.get('pass_rate', '-'):>24}{cb.get('pass_rate', '-'):>24}")
            if gate == "tools":
                lines.append(f"{gate + ' malformed @' + ctx:34}{ca.get('malformed_rate', '-'):>24}{cb.get('malformed_rate', '-'):>24}")
        for mode in ("blocking", "streamed") if gate == "tools" else ():
            sa, sb = (a[gate].get("by_stream") or {}).get(mode, {}), (b[gate].get("by_stream") or {}).get(mode, {})
            if sa or sb:
                lines.append(f"{'tools malformed, ' + mode:34}{sa.get('malformed_rate', '-'):>24}{sb.get('malformed_rate', '-'):>24}")
        rule = (rules.get("versus_incumbent") or {}).get(gate)
        if rule:
            delta = overall(a[gate], "pass_rate") - overall(b[gate], "pass_rate")
            good = delta >= rule["min_pass_rate_delta"]
            ok = ok and good
            lines.append(f"  {'ok  ' if good else 'FAIL'} {gate} pass-rate delta {delta:+.3f} (rule: >= {rule['min_pass_rate_delta']:+.2f})")
    prule = (rules.get("versus_incumbent") or {}).get("perf")
    if prule and "perf" in a and "perf" in b:
        ca, cb = find_cell(a["perf"], prule["cell"]), find_cell(b["perf"], prule["cell"])
        if ca and cb and ca["ttft_p95"] and cb["ttft_p95"]:
            r_ttft = ca["ttft_p95"] / cb["ttft_p95"]
            r_tps = ca["aggregate_output_tps"] / cb["aggregate_output_tps"]
            lines.append(f"{'perf ' + str(prule['cell']):34}")
            lines.append(f"{'  ttft p95 (s)':34}{ca['ttft_p95']:>24}{cb['ttft_p95']:>24}")
            lines.append(f"{'  aggregate output tok/s':34}{ca['aggregate_output_tps']:>24}{cb['aggregate_output_tps']:>24}")
            for good, text in ((r_ttft <= prule["max_ttft_p95_ratio"], f"ttft p95 ratio {r_ttft:.2f} (rule: <= {prule['max_ttft_p95_ratio']})"),
                               (r_tps >= prule["min_aggregate_output_tps_ratio"], f"throughput ratio {r_tps:.2f} (rule: >= {prule['min_aggregate_output_tps_ratio']})")):
                ok = ok and good
                lines.append(f"  {'ok  ' if good else 'FAIL'} {text}")
        else:
            lines.append(f"perf: no cell matching {prule['cell']} on both sides")
            ok = False
    elif prule:
        lines.append("perf: run missing on one side")
        ok = False
    lines.append("")
    lines.append(f"VERDICT: {'promote' if ok else 'do not promote'} {candidate} over {incumbent}")
    return lines, ok
