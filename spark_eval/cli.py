from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

from . import compare as cmp
from . import judge as judge_mod
from . import manifest
from .client import Client
from .gates import bringup, cases, perf

ROOT = Path(__file__).resolve().parent.parent
CASE_GATES = {"smoke": "smoke.yaml", "tools": "tools.yaml", "longctx": "longctx.yaml", "quality": "quality"}
ORDER = ["bringup", "smoke", "tools", "longctx", "quality", "perf", "soak"]


def ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def floors_ok(summary: dict, floor: dict) -> list[str]:
    fails = []
    for ctx, s in summary.items():
        if s["pass_rate"] < floor.get("min_pass_rate", 0):
            fails.append(f"pass rate {s['pass_rate']} at context {ctx} is below {floor['min_pass_rate']}")
        if s["malformed_rate"] > floor.get("max_malformed_rate", 1):
            fails.append(f"malformed rate {s['malformed_rate']} at context {ctx} is above {floor['max_malformed_rate']}")
        if s["error_rate"] > floor.get("max_error_rate", 1):
            fails.append(f"error rate {s['error_rate']} at context {ctx} is above {floor['max_error_rate']}")
    return fails


def run_case_gate(client, gate, suites_dir, contexts, repeat, only, floor, stream_modes=None):
    target = suites_dir / CASE_GATES[gate]
    files = sorted(target.glob("*.yaml")) if target.is_dir() else [target]
    rows = []
    for f in files:
        rows += cases.run_suite(client, f, contexts, repeat, only, stream_modes=stream_modes)["rows"]
    summary = cases.summarize(rows) if rows else {}
    fails = floors_ok(summary, floor) if rows else ["suite has no cases"]
    by_stream = {("streamed" if k == "True" else "blocking"): v for k, v in cases.summarize(rows, "stream").items()} if rows else {}
    if len(by_stream) == 2:
        for mode, v in by_stream.items():
            print(f"   {mode:9} pass {v['pass_rate']}  malformed {v['malformed_rate']}")
    return {"passed": not fails, "failures": fails, "by_context": summary, "by_stream": by_stream, "rows": rows}


def cmd_run(a) -> int:
    client = Client(a.base_url, a.model, a.api_key or os.environ.get("OPENAI_API_KEY"))
    rules = yaml.safe_load(Path(a.rules).read_text())
    suites_dir = Path(a.suites)
    man = manifest.build(client, a.label, a.mode, Path(a.recipe) if a.recipe else None, a.image,
                         a.revision, a.notes, suites_dir, a.launch, [Path(x) for x in a.artifact])
    out = manifest.run_dir(Path(a.results), man)
    print(f"run dir: {out}")
    gates = [g for g in ORDER if g in a.gates.split(",")]
    if a.mode == "lab" and ({"perf", "soak"} & set(gates)) and not a.force:
        print("perf and soak need --mode exclusive. Another model on the GPU skews the numbers. Use --force to override.")
        return 2
    contexts = {"smoke": [0], "tools": ints(a.tool_contexts), "longctx": ints(a.longctx_contexts), "quality": [0]}
    status = 0
    for gate in gates:
        print(f"\n== {gate}")
        if gate == "bringup":
            res = bringup.run(client, a.min_context, Path(a.server_log) if a.server_log else None)
        elif gate in CASE_GATES:
            res = run_case_gate(client, gate, suites_dir, contexts[gate], a.repeat, a.only, (rules.get("floors") or {}).get(gate, {}),
                                [False, True] if gate == "tools" and a.tool_stream == "both" else None)
        elif gate == "perf":
            res = perf.run(client, ints(a.input_lens), ints(a.concurrency), a.max_tokens, a.requests_per_worker, a.probe_tokens)
        else:
            res = perf.soak(client, a.soak_minutes, a.soak_concurrency, ints(a.input_lens), a.max_tokens)
        (out / f"{gate}.json").write_text(json.dumps(res, indent=2))
        print(f"-- {gate}: {'PASS' if res['passed'] else 'FAIL'}" + "".join(f"\n   {f}" for f in res.get("failures", [])))
        if not res["passed"]:
            status = 1
            if not a.keep_going:
                print("stopping at the first failed gate (use --keep-going to run the rest)")
                break
    return status


def cmd_compare(a) -> int:
    rules = yaml.safe_load(Path(a.rules).read_text())
    lines, ok = cmp.compare(Path(a.results), a.candidate, a.incumbent, rules)
    print("\n".join(lines))
    return 0 if ok else 1


def cmd_judge(a) -> int:
    root = Path(a.results)
    runs = [cmp.newest_gate_files(root, label).get("quality") for label in (a.candidate, a.incumbent)]
    if not all(runs):
        print("both labels need a quality run first")
        return 2
    j = Client(a.judge_url, a.judge_model, a.judge_key or os.environ.get("JUDGE_API_KEY"))
    res = judge_mod.run(j, runs[0].parent, runs[1].parent)
    (runs[0].parent / f"judge_vs_{a.incumbent}.json").write_text(json.dumps(res, indent=2))
    print(f"\n{a.candidate} wins {res['tally']['a']}, {a.incumbent} wins {res['tally']['b']}, ties {res['tally']['tie']}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="spark-eval")
    sub = p.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--results", default=str(ROOT / "results"))
    common.add_argument("--rules", default=str(ROOT / "promotion.yaml"))

    r = sub.add_parser("run", parents=[common], help="run gates against an endpoint")
    r.add_argument("--label", required=True, help="short name for the model+recipe, e.g. granite-4.2-30b")
    r.add_argument("--base-url", required=True, help="e.g. http://spark:8001/v1")
    r.add_argument("--model", required=True, help="served model name")
    r.add_argument("--api-key")
    r.add_argument("--gates", default="bringup,smoke,tools,longctx,quality")
    r.add_argument("--mode", choices=["lab", "exclusive"], default="lab")
    r.add_argument("--recipe", help="path to the sparkrun recipe file that launched the server")
    r.add_argument("--tool-stream", choices=["both", "suite"], default="both",
                   help="both: run every tools case blocking and streamed. suite: use each case's own setting")
    r.add_argument("--artifact", action="append", default=[],
                   help="extra file that is part of the system under test, e.g. a tool parser plugin or chat template. Repeatable")
    r.add_argument("--launch", help="the exact sparkrun command used, including -o overrides")
    r.add_argument("--image", help="container image, pinned by digest")
    r.add_argument("--revision", help="model revision (HF commit SHA)")
    r.add_argument("--server-log", help="saved server log, e.g. from `sparkrun logs`")
    r.add_argument("--notes")
    r.add_argument("--suites", default=str(ROOT / "suites"))
    r.add_argument("--min-context", type=int, default=131072)
    r.add_argument("--tool-contexts", default="0,32000,100000")
    r.add_argument("--longctx-contexts", default="32000,64000,120000")
    r.add_argument("--repeat", type=int, default=1)
    r.add_argument("--only", nargs="*", help="case ids to run")
    r.add_argument("--keep-going", action="store_true")
    r.add_argument("--force", action="store_true")
    r.add_argument("--input-lens", default="2000,16000,64000")
    r.add_argument("--concurrency", default="1,2,4,8")
    r.add_argument("--max-tokens", type=int, default=256)
    r.add_argument("--requests-per-worker", type=int, default=3)
    r.add_argument("--probe-tokens", type=int, default=16000)
    r.add_argument("--soak-minutes", type=float, default=60)
    r.add_argument("--soak-concurrency", type=int, default=4)
    r.set_defaults(fn=cmd_run)

    c = sub.add_parser("compare", parents=[common], help="candidate versus incumbent, with a verdict")
    c.add_argument("candidate")
    c.add_argument("incumbent")
    c.set_defaults(fn=cmd_compare)

    j = sub.add_parser("judge", parents=[common], help="blind pairwise judging of quality answers")
    j.add_argument("candidate")
    j.add_argument("incumbent")
    j.add_argument("--judge-url", required=True)
    j.add_argument("--judge-model", required=True)
    j.add_argument("--judge-key")
    j.set_defaults(fn=cmd_judge)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
