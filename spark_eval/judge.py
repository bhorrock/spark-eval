"""Blind pairwise judging of quality-gate answers by a stronger model.

Each pair is judged twice with the order swapped. A win only counts when both
orders agree, which cancels the judge's position bias.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .client import Client

PROMPT = """You are grading two answers to the same task from a working software developer.
Judge correctness first, then whether the answer could be used as written. Ignore length and style.

TASK:
{task}

ANSWER 1:
{one}

ANSWER 2:
{two}

Reply with JSON only: {{"winner": "1" | "2" | "tie", "reason": "<one sentence>"}}"""


def verdict(judge: Client, task: str, one: str, two: str) -> str:
    res = judge.chat([{"role": "user", "content": PROMPT.format(task=task, one=one, two=two)}], temperature=0, max_tokens=2000)
    m = re.search(r"\{.*\}", res.content or "", re.S)
    try:
        return str(json.loads(m.group(0)).get("winner", "tie")) if m else "tie"
    except json.JSONDecodeError:
        return "tie"


def run(judge: Client, run_a: Path, run_b: Path, log=print) -> dict:
    def answers(d):
        rows = json.loads((d / "quality.json").read_text())["rows"]
        return {r["id"]: r for r in rows if r["context"] == 0 and r["rep"] == 0}
    a, b = answers(run_a), answers(run_b)
    tally, detail = {"a": 0, "b": 0, "tie": 0}, []
    for cid in sorted(set(a) & set(b)):
        task = a[cid].get("task") or cid
        ta, tb = a[cid]["response"]["content"], b[cid]["response"]["content"]
        first, second = verdict(judge, task, ta, tb), verdict(judge, task, tb, ta)
        winner = "a" if (first, second) == ("1", "2") else "b" if (first, second) == ("2", "1") else "tie"
        tally[winner] += 1
        detail.append({"id": cid, "winner": winner})
        log(f"  {cid}: {winner}")
    return {"a": str(run_a), "b": str(run_b), "tally": tally, "cases": detail}
