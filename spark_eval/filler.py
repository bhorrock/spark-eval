"""Deterministic padding text for long-context cases.

The text is seeded and non-repeating, so prefix caching cannot hide the cost
of a long prompt unless a case reuses the same seed on purpose.
"""
from __future__ import annotations

import random

SUBJECTS = ["The scheduler", "A saved search", "The map stage", "Each vendor bill", "The deploy script",
            "A customer record", "The reduce stage", "The nightly job", "Every sales order", "The cache layer",
            "A custom field", "The retry queue", "The audit log", "A sublist line", "The import batch"]
VERBS = ["rejects", "reloads", "skips", "duplicates", "rewrites", "locks", "defers", "truncates",
         "validates", "reorders", "archives", "flags", "merges", "throttles", "recomputes"]
OBJECTS = ["the pending rows", "its governance budget", "stale tokens", "the last checkpoint", "two subsidiaries",
           "unposted journals", "the currency table", "orphaned file handles", "the item fulfilment",
           "inactive locations", "the rate limit window", "a partial payload", "the tax schedule"]
TAILS = ["when usage passes 80 percent", "after the second retry", "before month end close",
         "if the role lacks permission", "during the Tuesday window", "once the queue drains",
         "unless a flag overrides it", "while the index rebuilds", "on accounts created before 2019",
         "whenever the script yields"]


def paragraph(rng: random.Random, n_sentences: int = 6) -> str:
    out = []
    for _ in range(n_sentences):
        out.append(f"{rng.choice(SUBJECTS)} {rng.choice(VERBS)} {rng.choice(OBJECTS)} {rng.choice(TAILS)}.")
    return f"Note {rng.randint(1000, 99999)}. " + " ".join(out)


def raw_text(n_chars: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    paras, size = [], 0
    while size < n_chars:
        p = paragraph(rng)
        paras.append(p)
        size += len(p) + 2
    return paras


def build(target_tokens: int, seed: int = 0, needles: list[tuple[float, str]] | None = None,
          count_tokens=None) -> str:
    """Return about `target_tokens` of text with needles placed at fractional depths.

    `count_tokens` is a callable(text) -> int | None. With vLLM it is
    Client.count_tokens. Without it the estimate is 4.2 characters per token.
    """
    if target_tokens <= 0:
        return "\n\n".join(t for _, t in (needles or []))
    chars_per_token = 4.2
    paras = raw_text(int(target_tokens * chars_per_token), seed)
    if count_tokens:
        sample = "\n\n".join(paras[: min(len(paras), 200)])
        n = count_tokens(sample)
        if n:
            chars_per_token = len(sample) / n
            paras = raw_text(int(target_tokens * chars_per_token), seed)
    for depth, text in sorted(needles or [], key=lambda x: -x[0]):
        paras.insert(int(len(paras) * max(0.0, min(1.0, depth))), text)
    return "\n\n".join(paras)
