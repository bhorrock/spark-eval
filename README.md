# spark-eval

Acceptance tests for candidate models on the DGX Spark. The question it answers:
is this model, served by this recipe, good enough to replace the daily driver?

It speaks only the OpenAI API. Point it at vLLM today. Point it at Bifrost later,
and any result that changes is the proxy's doing.

## How the Spark is divided

The daily driver stays resident in about half the memory. The other half is one
shared slot. At any moment the slot holds a candidate model, or ComfyUI, or
Blender, or nothing.

| Mode | What runs | Gates allowed |
|---|---|---|
| lab | daily driver + candidate in the slot | bringup, smoke, tools, longctx, quality |
| exclusive | candidate alone, daily driver stopped | perf, soak (and any of the above) |

Two vLLM instances split memory cleanly but share one GPU. Latency measured in
lab mode is contention, so the CLI refuses perf and soak there unless you pass `--force`.

Start the daily driver first and wait for it to finish loading before starting the
candidate. vLLM profiles free memory at startup and a second process allocating
at the same moment throws the profile off. If your vLLM build accepts an explicit KV
cache size in bytes, set it on both instances. A fixed size behaves better than a
fraction on a shared pool.

## The gates

Cheapest first. A run stops at the first failed gate.

1. `bringup`. Server healthy, right model, `max_model_len` at least 131072. With
   `--server-log` it also reads KV capacity and "maximum concurrency" from the vLLM
   log and fails if one 128K request does not fit.
2. `smoke`. 12 protocol cases: templates, streaming, JSON modes, one tool call,
   tool results, `tool_choice: none`. All must pass. Failures here are nearly always
   serving flags.
3. `tools`. 14 cases at contexts 0, 32K and 100K. Scored by JSON Schema, no judge.
   Reports `pass_rate` and `malformed_rate` per context. Malformed means the protocol
   broke: bad JSON, unknown tool, schema violation, or parser markup such as
   `<tool_call>` or `<function=` left in `content`.
   Every case runs twice, blocking and streamed (`--tool-stream suite` turns that off).
   vLLM parsers have two code paths and the streaming one is where new parsers break.
4. `longctx`. Needles at three depths, a three-fact sum, a fact that is absent, and
   a needle that must feed a tool call. Contexts 32K, 64K, 120K.
5. `quality`. Your own prompts in `suites/quality/*.yaml`. Four starters are included.
   Use `exec` checks (node or python) where you can, `spark-eval judge` where you cannot.
6. `perf`. Input length x concurrency grid with fresh filler text per request, so the
   prefix cache starts cold. Records TTFT p50/p95, decode tok/s per request, aggregate
   tok/s, and deltas of vLLM's prefix cache, spec decode and preemption counters.
   Ends with a prefix cache probe: one 16K prompt sent twice, cold TTFT against warm.
7. `soak`. Mixed load for an hour. Fails on any error or on throughput falling 15%.

To test MTP, run perf from `qwen3.6-35b-a3b-nvfp4` and its `-nomtp` twin under two labels
and compare them. Speculative decoding tends to help at concurrency 1 to 2 and fade by 8.

## Use

```bash
pip install -e .

# 1. Launch the candidate in the lab slot
LAUNCH="sparkrun run recipes/granite-4.2-30b-nvfp4.yaml --tp 1 -o port=8001 -o gpu_memory_utilization=0.40"
$LAUNCH
sparkrun logs granite-4.2-30b-nvfp4 > /tmp/granite.log     # Ctrl+C detaches

# 2. Functional gates, lab mode
spark-eval run --label granite-4.2-30b \
  --base-url http://spark:8001/v1 --model granite-4.2-30b \
  --recipe recipes/granite-4.2-30b-nvfp4.yaml --launch "$LAUNCH" \
  --server-log /tmp/granite.log

# 3. Overnight, daily driver stopped
spark-eval run --label granite-4.2-30b --mode exclusive --gates perf,soak \
  --base-url http://spark:8001/v1 --model granite-4.2-30b \
  --recipe recipes/granite-4.2-30b-nvfp4.yaml --launch "<the exclusive-slot command>"

# 4. Same two steps for the incumbent under --label qwen3.6-35b-a3b, then:
spark-eval compare granite-4.2-30b qwen3.6-35b-a3b
spark-eval judge   granite-4.2-30b qwen3.6-35b-a3b --judge-url https://... --judge-model ...
```

`compare` takes the newest result per gate for each label, so the lab run and the
exclusive run combine. It exits 0 on "promote" and 1 otherwise.

Debugging one case: `--gates tools --tool-contexts 0 --only quotes_and_newlines_in_arg`.
The full response is in `tools.json` under `rows[].response`.

## Parsers that ship with the model

Granite's tool parser is a Python file in the model repo, loaded with `--tool-parser-plugin`.
That file is part of what you are testing, so tell the manifest about it:

```bash
spark-eval run ... --artifact ~/.cache/huggingface/hub/models--<org>--<name>/snapshots/<sha>/<parser>.py
```

Its hash goes into the manifest and into the result folder name. Add a chat template the
same way if you override one. If the server refuses to start with an unknown parser name,
the plugin did not load. If it starts and `smoke` fails on `single_tool_call`, it loaded and
does not work.

When vLLM ships the parser, copy the recipe to `granite-4.2-30b-nvfp4-builtin.yaml`, delete
the `--tool-parser-plugin` line, bump the image digest, and run the tools gate under a new
label. `spark-eval compare granite-4.2-30b-builtin granite-4.2-30b` then shows whether the
upstream port behaves like the plugin you validated, split by blocking and streamed.

## Results

```
results/<label>/<recipe-sha8>/<timestamp>/
  manifest.json    recipe text and hash, image, model revision, server version, suite hashes
  <gate>.json      summary plus every request and response
```

Commit results. They are small, and the incumbent's files are the baseline that every
later candidate is measured against. Change a recipe flag and the sha8 changes, so
tuning runs never overwrite each other.

## Rules

`promotion.yaml` holds the floors and the versus-incumbent rules. Edit it before a
comparison, not after.

## Tests

`python -m unittest tests.test_end_to_end` runs the whole pipeline against a fake server,
including a mode that leaks Qwen-style XML into `content` to prove the malformed counter fires.
