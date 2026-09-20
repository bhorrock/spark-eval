# Recipes

sparkrun v2 recipes. `.sparkrun/registry.yaml` at the repo root declares this folder as
a registry named `lab`, so once the repo is on your git server:

```bash
sparkrun registry add <git-url-of-this-repo> --trust
sparkrun list @lab
sparkrun run @lab/granite-4.2-30b-nvfp4 --tp 1
```

Before then, run a recipe by file path: `sparkrun run recipes/granite-4.2-30b-nvfp4.yaml --tp 1`.

| File | Role |
|---|---|
| `qwen3.6-35b-a3b-nvfp4.yaml` | incumbent, MTP on |
| `qwen3.6-35b-a3b-nvfp4-nomtp.yaml` | same, speculative decoding off, for the perf A/B |
| `granite-4.2-30b-nvfp4.yaml` | challenger, tool parser loaded from the model repo |
| `nemotron-3-nano-30b-a3b-nvfp4-eugr.yaml` | challenger on the eugr nightly image |
| `nemotron-3-nano-30b-a3b-nvfp4-scitrera.yaml` | same model and revision on scitrera 0.17.0, a container A/B |
| `nemotron-3.5-lightning-30b-a3b-nvfp4.yaml` | challenger, DSpark drafter |
| `nemotron-3.5-lightning-30b-a3b-nvfp4-nospec.yaml` | same, no drafter |
| `qwen3.8-27b-nvfp4.yaml` | challenger, dense 27B hybrid, MTP on |
| `qwen3.8-27b-nvfp4-nomtp.yaml` | same, no speculation |
| `qwen3.8-27b-nvfp4-dflash2.yaml` | same, z-lab DFlash2 drafter in place of MTP |

All of them pass `sparkrun recipe validate` (sparkrun 0.3.9). The Granite and Nemotron-3-Nano
recipes keep one `inline-script` suggestion for the parser lookup, which does not block a launch.

The Qwen 3.6 and Granite files still have `CHANGEME` values I could not know. The Nemotron
and Qwen 3.8 files have none. Their model revisions are the HF `main` commits as of
2026-09-19 and the images are pinned by digest (eugr nightly `20260918`, scitrera `0.17.0-t5`).
None of the seven has been launched. They were adapted from upstream recipes and checked
by rendering the command, so expect bringup to find at least one flag the image rejects.

Where each came from:

- Nemotron-3-Nano, eugr. `experimental-recipes/eugr-vllm/nemotron-3-nano-nvfp4-vllm.yaml` in
  the spark-arena registry. Upstream's `mods/nemotron-nano` fetches the reasoning parser from
  the model repo's `main` with wget at launch. Here it comes out of the HF cache at the
  pinned revision, so the mod is gone.
- Nemotron-3-Nano, scitrera. The recipe that shipped inside sparkrun 0.0.15. It is not in any
  registry now and the image has not moved since March 2026, so this file leaves out
  `--moe-backend` and `--load-format` rather than guess what vLLM 0.17 accepts.
- Nemotron 3.5 Lightning. `recipes/nemotron-3.5-lightning.yaml` in eugr/spark-vllm-docker,
  a v1 recipe for eugr's own launcher, ported to v2. Mamba flags copied as a set.
- Qwen3.8-27B. `official-recipes/qwen3.8/qwen3.8-27b-fp8-mtp-vllm.yaml`, with NVIDIA's NVFP4
  checkpoint swapped in. That recipe already runs `tensor_parallel: 1` on one Spark, and
  the NVFP4 checkpoint keeps its MTP head. DFlash2 settings are from eugr's recipe.

A drafter is a second HF repo. sparkrun finds it by reading `model` and `revision` out of
`speculative_config` in `defaults`, which is why that JSON lives there and not inline in
`command:`. A drafter without a `revision` downloads unpinned and then hits the same offline
`refs/` failure described under Conventions.

## Slots

One recipe file serves every slot. The slot is a pair of overrides:

| Slot | Overrides |
|---|---|
| driver | `-o port=8000 -o gpu_memory_utilization=0.45` |
| lab | `-o port=8001 -o gpu_memory_utilization=0.40` |
| exclusive | `-o port=8001 -o gpu_memory_utilization=0.80` |

0.45 + 0.40 leaves about 15% of the 121 GB sparkrun counts as usable for the OS, the
containers and CUDA graphs. Treat those numbers as a first guess and tune them.

Pass the same command to spark-eval with `--launch "sparkrun run ... -o ..."`. It is stored
in the manifest and folded into the recipe hash, so a lab run and an exclusive run of the
same file land in different result folders.

## Conventions

- Every serve flag is written in `command:`. Without a template, sparkrun builds the command
  from a fixed flag map and drops `defaults` keys it does not know. An explicit template
  also makes two recipes diffable line by line.
- Every tunable is a `{placeholder}` backed by `defaults`, so `-o key=value` can change it
  without editing the file.
- JSON flags are written with plain braces. sparkrun v2 passes them through and still
  resolves placeholders inside them, as in the `--speculative-config` line.
- `model_revision` is pinned and also passed as `--revision {model_revision}`. The validator
  flagged the first draft for omitting it: the container runs with `HF_HUB_OFFLINE=1`, a
  SHA download writes no ref in the HF cache, and vLLM then fails to find the model.
- `container:` is pinned by digest. A nightly tag moves under you.
- The driver recipe sets `restart_policy: unless-stopped`. Challengers do not, so a crashed
  candidate stays down and leaves its log.
- No lifecycle hooks. spark-eval's bringup gate does the health check, and hooks from an
  untrusted registry need `--trust` anyway.

## Parser plugins from the model repo

The Granite recipe loads its tool parser from the model's own files. The command looks the
file up with `huggingface_hub.hf_hub_download(model, file, revision)`, which returns the path
inside the mounted cache. I checked that this works with `HF_HUB_OFFLINE=1` against a cache
filled by a commit-SHA download, which is the state sparkrun leaves it in. Two conditions:

- `tool_parser_file` must be a file sparkrun actually downloaded. If the launch fails with
  LocalEntryNotFoundError, the file was filtered out of the download.
- `tool_call_parser` is the name the plugin registers in its `@ToolParserManager.register_module`
  line. It is often not the file name.

`sparkrun recipe validate` reports one suggestion for this recipe (`inline-script`). It is
right in general, and the alternative it offers (a `mods:` directory) is the better home once
you have more than one model needing this. For one model, one line is easier to read.

## Adding a candidate

Copy the Granite file, change `model`, `model_revision`, `served_model_name`, the parsers
and the description. Run `sparkrun recipe validate <file>` and `sparkrun recipe vram <file>`,
commit, then launch. Commit before you launch, so the result folder points at a recipe that
exists in git.
