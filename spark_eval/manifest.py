"""Run manifest. Every result file sits next to one of these, so a number can
always be traced to the recipe, image and flags that produced it."""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import time
from pathlib import Path

import httpx
import yaml

from .client import Client, root_url


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(cmd: list[str]) -> str | None:
    if not shutil.which(cmd[0]):
        return None
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=15).stdout.strip() or None
    except (subprocess.SubprocessError, OSError):
        return None


def server_version(base_url: str) -> str | None:
    try:
        r = httpx.get(f"{root_url(base_url)}/version", timeout=10)
        if r.status_code == 200:
            return r.json().get("version")
    except (httpx.HTTPError, ValueError):
        pass
    return None


def build(client: Client, label: str, mode: str, recipe: Path | None, image: str | None,
          model_revision: str | None, notes: str | None, suites_dir: Path,
          launch: str | None = None, artifacts: list[Path] | None = None) -> dict:
    if recipe:                                   # the recipe is the source of truth for both pins
        try:
            doc = yaml.safe_load(recipe.read_text()) or {}
            image = image or doc.get("container")
            model_revision = model_revision or doc.get("model_revision")
        except yaml.YAMLError:
            pass
    arts = [{"path": str(p), "sha256": sha256_file(p)} for p in artifacts or []]
    art_digest = "".join(a["sha256"] for a in arts)
    served = {}
    try:
        for m in client.models():
            if m.get("id") == client.model:
                served = {k: m.get(k) for k in ("id", "max_model_len", "root") if k in m}
    except httpx.HTTPError:
        pass
    return {
        "label": label,
        "mode": mode,                      # lab = beside the daily driver, exclusive = alone on the GPU
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "endpoint": client.base_url,
        "model": client.model,
        "served_model": served,
        "model_revision": model_revision,  # HF commit SHA
        "image": image,                    # pin by digest, e.g. repo@sha256:...
        "server_version": server_version(client.base_url),
        # The hash covers the launch command and artifacts too. Both change what the recipe does.
        "recipe": {"path": str(recipe), "text": recipe.read_text(),
                   "sha256": hashlib.sha256(recipe.read_bytes() + (launch or "").encode() + art_digest.encode()).hexdigest()} if recipe else None,
        "launch": launch,
        "artifacts": arts,                 # parser plugins, chat templates: code that ships outside the image
        "suites": {p.name: sha256_file(p)[:12] for p in sorted(suites_dir.glob("*.yaml"))},
        "harness_commit": _run(["git", "rev-parse", "--short", "HEAD"]),
        "client_host": platform.node(),
        "gpu": _run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]),
        "notes": notes,
    }


def run_dir(results_root: Path, manifest: dict) -> Path:
    rhash = (manifest.get("recipe") or {}).get("sha256", "norecipe")[:8]
    d = results_root / manifest["label"] / rhash / time.strftime("%Y%m%d-%H%M%S")
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return d


def latest_run(results_root: Path, label: str) -> Path | None:
    runs = sorted((results_root / label).glob("*/*/manifest.json"), key=lambda p: p.parent.name)
    return runs[-1].parent if runs else None
