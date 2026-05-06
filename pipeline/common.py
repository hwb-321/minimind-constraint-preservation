from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_ROOT = Path(__file__).resolve().parent


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def repo_path(value: Any) -> str:
    if not isinstance(value, str):
        return str(value)
    path = Path(value)
    if path.is_absolute():
        return str(path)
    if value.startswith("./") or value.startswith("../"):
        return str((REPO_ROOT / path).resolve())
    return value


def append_args(cmd: list[str], args: dict[str, Any]) -> list[str]:
    for key, value in args.items():
        if value is None or value is False:
            continue
        flag = "--" + key
        if value is True:
            cmd.append(flag)
        elif isinstance(value, list):
            cmd.append(flag)
            cmd.extend(repo_path(item) for item in value)
        else:
            cmd.extend([flag, repo_path(value)])
    return cmd


def build_command(stage: dict[str, Any]) -> list[str]:
    runtime = stage.get("runtime", {}) or {}
    script = str((REPO_ROOT / stage["script"]).resolve())
    nproc = int(runtime.get("nproc_per_node", 1))
    distributed = bool(stage.get("distributed", False)) and nproc > 1

    if distributed:
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            str(nproc),
            "--master_port",
            str(runtime.get("master_port", 29501)),
            script,
        ]
    else:
        cmd = [sys.executable, "-u", script]
    return append_args(cmd, stage.get("args", {}) or {})


def run_stage(config_path: Path, dry_run: bool = False) -> str:
    stage = load_yaml(config_path)
    runtime = stage.get("runtime", {}) or {}

    env = os.environ.copy()
    cuda_visible_devices = str(runtime.get("cuda_visible_devices", "") or "")
    if cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    env["PYTHONUNBUFFERED"] = "1"

    cmd = build_command(stage)
    print("$ " + " ".join(cmd), flush=True)
    if dry_run:
        return ""

    proc = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    stdout_lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        stdout_lines.append(line)
        print(line, end="", flush=True)
    return_code = proc.wait()
    stdout = "".join(stdout_lines)
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd, output=stdout)
    return stdout


def parse_accuracy(stdout: str) -> str:
    for line in stdout.splitlines():
        if line.startswith("Accuracy:"):
            return line.removeprefix("Accuracy:").strip()
    return ""
