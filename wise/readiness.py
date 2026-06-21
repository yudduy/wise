"""Local readiness checks for live WISE reproduction."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Mapping, Sequence


ENV_VARS = ("OPENAI_API_KEY", "WISE_OPENAI_MODEL")
PATH_VARS = (
    "WISE_MRSTEVE_ROOT",
    "WISE_MINECLIP_CHECKPOINT",
    "WISE_STEVE1_WEIGHTS",
    "WISE_VPT_MODEL",
    "WISE_VPT_NAV_CHECKPOINT",
)
MRSTEVE_FILES = (
    "main.py",
    "prepare_models.sh",
    "task_specs.yaml",
    "config/main.yaml",
    "scripts/get_stats.py",
)


def check(env: Mapping[str, str] = os.environ) -> dict[str, object]:
    missing_env = [name for name in ENV_VARS if not env.get(name)]
    missing_paths: list[str] = []
    for name in PATH_VARS:
        value = env.get(name)
        if not value:
            missing_paths.append(name)
        elif not Path(value).exists():
            missing_paths.append(f"{name}={value} (not found)")
    mrsteve_root = env.get("WISE_MRSTEVE_ROOT")
    missing_mrsteve = []
    if mrsteve_root and Path(mrsteve_root).exists():
        missing_mrsteve = [
            rel for rel in MRSTEVE_FILES
            if not (Path(mrsteve_root) / rel).exists()
        ]
    ready = not missing_env and not missing_paths and not missing_mrsteve
    return {
        "ready": ready,
        "missing_env": missing_env,
        "missing_paths": missing_paths,
        "missing_mrsteve_files": missing_mrsteve,
        "next_step": "run one MrSteve/Steve-1 episode" if ready else "set/download missing assets",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    report = check()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
