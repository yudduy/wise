"""Command helpers for running MrSteve outside this repo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


DEFAULT_TASK = "log_water_bucket_aba_randinit"


@dataclass(frozen=True)
class MrSteveCommand:
    cwd: Path
    argv: tuple[str, ...]

    def display(self) -> str:
        return " ".join(self.argv)


def episode_command(
    root: str | Path,
    *,
    agent: str,
    task: str = DEFAULT_TASK,
    episodes: int = 1,
    extra: Sequence[str] = (),
) -> MrSteveCommand:
    if agent not in {"steve1", "mrsteve"}:
        raise ValueError(f"unsupported MrSteve agent: {agent}")
    return MrSteveCommand(
        cwd=Path(root),
        argv=(
            "uv",
            "run",
            "main.py",
            f"task={task}",
            f"agent={agent}",
            f"n_episodes={episodes}",
            *extra,
        ),
    )


def steve1_smoke(root: str | Path, *, task: str = DEFAULT_TASK) -> MrSteveCommand:
    return episode_command(root, agent="steve1", task=task, episodes=1)


def mrsteve_smoke(root: str | Path, *, task: str = DEFAULT_TASK) -> MrSteveCommand:
    return episode_command(root, agent="mrsteve", task=task, episodes=1)


def stats_command(root: str | Path, output_glob: str) -> MrSteveCommand:
    return MrSteveCommand(
        cwd=Path(root),
        argv=("uv", "run", "scripts/get_stats.py", output_glob),
    )
