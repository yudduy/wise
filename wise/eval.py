"""Offline smoke and paper-target regression gates for WISE."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .explore import GridMap, ProgressiveExplorer
from .memory import CAN_OBTAIN, CausalEventGraph, Observation, Pose, ShortTermGeometricMemory, Task
from .scheduler import OpportunisticTaskScheduler


ABA_SPARSE = "aba-sparse"
ABC_SPARSE = "abc-sparse"


@dataclass(frozen=True)
class RegressionTarget:
    success_rate: float
    avg_timesteps: float


@dataclass(frozen=True)
class EpisodeResult:
    task: str
    success: bool
    timesteps: int
    details: dict[str, object]


PAPER_TARGETS = {
    ABA_SPARSE: RegressionTarget(success_rate=0.62, avg_timesteps=5981),
    ABC_SPARSE: RegressionTarget(success_rate=0.77, avg_timesteps=4620),
}


def summarize(results: Sequence[EpisodeResult], target: RegressionTarget, *, mode: str) -> dict[str, object]:
    if not results:
        return {
            "episodes": 0,
            "success_rate": 0.0,
            "avg_timesteps": None,
            "paper_target": asdict(target),
            "regression_gate": "no_results",
        }
    success_rate = sum(result.success for result in results) / len(results)
    avg_timesteps = sum(result.timesteps for result in results if result.success) / max(1, sum(result.success for result in results))
    gate = (
        success_rate >= target.success_rate and avg_timesteps <= target.avg_timesteps
        if mode == "live"
        else None
    )
    return {
        "episodes": len(results),
        "success_rate": success_rate,
        "avg_timesteps": avg_timesteps,
        "paper_target": asdict(target),
        "regression_gate": gate if gate is not None else "skipped_offline_fixture",
    }


def build_offline_report(tasks: Iterable[str] = (ABA_SPARSE, ABC_SPARSE)) -> dict[str, object]:
    report: dict[str, object] = {"mode": "offline", "results": [], "summary": {}}
    for task in tasks:
        results = [offline_episode(task)]
        report["results"].extend(asdict(result) for result in results)  # type: ignore[union-attr]
        report["summary"][task] = summarize(results, PAPER_TARGETS[task], mode="offline")  # type: ignore[index]
    return report


def offline_episode(task: str) -> EpisodeResult:
    if task == ABA_SPARSE:
        return _offline_aba()
    if task == ABC_SPARSE:
        return _offline_abc()
    raise ValueError(f"unknown task: {task}")


def missing_live_requirements(env: Mapping[str, str] = os.environ) -> list[str]:
    missing: list[str] = []
    for name in ("WISE_MINEDOJO_READY", "OPENAI_API_KEY", "WISE_OPENAI_MODEL"):
        if not env.get(name):
            missing.append(name)
    for name in ("WISE_MINECLIP_CHECKPOINT", "WISE_VPT_NAV_CHECKPOINT"):
        value = env.get(name)
        if not value:
            missing.append(name)
        elif not Path(value).exists():
            missing.append(f"{name}={value} (not found)")
    return missing


def require_live_ready(env: Mapping[str, str] = os.environ) -> None:
    missing = missing_live_requirements(env)
    if missing:
        raise RuntimeError("live WISE regression requires: " + ", ".join(missing))


def run_live_regression(task: str, episodes: int) -> dict[str, object]:
    require_live_ready()
    raise RuntimeError(
        "live MineDojo/MineRL runner is not wired in this scaffold yet; "
        f"expected to run {episodes} {task} episodes and compare with Table 2."
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="run deterministic offline smoke")
    parser.add_argument("--require-live", action="store_true", help="require live MineDojo/MineRL/OpenAI assets")
    parser.add_argument("--task", choices=(ABA_SPARSE, ABC_SPARSE), help="task to run; default is both offline tasks")
    parser.add_argument("--episodes", type=int, default=50, help="episode count for live regression")
    args = parser.parse_args(argv)

    try:
        if args.require_live:
            if not args.task:
                parser.error("--require-live needs --task")
            report = run_live_regression(args.task, args.episodes)
        elif args.offline:
            report = build_offline_report([args.task] if args.task else (ABA_SPARSE, ABC_SPARSE))
        else:
            parser.error("choose --offline or --require-live")
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _offline_aba() -> EpisodeResult:
    memory = ShortTermGeometricMemory()
    graph = CausalEventGraph()
    resource_a = Observation("water-source", Pose(12, z=4), embedding=(1.0, 0.0), entities=("water",), t=1)
    memory.add(resource_a)
    graph.add_observation(resource_a)
    hits = memory.retrieve(Task("return to water", "water", embedding=(0.0, 1.0)), graph)
    grid = GridMap(8, 8)
    grid.mark_visited((0, 0))
    ProgressiveExplorer(grid).step((0, 0))
    return EpisodeResult(
        task=ABA_SPARSE,
        success=bool(hits),
        timesteps=3,
        details={"offline_fixture": True, "retrieved": hits[0].observation.id if hits else None, "coverage": grid.coverage},
    )


def _offline_abc() -> EpisodeResult:
    memory = ShortTermGeometricMemory()
    graph = CausalEventGraph()
    cow = Observation("cow-encounter", Pose(3, z=4), embedding=(1.0, 0.0), entities=("cow",), t=500)
    memory.add(cow)
    graph.add_observation(cow)
    graph.add_edge("cow", CAN_OBTAIN, "beef")
    tasks = [
        Task("find water", "water", urgency=0.1, embedding=(0.0, 1.0)),
        Task("collect logs", "logs", urgency=0.1, embedding=(0.0, 1.0)),
        Task("obtain beef", "beef", urgency=0.1, embedding=(0.0, 1.0)),
    ]
    order = OpportunisticTaskScheduler().reorder(tasks, memory, graph, Pose(0, z=0))
    return EpisodeResult(
        task=ABC_SPARSE,
        success=order[0].target == "beef",
        timesteps=3,
        details={"offline_fixture": True, "order": [task.name for task in order]},
    )


if __name__ == "__main__":
    raise SystemExit(main())
