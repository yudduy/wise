"""Offline smoke and paper-target regression gates for WISE."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

from .explore import GridMap, ProgressiveExplorer
from .memory import CAN_OBTAIN, CausalEventGraph, Observation, Pose, ShortTermGeometricMemory, Task
from .readiness import check as readiness_check
from .scheduler import OpportunisticTaskScheduler
from .tasks import ABC_SPARSE, ABA_SPARSE, SPARSE_TASKS, selected_seeds

FULL_WISE = "full"
NO_CAUSAL_GRAPH = "no-causal-graph"
NO_SCHEDULER = "no-scheduler"
NO_EXPLORATION = "no-exploration"
STEVE1_BASELINE = "steve1"
MRSTEVE_BASELINE = "mrsteve"
VARIANTS = (FULL_WISE, NO_CAUSAL_GRAPH, NO_SCHEDULER, NO_EXPLORATION, STEVE1_BASELINE, MRSTEVE_BASELINE)


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
    variant: str = FULL_WISE
    failure: str = ""
    gpt_calls: int = 0


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
    success_count = sum(result.success for result in results)
    avg_timesteps = (
        sum(result.timesteps for result in results if result.success) / success_count
        if success_count
        else None
    )
    gate = (
        success_rate >= target.success_rate and avg_timesteps <= target.avg_timesteps
        if mode == "live"
        else None
    )
    return {
        "episodes": len(results),
        "success_rate": success_rate,
        "avg_timesteps": avg_timesteps,
        "failures": [result.failure for result in results if result.failure],
        "gpt_calls": sum(result.gpt_calls for result in results),
        "paper_target": asdict(target),
        "regression_gate": gate if gate is not None else "skipped_offline_fixture",
    }


def build_offline_report(tasks: Iterable[str] = (ABA_SPARSE, ABC_SPARSE), *, episodes: int = 1, variant: str = FULL_WISE) -> dict[str, object]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    report: dict[str, object] = {"mode": "offline", "variant": variant, "results": [], "summary": {}}
    for task in tasks:
        results = [offline_episode(task, seed=seed, variant=variant) for seed in selected_seeds(task, episodes)]
        report["results"].extend(asdict(result) for result in results)  # type: ignore[union-attr]
        report["summary"][task] = summarize(results, PAPER_TARGETS[task], mode="offline")  # type: ignore[index]
    return report


def offline_episode(task: str, *, seed: int = 0, variant: str = FULL_WISE) -> EpisodeResult:
    if task == ABA_SPARSE:
        return _offline_aba(seed, variant)
    if task == ABC_SPARSE:
        return _offline_abc(seed, variant)
    raise ValueError(f"unknown task: {task}")


def missing_live_requirements(env: Mapping[str, str] = os.environ) -> list[str]:
    report = readiness_check(env)
    return [
        *report["missing_env"],  # type: ignore[list-item]
        *report["missing_paths"],  # type: ignore[list-item]
        *[f"WISE_MRSTEVE_ROOT/{path}" for path in report["missing_mrsteve_files"]],  # type: ignore[index]
    ]


def require_live_ready(env: Mapping[str, str] = os.environ) -> None:
    missing = missing_live_requirements(env)
    if missing:
        raise RuntimeError("live WISE regression requires: " + ", ".join(missing))


def run_live_regression(task: str, episodes: int, variant: str) -> dict[str, object]:
    require_live_ready()
    raise RuntimeError(
        "live MineDojo/MineRL runner is not wired in this scaffold yet; "
        f"expected to run {episodes} {task} episodes for {variant} and compare with Table 2."
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="run deterministic offline smoke")
    parser.add_argument("--require-live", action="store_true", help="require live MineDojo/MineRL/OpenAI assets")
    parser.add_argument("--task", choices=(ABA_SPARSE, ABC_SPARSE), help="task to run; default is both offline tasks")
    parser.add_argument("--episodes", type=int, help="episode count; defaults to 1 offline and 50 live")
    parser.add_argument("--variant", choices=VARIANTS, default=FULL_WISE, help="agent/ablation variant")
    args = parser.parse_args(argv)

    try:
        if args.require_live:
            if not args.task:
                parser.error("--require-live needs --task")
            report = run_live_regression(args.task, args.episodes or 50, args.variant)
        elif args.offline:
            report = build_offline_report([args.task] if args.task else (ABA_SPARSE, ABC_SPARSE), episodes=args.episodes or 1, variant=args.variant)
        else:
            parser.error("choose --offline or --require-live")
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _offline_aba(seed: int, variant: str) -> EpisodeResult:
    spec = SPARSE_TASKS[ABA_SPARSE]
    memory = ShortTermGeometricMemory()
    graph = CausalEventGraph()
    resource_a = Observation("water-source", Pose(12, z=4), embedding=(1.0, 0.0), entities=("water",), t=1)
    memory.add(resource_a)
    graph.add_observation(resource_a)
    hits = memory.retrieve(Task("return to water", "water", embedding=(0.0, 1.0)), graph)
    grid = GridMap(8, 8)
    if variant != NO_EXPLORATION:
        grid.mark_visited((0, 0))
        ProgressiveExplorer(grid).step((0, 0))
    success = bool(hits) and spec.succeeded({"water_bucket": 1}) and variant not in {STEVE1_BASELINE}
    return EpisodeResult(
        task=ABA_SPARSE,
        success=success,
        timesteps=3,
        details={"offline_fixture": True, "seed": seed, "retrieved": hits[0].observation.id if hits else None, "coverage": grid.coverage, "mrsteve_task": spec.mrsteve_task},
        variant=variant,
        failure="" if success else "offline_ablation_failed",
        gpt_calls=1 if variant not in {STEVE1_BASELINE, MRSTEVE_BASELINE, NO_CAUSAL_GRAPH} else 0,
    )


def _offline_abc(seed: int, variant: str) -> EpisodeResult:
    spec = SPARSE_TASKS[ABC_SPARSE]
    memory = ShortTermGeometricMemory()
    graph = CausalEventGraph()
    cow = Observation("cow-encounter", Pose(3, z=4), embedding=(1.0, 0.0), entities=("cow",), t=500)
    memory.add(cow)
    graph.add_observation(cow)
    if variant != NO_CAUSAL_GRAPH:
        graph.add_edge("cow", CAN_OBTAIN, "beef")
    tasks = [
        Task("find water", "water", urgency=0.1, embedding=(0.0, 1.0)),
        Task("collect logs", "logs", urgency=0.1, embedding=(0.0, 1.0)),
        Task("obtain beef", "beef", urgency=0.1, embedding=(0.0, 1.0)),
    ]
    order = list(tasks) if variant in {NO_SCHEDULER, STEVE1_BASELINE, MRSTEVE_BASELINE} else OpportunisticTaskScheduler().reorder(tasks, memory, graph, Pose(0, z=0))
    success = order[0].target == "beef" and spec.succeeded({"beef": 1})
    return EpisodeResult(
        task=ABC_SPARSE,
        success=success,
        timesteps=3,
        details={"offline_fixture": True, "seed": seed, "order": [task.name for task in order], "mrsteve_task": spec.mrsteve_task},
        variant=variant,
        failure="" if success else "offline_ablation_failed",
        gpt_calls=1 if variant not in {STEVE1_BASELINE, MRSTEVE_BASELINE, NO_CAUSAL_GRAPH} else 0,
    )


if __name__ == "__main__":
    raise SystemExit(main())
