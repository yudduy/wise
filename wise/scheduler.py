"""Opportunistic task scheduling for WISE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .memory import CausalEventGraph, Pose, ShortTermGeometricMemory, Task


@dataclass(frozen=True)
class TaskScore:
    task: Task
    score: float
    urgency: float
    causal_relevance: float
    nav_cost: float


class OpportunisticTaskScheduler:
    def __init__(
        self,
        *,
        urgency_weight: float = 0.3,
        causal_weight: float = 0.5,
        nav_weight: float = 0.2,
        max_nav_distance: float = 100.0,
    ):
        self.urgency_weight = urgency_weight
        self.causal_weight = causal_weight
        self.nav_weight = nav_weight
        self.max_nav_distance = max_nav_distance

    def reorder(
        self,
        tasks: Sequence[Task],
        memory: ShortTermGeometricMemory,
        graph: CausalEventGraph,
        current_pose: Pose,
        *,
        completed: Iterable[str] = (),
    ) -> list[Task]:
        indexed = [(index, task) for index, task in enumerate(tasks) if not task.completed]
        executable = [(index, task) for index, task in indexed if task.executable(completed)]
        blocked = [task for _, task in indexed if not task.executable(completed)]
        ranked = sorted(
            executable,
            key=lambda item: (-self.score(item[1], memory, graph, current_pose).score, item[0]),
        )
        return [task for _, task in ranked] + blocked

    def score(
        self,
        task: Task,
        memory: ShortTermGeometricMemory,
        graph: CausalEventGraph,
        current_pose: Pose,
    ) -> TaskScore:
        hits = memory.retrieve(task, graph, top_k=1)
        causal = hits[0].causal_score if hits else 0.0
        nav_cost = 1.0
        if hits:
            nav_cost = min(1.0, current_pose.distance_to(hits[0].observation.pose) / self.max_nav_distance)
        score = (
            self.urgency_weight * task.urgency
            + self.causal_weight * causal
            - self.nav_weight * nav_cost
        )
        return TaskScore(task, score, task.urgency, causal, nav_cost)
