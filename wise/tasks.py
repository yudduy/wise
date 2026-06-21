"""Sparse-task specs and evaluator helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


ABA_SPARSE = "aba-sparse"
ABC_SPARSE = "abc-sparse"


@dataclass(frozen=True)
class SparseTaskSpec:
    name: str
    mrsteve_task: str
    goals: tuple[str, ...]
    success_inventory: Mapping[str, int]
    timeout_steps: int
    seeds: tuple[int, ...] = tuple(range(50))

    def succeeded(self, inventory: Mapping[str, int]) -> bool:
        return all(int(inventory.get(item, 0)) >= count for item, count in self.success_inventory.items())

    def timed_out(self, timestep: int) -> bool:
        return timestep >= self.timeout_steps


SPARSE_TASKS = {
    ABA_SPARSE: SparseTaskSpec(
        name=ABA_SPARSE,
        mrsteve_task="log_water_bucket_aba_randinit",
        goals=("find first resource", "find second resource", "return to first resource"),
        success_inventory={"water_bucket": 1},
        timeout_steps=10_000,
    ),
    ABC_SPARSE: SparseTaskSpec(
        name=ABC_SPARSE,
        mrsteve_task="beef_log_water_abc_sparse",
        goals=("find water", "collect logs", "obtain beef"),
        success_inventory={"beef": 1},
        timeout_steps=10_000,
    ),
}


def task_spec(name: str) -> SparseTaskSpec:
    try:
        return SPARSE_TASKS[name]
    except KeyError as error:
        raise ValueError(f"unknown sparse task: {name}") from error


def selected_seeds(name: str, episodes: int) -> tuple[int, ...]:
    seeds = task_spec(name).seeds
    if episodes > len(seeds):
        raise ValueError(f"{name} only defines {len(seeds)} seeds")
    return seeds[:episodes]
