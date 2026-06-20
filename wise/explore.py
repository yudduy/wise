"""Multi-scale progressive exploration scaffold."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable


Cell = tuple[int, int]


@dataclass(frozen=True)
class Region:
    x0: int
    y0: int
    x1: int
    y1: int
    depth: int

    @property
    def center(self) -> Cell:
        return ((self.x0 + self.x1 - 1) // 2, (self.y0 + self.y1 - 1) // 2)

    def contains(self, cell: Cell) -> bool:
        x, y = cell
        return self.x0 <= x < self.x1 and self.y0 <= y < self.y1


class GridMap:
    def __init__(self, width: int, height: int, *, blocked: Iterable[Cell] = ()):
        self.width = width
        self.height = height
        self.blocked = set(blocked)
        self.visited: set[Cell] = set()

    @property
    def coverage(self) -> float:
        reachable = self.width * self.height - len(self.blocked)
        return len(self.visited) / reachable if reachable else 1.0

    def mark_visited(self, cell: Cell) -> None:
        if self.reachable(cell):
            self.visited.add(cell)

    def cells(self) -> list[Cell]:
        return [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if self.reachable((x, y))
        ]

    def unvisited(self, region: Region | None = None) -> list[Cell]:
        return [
            cell for cell in self.cells()
            if cell not in self.visited and (region is None or region.contains(cell))
        ]

    def neighbors(self, cell: Cell) -> list[Cell]:
        x, y = cell
        candidates = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        return [candidate for candidate in candidates if self.reachable(candidate)]

    def reachable(self, cell: Cell) -> bool:
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height and cell not in self.blocked


class ProgressiveExplorer:
    def __init__(
        self,
        grid: GridMap,
        *,
        max_depth: int = 3,
        quadtree_depth_weight: float = 0.6,
        quadtree_distance_penalty: float = 0.4,
        stagnation_blocks: float = 5.0,
        stagnation_seconds: float = 30.0,
    ):
        self.grid = grid
        self.max_depth = max_depth
        self.quadtree_depth_weight = quadtree_depth_weight
        self.quadtree_distance_penalty = quadtree_distance_penalty
        self.stagnation_blocks = stagnation_blocks
        self.stagnation_seconds = stagnation_seconds

    def next_target(self, current: Cell) -> Cell | None:
        region = self.quadtree_region(current)
        return (
            self.frontier_target(current, region)
            or self.quadtree_target(current, region)
            or self.voronoi_target(current)
        )

    def step(self, current: Cell) -> Cell | None:
        target = self.next_target(current)
        if target is not None:
            self.grid.mark_visited(target)
        return target

    def quadtree_region(self, current: Cell) -> Region | None:
        regions = [region for region in self._regions(Region(0, 0, self.grid.width, self.grid.height, 0)) if self.grid.unvisited(region)]
        if not regions:
            return None
        return max(regions, key=lambda region: self._region_score(region, current))

    def quadtree_target(self, current: Cell, region: Region | None = None) -> Cell | None:
        cells = self.grid.unvisited(region)
        if not cells:
            return None
        center = region.center if region else current
        return min(cells, key=lambda cell: (_distance(cell, center), _distance(cell, current)))

    def frontier_target(self, current: Cell, region: Region | None = None) -> Cell | None:
        frontiers = [
            cell for cell in self.grid.unvisited(region)
            if any(neighbor in self.grid.visited for neighbor in self.grid.neighbors(cell))
        ]
        if not frontiers:
            return None
        return max(frontiers, key=lambda cell: self._frontier_score(cell, current))

    def voronoi_target(self, current: Cell) -> Cell | None:
        cells = self.grid.unvisited()
        if not cells:
            return None
        if not self.grid.visited:
            return min(cells, key=lambda cell: _distance(cell, current))
        return max(cells, key=lambda cell: (min(_distance(cell, visited) for visited in self.grid.visited), -_distance(cell, current)))

    def stagnated(self, blocks_moved: float, seconds: float) -> bool:
        return blocks_moved < self.stagnation_blocks and seconds >= self.stagnation_seconds

    def _regions(self, region: Region) -> list[Region]:
        if region.depth >= self.max_depth or region.x1 - region.x0 <= 1 or region.y1 - region.y0 <= 1:
            return [region]
        mid_x = (region.x0 + region.x1) // 2
        mid_y = (region.y0 + region.y1) // 2
        children = [
            Region(region.x0, region.y0, mid_x, mid_y, region.depth + 1),
            Region(mid_x, region.y0, region.x1, mid_y, region.depth + 1),
            Region(region.x0, mid_y, mid_x, region.y1, region.depth + 1),
            Region(mid_x, mid_y, region.x1, region.y1, region.depth + 1),
        ]
        return [leaf for child in children for leaf in self._regions(child) if child.x0 < child.x1 and child.y0 < child.y1]

    def _region_score(self, region: Region, current: Cell) -> float:
        area = max(1, (region.x1 - region.x0) * (region.y1 - region.y0))
        unexplored = len(self.grid.unvisited(region)) / area
        depth = region.depth / max(1, self.max_depth)
        distance = _distance(current, region.center) / max(1.0, sqrt(self.grid.width ** 2 + self.grid.height ** 2))
        return self.quadtree_depth_weight * depth * unexplored - self.quadtree_distance_penalty * distance

    def _frontier_score(self, cell: Cell, current: Cell) -> float:
        coverage_gain = len([neighbor for neighbor in self.grid.neighbors(cell) if neighbor not in self.grid.visited]) / 4.0
        novelty = min((_distance(cell, visited) for visited in self.grid.visited), default=0.0)
        novelty /= max(1.0, sqrt(self.grid.width ** 2 + self.grid.height ** 2))
        distance = _distance(cell, current) / max(1.0, sqrt(self.grid.width ** 2 + self.grid.height ** 2))
        return coverage_gain + novelty - distance


def _distance(left: Cell, right: Cell) -> float:
    return sqrt((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2)
