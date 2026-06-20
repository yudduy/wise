import unittest
import asyncio

from wise.memory import (
    CAN_OBTAIN,
    CO_OCCURS_WITH,
    CausalEventGraph,
    Observation,
    Pose,
    ShortTermGeometricMemory,
    Task,
    image_entropy,
)
from wise.explore import GridMap, ProgressiveExplorer
from wise.eval import ABC_SPARSE, ABA_SPARSE, PAPER_TARGETS, EpisodeResult, build_offline_report, missing_live_requirements, summarize
from wise.scheduler import OpportunisticTaskScheduler
from wise.vlm import AsyncGraphBuilder, FixtureVLMClient


class PackageSmokeTests(unittest.TestCase):
    def test_package_imports(self):
        import wise

        self.assertEqual(wise.__all__, [])


class MemoryTests(unittest.TestCase):
    def test_dp_means_splits_distant_event_embeddings(self):
        memory = ShortTermGeometricMemory(dp_lambda=0.15)
        memory.add(Observation("near-a", Pose(0, z=0), embedding=(1.0, 0.0), t=1))
        memory.add(Observation("near-b", Pose(1, z=0), embedding=(0.99, 0.01), t=2))
        memory.add(Observation("far", Pose(1, z=0), embedding=(0.0, 1.0), t=3))

        self.assertEqual(len(memory.clusters()), 2)

    def test_eviction_removes_oldest_from_largest_cluster(self):
        memory = ShortTermGeometricMemory(max_observations=2, dp_lambda=0.15)
        memory.add(Observation("old", Pose(0), embedding=(1.0, 0.0), t=1))
        memory.add(Observation("mid", Pose(0), embedding=(0.99, 0.01), t=2))
        memory.add(Observation("new", Pose(0), embedding=(0.98, 0.02), t=3))

        self.assertNotIn("old", memory.observations)
        self.assertEqual(set(memory.observations), {"mid", "new"})

    def test_keyframes_include_cluster_representatives_and_entropy_frames(self):
        memory = ShortTermGeometricMemory(entropy_threshold=0.15)
        memory.add(Observation("plain", Pose(0), embedding=(1.0, 0.0), frame=bytes([0, 0, 0, 0]), t=1))
        memory.add(Observation("rich", Pose(0), embedding=(0.0, 1.0), frame=bytes(range(16)), t=2))

        self.assertEqual(image_entropy(bytes([0, 0, 0, 0])), 0.0)
        self.assertIn("rich", {observation.id for observation in memory.keyframes()})

    def test_causal_recall_beats_poor_visual_similarity(self):
        memory = ShortTermGeometricMemory()
        graph = CausalEventGraph()
        observation = Observation(
            "cow-1",
            Pose(5, z=5),
            embedding=(1.0, 0.0),
            entities=("cow",),
        )
        memory.add(observation)
        graph.add_observation(observation)
        graph.add_edge("cow", CAN_OBTAIN, "beef")

        hits = memory.retrieve(Task("obtain beef", "beef", embedding=(0.0, 1.0)), graph)

        self.assertEqual(hits[0].observation.id, "cow-1")
        self.assertEqual(hits[0].visual_score, 0.0)
        self.assertEqual(hits[0].causal_score, 1.0)
        self.assertEqual(hits[0].reason, "cow CAN_OBTAIN beef")


class VLMTests(unittest.TestCase):
    def test_async_fixture_builder_updates_entities_causal_edges_and_cooccurrence(self):
        graph = CausalEventGraph()
        observation = Observation("obs-1", Pose(2, z=3), frame="cow-frame", t=7)
        client = FixtureVLMClient(
            {
                "cow-frame": {
                    "entities": ["cow", "grass"],
                    "causal_edges": [("cow", CAN_OBTAIN, "beef")],
                    "co_occurs": [("cow", "grass")],
                }
            }
        )

        asyncio.run(AsyncGraphBuilder(graph, client).process([observation]))

        self.assertEqual(set(observation.entities), {"cow", "grass"})
        self.assertEqual(graph.causal_sources(Task("obtain beef", "beef")), {"cow"})
        self.assertIn("grass", graph.out_edges[("cow", CO_OCCURS_WITH)])


class SchedulerTests(unittest.TestCase):
    def test_scheduler_reorders_order_free_opportunity(self):
        memory = ShortTermGeometricMemory()
        graph = CausalEventGraph()
        observation = Observation("cow-1", Pose(3, z=4), embedding=(1.0, 0.0), entities=("cow",))
        memory.add(observation)
        graph.add_observation(observation)
        graph.add_edge("cow", CAN_OBTAIN, "beef")
        tasks = [
            Task("find water", "water", urgency=0.1, embedding=(0.0, 1.0)),
            Task("collect logs", "logs", urgency=0.1, embedding=(0.0, 1.0)),
            Task("obtain beef", "beef", urgency=0.1, embedding=(0.0, 1.0)),
        ]

        order = OpportunisticTaskScheduler().reorder(tasks, memory, graph, Pose(0, z=0))

        self.assertEqual(order[0].name, "obtain beef")

    def test_scheduler_does_not_violate_dependencies(self):
        memory = ShortTermGeometricMemory()
        graph = CausalEventGraph()
        observation = Observation("cow-1", Pose(3, z=4), embedding=(1.0, 0.0), entities=("cow",))
        memory.add(observation)
        graph.add_observation(observation)
        graph.add_edge("cow", CAN_OBTAIN, "beef")
        tasks = [
            Task("find water", "water", urgency=0.1),
            Task("obtain beef", "beef", dependencies=("logs",), urgency=0.1),
            Task("collect logs", "logs", urgency=0.1),
        ]

        names = [
            task.name for task in OpportunisticTaskScheduler().reorder(tasks, memory, graph, Pose(0, z=0), completed=())
        ]

        self.assertGreater(names.index("obtain beef"), names.index("collect logs"))


class ExplorationTests(unittest.TestCase):
    def test_progressive_explorer_increases_coverage_without_revisiting(self):
        grid = GridMap(8, 8)
        grid.mark_visited((0, 0))
        explorer = ProgressiveExplorer(grid)
        before = grid.coverage
        targets = [explorer.step(target) for target in [(0, 0), (1, 0), (2, 0)]]

        self.assertTrue(all(target is not None for target in targets))
        self.assertGreater(grid.coverage, before)
        self.assertEqual(len(grid.visited), 4)

    def test_explorer_exposes_quadtree_frontier_and_voronoi_tiers(self):
        grid = GridMap(4, 4)
        grid.mark_visited((1, 1))
        explorer = ProgressiveExplorer(grid, max_depth=2)

        self.assertIsNotNone(explorer.quadtree_region((1, 1)))
        self.assertIn(explorer.frontier_target((1, 1)), {(0, 1), (1, 0), (1, 2), (2, 1)})
        self.assertIn(explorer.voronoi_target((1, 1)), grid.unvisited())
        self.assertTrue(explorer.stagnated(4.9, 30.0))


class EvalTests(unittest.TestCase):
    def test_offline_report_exercises_aba_and_abc_without_claiming_regression(self):
        report = build_offline_report()

        self.assertEqual(report["mode"], "offline")
        self.assertEqual(set(report["summary"]), {ABA_SPARSE, ABC_SPARSE})
        self.assertEqual(report["summary"][ABC_SPARSE]["regression_gate"], "skipped_offline_fixture")
        abc = [result for result in report["results"] if result["task"] == ABC_SPARSE][0]
        self.assertEqual(abc["details"]["order"][0], "obtain beef")

    def test_live_target_comparison_uses_paper_numbers(self):
        good = [EpisodeResult(ABC_SPARSE, True, 4500, {}) for _ in range(50)]
        bad = [EpisodeResult(ABC_SPARSE, index < 10, 9000, {}) for index in range(50)]

        self.assertTrue(summarize(good, PAPER_TARGETS[ABC_SPARSE], mode="live")["regression_gate"])
        self.assertFalse(summarize(bad, PAPER_TARGETS[ABC_SPARSE], mode="live")["regression_gate"])

    def test_live_requirements_name_missing_assets(self):
        self.assertIn("WISE_MINEDOJO_READY", missing_live_requirements({}))
        self.assertIn("WISE_MINECLIP_CHECKPOINT", missing_live_requirements({}))


if __name__ == "__main__":
    unittest.main()
