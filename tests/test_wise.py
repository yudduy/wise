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


if __name__ == "__main__":
    unittest.main()
