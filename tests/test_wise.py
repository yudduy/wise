import unittest
import asyncio
import tempfile
from pathlib import Path

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
from wise.embedding import FixtureEmbedder, MineCLIPEmbedder, add_embedded_observation
from wise.explore import GridMap, ProgressiveExplorer
from wise.eval import ABC_SPARSE, ABA_SPARSE, NO_SCHEDULER, PAPER_TARGETS, EpisodeResult, build_offline_report, missing_live_requirements, summarize
from wise.mrsteve import mrsteve_smoke, stats_command, steve1_smoke
from wise.observation import adapt_minedojo_observation
from wise.provision import plan as provision_plan, write_env
from wise.readiness import check as readiness_check
from wise.scheduler import OpportunisticTaskScheduler
from wise.tasks import SPARSE_TASKS, selected_seeds, task_spec
from wise.vlm import AsyncGraphBuilder, CachedVLMClient, FixtureVLMClient, GraphUpdate, JsonlGraphCache


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

    def test_cached_vlm_client_reuses_jsonl_graph_update(self):
        class CountingClient:
            def __init__(self):
                self.calls = 0

            async def analyze(self, observation):
                self.calls += 1
                return GraphUpdate(entities=("cow",), causal_edges=(("cow", CAN_OBTAIN, "beef"),))

        with tempfile.TemporaryDirectory() as tmp:
            client = CountingClient()
            cache = JsonlGraphCache(Path(tmp) / "graph.jsonl")
            cached = CachedVLMClient(client, cache, prompt_version="p1", model="fixture")
            observation = Observation("obs", Pose(0), frame="frame")

            first = asyncio.run(cached.analyze(observation))
            second = asyncio.run(cached.analyze(observation))

        self.assertEqual(first, second)
        self.assertEqual(client.calls, 1)


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
        self.assertIn("WISE_MRSTEVE_ROOT", missing_live_requirements({}))
        self.assertIn("WISE_MINECLIP_CHECKPOINT", missing_live_requirements({}))

    def test_offline_report_respects_episode_count_and_seeds(self):
        report = build_offline_report([ABC_SPARSE], episodes=5)

        self.assertEqual(report["summary"][ABC_SPARSE]["episodes"], 5)
        self.assertEqual([result["details"]["seed"] for result in report["results"]], [0, 1, 2, 3, 4])

    def test_offline_ablation_reports_failures_and_gpt_calls(self):
        report = build_offline_report([ABC_SPARSE], episodes=1, variant=NO_SCHEDULER)

        self.assertEqual(report["variant"], NO_SCHEDULER)
        self.assertFalse(report["results"][0]["success"])
        self.assertEqual(report["summary"][ABC_SPARSE]["failures"], ["offline_ablation_failed"])
        self.assertEqual(report["summary"][ABC_SPARSE]["gpt_calls"], 1)


class SparseTaskTests(unittest.TestCase):
    def test_sparse_task_specs_encode_success_and_timeout(self):
        spec = task_spec(ABC_SPARSE)

        self.assertEqual(spec, SPARSE_TASKS[ABC_SPARSE])
        self.assertTrue(spec.succeeded({"beef": 1}))
        self.assertFalse(spec.succeeded({"beef": 0}))
        self.assertTrue(spec.timed_out(spec.timeout_steps))
        self.assertEqual(selected_seeds(ABC_SPARSE, 3), (0, 1, 2))


class ReadinessTests(unittest.TestCase):
    def test_readiness_passes_with_expected_files_and_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "MrSteve"
            for rel in ("config/main.yaml", "scripts/get_stats.py"):
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("", encoding="utf-8")
            for rel in ("main.py", "prepare_models.sh", "task_specs.yaml"):
                (root / rel).write_text("", encoding="utf-8")
            files = {}
            for name in ("mineclip.ckpt", "steve1.weights", "2x.model", "vpt_nav.weights"):
                path = Path(tmp) / name
                path.write_text("", encoding="utf-8")
                files[name] = str(path)

            report = readiness_check(
                {
                    "OPENAI_API_KEY": "test",
                    "WISE_OPENAI_MODEL": "gpt-4o-test",
                    "WISE_MRSTEVE_ROOT": str(root),
                    "WISE_MINECLIP_CHECKPOINT": files["mineclip.ckpt"],
                    "WISE_STEVE1_WEIGHTS": files["steve1.weights"],
                    "WISE_VPT_MODEL": files["2x.model"],
                    "WISE_VPT_NAV_CHECKPOINT": files["vpt_nav.weights"],
                }
            )

        self.assertTrue(report["ready"])
        self.assertEqual(report["next_step"], "run one MrSteve/Steve-1 episode")


class MrSteveCommandTests(unittest.TestCase):
    def test_steve1_and_mrsteve_smoke_commands(self):
        self.assertEqual(
            steve1_smoke("/repo").argv,
            (
                "uv",
                "run",
                "main.py",
                "task=log_water_bucket_aba_randinit",
                "agent=steve1",
                "n_episodes=1",
            ),
        )
        self.assertEqual(mrsteve_smoke("/repo").argv[4], "agent=mrsteve")

    def test_stats_command_uses_mrsteve_stats_script(self):
        self.assertEqual(
            stats_command("/repo", "outputs/task/agent/*").argv,
            ("uv", "run", "scripts/get_stats.py", "outputs/task/agent/*"),
        )


class ProvisionTests(unittest.TestCase):
    def test_provision_plan_names_clone_setup_and_smoke_commands(self):
        planned = provision_plan("/repo/MrSteve")

        self.assertEqual(planned["clone"], "git clone https://github.com/frechele/MrSteve /repo/MrSteve")
        self.assertIn("uv run bash prepare_models.sh", planned["setup"])
        self.assertIn("agent=steve1", planned["smoke"][0])
        self.assertIn("agent=mrsteve", planned["smoke"][1])

    def test_write_env_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env.wise"
            write_env(env_path, "/repo/MrSteve")

            self.assertIn("WISE_MRSTEVE_ROOT=/repo/MrSteve", env_path.read_text(encoding="utf-8"))
            with self.assertRaises(FileExistsError):
                write_env(env_path, "/other")


class ObservationAdapterTests(unittest.TestCase):
    def test_minedojo_dict_adapts_to_wise_observation_without_inventory_leak(self):
        adapted = adapt_minedojo_observation(
            {
                "rgb": "frame-token",
                "location_stats": {"x": 1, "y": 64, "z": -3, "yaw": 90, "pitch": 10},
                "nearby_entities": [{"name": "cow"}, {"type": "oak_log"}, "water"],
                "inventory": {"beef": 0},
            },
            observation_id="obs-7",
            timestep=42,
            embedding=(0.1, 0.2),
        )

        self.assertEqual(adapted.observation.id, "obs-7")
        self.assertEqual(adapted.observation.frame, "frame-token")
        self.assertEqual(adapted.observation.pose.x, 1.0)
        self.assertEqual(adapted.observation.pose.yaw, 90.0)
        self.assertEqual(adapted.observation.entities, ("cow", "oak_log", "water"))
        self.assertEqual(adapted.observation.embedding, (0.1, 0.2))
        self.assertEqual(adapted.evaluator_state, {"inventory": {"beef": 0}})


class EmbeddingTests(unittest.TestCase):
    def test_fixture_embedder_wires_observation_into_memory(self):
        memory = ShortTermGeometricMemory()
        observation = Observation("obs", Pose(0), frame="frame-a")

        add_embedded_observation(memory, observation, FixtureEmbedder({"frame-a": (0.3, 0.7)}))

        self.assertEqual(observation.embedding, (0.3, 0.7))
        self.assertIn("obs", memory.observations)

    def test_mineclip_placeholder_validates_checkpoint_and_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "mineclip.ckpt"
            checkpoint.write_text("", encoding="utf-8")
            embedder = MineCLIPEmbedder(checkpoint)

            with self.assertRaisesRegex(RuntimeError, "MineCLIP runtime is not wired"):
                embedder.embed("frame")


if __name__ == "__main__":
    unittest.main()
