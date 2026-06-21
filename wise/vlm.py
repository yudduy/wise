"""Async VLM graph construction for WISE."""

from __future__ import annotations

import asyncio
import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from .memory import CAN_OBTAIN, CO_OCCURS_WITH, CausalEventGraph, Observation


@dataclass(frozen=True)
class GraphUpdate:
    entities: tuple[str, ...] = ()
    causal_edges: tuple[tuple[str, str, str], ...] = ()
    co_occurs: tuple[tuple[str, str], ...] = ()
    node_types: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if self.node_types is None:
            object.__setattr__(self, "node_types", {})

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "GraphUpdate":
        return cls(
            entities=tuple(str(entity) for entity in value.get("entities", ())),
            causal_edges=tuple(_edge(edge) for edge in value.get("causal_edges", ())),
            co_occurs=tuple(_pair(pair) for pair in value.get("co_occurs", ())),
            node_types={str(k): str(v) for k, v in dict(value.get("node_types", {})).items()},
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "entities": list(self.entities),
            "causal_edges": [list(edge) for edge in self.causal_edges],
            "co_occurs": [list(pair) for pair in self.co_occurs],
            "node_types": dict(self.node_types or {}),
        }


class VLMClient(Protocol):
    async def analyze(self, observation: Observation) -> GraphUpdate:
        ...


class FixtureVLMClient:
    def __init__(self, fixtures: Mapping[str, Mapping[str, object]]):
        self.fixtures = fixtures

    async def analyze(self, observation: Observation) -> GraphUpdate:
        fixture_key = observation.frame if isinstance(observation.frame, str) else observation.id
        try:
            return GraphUpdate.from_mapping(self.fixtures[str(fixture_key)])
        except KeyError as error:
            raise KeyError(f"missing VLM fixture for {fixture_key!r}") from error


class JsonlGraphCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def get(self, key: str) -> GraphUpdate | None:
        if not self.path.exists():
            return None
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("key") == key:
                    return GraphUpdate.from_mapping(record["parsed_update"])
        return None

    def put(self, key: str, observation: Observation, update: GraphUpdate, *, prompt_version: str, model: str, raw_response: object = None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "key": key,
            "observation_id": observation.id,
            "frame_id": observation.frame if isinstance(observation.frame, str) else observation.id,
            "prompt_version": prompt_version,
            "model": model,
            "raw_response": raw_response,
            "parsed_update": update.to_mapping(),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


class CachedVLMClient:
    def __init__(
        self,
        client: VLMClient,
        cache: JsonlGraphCache,
        *,
        prompt_version: str = "wise-v1",
        model: str = "fixture",
    ):
        self.client = client
        self.cache = cache
        self.prompt_version = prompt_version
        self.model = model

    async def analyze(self, observation: Observation) -> GraphUpdate:
        cache_key = self.key(observation)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        update = await self.client.analyze(observation)
        self.cache.put(cache_key, observation, update, prompt_version=self.prompt_version, model=self.model)
        return update

    def key(self, observation: Observation) -> str:
        frame_id = observation.frame if isinstance(observation.frame, str) else observation.id
        return f"{self.prompt_version}:{self.model}:{observation.id}:{frame_id}"


class OpenAIVLMClient:
    """Minimal env-gated GPT-4o-compatible client."""

    def __init__(self, *, api_key: str, model: str, endpoint: str = "https://api.openai.com/v1/responses"):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required")
        if not model:
            raise ValueError("WISE_OPENAI_MODEL is required")
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint

    @classmethod
    def from_env(cls) -> "OpenAIVLMClient":
        return cls(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            model=os.environ.get("WISE_OPENAI_MODEL", ""),
            endpoint=os.environ.get("WISE_OPENAI_ENDPOINT", "https://api.openai.com/v1/responses"),
        )

    async def analyze(self, observation: Observation) -> GraphUpdate:
        return await asyncio.to_thread(self._request, observation)

    def _request(self, observation: Observation) -> GraphUpdate:
        payload = {
            "model": self.model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Extract Minecraft entities and causal relations as JSON with keys "
                                "entities, causal_edges, co_occurs, node_types. "
                                f"Observation id={observation.id}, frame={observation.frame!r}."
                            ),
                        }
                    ],
                }
            ],
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            body = json.loads(response.read().decode())
        text = _extract_text(body)
        try:
            return GraphUpdate.from_mapping(json.loads(text))
        except json.JSONDecodeError as error:
            raise ValueError(f"VLM response was not JSON: {text[:200]}") from error


class AsyncGraphBuilder:
    def __init__(self, graph: CausalEventGraph, client: VLMClient, *, workers: int = 1):
        self.graph = graph
        self.client = client
        self.workers = max(1, workers)
        self.queue: asyncio.Queue[Observation | None] = asyncio.Queue()
        self.tasks: list[asyncio.Task[None]] = []

    async def process(self, observations: Sequence[Observation]) -> CausalEventGraph:
        await self.start()
        for observation in observations:
            await self.queue.put(observation)
        await self.queue.join()
        await self.stop()
        return self.graph

    async def start(self) -> None:
        if not self.tasks:
            self.tasks = [asyncio.create_task(self._worker()) for _ in range(self.workers)]

    async def stop(self) -> None:
        for _ in self.tasks:
            await self.queue.put(None)
        await asyncio.gather(*self.tasks)
        self.tasks = []

    async def _worker(self) -> None:
        while True:
            observation = await self.queue.get()
            try:
                if observation is None:
                    return
                apply_update(self.graph, observation, await self.client.analyze(observation))
            finally:
                self.queue.task_done()


def apply_update(graph: CausalEventGraph, observation: Observation, update: GraphUpdate) -> Observation:
    observation.entities = tuple(dict.fromkeys((*observation.entities, *update.entities)))
    for entity, node_type in dict(update.node_types or {}).items():
        graph.add_node(entity, node_type)
    graph.add_observation(observation)
    for source, relation, target in update.causal_edges:
        graph.add_edge(source, relation, target)
    for source, target in update.co_occurs:
        graph.add_edge(source, CO_OCCURS_WITH, target)
    return observation


def _edge(value: object) -> tuple[str, str, str]:
    if isinstance(value, Mapping):
        return (str(value["source"]), str(value.get("relation", CAN_OBTAIN)), str(value["target"]))
    source, relation, target = value  # type: ignore[misc]
    return (str(source), str(relation), str(target))


def _pair(value: object) -> tuple[str, str]:
    if isinstance(value, Mapping):
        return (str(value["source"]), str(value["target"]))
    source, target = value  # type: ignore[misc]
    return (str(source), str(target))


def _extract_text(body: Mapping[str, object]) -> str:
    if "output_text" in body:
        return str(body["output_text"])
    output = body.get("output", [])
    for item in output if isinstance(output, list) else []:
        for content in item.get("content", []):  # type: ignore[union-attr]
            if content.get("type") in {"output_text", "text"}:  # type: ignore[union-attr]
                return str(content.get("text", ""))  # type: ignore[union-attr]
    raise ValueError("OpenAI response did not include output text")
