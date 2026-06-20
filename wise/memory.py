"""WISE memory: PEM-style geometry plus a causal event graph."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from math import floor, log2, sqrt
from typing import Iterable, Sequence


CAN_OBTAIN = "CAN_OBTAIN"
CO_OCCURS_WITH = "CO_OCCURS_WITH"
POSITION = "POSITION"
TIMESTEP = "TIMESTEP"


def key(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sqrt(sum(a * a for a in left))
    right_norm = sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return 1.0 - cosine(left, right)


def mean_embedding(observations: Sequence["Observation"]) -> tuple[float, ...]:
    if not observations or not observations[0].embedding:
        return ()
    width = len(observations[0].embedding)
    totals = [0.0] * width
    count = 0
    for observation in observations:
        if len(observation.embedding) != width:
            continue
        count += 1
        for index, value in enumerate(observation.embedding):
            totals[index] += value
    if count == 0:
        return ()
    return tuple(value / count for value in totals)


def image_entropy(frame: object) -> float:
    if frame is None:
        return 0.0
    if isinstance(frame, str):
        data = frame.encode()
    elif isinstance(frame, (bytes, bytearray)):
        data = bytes(frame)
    else:
        try:
            data = bytes(int(value) & 0xFF for value in frame)  # type: ignore[arg-type]
        except TypeError:
            return 0.0
    if not data:
        return 0.0
    counts = Counter(data)
    denominator = log2(max(2, len(counts)))
    entropy = 0.0
    for count in counts.values():
        p = count / len(data)
        entropy -= p * log2(p)
    return entropy / denominator


@dataclass(frozen=True)
class Pose:
    x: float
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0

    def distance_to(self, other: "Pose") -> float:
        return sqrt((self.x - other.x) ** 2 + (self.z - other.z) ** 2)


@dataclass
class Observation:
    id: str
    pose: Pose
    embedding: tuple[float, ...] = ()
    frame: object = None
    t: int = 0
    entities: tuple[str, ...] = ()


@dataclass
class Task:
    name: str
    target: str = ""
    dependencies: tuple[str, ...] = ()
    urgency: float = 0.0
    embedding: tuple[float, ...] = ()
    completed: bool = False

    @property
    def goal(self) -> str:
        return key(self.target or self.name.split()[-1])

    def executable(self, completed: Iterable[str]) -> bool:
        done = {key(item) for item in completed}
        return all(key(item) in done for item in self.dependencies)


@dataclass(frozen=True)
class MemoryHit:
    observation: Observation
    score: float
    visual_score: float
    causal_score: float
    reason: str


@dataclass
class EventCluster:
    id: str
    observations: list[Observation] = field(default_factory=list)
    centroid: tuple[float, ...] = ()

    def add(self, observation: Observation) -> None:
        self.observations.append(observation)
        self.centroid = mean_embedding(self.observations)

    def remove(self, observation: Observation) -> None:
        self.observations.remove(observation)
        self.centroid = mean_embedding(self.observations)

    def representative(self) -> Observation:
        if not self.centroid:
            return self.observations[0]
        return max(self.observations, key=lambda observation: cosine(observation.embedding, self.centroid))


@dataclass
class PlaceBucket:
    key: tuple[int, int, int]
    event_clusters: list[EventCluster] = field(default_factory=list)


class ShortTermGeometricMemory:
    """PEM-shaped short-term memory with DP-means event clusters."""

    def __init__(
        self,
        *,
        max_observations: int = 1000,
        place_resolution: float = 8.0,
        yaw_resolution: float = 45.0,
        dp_lambda: float = 0.15,
        entropy_threshold: float = 0.15,
    ):
        self.max_observations = max_observations
        self.place_resolution = place_resolution
        self.yaw_resolution = yaw_resolution
        self.dp_lambda = dp_lambda
        self.entropy_threshold = entropy_threshold
        self.observations: dict[str, Observation] = {}
        self.order: list[str] = []
        self.place_buckets: dict[tuple[int, int, int], PlaceBucket] = {}

    def add(self, observation: Observation) -> None:
        if observation.id in self.observations:
            self.remove(observation.id)
        self.observations[observation.id] = observation
        self.order.append(observation.id)
        bucket = self.place_buckets.setdefault(self._place_key(observation.pose), PlaceBucket(self._place_key(observation.pose)))
        cluster = self._nearest_cluster(bucket, observation)
        if cluster is None:
            cluster = EventCluster(f"{bucket.key}:{len(bucket.event_clusters)}")
            bucket.event_clusters.append(cluster)
        cluster.add(observation)
        self._evict_over_capacity()

    def remove(self, observation_id: str) -> None:
        observation = self.observations.pop(observation_id)
        self.order.remove(observation_id)
        for bucket_key, bucket in list(self.place_buckets.items()):
            for cluster in list(bucket.event_clusters):
                if observation in cluster.observations:
                    cluster.remove(observation)
                if not cluster.observations:
                    bucket.event_clusters.remove(cluster)
            if not bucket.event_clusters:
                del self.place_buckets[bucket_key]

    def clusters(self) -> list[EventCluster]:
        return [cluster for bucket in self.place_buckets.values() for cluster in bucket.event_clusters]

    def search(self, query_embedding: Sequence[float], top_k: int = 10) -> list[tuple[Observation, float]]:
        ranked = [
            (observation, cosine(query_embedding, observation.embedding))
            for observation in self.observations.values()
        ]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:top_k]

    def retrieve(
        self,
        task: Task,
        graph: "CausalEventGraph",
        *,
        query_embedding: Sequence[float] | None = None,
        lambda_weight: float = 0.5,
        top_k: int = 10,
    ) -> list[MemoryHit]:
        query = tuple(query_embedding or task.embedding)
        hits: list[MemoryHit] = []
        for observation in self.observations.values():
            visual = cosine(query, observation.embedding)
            causal = 1.0 if graph.causal_match(observation, task) else 0.0
            score = lambda_weight * visual + (1.0 - lambda_weight) * causal
            if score > 0.0:
                hits.append(MemoryHit(observation, score, visual, causal, graph.reason(observation, task) if causal else "visual"))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]

    def keyframes(self) -> list[Observation]:
        selected: dict[str, Observation] = {}
        for cluster in self.clusters():
            selected.setdefault(cluster.representative().id, cluster.representative())
        for observation in self.observations.values():
            if image_entropy(observation.frame) >= self.entropy_threshold:
                selected.setdefault(observation.id, observation)
        return sorted(selected.values(), key=lambda observation: (observation.t, observation.id))

    def _place_key(self, pose: Pose) -> tuple[int, int, int]:
        return (
            floor(pose.x / self.place_resolution),
            floor(pose.z / self.place_resolution),
            floor((pose.yaw % 360.0) / self.yaw_resolution),
        )

    def _nearest_cluster(self, bucket: PlaceBucket, observation: Observation) -> EventCluster | None:
        if not bucket.event_clusters:
            return None
        nearest = min(bucket.event_clusters, key=lambda cluster: cosine_distance(observation.embedding, cluster.centroid))
        if cosine_distance(observation.embedding, nearest.centroid) > self.dp_lambda:
            return None
        return nearest

    def _evict_over_capacity(self) -> None:
        while len(self.observations) > self.max_observations:
            largest = max(self.clusters(), key=lambda cluster: len(cluster.observations))
            oldest = min(largest.observations, key=lambda observation: (observation.t, self.order.index(observation.id)))
            self.remove(oldest.id)


@dataclass(frozen=True)
class GraphEdge:
    source: str
    relation: str
    target: str


class CausalEventGraph:
    """Small semantic graph for WISE causal retrieval."""

    def __init__(self):
        self.nodes: dict[str, dict[str, object]] = {}
        self.edges: set[GraphEdge] = set()
        self.out_edges: dict[tuple[str, str], set[str]] = {}
        self.in_edges: dict[tuple[str, str], set[str]] = {}
        self.observation_positions: dict[str, Pose] = {}
        self.observation_entities: dict[str, set[str]] = {}
        self.observations_by_entity: dict[str, set[str]] = {}

    def add_node(self, name: str, node_type: str = "entity", **attrs: object) -> str:
        node = key(name)
        self.nodes.setdefault(node, {"type": node_type})
        self.nodes[node].update(attrs)
        return node

    def add_edge(self, source: str, relation: str, target: str) -> None:
        source_key = self.add_node(source)
        target_key = self.add_node(target)
        edge = GraphEdge(source_key, relation, target_key)
        self.edges.add(edge)
        self.out_edges.setdefault((source_key, relation), set()).add(target_key)
        self.in_edges.setdefault((target_key, relation), set()).add(source_key)

    def add_observation(self, observation: Observation) -> None:
        observation_node = self.add_node(observation.id, "observation", t=observation.t)
        self.observation_positions[observation.id] = observation.pose
        entities = {key(entity) for entity in observation.entities}
        self.observation_entities[observation.id] = entities
        self.add_edge(observation_node, POSITION, f"{observation.pose.x:.3f},{observation.pose.z:.3f}")
        self.add_edge(observation_node, TIMESTEP, str(observation.t))
        for entity in entities:
            self.add_node(entity, "entity")
            self.observations_by_entity.setdefault(entity, set()).add(observation.id)
            self.add_edge(entity, POSITION, observation.id)

    def causal_sources(self, task: Task) -> set[str]:
        return set(self.in_edges.get((task.goal, CAN_OBTAIN), set()))

    def causal_match(self, observation: Observation, task: Task) -> bool:
        entities = self.observation_entities.get(observation.id, {key(entity) for entity in observation.entities})
        return task.goal in entities or bool(entities & self.causal_sources(task))

    def reason(self, observation: Observation, task: Task) -> str:
        entities = self.observation_entities.get(observation.id, {key(entity) for entity in observation.entities})
        for entity in sorted(entities & self.causal_sources(task)):
            return f"{entity} {CAN_OBTAIN} {task.goal}"
        if task.goal in entities:
            return f"observed {task.goal}"
        return "no causal match"
