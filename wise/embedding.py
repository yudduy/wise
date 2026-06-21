"""Embedding boundary for WISE memory."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Protocol, Sequence

from .memory import Observation, ShortTermGeometricMemory


class FrameEmbedder(Protocol):
    def embed(self, frame: object) -> tuple[float, ...]:
        ...


class FixtureEmbedder:
    def __init__(self, embeddings: Mapping[object, Sequence[float]]):
        self.embeddings = embeddings

    def embed(self, frame: object) -> tuple[float, ...]:
        try:
            return tuple(float(value) for value in self.embeddings[frame])
        except KeyError as error:
            raise KeyError(f"missing fixture embedding for {frame!r}") from error


class MineCLIPEmbedder:
    def __init__(self, checkpoint: str | Path):
        self.checkpoint = Path(checkpoint)
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"MineCLIP checkpoint not found: {self.checkpoint}")

    def embed(self, frame: object) -> tuple[float, ...]:
        raise RuntimeError(
            "MineCLIP runtime is not wired in this scaffold yet; install MineCLIP "
            "and replace MineCLIPEmbedder.embed with the real model call."
        )


def add_embedded_observation(
    memory: ShortTermGeometricMemory,
    observation: Observation,
    embedder: FrameEmbedder,
) -> Observation:
    observation.embedding = embedder.embed(observation.frame)
    memory.add(observation)
    return observation
