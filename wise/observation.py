"""Adapters from MineDojo/MrSteve observations into WISE observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .memory import Observation, Pose


@dataclass(frozen=True)
class AdaptedObservation:
    observation: Observation
    evaluator_state: dict[str, object]


def adapt_minedojo_observation(
    raw: Mapping[str, object],
    *,
    observation_id: str,
    timestep: int,
    embedding: Sequence[float] = (),
) -> AdaptedObservation:
    observation = Observation(
        id=observation_id,
        pose=_pose(raw),
        embedding=tuple(float(value) for value in embedding),
        frame=raw.get("rgb", raw.get("pov", raw.get("frame"))),
        t=timestep,
        entities=_entities(raw),
    )
    evaluator_state = {
        key: raw[key]
        for key in ("inventory", "equipped_items", "life_stats")
        if key in raw
    }
    return AdaptedObservation(observation, evaluator_state)


def _pose(raw: Mapping[str, object]) -> Pose:
    stats = raw.get("location_stats", raw)
    if isinstance(stats, Mapping):
        return Pose(
            x=float(stats.get("x", stats.get("pos_x", 0.0))),
            y=float(stats.get("y", stats.get("pos_y", 0.0))),
            z=float(stats.get("z", stats.get("pos_z", 0.0))),
            yaw=float(stats.get("yaw", 0.0)),
            pitch=float(stats.get("pitch", 0.0)),
        )
    if isinstance(stats, Sequence) and len(stats) >= 3 and not isinstance(stats, (str, bytes, bytearray)):
        yaw = float(stats[3]) if len(stats) > 3 else 0.0
        pitch = float(stats[4]) if len(stats) > 4 else 0.0
        return Pose(float(stats[0]), float(stats[1]), float(stats[2]), yaw, pitch)
    return Pose(0.0)


def _entities(raw: Mapping[str, object]) -> tuple[str, ...]:
    values = raw.get("entities", raw.get("nearby_entities", ()))
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        return ()
    entities: list[str] = []
    for value in values:
        if isinstance(value, Mapping):
            name = value.get("name", value.get("type"))
            if name:
                entities.append(str(name))
        else:
            entities.append(str(value))
    return tuple(dict.fromkeys(entities))
