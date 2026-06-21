# WISE

Reproduction scaffold for [WISE: A Long-Horizon Agent in Minecraft with
Why-Which Reasoning](https://arxiv.org/html/2606.12852v1).

Code-release check, June 19 2026: exact GitHub searches for the title,
`Which-Why Informed Semantic Explorer`, and `2606.12852` did not surface an
official implementation. The visible GitHub hit was a paper index, not source
code, so this repo reimplements from the arXiv v1 method and appendix.

## Quickstart

```bash
uv run python -m unittest
uv run python -m wise.readiness
uv run python -m wise.eval --offline
```

## Reproduction Targets

Paper Table 2 targets are encoded as regression gates:

- ABA-Sparse: 62% success, 5,981 average timesteps.
- ABC-Sparse: 77% success, 4,620 average timesteps.

Live validation is intentionally explicit:

```bash
uv run python -m wise.eval --task aba-sparse --episodes 50 --require-live
uv run python -m wise.eval --task abc-sparse --episodes 50 --require-live
```

The offline mode is a deterministic smoke path. It exercises the same module
boundaries without pretending to reproduce Minecraft numbers.

## Shape

- `wise.memory`: PEM-style place/event memory, DP-means clusters, hybrid
  keyframe selection, causal graph, and lambda-weighted retrieval.
- `wise.vlm`: async graph-construction queue, offline fixture client, and
  env-gated OpenAI client.
- `wise.scheduler`: dependency-safe opportunistic task scoring with paper
  weights.
- `wise.explore`: quadtree, frontier, and Voronoi grid exploration scaffold.
- `wise.mrsteve`: external MrSteve command construction for smoke runs.
- `wise.observation` and `wise.embedding`: MineDojo observation and MineCLIP
  embedding boundaries.
- `wise.readiness` and `wise.eval`: live asset checks, ABA/ABC smoke runs, and
  paper-target gates.

## Live Requirements

Live regression is intentionally blocked until these are explicit:

- `WISE_MRSTEVE_ROOT=/path/to/MrSteve`
- `WISE_MINECLIP_CHECKPOINT=/path/to/checkpoint`
- `WISE_STEVE1_WEIGHTS=/path/to/checkpoint`
- `WISE_VPT_MODEL=/path/to/2x.model`
- `WISE_VPT_NAV_CHECKPOINT=/path/to/checkpoint`
- `OPENAI_API_KEY=...`
- `WISE_OPENAI_MODEL=...`

The scaffold does not fabricate the paper's 1534-entry grounding database; tests
use a tiny fixture and the live path should load the real database when it is
available.
