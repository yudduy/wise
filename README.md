# WISE

Minimal scaffold for [WISE: A Long-Horizon Agent in Minecraft with Why-Which
Reasoning](https://arxiv.org/html/2606.12852v1).

This repo does not reproduce the paper yet. It implements the WISE-shaped
offline harness and the readiness checks needed before live MineDojo/MrSteve
runs can be trusted.

## Quickstart

```bash
uv run python -m unittest
uv run python -m wise.eval --offline
uv run python -m wise.readiness
```

## Reproduction

WISE code/assets were not found in exact arXiv-title, arXiv-id, AlphaXiv,
GitHub, or Hugging Face checks. The first real gate is one MrSteve/STEVE-1
episode, not training a controller.

Required live env:

```bash
export WISE_MRSTEVE_ROOT=/path/to/MrSteve
export WISE_MINECLIP_CHECKPOINT=/path/to/mineclip.ckpt
export WISE_STEVE1_WEIGHTS=/path/to/steve1.weights
export WISE_VPT_MODEL=/path/to/2x.model
export WISE_VPT_NAV_CHECKPOINT=/path/to/vpt_nav.weights
export OPENAI_API_KEY=...
export WISE_OPENAI_MODEL=...
```

Live command shape:

```bash
uv run python -m wise.provision --mrsteve-root ../MrSteve
uv run python -m wise.provision --mrsteve-root ../MrSteve --write-env
uv run python -m wise.mrsteve --root ../MrSteve
uv run python -m wise.readiness --mrsteve-only --env-file .env.wise
uv run python -m wise.eval --require-live --task aba-sparse --variant mrsteve --env-file .env.wise
uv run python -m wise.eval --task aba-sparse --variant mrsteve --mrsteve-stats stats.txt
```

Paper targets encoded in `wise.eval`: ABA-Sparse `62% / 5981`, ABC-Sparse
`77% / 4620`.

## Shape

- `wise.memory`: PEM-style memory, causal graph, and lambda-weighted retrieval.
- `wise.vlm`: async graph construction and JSONL cache.
- `wise.scheduler`: dependency-safe opportunistic task scoring.
- `wise.explore`: quadtree, frontier, and Voronoi grid exploration scaffold.
- `wise.readiness`, `wise.provision`, `wise.mrsteve`: live setup helpers.
- `wise.eval`: offline smoke, paper targets, and ablation reporting.

Paper lineage, kept short: VPT -> STEVE-1 -> MrSteve -> WISE for the controller
stack; Voyager/GITM/JARVIS-1/MP5/Optimus-1/CoALA/ADAM for memory and planning;
frontier/SLAM/Voronoi work for exploration.
