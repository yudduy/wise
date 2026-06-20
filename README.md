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
