# WISE Reproduction Plan

This plan targets the first reproducible milestone: run the WISE harness on the
same controller family used by the paper, then compare ABA-Sparse and ABC-Sparse
against the reported numbers.

## Source Check

- WISE: arXiv `2606.12852`, no public implementation found in exact title,
  arXiv-id, AlphaXiv, GitHub, or Hugging Face searches as of 2026-06-20.
- MrSteve: public repo with MineCLIP, Steve-1, and VPT-Nav weight download
  script. DeepWiki confirms the main run surface is `main.py`, Hydra configs,
  `task_specs.yaml`, MineDojo/VPT wrappers, and `scripts/get_stats.py`.
- STEVE-1, OpenAI VPT, and MineCLIP all have public upstream repos. Use them
  through MrSteve first; do not retrain before proving the released loop runs.

## Required Local Assets

Set these before trying live regression:

```bash
export WISE_MRSTEVE_ROOT=/path/to/MrSteve
export WISE_MINECLIP_CHECKPOINT=/path/to/mineclip.ckpt
export WISE_STEVE1_WEIGHTS=/path/to/steve1.weights
export WISE_VPT_MODEL=/path/to/2x.model
export WISE_VPT_NAV_CHECKPOINT=/path/to/vpt_nav.weights
export OPENAI_API_KEY=...
export WISE_OPENAI_MODEL=...
```

Check readiness:

```bash
uv run python -m wise.readiness
```

Expected first result on a clean machine is failure with named missing paths.

Print the provisioning plan and optionally create a local env template:

```bash
uv run python -m wise.provision --mrsteve-root ../MrSteve
uv run python -m wise.provision --mrsteve-root ../MrSteve --write-env
```

Build the exact MrSteve smoke commands without running them:

```bash
python - <<'PY'
from wise.mrsteve import steve1_smoke, mrsteve_smoke
print(steve1_smoke("/path/to/MrSteve").display())
print(mrsteve_smoke("/path/to/MrSteve").display())
PY
```

## Step-by-Step Gates

1. **MrSteve baseline runs.**
   - Clone/setup MrSteve outside this repo.
   - Run its `prepare_models.sh`.
   - Verify `uv run main.py task=log_water_bucket_aba_randinit agent=steve1 n_episodes=1`.
   - Verify `uv run main.py task=log_water_bucket_aba_randinit agent=mrsteve n_episodes=1`.

2. **Adapter captures real observations.**
   - Read frame, pose, yaw/pitch, timestep, and episode metadata from the
     MrSteve/MineDojo wrapper.
   - Feed frames into MineCLIP and store `Observation` records in `wise.memory`.

3. **WISE graph runs live but cached.**
   - Select keyframes from memory.
   - Send keyframes to GPT-4o asynchronously.
   - Save every graph update to JSONL before using it in an episode.

4. **ABA/ABC task reconstruction.**
   - Start from MrSteve `task_specs.yaml`.
   - Reconstruct ABA-Sparse and ABC-Sparse task variants from the WISE paper.
   - Keep evaluator success predicates separate from agent observations.

5. **50-episode measurement.**
   - Run ABA-Sparse and ABC-Sparse for 50 seeds each.
   - Report success rate, average timesteps, failure reasons, and GPT calls.
   - Compare against WISE Table 2 targets encoded in `wise.eval`.

6. **Ablations.**
   - Full WISE.
   - No causal graph.
   - No opportunistic scheduler.
   - No exploration.
   - Steve-1/MrSteve baselines from the same task runner.
   - Offline command shape:
     `uv run python -m wise.eval --offline --task abc-sparse --episodes 5 --variant no-scheduler`

## Do Not Train Yet

Training is only justified after the released MrSteve/STEVE-1/VPT-Nav/MineCLIP
path fails for a concrete reason. The first expensive operation should be a
live 1-episode smoke run, not model training.
