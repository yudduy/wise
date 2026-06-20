# WISE Paper Map

WISE is a low-level Minecraft controller extension, not a new high-level
planner. The useful scaffold is the smallest version that preserves the paper's
three connected modules: causal memory, opportunistic scheduling, and
multi-scale exploration.

## Controller Lineage

- VPT learns Minecraft behavior from unlabeled videos plus inverse dynamics.
- STEVE-1 instruction-tunes VPT with text-conditioned MineCLIP latents.
- MrSteve adds Place Event Memory for what-where-when recall.
- WISE keeps that low-level-controller framing but adds why/which causal
  reasoning over memories.

## Memory And Planning Lineage

- Voyager stores executable skills and uses LLM feedback loops.
- GITM uses text-based knowledge and memory for Minecraft planning.
- JARVIS-1 and MP5 add multimodal memory and active perception.
- Optimus-1 separates world knowledge from multimodal experience memory.
- CoALA gives the broader language-agent memory/action framing.
- ADAM builds causal world knowledge, but WISE applies causal memory directly
  to low-level retrieval and scheduling.

## Exploration Lineage

- Count-based exploration is simple but can loop locally.
- Frontier exploration pushes boundaries between explored and unknown regions.
- SLAM/topological memory systems keep spatial structure for navigation.
- Voronoi-style local completion fills interior gaps that frontier search misses.

## Implementation Interpretation

- Reuse the PEM shape: place buckets, event clusters, MineCLIP-style embeddings,
  DP-means clustering, and top-k similarity retrieval.
- Add WISE's Causal Event Graph: VLM-extracted entity/action/environment nodes,
  position/timestep/causal edges, and incoming `CAN_OBTAIN` traversal for tasks.
- Preserve the retrieval equation: `lambda * sim + (1 - lambda) * CausalMatch`,
  defaulting to `lambda = 0.5` and `top_k = 10`.
- Preserve the scheduler weights: urgency `0.3`, causal relevance `0.5`, and
  navigation cost `0.2`.
- Preserve the exploration hierarchy: quadtree global target, frontier regional
  refinement, Voronoi local completion, then stagnation recovery.

## Regression Targets

- Table 2 ABA-Sparse: 62% success, 5,981 average timesteps.
- Table 2 ABC-Sparse: 77% success, 4,620 average timesteps.
- Table 3 exploration ablation: full quadtree + frontier + Voronoi reaches 0.97
  coverage at 10,000 simulated steps.

Offline fixtures only prove module wiring. Live MineDojo/MineRL/VPT/MineCLIP
runs are required before claiming paper reproduction.
