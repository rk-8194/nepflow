# Information-Entropy Selection: Computational Glossary

This glossary describes terminology used by NEPflow's **exact k-nearest-neighbour (kNN)** calculation during information-entropy bandwidth calibration. Here, a *descriptor* is a numerical representation of an atomic environment, and a *location* is a distinct descriptor vector after exact duplicates have been grouped.

## CPU tile

A **CPU tile** is a small block of the distance-calculation workload processed together on the CPU rather than computing or storing the entire source-versus-reference distance matrix at once. Tiling limits working memory and allows results to be accumulated incrementally.

In NEPflow's `TiledCPUNeighbourIndex`, the CPU implementation visits one source descriptor at a time and compares it against successive **target tiles** of other descriptors. For example, a target tile of 2,048 means that one source can be compared with up to 2,048 reference descriptors in a batch. CPU tiles **do not necessarily form a 2,048 × 2,048 matrix**.

A *CPU tile* is a batching concept; it does not mean a CPU core, thread, or a distinct computer node.

## GPU tile

A **GPU tile** is a block of source–target distance calculations executed together on one GPU, using vectorised operations. Rather than storing the full distance matrix of all descriptors against all descriptors, the GPU evaluates many smaller tiles and merges the nearest-neighbour candidates.

NEPflow's exact tiled GPU backend has two tile dimensions:

- **Source tile:** the number of atomic environments whose nearest neighbours are being calculated together.
- **Target tile:** the number of reference environments compared against those sources in that step.

For example, a **2,048 × 2,048** GPU tile evaluates up to **4,194,304 pairwise distances**, each across the descriptor dimensions (42 in the October 2026 production dataset). The backend processes further target tiles to cover the full reference pool and further source tiles to cover all source environments.

A GPU tile is **not** a GPU device; one GPU processes many tiles. Actual tile dimensions can be reduced by memory constraints. GPU compute tile size must remain conceptually separate from checkpoint-write frequency.

## CPU certification

**CPU certification** verifies the exactness of the nearest neighbours identified by the GPU for **one source location**. NEPflow recalculates distances to the retained GPU candidate neighbours using its canonical CPU distance calculation, orders them with the required deterministic tie-breaking, and checks the GPU's conservative exclusion bounds to establish that no discarded location could change the nearest-`k` result.

When that proof succeeds, the neighbour result is accepted as certified. A CPU certification is **not normally a full CPU search** against every descriptor; that expensive operation is a separate *full CPU rescan* used only as an exceptional fallback.

## Boundary refinement

A **boundary refinement** occurs when initial GPU results do not give a sufficient mathematical proof of the nearest-`k` boundary. For example, the eighth and ninth nearest environments may have identical or extremely similar distances, or the floating-point error intervals may overlap.

Instead of immediately recomputing and sorting distances to the entire dataset on the CPU, NEPflow makes a further **GPU-assisted pass** over reference locations to find all candidates that could compete at the boundary. It then recalculates the distances of those candidates with the canonical CPU calculation and applies deterministic tie-breaking to resolve the exact nearest neighbours.

The GPU candidate filter must conservatively include **every potentially competitive location**, accounting for floating-point error in both GPU and canonical CPU calculations. If exactness cannot be established, the operation must fail explicitly rather than silently accept an approximate result.

## Related terms

- **Source location:** the descriptor whose neighbours are being sought.
- **Target/reference location:** a descriptor compared against the source.
- **Full CPU rescan:** an exhaustive canonical CPU distance calculation and ordering against all distinct locations for a particular source; substantially more expensive than ordinary CPU certification.
- **Checkpoint batch:** a completed, certified group of source results that may be persisted for restart. Its size/frequency is an I/O and recovery concern, not a definition of the CPU or GPU computation tile.
- **Completed locations vs GPU evaluations:** GPU evaluation counters can advance while completed/certified-location counters remain at zero, because the implementation performs the exhaustive GPU distance pass before its CPU final-certification phase.

## Implementation references

- `src/nepflow/stages/selection/algorithms/information_entropy/neighbours.py` — `TiledCPUNeighbourIndex`, canonical CPU distance operations.
- `src/nepflow/stages/selection/algorithms/information_entropy/gpu_neighbours.py` — `ExactTiledGPUNeighbourIndex`, GPU source/target tiles, `_certify_final()`, `_adaptive_gpu_boundary_refinement()`, and multi-GPU checkpoint publication.
