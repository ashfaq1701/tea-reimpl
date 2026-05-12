# tea-implementation

An independent, open-source reimplementation of **TEA** (Huan et al., EuroSys '23) — a CPU temporal-graph random-walk engine built from PAT + HPAT + Auxiliary-Index hybrid sampling.
Used as the CPU baseline for cross-architecture comparison in the Tempest paper, since the original TEA source is not publicly available.

## Citation

> Chengying Huan, Shuaiwen Leon Song, Santosh Pandey, Hang Liu, Yongchao Liu, Baptiste Lepers, Changhua He, Kang Chen, Jinlei Jiang, Yongwei Wu. **TEA: A General-Purpose Temporal Graph Random Walk Engine.** In *Eighteenth European Conference on Computer Systems (EuroSys '23)*, May 8–12, 2023, Rome, Italy. ACM, 17 pages. https://doi.org/10.1145/3552326.3567491

## Why this is a reliable TEA reimplementation

- **Candidate edge set Γ_t(u)** (paper §2.1, §2.3). *Paper:* out-edges of `u` with `t' > t_prev`, time-DESC per vertex. *Ours:* `TemporalGraph` CSR with per-vertex DESC sort and `candidate_set_len` via strict-`>` binary search.

- **Linear bias** (paper §2.3.I, Fig 5). *Paper:* `δ(e) = rank(e)`, newest=D, oldest=1; example gives weights `7,6,5,4,3,2,1`. *Ours:* `LinearBias::compute_weights` returns those exact ranks from position in the time-DESC slice.

- **Exponential bias with static-weight cancellation** (paper §2.3.II, Eq 3). *Paper:* `δ(e) = exp(t_i)` after cancelling the `−t` term in the normalizer, so weights are static post-preprocess. *Ours:* `ExponentialBias` stores `δ = exp(t_i − t_max_u)`, softmax-equivalent to `exp(t_i)`; never overflows; tables built once.

- **Temporal node2vec rejection on β** (paper §2.3.III, Alg 2 lines 19–21). *Paper:* `β ∈ {1/p, 1, 1/q}`, accept proposal with prob proportional to β. *Ours:* `Node2VecBias::accept_ratio` returns `β/β_max ∈ (0,1]`; `rejection_loop` accepts on `u01 < accept_ratio` with retry cap.

- **PAT data structure** (paper §3.2). *Paper:* partition edges into trunks of size `trunkSize ≈ ⌊√D⌋`, alias table per trunk, prefix sum across trunks. *Ours:* `Pat<BiasT>` with `trunk_size = max(⌈√D⌉, 8)`, flat per-vertex alias arena, flat per-vertex inclusive cumsum arena.

- **PAT sampling Cases ① and ②** (paper §3.2). *Paper:* full trunks → ITS over cumsum + alias-sample inside; partial last trunk → rebuild prefix sum inside, ITS within. *Ours:* `sample_pat` computes `num_full = G/T`, `partial = G%T`; on partial>0, `bias.compute_weights(..., slice_start_pos)` recomputes weights with the same per-vertex normalization used at build, then on-the-fly cumsum + ITS.

- **HPAT hierarchy** (paper §3.3, Eq 5–7). *Paper:* `K = ⌊log₂ D⌋`, level-k trunks of size `2^k`, `⌊D/2^k⌋` trunks per level, trunk `(k,i)` covers edges `[i·2^k+1, (i+1)·2^k]`. *Ours:* `Hpat<BiasT>` with per-vertex `K_max_`, level-k offsets via `hpat_alias_level_offset`, trunk `(k,i)` covers `[i·2^k, (i+1)·2^k)` (0-indexed; same partition).

- **HPAT sampling via binary cover** (paper §3.3, Fig 6b). *Paper:* candidate prefix `L` is binary-decomposed; example `L=7 = 4+2+1 → {τ²·⁰, τ¹·², τ⁰·⁶}`; ITS picks one trunk, alias samples inside. *Ours:* `decompose_to_trunks` produces exactly `[{2,0}, {1,2}, {0,6}]` for `L=7, K=2`; `sample_hpat` walks the cover, builds a tiny ITS cumsum, picks, alias-samples.

- **Auxiliary Index** (paper §3.4). *Paper:* precomputed `(u, L) → cover` table reduces trunk-finding from `O(log D)` to `O(1)`. *Ours:* `tea::AuxIndex` is a CSR-of-CSR table indexed by `(u, L)`, built in parallel per §4.2 ("embarrassingly parallel"); `sample_hpat` consults `hpat.aux().cover_for(u, G)` when active. Memory-budget fallback (`kAuxIndexMaxBytes`) keeps huge graphs viable; bit-identical edge picks vs the on-the-fly path are asserted in `tests/test_aux_index.cpp`.

- **Walk loop with rejection** (paper §4.1, Alg 2). *Paper:* outer step loop, inner rejection on `Dynamic_parameter`; for non-node2vec biases `Dynamic_parameter ≡ 1` (always accept). *Ours:* `walk_engine.hpp` step loop; `rejection_loop` is only instantiated for `Node2VecBias` — for linear/exp the rejection body collapses to a single accept under dead-code elimination.

- **Parallel lock-free preprocessing** (paper §4.2). *Paper:* "calculate the position of each alias table before construction; assign a thread to construct each alias table and store the resultant alias table in the designated memory position without contentions." *Ours:* `Pat::build`, `Hpat::build`, `AuxIndex::build`, `NeighborSets::build` all follow this pattern — serial prefix sum computes per-vertex offsets into global flat arenas, then `#pragma omp parallel for schedule(dynamic, 64)` writes per-vertex slices with zero atomics in the hot path.

- **§3.3 ad-hoc optimizations.** Both implemented. *Paper:* "if the out-degree of a vertex is relatively low, we can simply build alias tables for its specific out edges" and "if the temporal information of certain neighbors is earlier than all the incoming edges, we can simply discard them." *Ours:* vertices with `D ≤ kHpatDegreeThreshold=64` skip the HPAT hierarchy and use a single AliasTable of size D (always-on; verified by `test_solo.cpp` to halve per-vertex alias storage and produce identical distributions to PAT under χ²). Outgoing edges with `t ≤ min_incoming_time(u)` are excluded from first-hop candidates via the `TEA_TEMPORAL_START=1` env var (opt-in; verified by `test_temporal_start.cpp`).

- **Correctness coverage.** χ² distribution tests for every bias vs the analytic `P((u, v_i))`; HPAT-vs-PAT two-sample χ² confirming both sampler variants produce statistically identical distributions; cover identity for all `(u, L)` against on-the-fly decomposition; walk-engine determinism across 1/4/16 OpenMP threads. 100+ test cases across 11 binaries.

## Explicitly out of scope

These TEA features are not reimplemented; their absence is documented and orthogonal to the in-memory single-machine comparison Tempest's Table 4 makes:

- **Streaming updates** (paper §3.5) — Tempest's comparison is on static snapshots.
- **Out-of-core execution** (paper §3.2 final paragraph, §4.1) — Tempest is in-memory; the published 6158× speedups are in-memory rows of paper Table 4.
- **Programmability API** (paper §4, Table 2: `Dynamic_weight()`, `Dynamic_parameter()`, `Edges_interval()`) — a usability claim, not a performance one. Our three biases (Linear, Exponential, Temporal node2vec) are hardcoded as C++ templates; the compiled binary is what gets benchmarked.

## Build & run

```sh
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build -j

# Run walks
./build/tea_walk <csv_path> exponential tea_hpat 1 20 80 -1
#                           ^picker     ^variant ^dir ^wpn ^mwl ^timescale_bound
```

`TEA_DISABLE_AUX=1` disables the precomputed Auxiliary Index (forces on-the-fly cover decomposition) — useful for the §3.4 ablation row in the paper.
