# tea-reimpl

An independent, open-source reimplementation of **TEA** (Huan et al., EuroSys '23) — a CPU temporal-graph random-walk engine built from PAT + HPAT + Auxiliary-Index hybrid sampling.
Used as the CPU baseline for cross-architecture comparison in the Tempest paper, since the original TEA source is not publicly available.

## Citation

> Chengying Huan, Shuaiwen Leon Song, Santosh Pandey, Hang Liu, Yongchao Liu, Baptiste Lepers, Changhua He, Kang Chen, Jinlei Jiang, Yongwei Wu. **TEA: A General-Purpose Temporal Graph Random Walk Engine.** In *Eighteenth European Conference on Computer Systems (EuroSys '23)*, May 8–12, 2023, Rome, Italy. ACM, 17 pages. https://doi.org/10.1145/3552326.3567491

## Why this is a reliable TEA reimplementation

- **Candidate edge set Γ_{t_prev}(u)** (paper §2.1, §2.3). *Paper:* the set of edges incident to `u` that respect temporal continuity with the walker's last edge. *Ours:* `TemporalGraph` stores per-vertex time-ASCENDING adjacency in a CSR; `candidate_set_len(u, t_prev)` returns the size of the admissible prefix via a strict-`<` `lower_bound`. PAT/HPAT/AuxIndex all operate on this prefix.

- **Linear bias** (paper §2.3.I, Fig 5). *Paper:* `δ(e) = rank(e)`, newest edge has rank D, oldest has rank 1 — the most-recent-in-time edge always carries the highest weight. *Ours:* `LinearBias::compute_weights` returns `position + 1` on the per-vertex ASC slice, so rank D lands on the most-recent edge and rank 1 on the oldest — same per-edge weight assignment the paper specifies.

- **Exponential bias with static-weight cancellation** (paper §2.3.II, Eq 3). *Paper:* `δ(e) = exp(t_i)` after the `−t_cur` cancellation, so the per-edge weights are static post-preprocess and PAT/HPAT alias tables don't need a per-step rebuild. *Ours:* `ExponentialBias` stores `δ = exp(t_i − t_max_u)`, softmax-equivalent to `exp(t_i)` by shift-invariance; never overflows on raw unix timestamps; tables built once.

- **Temporal node2vec rejection on β** (paper §2.3.III, Alg 2 lines 19–21). *Paper:* `β ∈ {1/p, 1, 1/q}` keyed on the candidate's relationship to the walker's previous vertex (return / 1-hop neighbor / 2-hop), accept the static-exp proposal with probability proportional to β. *Ours:* `Node2VecBias::accept_ratio` returns `β / β_max ∈ (0, 1]`; `rejection_loop` accepts on `u01 < accept_ratio` with a retry cap. `NeighborSets::build` builds the **undirected** neighborhood of each vertex (for an input edge `(a, b, t)`, both `N(a) ⊇ {b}` and `N(b) ⊇ {a}`), matching the original node2vec paper's `N(w)` convention.

- **PAT data structure** (paper §3.2). *Paper:* partition each vertex's edge list into trunks of size `trunkSize ≈ ⌊√D⌋`, build an alias table per trunk, store the inclusive prefix sum of per-trunk totals. *Ours:* `Pat<BiasT>` with `trunk_size = max(⌈√D⌉, kPatTrunkSizeMin=8)`, a flat per-vertex alias arena `alias_arena_[E]`, and a flat per-vertex inclusive cumsum arena `cumsum_arena_[Σ⌈D/T⌉]`.

- **PAT sampling Cases ① and ②** (paper §3.2). *Paper:* trunks that are entirely within the candidate set are sampled via ITS over the prefix sum + alias inside; the trunk straddling the cutoff (the "partial" trunk) gets its prefix sum rebuilt inside. *Ours:* `sample_pat` computes `num_full = G/T`, `partial = G%T`; on `partial > 0`, `bias.compute_weights(..., slice_start_pos)` recomputes the partial-trunk weights with the same per-vertex normalization used at build, then on-the-fly cumsum + ITS.

- **HPAT hierarchy** (paper §3.3, Eq 5–7). *Paper:* `K = ⌊log₂ D⌋`, level-k trunks of size `2^k`, `⌊D/2^k⌋` trunks per level; trunk `(k, i)` covers a `2^k`-edge contiguous chunk. *Ours:* `Hpat<BiasT>` with per-vertex `K_max_`, level-k offsets via `hpat_alias_level_offset`; trunk `(k, i)` covers `[i·2^k, (i+1)·2^k)` (0-indexed, same partition the paper writes 1-indexed).

- **HPAT sampling via binary cover** (paper §3.3, Fig 6b). *Paper:* the candidate prefix of length `L` is binary-decomposed into `popcount(L)` trunks (e.g. `L=7 = 4+2+1 → {τ²·⁰, τ¹·², τ⁰·⁶}`); a small ITS picks the trunk, then alias-sample inside. *Ours:* `decompose_to_trunks` produces exactly `[{2,0}, {1,2}, {0,6}]` for `L=7, K=2`; `sample_hpat` walks the cover, builds a tiny ITS cumsum, picks, alias-samples. No partial-trunk case — the cover is always exact.

- **Auxiliary Index** (paper §3.4). *Paper:* a precomputed `(u, L) → cover` table reduces trunk-finding from `O(log D)` to `O(1)`. *Ours:* `tea::AuxIndex` is a CSR-of-CSR table indexed by `(u, L)`, built in parallel per §4.2 ("embarrassingly parallel"); `sample_hpat` consults `hpat.aux().cover_for(u, G)` when active. A memory-budget fallback (`kAuxIndexMaxBytes`) keeps huge graphs viable, and the AuxIndex vs on-the-fly paths produce bit-identical edge picks (asserted in `tests/test_aux_index.cpp`).

- **Walk loop with rejection** (paper §4.1, Alg 2). *Paper:* outer step loop, inner rejection on `Dynamic_parameter`; for non-node2vec biases `Dynamic_parameter ≡ 1` (always accept). *Ours:* `walk_engine.hpp` step loop; `rejection_loop` is only instantiated for `Node2VecBias` — for linear/exp the rejection body collapses to a single accept under dead-code elimination.

- **Parallel lock-free preprocessing** (paper §4.2). *Paper:* "calculate the position of each alias table before construction; assign a thread to construct each alias table and store the resultant alias table in the designated memory position without contentions." *Ours:* `Pat::build`, `Hpat::build`, `AuxIndex::build`, `NeighborSets::build` all follow this pattern — a serial prefix sum computes per-vertex offsets into global flat arenas, then `#pragma omp parallel for schedule(dynamic, 64)` writes per-vertex slices with zero atomics in the hot path.

- **§3.3 ad-hoc optimization — low-degree solo path.** *Paper:* "if the out-degree of a vertex is relatively low, we can simply build alias tables for its specific out edges." *Ours:* vertices with `D ≤ kHpatDegreeThreshold = 64` skip the HPAT hierarchy and use a single AliasTable of size D (always-on; `tests/test_solo.cpp` verifies the same per-vertex distribution as PAT under χ² and confirms the halved per-vertex alias storage).

- **Correctness coverage.** χ² distribution tests for every bias against the analytic `P((u, v_i))`; HPAT-vs-PAT two-sample χ² confirming both sampler variants produce statistically identical distributions; AuxIndex cover identity for all `(u, L)` against on-the-fly decomposition; walk-engine determinism across 1/4/16 OpenMP threads (byte-identical output); a NeighborSets regression test that catches in-neighbors-only adjacency from being silently reintroduced; an early-termination invariant test (ported from Tempest's `WalkTerminalEdgesTest`) asserting any walk shorter than `max_walk_len` did so because the candidate set was genuinely empty — not because the walk loop bailed out early. 130+ test cases across 12 binaries.

## Design efficiency of this reimplementation

Correctness establishes that we implement the paper.  This section is about why the implementation is also *fast* — the design decisions that distinguish it from a naive C++ port.

### Memory layout

- **SoA over AoS, one global flat arena per concept.** Per-vertex edges live in three parallel flat arrays (`targets_[E]`, `timestamps_[E]`, plus a CSR `offsets_[N+1]`). Same shape for PAT (`alias_arena_[E]`, `cumsum_arena_[Σ⌈D/T⌉]`), HPAT (`alias_arena_[~E·log D]`, `trunk_totals_[~E·log D / 2]`), AuxIndex (CSR-of-CSR over `(u, L)` cover entries), and NeighborSets (CSR sorted neighbor lists). No `vector<vector<T>>` anywhere — that pattern fragments the heap, defeats the hardware prefetcher, and pays a TLB miss for every vertex lookup.

- **Packed `AliasEntry` = 8 bytes.** `{uint32_t alias; float prob;}` instead of a naive `{int32_t alias; double prob;}` which pads to 16. Halves the cache footprint of the dominant arena, and 24 mantissa bits is far more precision than alias-table thresholds need.

- **`tea::span<const T>` views, never copies.** Every per-vertex accessor returns a non-owning span into the global arena. Zero copy in the hot path. (Hand-rolled C++17 span since `std::span` is C++20.)

### Hot-path compute

- **Sampler templated on `BiasT`.** Bias type is resolved at compile time; each `(BiasT, Variant)` instantiation produces a tight monomorphized sampler the compiler can fully inline. No virtual dispatch, no `std::function`, no runtime `enum + switch` inside the per-step loop.

- **Inner loop in headers, force-inlined.** `AliasView::sample`, `sample_pat`, `sample_hpat`, and the bias `compute_weights` all live in headers. The compiler sees across alias / bias / sampler boundaries and schedules registers across the whole step.

- **Branchless alias sample.** `return (prob_rng < e.prob) ? bucket_idx : e.alias;` — one compare, expected to lower to a `cmov` on x86. No conditional jump in the innermost step.

- **Zero heap allocations during sampling.** Per-walk scratch (`SamplerScratch::partial_weights`, `partial_cumsum`) is allocated once per thread and reused across millions of sample calls. HPAT cover decomposition writes into a stack array (`TrunkRef[64]`). No `malloc` inside the hot loop.

- **`AliasBuildScratch` reused across millions of trunk builds.** Vose's algorithm needs three scratch vectors; allocating them per trunk during a parallel preprocess on a 40M-edge graph would dominate the build. Per-thread scratch is allocated once at the start of the parallel-for and reused for every trunk that thread owns.

- **Compile-time `needs_prev_vertex<BiasT>()`.** The rejection loop for node2vec exists only in the Node2Vec instantiation; for Linear/Exponential the loop body collapses to a single proposal under dead-code elimination.

### Threading

- **OpenMP `parallel for schedule(dynamic, 64)`** over walk seeds. Dynamic schedule handles uneven walk-length distributions; chunk = 64 amortizes scheduler overhead.

- **Per-walk RNG seeded from `(global_seed, walk_idx)`.** Each walk has an independent Pcg64 stream that doesn't depend on which thread runs it. Result: walks are *deterministic* regardless of OMP scheduling, verified by an end-to-end test that runs the same input at 1/4/16 threads and asserts byte-identical output.

- **Per-thread output buffer slices, no false sharing.** Each walk's output occupies a `max_walk_len`-sized slot in a pre-allocated buffer indexed by `walk_idx * max_walk_len`. Writes are lock-free and the slot stride keeps neighbouring walks out of each other's cache lines.

- **Lock-free parallel preprocessing.** All four builders (`Pat::build`, `Hpat::build`, `AuxIndex::build`, `NeighborSets::build`) follow the same pattern: a serial prefix sum computes per-vertex offsets into the global flat arenas, then `#pragma omp parallel for schedule(dynamic, 64)` writes into per-vertex slices — zero atomics in the hot path. This mirrors paper §4.2's "calculate the position of each alias table before construction, then assign a thread to construct each alias table … without contentions."

### Algorithmic switches

- **AuxiliaryIndex with memory-budget fallback** (paper §3.4). The precomputed `(u, L) → cover` table is built when its projected memory fits `kAuxIndexMaxBytes` (default 4 GB); otherwise the sampler transparently falls back to on-the-fly binary decomposition (`decompose_to_trunks`, `O(log D)` per call, zero storage). Both paths produce **bit-identical edge picks** under the same RNG stream, asserted by `tests/test_aux_index.cpp`. `TEA_DISABLE_AUX=1` forces the fallback for ablation runs.

- **TEA §3.3 ad-hoc optimization — low-degree solo path.** Vertices with `D ≤ kHpatDegreeThreshold = 64` skip the HPAT hierarchy and store a single AliasTable of size D. Halves per-vertex alias storage on the long tail of low-degree vertices. Sample-time: alias-sample directly when the candidate set spans the full degree, partial-prefix ITS recompute otherwise (same path PAT's partial last trunk uses).

- **Static-weight cancellation trick for ExponentialBias** (paper §2.3 II, Eq 3). `δ(e_i) = exp(t_i − t_cur)` formally depends on the walker's arrival time, but the normalizer cancels: `P(e_i) = exp(t_i) / Σ exp(t_j)`. We store `exp(t_i − t_max_u)` (softmax-equivalent to `exp(t_i)`, never overflows) once at preprocess; the PAT/HPAT alias tables are *static* across the entire walk. No per-step rebuild.

- **Undirected adjacency for the node2vec β-check.** `NeighborSets::build(graph, input_was_directed)` augments the per-vertex inbound CSR with a transpose pass when the input graph was directed, so `N(w)` is the undirected neighborhood (every edge contributes both endpoints' adjacency). Matches the original node2vec `N(w)` convention and Tempest's `build_node_adjacency_csr_std`; verified by `tests/test_node2vec.cpp`'s `DirectedOutEdgeContributesToUndirectedNeighborhood` case.

### Numerical

- **`exp((t_i − t_max_u) · scale)` with `scale = 1` by default.** Raw unix timestamps (~1.7×10⁹) trivially overflow `exp()`; subtracting the per-vertex max anchors the largest weight at `exp(0) = 1` and is *exactly* equivalent to the paper's distribution (softmax shift-invariance). Older edges may underflow to 0 — that is the correct behaviour of pure exponential bias on unix-timescale data, not a numerical bug.

- **`timescale_bound > 0` re-enables a controlled rescale** for Tempest-compatibility: `scale = timescale_bound / (t_max_u − t_min_u)`. This *does* change the distribution (it compresses the per-vertex time span to a fixed magnitude before `exp`), and the runtime banner reports which mode each run used. Used only in cross-comparison runs where Tempest's exp is also rescaled by the same knob.

### Build-time hygiene

- **C++17 with `-O3 -march=native -fopenmp`.** Matches the standard Tempest uses; no exotic toolchain dependencies. Bias structs are stateless except for one `double timescale_bound`, so the compiler aggressively constant-folds.

- **Headers-only public API.** All hot types are header-only templates; only `graph.cpp` is a translation unit. Link-time is dominated by gtest.

- **No SIMD intrinsics in the sampler inner loop.** Deliberate — adding hand-tuned vectorisation here would invite reviewers to call the baseline sandbagged (the original TEA paper doesn't describe SIMD). The compiler is free to autovectorize the cumsum/recompute loops in `partial_weights`, but the alias-sample is a single branchless operation already.

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

`tea_walk` prints two timing figures per run: a wall-time figure that brackets start-list construction + output-buffer allocation + the walk loop (Tempest-comparable), and a pure walk-loop time for just the parallel sampler invocation. Both `Walks done: … (W s)` and `Walk loop time: L s` appear in stdout.

`TEA_DISABLE_AUX=1` disables the precomputed Auxiliary Index (forces on-the-fly cover decomposition) — useful for the §3.4 ablation row in the paper.
