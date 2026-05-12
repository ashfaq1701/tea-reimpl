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

## Design efficiency of this reimplementation

Correctness establishes that we implement the paper.  This section is about why the implementation is also *fast* — the design decisions that distinguish it from a naive C++ port.

### Memory layout

- **SoA over AoS, one global flat arena per concept.** Per-vertex outgoing edges live in three parallel flat arrays (`targets_[E]`, `timestamps_[E]`, plus a CSR `offsets_[N+1]`).  Same shape for PAT (`alias_arena_[E]`, `cumsum_arena_[Σ⌈D/T⌉]`), HPAT (`alias_arena_[~E·log D]`, `trunk_totals_[~E·log D / 2]`), AuxIndex (CSR-of-CSR over `(u, L)` cover entries), and NeighborSets (CSR sorted neighbor lists).  No `vector<vector<T>>` anywhere — that pattern fragments the heap, defeats the hardware prefetcher, and pays a TLB miss for every vertex lookup.

- **Packed `AliasEntry` = 8 bytes.** `{uint32_t alias; float prob;}` instead of a naive `{int32_t alias; double prob;}` which pads to 16.  Halves the cache footprint of the dominant arena, and 24 mantissa bits is far more precision than alias-table thresholds need.

- **`tea::span<const T>` views, never copies.** Every per-vertex accessor returns a non-owning span into the global arena.  Zero copy in the hot path.  (Hand-rolled C++17 span since `std::span` is C++20.)

### Hot-path compute

- **Sampler templated on `BiasT`.** Bias type is resolved at compile time; each `(BiasT, Variant)` instantiation produces a tight monomorphized sampler the compiler can fully inline.  No virtual dispatch, no `std::function`, no runtime `enum + switch` inside the per-step loop.

- **Inner loop in headers, force-inlined.** `AliasView::sample`, `Pat<BiasT>::sample_*`, `Hpat<BiasT>::sample_*`, and the bias `compute_weights` all live in headers.  The compiler sees across alias / bias / sampler boundaries and schedules registers across the whole step.

- **Branchless alias sample.** `return (prob_rng < e.prob) ? bucket_idx : e.alias;` — one compare, one cmov on x86.  No conditional jump in the innermost step.

- **Zero heap allocations during sampling.** Per-walk scratch (`SamplerScratch::partial_weights`, `partial_cumsum`) is allocated once per thread and reused across millions of sample calls.  HPAT cover decomposition writes into a stack array (`TrunkRef[64]`).  No `malloc` inside the hot loop.

- **`AliasBuildScratch` reused across millions of trunk builds.** Vose's algorithm needs three scratch vectors; allocating them per trunk during a parallel preprocess on a 40M-edge graph would dominate the build.  Per-thread scratch is allocated once at the start of the parallel-for and reused for every trunk that thread owns.

- **Compile-time `needs_prev_vertex<BiasT>()`.** The rejection loop for node2vec exists only in the Node2Vec instantiation; for Linear/Exponential the loop body collapses to a single proposal under dead-code elimination.

### Threading

- **OpenMP `parallel for schedule(dynamic, 64)`** over walk seeds.  Dynamic schedule handles uneven walk-length distributions; chunk = 64 amortizes scheduler overhead.

- **Per-walk RNG seeded from `(global_seed, walk_idx)`.** Each walk has an independent Pcg64 stream that doesn't depend on which thread runs it.  Result: walks are *deterministic* regardless of OMP scheduling, verified by an end-to-end test that runs the same input at 1/4/16 threads and asserts byte-identical output.

- **`alignas(64) PerThreadRng`** and per-thread output buffer slices.  No two threads ever touch the same cache line during the walk loop — kills false sharing.  Each walk's output occupies a `max_walk_len`-sized slot in a pre-allocated buffer indexed by `walk_idx * max_walk_len`, so writes are lock-free.

- **Lock-free parallel preprocessing.** All four builders (`Pat::build`, `Hpat::build`, `AuxIndex::build`, `NeighborSets::build`) follow the same pattern: a serial prefix-sum computes per-vertex offsets into the global flat arenas, then `#pragma omp parallel for schedule(dynamic, 64)` writes into per-vertex slices — zero atomics in the hot path.  This mirrors paper §4.2's "calculate the position of each alias table before construction, then assign a thread to construct each alias table … without contentions."

### Algorithmic switches

- **AuxiliaryIndex with memory-budget fallback** (paper §3.4).  The precomputed `(u, L) → cover` table is built when its projected memory fits `kAuxIndexMaxBytes` (default 4 GB); otherwise the sampler transparently falls back to on-the-fly binary decomposition (`decompose_to_trunks`, `O(log D)` per call, zero storage).  Both paths produce **bit-identical edge picks** under the same RNG stream, asserted by `tests/test_aux_index.cpp`.  An env-var (`TEA_DISABLE_AUX=1`) forces the fallback for ablation runs.

- **TEA §3.3 ad-hoc optimization #1 — low-degree solo path.** Vertices with `D ≤ kHpatDegreeThreshold = 64` skip the HPAT hierarchy and store a single AliasTable of size D.  Halves per-vertex alias storage on the long tail of low-degree vertices.  Sample-time: alias-sample directly when `|Γ_t(u)| = D`, partial-prefix ITS recompute when `|Γ_t(u)| < D` (same path PAT's partial last trunk uses).

- **TEA §3.3 ad-hoc optimization #2 — discard older-than-Γ_t neighbors.** Opt-in via `TEA_TEMPORAL_START=1`.  When enabled, walks start with `t_prev = min_incoming_time_of(u_start)` instead of the `INT64_MIN` sentinel, so the existing candidate-set computation naturally excludes outgoing edges whose `t ≤ t_min_in` — those are unreachable from any walker arriving via an incoming edge anyway.

- **Static-weight cancellation trick for ExponentialBias** (paper §2.3 II, Eq 3).  `δ(e_i) = exp(t_i − t_cur)` formally depends on the walker's arrival time, but the normalizer cancels: `P(e_i) = exp(t_i) / Σ exp(t_j)`.  We store `exp(t_i − t_max_u)` (softmax-equivalent to `exp(t_i)`, never overflows) once at preprocess; the PAT/HPAT alias tables are *static* across the entire walk.  No per-step rebuild.

### Numerical

- **`exp((t_i − t_max_u) · scale)` with `scale = 1` by default.** Raw unix timestamps (~1.7×10⁹) trivially overflow `exp()`; subtracting the per-vertex max anchors the largest weight at `exp(0) = 1` and is *exactly* equivalent to the paper's distribution (softmax shift-invariance).  Older edges may underflow to 0 — that is the correct behaviour of pure exponential bias on unix-timescale data, not a numerical bug.

- **`timescale_bound > 0` re-enables a controlled rescale** for Tempest-compatibility: `scale = timescale_bound / (t_max_u − t_min_u)`.  This *does* change the distribution (it compresses the per-vertex time span to a fixed magnitude before `exp`), and the runtime banner reports which mode each run used.  Used only in cross-comparison runs where Tempest's exp is also rescaled by the same knob.

### Build-time hygiene

- **C++17 with `-O3 -march=native -fopenmp`.** Matches the standard Tempest uses; no exotic toolchain dependencies.  Bias structs are stateless except for one `double timescale_bound`, so the compiler aggressively constant-folds.

- **Headers-only public API.**  All hot types are header-only templates; only `graph.cpp` is a translation unit.  Link-time is dominated by gtest.

- **No SIMD intrinsics in the sampler inner loop.** Deliberate — adding hand-tuned vectorisation here would invite reviewers to call the baseline sandbagged (the original TEA paper doesn't describe SIMD).  The compiler is free to autovectorize the cumsum/recompute loops in `partial_weights`, but the alias-sample is a single branchless operation already.

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
