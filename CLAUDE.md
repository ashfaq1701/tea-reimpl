# TEA Reimplementation — Plan & Roadmap

> Independent reimplementation of TEA (Huan et al., EuroSys '23) for use as a
> publication-grade CPU baseline against Tempest. Built strictly from the
> algorithm and data-structure specifications in `paper.pdf`. No code from the
> original authors is available — this file is the authoritative spec we work
> from.

---

## Performance architecture contract

These are the design decisions every phase must honor. Reviewing a PR against
this section is faster than re-deriving the rationale from scratch each time.

### Memory layout
- **SoA over AoS.** Per-vertex outgoing edges are stored as three parallel
  flat arrays (`targets[]`, `timestamps[]`, `weights[]`), one giant array each
  spanning all edges of all vertices, indexed by a CSR `vertex_offsets[N+1]`.
  No `vector<vector<T>>` anywhere — that pattern fragments the heap, defeats
  hardware prefetch, and pays a TLB miss for every vertex lookup.
- **Global flat arenas, one per concept.** All trunks of all vertices share one
  `alias_entries[]` arena and one `trunk_cumsums[]` arena, with per-vertex
  metadata = `(offset_in_arena, count, level_offsets[K+1])`. Same for HPAT,
  same for the auxiliary index, same for the per-vertex neighbor sets used in
  Node2Vec. Build-time: each vertex's worker thread writes lock-free into its
  own arena slice.
- **Packed `AliasEntry` = 8 bytes.** `{uint32_t alias; float prob;}`. A naive
  `{int32_t alias; double prob;}` is 12 (and pads to 16) — half the cache
  footprint matters at scale. 24 mantissa bits is more than enough precision
  for alias-table thresholds.
- **`span<T>` views, never copies.** Every API that exposes per-vertex data
  hands out `tea::span<const T>` into the global arenas. Zero copy. (`tea::span`
  is our own C++17-friendly span — std::span is C++20 only.)

### Hot-path compute
- **Sampler templated on `BiasT`.** Bias type is resolved at compile time so
  there is no virtual dispatch inside the per-step loop. Each `(BiasT, Variant)`
  instantiation produces a tight monomorphized sampler the compiler can fully
  inline. No `std::function`, no `virtual`, no `enum + switch` in the inner
  loop — these all become full kernels.
- **Inner loop in headers.** `AliasTable::sample` and `Sampler<BiasT>::sample`
  live in headers (force-inlined). The compiler must see across the alias /
  bias / sampler boundary to schedule registers well.
- **No heap allocations during sampling.** Partial-trunk on-the-fly prefix sums
  use stack arrays sized to `kPatTrunkSizeMax`. Per-walk output goes to a
  thread-local slice of a pre-allocated buffer.
- **Branchless alias sample.**
  `return (prob_rng < e.prob) ? bucket_idx : e.alias;` — no conditional jump.

### Threading
- **`alignas(64) PerThreadRng` + per-thread output buffer.** No two threads
  ever write into the same cache line during the walk loop. Kills false sharing.
- **OpenMP `parallel for schedule(dynamic, 64)`** over walk seeds. Dynamic
  schedule handles uneven walk-length distributions; chunk=64 amortizes the
  scheduler's overhead.
- **First-touch NUMA allocation.** On multi-socket nodes, each thread builds
  the per-vertex data for its assigned vertex range, so the data lands on its
  local NUMA node. Runtime banner reports the detected topology.
- **Lock-free preprocessing.** Each vertex is built independently; writes only
  go to that vertex's slice of the global arenas. No mutex, no atomic.

### Algorithmic switches
- **TEA §3.3 ad-hoc optimizations.** Both implemented. (a) *Low-degree single
  AliasTable*: vertices with `D ≤ kHpatDegreeThreshold` (= 64) skip the HPAT
  hierarchy and store a single AliasTable of size D, halving per-vertex alias
  storage. Sample-time: alias-sample directly when |Γ_t(u)|=D, partial-prefix
  ITS recompute when |Γ_t(u)|<D. Always-on. (b) *Discard older-than-Γ_t
  neighbors*: opt-in via `TEA_TEMPORAL_START=1` env var. When enabled, walks
  start with `t_prev = min_incoming_time_of(u_start)` instead of the
  `INT64_MIN` sentinel, so the sampler's existing candidate-set computation
  naturally excludes outgoing edges with `t ≤ t_min_in`. Default off
  preserves the "every outgoing edge is a first-hop candidate" semantics
  used by the existing test invariants.
- **AuxiliaryIndex: explicit OR on-the-fly, runtime-selected.** Explicit table
  (paper §3.4 precomputed lookup) is built by default; on-the-fly fallback
  (binary decomposition, O(log D) per call, zero storage) kicks in when the
  explicit table would exceed `kAuxIndexMaxBytes` (= 4 GB) OR when
  `TEA_DISABLE_AUX=1` is set in the environment.  Both paths produce
  bit-identical edge picks (asserted by `test_aux_index.cpp`); the env knob
  exists for §8 ablation runs.  On laptop-scale hardware the aux table loses
  ~20% throughput because its arenas (multi-GB) miss L2/L3 while the inline
  decompose stays in L1; on the paper's Xeon, aux is reported to add ~3×.
  The choice is reported in the run banner (`aux=on/off`).
- **Compile-time `needs_prev_vertex<BiasT>()`.** Node2Vec's rejection loop
  exists only in the Node2Vec instantiation; for Linear/Exponential the loop
  body collapses to a single proposal under DCE.

### Numerical (revised after Phase 2 self-review against paper §2.3-II)

- **ExponentialBias uses `exp(t_i − t_max_u)` directly. No rescaling.** Raw
  unix timestamps (~1.7e9) overflow `exp()`; subtracting `t_max_u` (the
  per-vertex max) makes the largest weight equal to `exp(0) = 1`, never
  overflows, and preserves the paper's distribution exactly (softmax is
  shift-invariant). Older edges with very large negative exponents underflow
  to 0 — that is the correct behavior of pure exponential bias on
  unix-timescale data, NOT a numerical bug. An earlier draft proposed
  "rescale to `[0, 80]`" but that distorts the softmax temperature and is
  NOT what the paper specifies; removed.

- **`timescale_bound > 0` (CLI arg parity with Tempest) re-enables a
  controlled rescale**: `scale = timescale_bound / (t_max_u − t_min_u)`.
  This DOES change the distribution; the runtime banner reports which mode
  the run used. Used in cross-comparison runs where Tempest's exp is also
  rescaled by the same knob.

### What we deliberately do not do
- No SIMD intrinsics in the sample inner loop. Reviewers might say we sandbagged
  KnightKing/GraphWalker by adding SIMD that the paper didn't describe. Stay
  within the paper's spec.
- No GPU port. Defeats the whole point of a CPU-class baseline for Tempest.
- No abseil pull at Phase 0. Added only at Phase 5.1 when neighbor sets
  actually need it. Keeps `cmake -B build` configure-time down.

---

## 0. Why we're doing this

The original TEA implementation is unobtainable ("lost" per the authors). The
Tempest paper currently compares to TEA on published numbers only, which prior
reviews flag as a major weakness. This reimplementation:

1. Lets us run TEA on the **same hardware** as Tempest, on the **same datasets**,
   for an apples-to-apples comparison.
2. Provides a **reproducibility check** on TEA's published 954×–6,158× speedup
   claims over KnightKing/GraphWalker. KnightKing and GraphWalker are
   open-source; if our reimpl of TEA hits the same ratio against them, the
   algorithmic claim is validated. If not, we report the discrepancy
   neutrally — we don't accuse, we just measure.
3. Is open-sourced alongside Tempest, so reviewers and downstream readers can
   inspect and verify.

This is a **CPU-only** reimplementation, matching TEA's design (heavy use of
hashmaps + alias tables, out-of-core via disk trunks). Tempest's GPU vs our
TEA's CPU is positioned as an **architecture-class** comparison, not an
algorithm comparison.

---

## 1. Scope decisions

### In scope

- **PAT** (Persistent Alias Table, §3.2): single-level trunks of size `trunkSize`.
- **HPAT** (Hierarchical PAT, §3.3): multi-level trunks at sizes `2^k`.
- **Auxiliary Index** (§3.4): O(1) trunk lookup for HPAT.
- **Three bias variants** (§2.3):
  - **Linear**: `w(e) = rank(e)` (CTDNE-style).
  - **Exponential**: `w(e) = exp(t_e)` (with the *t_cur cancellation trick* so
    weights remain static after preprocess).
  - **Temporal node2vec**: same `exp(t_e)` static weights, plus a **β rejection
    test** dependent on the previous walker vertex.
- **Per-vertex sorted edge lists**, time-descending.
- **Parallel preprocessing** with OpenMP (matches their lock-free construction).
- **Parallel walks** (one walker per thread; embarrassingly parallel).
- **CLI binary** mirroring `ablation_streaming.cpp`'s positional-arg shape so
  it slots cleanly into the existing benchmark harness.

### Out of scope (deliberately)

| Feature | Why we skip |
|---|---|
| Out-of-core (disk-backed HPAT trunks) | Tempest is an in-memory comparison; the 6158× claim numbers in Table 4 are also in-memory. |
| Incremental streaming updates (§3.5) | Tempest's streaming axis is separate; we compare static-snapshot walks. |
| Distributed execution | TEA itself doesn't support it; KnightKing's distributed mode is the apples-to-apples baseline we already have published numbers for. |
| Their high-level temporal-centric API (`Dynamic_weight`, `Dynamic_parameter`, `Edges_interval`) | Not needed for benchmarks — we hardcode the three bias variants directly. The API is a programmability claim; we measure speed. |
| `growth/edit/delicious/twitter` Koblenz format parsers | We feed it the same `u,i,ts` CSV format Tempest already uses. One parser, one source of truth. |

### Datasets

All datasets are already preprocessed (dense-encoded, sorted by ts ascending,
`u,i,ts` CSV header) in the sibling directory:

```
/home/ms2420/CLionProjects/other_datasets/
├── delicious.csv            ← TEA paper overlap (300M edges, 600K nodes)
├── growth.csv               ← TEA paper overlap (40M edges, 1.87M nodes)
├── ml_tgbl-coin.csv         ← Tempest primary baseline (22M edges, 700K nodes)
├── ml_tgbl-flight.csv       ← Tempest primary baseline (67M edges, 1M nodes)
├── ml_tgbl-comment.csv      ← Tempest primary baseline (44M edges, 994K nodes)
├── sx-stackoverflow.csv     ← second-batch (63M edges, 2.6M nodes)
├── wiki-talk-temporal.csv   ← second-batch (7.8M edges, 1.14M nodes)
├── ml_tgbl-review.csv       ← (4.8M edges, available but unused in the paper)
└── preprocess_other_datasets.py  ← provenance for the encoding
```

Same `u,i,ts` format Tempest's `ablation_streaming` already consumes, so the
TEA `csv_reader` can be a near-clone of Tempest's. **Five datasets matter for
the validation §9.2 table**: `ml_tgbl-coin`, `ml_tgbl-flight`, `ml_tgbl-comment`,
`delicious`, `growth`. The first three give us the direct cross-comparison
against the Tempest numbers; the last two give us overlap with the original
TEA paper's published numbers (Table 4 lists growth, edit, delicious, twitter
— we have growth and delicious in common). Avoid `twitter` (1.5B edges) —
out-of-core territory, out of scope.

---

## 2. Algorithm summary

### 2.1 Walk model

Temporal graph `G = (V, E, R)`. Each edge `e = (u, v, t)`. A walker at vertex `u`
arriving at time `t_prev` samples the next edge from the **candidate edge set**

```
Γ_{t_prev}(u) = { (u, v, t) ∈ E : t > t_prev }
```

with edge transition probability `P((u, v_i)) = δ(e_i) / Σ_j δ(e_j)`. The bias
function `δ` defines the three variants.

### 2.2 Sampling primitives (background)

- **ITS** (inverse-transform sampling): build prefix-sum of weights, binary-
  search for `r * total`. Build O(n), sample O(log n). Space O(n).
- **Alias method** (Vose): build O(n), sample O(1). Space O(n). But: rebuild
  needed every time the weight distribution changes — kills it for static
  graphs where weights depend on `t_cur`.
- **Rejection sampling**: propose uniformly, accept with prob `δ/δ_max`.
  Expected trials = `δ_max·n / Σδ`, can be enormous when distribution is
  skewed (KnightKing's weakness on exponential weights).

### 2.3 The TEA hybrid (the headline contribution)

Per vertex `u`, partition the time-descending edge list of `u` into **trunks**.
For each trunk:
- Build an **alias table** sized to that trunk (so within-trunk sampling is O(1)).
- Compute the **trunk's total weight**.

Per vertex, store a **prefix-sum across trunks**. At sample time:
1. Determine which trunks are *fully contained* in `Γ_{t_prev}(u)` (a prefix of
   trunks, because edges are time-descending).
2. **ITS** on the prefix-sum to pick a trunk.
3. **Alias** within the picked trunk to pick an edge.

If the trunk straddling the time cutoff is partial, build a fresh prefix-sum
inside that one partial trunk and ITS-sample within it.

**Net complexity**: O(log(D/trunkSize) + 1) for PAT, O(log(log D) + 1) for HPAT
with the auxiliary index. Vs O(log D) for plain ITS.

### 2.4 The static-weight trick (for exponential bias)

Naively, `δ(e_i) = exp(t_i − t_cur)` depends on `t_cur`, forcing rebuild every
step. But the normalizer cancels:

```
P(e_i) = exp(t_i − t_cur) / Σ exp(t_j − t_cur)
       = exp(t_i) / Σ exp(t_j)
```

So we store `w(e_i) = exp(t_i)` once at preprocess time. The PAT/HPAT becomes
*static*.

### 2.5 Temporal node2vec via rejection on top of PAT

Node2vec adds a `β` term that depends on the *previous* vertex `w`:

```
β_{w, v_i} = 1/p  if v_i == w        (returning)
             1    if v_i ∈ N(w)      (1-hop from w)
             1/q  otherwise          (2-hops from w)
```

Combined with exp weights:
```
P((u, v_i)) = β_{w, v_i} · exp(t_i) / Σ exp(t_j)
```

TEA samples from the static `exp(t_i)` PAT to propose an edge, then **rejection-
tests** it against β/β_max. This is Algorithm 2 lines 19–22 in the paper:

```
loop:
   (R, R')          ← (rand, rand)
   (v, t)           ← PAT.sample(R, u, t_prev)        # proposal
   if R' ≤ β_{w,v} / β_max: break                     # accept
# else retry
```

`β_max = max(1/p, 1, 1/q)`. Need O(1) `v ∈ N(w)?` check → per-vertex hash-set of
neighbors. This is *one* of TEA's "heavy hashmap" usages.

---

## 3. Core data structures (specs)

### 3.1 Edge representation

```cpp
struct Edge {
    int32_t u;
    int32_t v;
    int64_t t;
};
```

We assume vertex IDs are **dense-encoded** in `[0, N)` (matches Tempest's
preprocessing). No vertex-id hashmap needed for the graph itself.

### 3.2 Per-vertex edge list

```cpp
// For each vertex u, store outgoing edges sorted by t DESCENDING (newest first).
struct VertexEdges {
    std::vector<int32_t> targets;     // size = degree(u)
    std::vector<int64_t> timestamps;  // size = degree(u), monotone non-increasing
    std::vector<double>  weights;     // size = degree(u), static bias-dependent
};
```

Why time-descending: the candidate edge set `Γ_{t_prev}(u)` is then the
**prefix** of length `L` where `L = upper_bound(timestamps, t_prev)`. Prefix is
the natural shape for the trunk hierarchy (level-k trunks live in positions
`[i·2^k, (i+1)·2^k)`).

### 3.3 AliasTable

Vose's algorithm. For weights `w_1..w_n`:

```cpp
class AliasTable {
public:
    AliasTable(const double* weights, size_t n);   // build O(n)
    int32_t sample(double r1, double r2) const;    // O(1), returns index in [0, n)
private:
    std::vector<double>  prob_;   // size n
    std::vector<int32_t> alias_;  // size n
};
```

### 3.4 Trunk (one alias-table slot)

```cpp
struct Trunk {
    AliasTable alias;       // size = trunk_size
    double     total_weight;
};
```

### 3.5 PAT (single-level)

For vertex `u` with degree `D_u`, choose `trunkSize ≈ √D_u`:

```cpp
struct PAT_PerVertex {
    std::vector<Trunk>  trunks;       // size ≈ ⌈D_u / trunkSize⌉
    std::vector<double> trunk_cumsum; // prefix sum of trunk_total_weight, size = #trunks
    // For the partial last trunk that may straddle Γ_t(u)'s cutoff, we keep the
    // raw weights so we can build an on-the-fly prefix-sum at sample time.
    const std::vector<double>* edge_weights;  // back-reference into VertexEdges.weights
};
```

### 3.6 HPAT (multi-level)

Per Equation 5–7 in the paper, for vertex `u` with `n = D_u`:

```
K = ⌊log₂(n)⌋
At each level k ∈ [0, K]:
    trunks at this level have size 2^k
    number of full trunks at level k = ⌊n / 2^k⌋
    τ_u^{k,i} = edges[i·2^k .. (i+1)·2^k - 1]
```

```cpp
struct HPAT_PerVertex {
    std::vector<std::vector<Trunk>>  trunks_per_level;  // trunks_per_level[k][i]
    std::vector<std::vector<double>> cumsum_per_level;  // cumsum over trunks_per_level[k]
    const std::vector<double>* edge_weights;
};
```

Total storage per vertex: `Σ_k (n / 2^k · 2^k) = Σ_k n = (K+1)·n ≈ n·log(n)`. The
paper accepts this space-time tradeoff.

### 3.7 AuxiliaryIndex (HPAT only)

Given `L = |Γ_{t_prev}(u)|` (a value in `[0, D_u]`), the auxiliary index returns
**which set of HPAT trunks exactly covers** edges `[0, L)` in O(1).

Mechanism: binary decomposition of `L`. If `L = Σ b_k · 2^k`, then the cover is
the set `{τ_u^{k_1, i_1}, τ_u^{k_2, i_2}, ...}` where the `i`'s are computed from
the running sum of trunk sizes.

```cpp
struct AuxiliaryIndex_PerVertex {
    // For each value of L ∈ [0, D_u], precompute the (k, i) trunk list.
    // Implementation: complete binary tree of size D_u (paper Fig 6d).
    // For our reimpl we can store an O(D_u) flat array of trunk-id-lists,
    // each list of length at most log(D_u).
    std::vector<std::vector<std::pair<int8_t, int32_t>>> trunks_for_prefix;
    //  trunks_for_prefix[L] = list of (k, i) pairs that exactly cover edges [0..L)
};
```

For memory, the paper sketches a more clever representation (binary search tree
keyed by `L`) — but we can start with the flat array (O(D log D) space) and
optimize later if needed.

### 3.8 Bias

**Implemented in `include/tea/bias.hpp` as templated structs (not a virtual
hierarchy).** The sampler is templated on the bias type; bias methods are
header-inlined and resolved at compile time, so there is no virtual dispatch
in the hot loop.

Each bias exposes:
```cpp
struct XxxBias {
    static constexpr bool        needs_prev_vertex;
    static constexpr const char* name();
    struct PerVertexParams { ... };  // empty for Uniform; degree for Linear;
                                      // (t_pivot, scale) for Exponential
    PerVertexParams compute_per_vertex_params(span<const int64_t> ts_desc) const;
    void compute_weights(span<const int64_t> ts_slice,
                         const PerVertexParams& params,
                         int32_t slice_start_pos,
                         double* out) const;
    void compute_weights_full(span<const int64_t> ts_desc, double* out) const;
};
```

The `(params, slice_start_pos)` form lets PAT/HPAT recompute weights for the
partial trunk at sample time (paper §3.2 Case ②) with the *same* per-vertex
normalization that the alias tables were built with — otherwise the on-the-fly
prefix-sum would not agree with the persisted trunk cumsums.

**Bias concrete types** (in `include/tea/bias.hpp`):

- **UniformBias**: `δ(e) = 1`. PerVertexParams is empty.
- **LinearBias**: `δ(e) = rank(e)` = `degree − position_in_time_desc_list`.
  PerVertexParams stores `degree`. Slice-start-pos lets it compute ranks
  for any sub-range without seeing the full edge list.
- **ExponentialBias**: `δ(e) = exp((t − t_max_u) × scale)`. PerVertexParams
  stores `(t_pivot=t_max_u, scale)`. `scale=1` (proper TEA) by default;
  `scale = timescale_bound / span_u` when the Tempest-compat flag is set.
- **Node2VecBias** (Phase 5, `include/tea/node2vec_bias.hpp`):
  Same static_weight as ExponentialBias, plus an `accept_ratio(prev, v,
  neighbors)` for the β rejection test.

---

## 4. Source tree

```
tea-implementation/
├── CLAUDE.md                  ← this file (the plan)
├── README.md                  ← usage instructions, citation
├── paper.pdf                  ← reference
├── CMakeLists.txt
├── .gitignore
│
├── third_party/               ← vendored deps (commit minimally)
│   ├── abseil/                ← for absl::flat_hash_set (neighbor set in node2vec)
│   └── googletest/            ← for unit tests
│
├── include/tea/               ← public headers (all install-friendly)
│   ├── edge.hpp               ← Edge struct
│   ├── graph.hpp              ← TemporalGraph: vertex-indexed sorted edge lists
│   ├── neighbor_set.hpp       ← VertexNeighborSet: flat_hash_set wrapper
│   ├── alias.hpp              ← AliasTable (Vose)
│   ├── pat.hpp                ← PAT (single-level trunks)
│   ├── hpat.hpp               ← HPAT (multi-level trunks)
│   ├── aux_index.hpp          ← AuxiliaryIndex (HPAT trunk lookup in O(1))
│   ├── bias.hpp               ← Bias interface + Linear/Exponential/Node2Vec
│   ├── sampler.hpp            ← Sampler: Algorithm 2's inner loop, per walker
│   ├── walk_engine.hpp        ← WalkEngine: top-level walk-generation API
│   ├── rng.hpp                ← xorshift / pcg64 per-thread RNG
│   ├── walk.hpp               ← Walk = vector<(vertex, timestamp)>
│   └── config.hpp             ← compile-time constants (trunk size strategy, etc)
│
├── src/                       ← implementation (matches include/tea structure)
│   ├── graph.cpp
│   ├── alias.cpp
│   ├── pat.cpp
│   ├── hpat.cpp
│   ├── aux_index.cpp
│   ├── bias.cpp
│   ├── sampler.cpp
│   └── walk_engine.cpp
│
├── cli/
│   ├── tea_walk.cpp           ← Main runner. Positional args mirror
│   │                            ablation_streaming.cpp so it slots into
│   │                            the existing tempest-benchmarks harness.
│   ├── csv_reader.cpp/.hpp    ← Edge-stream CSV parser (same format as Tempest's)
│   └── timer.hpp              ← chrono-based timing harness
│
├── tests/                     ← gtest-based unit tests
│   ├── test_alias.cpp         ← Vose: sample distribution matches weights
│   ├── test_pat.cpp           ← PAT: build + sample on small graphs by hand
│   ├── test_hpat.cpp          ← HPAT: cover identity, sample distribution
│   ├── test_aux_index.cpp     ← Trunks-for-prefix cover identity for all L
│   ├── test_bias.cpp          ← Bias output values match paper Eq 2/3/4
│   ├── test_sampler.cpp       ← Sampler: full step matches a reference impl
│   ├── test_walk_engine.cpp   ← End-to-end short walks on toy graphs
│   └── test_graph.cpp         ← Time-descending sort, candidate-set boundary
│
└── bench/
    ├── bench_construction.cpp ← Preprocess timing, std-of-builds
    ├── bench_walk_throughput.cpp ← Walks/sec, edges/step on shared datasets
    └── bench_sampling_cost.cpp   ← Reproduce paper Fig 2 (avg #edges/step)
```

**External dependencies (vendored or required):**

| Dep | Purpose | License |
|---|---|---|
| `absl::flat_hash_set` | Per-vertex neighbor set for node2vec β lookup | Apache-2 |
| `OpenMP` | Preprocessing + per-walker parallelism | system, often built into compiler |
| `gtest` | Unit tests | BSD |
| `xxhash` *(optional)* | Faster hashing for the neighbor set | BSD |

Keep `third_party/` minimal — just abseil and gtest. OpenMP is system-provided.

**Build:**

- CMake ≥ 3.18.
- C++17 (matches Tempest's standard).
- gcc-12 or clang-15+ with `-O3 -fopenmp -march=native`.

---

## 5. Sampling algorithm — pseudocode

This is the implementation-level version of paper Algorithm 2 lines 1–25.

```
function Walk(u_start, t_start, walk_len, bias, graph):
    walks = []
    walks.append((u_start, t_start))
    (u, t) = (u_start, t_start)
    (prev_u, prev_t) = (-1, -1)                           # for node2vec

    for step in [1 .. walk_len - 1]:
        Γ_len = upper_bound(graph[u].timestamps, t)        # candidate-set length
        if Γ_len == 0: break                                # walk dies

        # Inner loop: keep proposing until accepted (rejection on β only)
        loop:
            (next_v, next_t) = sample_PAT_or_HPAT(graph, u, Γ_len, bias, rng)
            if bias.needs_prev_vertex():
                ratio = bias.accept_ratio(prev_u, next_v, graph[prev_u].neighbor_set)
                r = rng.uniform()
                if r > ratio: continue
            break

        prev_u, prev_t = u, t
        u, t = next_v, next_t
        walks.append((u, t))

    return walks

function sample_PAT_or_HPAT(graph, u, Γ_len, bias, rng):
    # PAT or HPAT path: pick a trunk via ITS, then sample within via alias.

    # 1. Identify covering trunks:
    if using HPAT:
        cover = aux_index[u].trunks_for_prefix[Γ_len]      # O(1)
    else:                                                   # PAT
        cover = first ⌊Γ_len/trunkSize⌋ trunks + partial last trunk

    # 2. Build cumsum across the trunks in `cover` (small: ≤ log D entries for HPAT)
    cumsum = [0]
    for trunk in cover:
        cumsum.append(cumsum.back() + trunk.total_weight_for_this_prefix_segment)
    total = cumsum.back()

    # 3. ITS to pick which trunk
    r = rng.uniform() * total
    k = upper_bound(cumsum, r) - 1
    chosen_trunk = cover[k]

    # 4. If chosen trunk is complete: alias-sample within it.
    #    If chosen trunk is partial: build a fresh prefix-sum on its prefix
    #    portion, ITS-sample within it.
    if chosen_trunk is complete:
        local_idx = chosen_trunk.alias.sample(rng.u(), rng.u())
        edge_idx_in_u = trunk_global_offset(chosen_trunk) + local_idx
    else:
        # Partial trunk: weights[0 .. trunk_partial_len]
        cumsum_inner = prefix_sum(trunk_partial_weights)
        r2 = rng.uniform() * cumsum_inner.back()
        local_idx = upper_bound(cumsum_inner, r2) - cumsum_inner.begin() - 1
        edge_idx_in_u = trunk_global_offset(chosen_trunk) + local_idx

    return (graph[u].targets[edge_idx_in_u], graph[u].timestamps[edge_idx_in_u])
```

The walker hot path is dominated by step 1 + step 3 + step 4, all of which are
small for HPAT (cover has ≤ log D entries; chosen trunk has ≤ 2^K entries).

---

## 6. Preprocessing pipeline

Given a CSV stream of `(u, i, t)` (already dense-encoded), produce a fully-built
PAT or HPAT + auxiliary index per vertex.

```
1. Parse CSV → Vec<Edge>. Single-threaded, but fast (memmap + std::from_chars).
2. Bucket edges by source vertex u: build vector<vector<EdgeRef>>.
   - Parallel: each thread handles a vertex-range.
3. For each vertex u:
   a. Sort its edge list by t DESCENDING. (radix sort by t for speed.)
   b. Compute static weights w(e_i) via the bias function.
   c. For each trunk level k ∈ [0..K]:
        for each trunk index i ∈ [0 .. ⌊D_u/2^k⌋):
            build alias table over weights[i·2^k .. (i+1)·2^k)
            total_weight_at_level_k_trunk_i = sum
        build cumsum across trunks at level k
   d. Build auxiliary index for u: for each L ∈ [0 .. D_u]:
        decompose L into binary, fill in covering-trunks list.
   e. For node2vec: build absl::flat_hash_set of u's distinct neighbors.
4. Return TemporalGraph + per-vertex HPAT + per-vertex AuxIndex.
```

**Parallelization**: trivially parallel across vertices (each vertex is
independent). Use `#pragma omp parallel for schedule(dynamic, 64)`. The paper
reports 12.8× speedup at 16 threads, near-linear scaling — same target for us.

**Memory budget** for HPAT on our datasets (assuming each edge appears in
`log(D)` trunks, alias entry = 12 B):

| Dataset | E | log(D) | HPAT mem |
|---|---|---|---|
| growth | 40M | ~17 | ~8 GB |
| coin | 22M | ~11 | ~3 GB |
| flight | 67M | ~12 | ~10 GB |
| tgbl-comment | 44M | ~15 | ~8 GB |
| delicious | 300M | ~22 | ~80 GB (PAT-only fallback) |

Decision: implement both PAT and HPAT; auto-select PAT when memory budget
exceeded; report which one was used per dataset.

---

## 7. Bias correctness — what each variant needs

| Variant | Static weight `δ(e_i)` | Dynamic β | Rejection? | Notes |
|---|---|---|---|---|
| Linear | `rank(e_i)` (position from newest = 1..D) | none | no | Per Eq 2 of the paper. Or use `t_i` if rank is awkward. |
| Exponential | `exp((t_i − t_max_u) × scale)` | none | no | `scale=1` (default, proper TEA, no temperature change) or `timescale_bound / span` (Tempest-compat). |
| Temporal node2vec | same as Exponential | β | **yes** | Rejection on β/β_max where β ∈ {1/p, 1, 1/q}. |

**Numerical resolution (post-Phase-2 review)**: timestamps in real datasets
are unix seconds (10-digit numbers). `exp(t)` for `t ≈ 1.7e9` is wildly out
of range. Resolution:
- **Subtract per-vertex `t_max_u`** so the largest weight is `exp(0) = 1`,
  never overflows. The softmax is shift-invariant — this preserves the
  paper's distribution exactly.
- Older edges may underflow `exp(t − t_max)` to 0 on long-span datasets.
  This is CORRECT under pure exponential bias: those edges genuinely have
  near-zero pick probability.
- Earlier draft proposed "rescale to `[0, 80]`" before exp. **Discarded**:
  scaling is NOT shift-invariant and silently changes the softmax
  temperature. We do not rescale by default.
- For cross-comparison against Tempest (which DOES rescale via its own
  `timescale_bound` argument), our CLI accepts the same arg and applies
  the equivalent rescale via `scale = timescale_bound / span`. The
  runtime banner reports which mode each run used.

---

## 8. CLI — interface

Minimal positional layout. Every accepted arg is read and used by the engine;
nothing is parsed-then-discarded. The benchmark runner skips Tempest's
GPU-only slots (`use_gpu`, `block_dim`, `w_threshold_warp`) and its
streaming/windowing slots (`num_batches`, `num_windows`) when invoking
`tea_walk` — none of those concepts apply to a CPU in-memory engine.

```
Usage:
  tea_walk <file_path>
           [picker=exponential]              ← {uniform, linear, exponential, temporal_node2vec}
           [kernel_launch_type=tea_hpat]     ← {tea_pat, tea_hpat}
           [is_directed=1]
           [num_walks_per_node=20]
           [max_walk_len=80]
           [timescale_bound=-1]              ← -1 = strict paper; >0 = Tempest-compat rescale
```

Output stdout uses the same regex-parseable markers the benchmark harness expects:

```
=== TEA Run ===
File: ...
Picker: exponential
Variant: tea_hpat
...
Throughput: 1.234e+06 walks/sec
Steps/sec: 1.234e+07 steps/sec
Final avg walk length: 37.46
```

This lets `run_ablation.py` add tea_pat / tea_hpat as extra `ALL_VARIANTS`
entries with no changes to the Python orchestrator.

---

## 9. Validation plan

Two orthogonal correctness/perf checks before integrating into the Tempest
paper:

### 9.1 Algorithmic correctness

- **Distributional check**: on a small toy graph (10 vertices, 50 edges),
  run 1M walks with each bias and compare the empirical edge-pick frequencies
  against the analytic `P((u, v_i))` from Eq 2/3/4. χ² test should pass at
  p > 0.01.
- **Cover identity**: for each vertex `u` and each `L ∈ [0, D_u]`, verify that
  the auxiliary-index trunk cover `{τ_u^{k_j, i_j}}` exactly partitions
  `[0, L)` with no gaps or overlaps.
- **Static-weight cancellation**: for ExponentialBias, sample 1M walks at
  `t_cur = t_min` and compare to sampling at `t_cur = t_max - 1`; the
  empirical edge-pick frequencies should match within Monte Carlo noise (the
  cancellation should make the distribution `t_cur`-invariant).

### 9.2 Performance reproduction

Run our reimpl on the four shared datasets and report:

| Metric | Our reimpl | TEA paper (their HW) | KnightKing (run by us) | GraphWalker (run by us) |
|---|---|---|---|---|
| Walks/sec | measured | published | run open-source | run open-source |
| Avg sampling cost (edges/step) | measured | published | measured | measured |
| Preprocess time | measured | published | measured | measured |

If our reimpl matches TEA's *ratio* over KnightKing/GraphWalker (which we can
run independently on the same hardware), the algorithm is reproduced. If not,
we present the absolute numbers and the ratio gap honestly.

The HW we'll target: a single A40-class node (matches Tempest's GPU runs),
running TEA on the CPUs of that node (typically 32-64 cores in a Xeon Gold or
EPYC). Document the exact CPU model in README.

---

## 10. Implementation order — step by step

Each step ends at a working, testable artifact. Don't move to the next until
the previous compiles, tests pass, and a simple end-to-end smoke runs.

### Phase 0 — Scaffold (½ day)

0.1. `CMakeLists.txt` with C++17, OpenMP, gtest fetched.
0.2. `include/tea/edge.hpp`, `walk.hpp`, `rng.hpp` (pcg64 wrapper).
0.3. `cli/tea_walk.cpp` with stub main that just parses positional args.
0.4. Smoke: builds, `tea_walk --help` shows usage.

### Phase 1 — Graph + Alias (1 day)

1.1. `csv_reader.{hpp,cpp}` parses the `u,i,ts` CSV format.
1.2. `graph.{hpp,cpp}`: `TemporalGraph` with per-vertex edge lists, time-desc
     sorted. Parallel preprocess.
1.3. `alias.{hpp,cpp}`: Vose alias table, build + sample.
1.4. Tests: `test_graph.cpp` (sort order, candidate-set boundary),
     `test_alias.cpp` (distribution matches weights via χ²).
1.5. Smoke: load a Koblenz CSV, print `N`, `E`, `max_degree`, sample with
     uniform weights on one vertex 1M times, verify uniform.

### Phase 2 — Bias (½ day)

2.1. `bias.{hpp,cpp}`: `Bias` interface + `LinearBias`, `ExponentialBias`.
2.2. Timestamp rescaling logic (per §7).
2.3. Test: `test_bias.cpp` confirms `static_weight(t)` matches Eq 2/3 for
     synthetic timestamps.

### Phase 3 — PAT (1 day)

3.1. `pat.{hpp,cpp}`: build single-level PAT from a `VertexEdges`.
3.2. `sampler.{hpp,cpp}` (first version, PAT-only): Algorithm 2 step 1–4 with
     PAT.
3.3. Test: `test_pat.cpp` (per-trunk weight totals correct), `test_sampler.cpp`
     (distribution check on a hand-built 3-vertex graph for linear bias).
3.4. Smoke: end-to-end walks on `growth` with `LinearBias`, check that
     **walks/sec is non-zero and walks visit valid edges** (assertion in
     debug build).

### Phase 4 — HPAT + Auxiliary Index (1 day)

4.1. `hpat.{hpp,cpp}`: multi-level trunks, level-k cumsums.
4.2. `aux_index.{hpp,cpp}`: trunks-for-prefix lookup table.
4.3. Switch `sampler.cpp` to use HPAT path when configured.
4.4. Test: `test_hpat.cpp` (per-level total weight matches sum of trunk totals),
     `test_aux_index.cpp` (cover identity for all `L`).
4.5. Smoke: HPAT walks on `growth` produce same empirical-distribution as PAT
     (within MC noise).

### Phase 5 — Node2Vec bias (½ day)

5.1. `neighbor_set.{hpp,cpp}`: per-vertex `absl::flat_hash_set<int32_t>`.
5.2. `Node2VecBias` (inherits ExponentialBias).
5.3. Sampler's inner loop adds the β rejection test.
5.4. Test: `test_bias.cpp` node2vec gets right β for prev=v / prev∈N(v) / else.

### Phase 6 — Parallel walks (½ day)

6.1. Walk engine: OpenMP parallel-for over walk seeds, per-thread RNG state.
6.2. Stable RNG seeding (matches Tempest's reproducibility expectations).
6.3. Bench: `bench_walk_throughput.cpp` reports walks/sec, steps/sec.
6.4. Smoke: scaling test — measure walks/sec at 1, 4, 16 threads, expect ≈
     linear scaling.

### Phase 7 — CLI integration with Tempest harness (½ day)

7.1. `cli/tea_walk.cpp` accepts the `ablation_streaming.cpp` positional-arg
     shape.
7.2. Output `Throughput: ...`, `Steps/sec: ...`, `Final avg walk length: ...`
     lines so `tempest-benchmarks/ablation_runner/common.py` can parse them.
7.3. Add a constants entry in `ablation_runner` for the TEA binary (separate
     binary path; same arg layout).

### Phase 8 — Validation (1–2 days)

8.1. Run on the four shared datasets, all three biases.
8.2. Run KnightKing on the same datasets (open-source, single-machine mode).
8.3. Run GraphWalker on the same datasets (open-source, single-machine).
8.4. Produce the comparison table from §9.2.
8.5. Write up the methodology paragraph for the paper.

**Total estimated effort**: ~6–8 working days for a focused engineer with
Claude Code, single-threaded. Can be cut to ~4 days if HPAT is deferred and
only PAT is implemented (still publishable as "we implement TEA's PAT
mechanism; HPAT is a follow-up optimization we did not reimplement").

---

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Faithful implementation is slower than published numbers, reviewer says "you implemented it wrong" | Open-source the reimpl. Document every algorithmic choice that's ambiguous in the paper. Run KnightKing/GraphWalker on the same node — if our reimpl ratio over them is close to the published ratio, the algorithm is right. |
| `exp(t)` overflow for real timestamps | Per-vertex `t_min` shift + global rescaling to `[0, 80]`. Documented in §7. |
| HPAT memory blows up on delicious (300M edges, 600K nodes) | Auto-fall-back to PAT (single level) when HPAT would exceed memory budget. Document which mode was used per dataset. |
| Node2Vec β rejection ratio degenerate when p << 1 (many rejections) | Cap loop iterations at e.g. 1000; on overflow, fall back to ITS-direct on β-weighted edges. Document. |
| Vendored abseil/gtest bloat | Keep abseil minimal (just `flat_hash_set`); gtest only built when `BUILD_TESTING=ON`. |

---

## 12. Definition of done

- All unit tests pass in CI (single Linux x86 toolchain target is fine).
- End-to-end CLI runs all three biases on all four shared datasets.
- Comparison table (§9.2) is filled in for at least three of the four datasets
  (delicious may need PAT-only fallback or be skipped if memory binds).
- Methodology paragraph for the paper is drafted in `README.md`.
- A reproducibility note documents the divergence (if any) between our
  reimpl numbers and the TEA paper's numbers, with the open-source codepath
  cited so reviewers can inspect.

---

## 13. Anti-goals (do not do these)

- Do not optimize beyond the paper's spec. We're reproducing, not improving.
- Do not implement out-of-core or streaming. They're out of scope.
- Do not use SIMD intrinsics in the alias-table inner loop unless we
  document that as a deliberate optimization equivalent to a footnote in the
  paper. Otherwise reviewers might say we sandbagged the baseline.
- Do not GPU-port. The whole point is CPU-class comparison.
- Do not vendor more than necessary. Build complexity hurts reproducibility.
