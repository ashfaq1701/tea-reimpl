// tea_verify_walks: end-to-end correctness verifier for backward walks.
//
// Loads a CSV, builds the graph and HPAT, runs walks via the same path
// as tea_walk, then iterates over EVERY emitted walk and asserts:
//
//   1. Slot 0 = (start_vertex, kSentinelStartTimestamp).
//   2. For k ≥ 1: walk[k].t < walk[k-1].t (strict causal decrease, with
//      sentinel handling for k=1).
//   3. For k ≥ 1: there exists an entry in graph.targets_of(walk[k-1].v)
//      with target = walk[k].v and ts = walk[k].t (the transition is a
//      real edge in the inbound CSR of walk[k-1].v).
//   4. Trailing slots after walk_len are sentinel-padded.
//
// On any failure: prints the offending walk + step and exits 2.
// On success: prints sample walks (5 by default) for manual inspection
// and reports verification statistics.
//
// Usage matches tea_walk: tea_verify_walks <csv> [picker=exponential]
//   [variant=tea_hpat] [is_directed=1] [walks_per_node=1] [max_walk_len=80]
//   [timescale_bound=100] [sample_walks=5]

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <vector>

#include "cli_args.hpp"
#include "tea/bias.hpp"
#include "tea/config.hpp"
#include "tea/csv_reader.hpp"
#include "tea/graph.hpp"
#include "tea/hpat.hpp"
#include "tea/neighbor_set.hpp"
#include "tea/node2vec_bias.hpp"
#include "tea/pat.hpp"
#include "tea/walk_engine.hpp"

namespace {

struct VerifyStats {
    int64_t walks_checked   = 0;
    int64_t edges_checked   = 0;  // total step assertions performed
    int64_t dead_at_start   = 0;
    int64_t max_walk_len_seen = 0;
};

// Returns true if (target, ts) appears in u's inbound adjacency.
// O(D_u) linear scan — fine for verification.
bool edge_in_inbound(const tea::TemporalGraph& g,
                     int32_t u, int32_t target_v, int64_t ts) {
    const auto tgt = g.targets_of(u);
    const auto t   = g.timestamps_of(u);
    for (std::size_t i = 0; i < tgt.size(); ++i) {
        if (tgt[i] == target_v && t[i] == ts) return true;
    }
    return false;
}

void die_on_step(int64_t walk_idx, int32_t step, const char* reason,
                 const tea::TemporalGraph& g,
                 const tea::NodeStep* slots, int32_t walk_len) {
    std::fprintf(stderr, "\n!!! VERIFICATION FAILED on walk %ld step %d: %s\n",
                 static_cast<long>(walk_idx), step, reason);
    std::fprintf(stderr, "Walk (len=%d):\n", walk_len);
    for (int32_t k = 0; k < walk_len; ++k) {
        std::fprintf(stderr, "  slot[%d] = (v=%d, t=%ld)\n",
                     k, slots[k].v, static_cast<long>(slots[k].t));
    }
    if (step >= 1) {
        const int32_t prev_v = slots[step - 1].v;
        std::fprintf(stderr, "Inbound adjacency of v=%d (the prev step's vertex):\n",
                     prev_v);
        const auto tgt = g.targets_of(prev_v);
        const auto t   = g.timestamps_of(prev_v);
        for (std::size_t i = 0; i < tgt.size(); ++i) {
            std::fprintf(stderr, "  src=%d  ts=%ld\n",
                         tgt[i], static_cast<long>(t[i]));
        }
    }
    std::exit(2);
}

// Run all assertions on a single walk.  Returns walk_len from
// walk_lens_out (== max_walk_len if it didn't die).
void verify_walk(const tea::TemporalGraph& g,
                 const tea::NodeStep* slots, int32_t walk_len,
                 int32_t max_walk_len, int64_t walk_idx,
                 VerifyStats& stats) {
    // 1. slot 0 = (start vertex, sentinel ts)
    if (slots[0].t != tea::kSentinelStartTimestamp) {
        die_on_step(walk_idx, 0, "slot 0 timestamp is not the sentinel",
                    g, slots, walk_len);
    }
    if (walk_len < 1 || walk_len > max_walk_len) {
        die_on_step(walk_idx, 0, "walk_len out of range",
                    g, slots, walk_len);
    }
    if (walk_len == 1) ++stats.dead_at_start;

    // 2-3. Each subsequent slot must be a real inbound edge of the prev
    //      slot with strictly smaller timestamp.
    int64_t prev_ts = tea::kSentinelStartTimestamp;
    int32_t prev_v  = slots[0].v;
    for (int32_t k = 1; k < walk_len; ++k) {
        const int32_t v_k = slots[k].v;
        const int64_t t_k = slots[k].t;

        if (v_k == tea::kWalkDeadSentinel) {
            die_on_step(walk_idx, k,
                        "live slot but vertex is kWalkDeadSentinel",
                        g, slots, walk_len);
        }
        // Strictly decreasing in time.  k=1 is t_1 < kSentinelStartTimestamp
        // (INT64_MAX), trivially satisfied for any real ts.
        if (!(t_k < prev_ts)) {
            die_on_step(walk_idx, k, "timestamp not strictly less than prev",
                        g, slots, walk_len);
        }
        // Edge must exist in the inbound CSR of prev_v.
        if (!edge_in_inbound(g, prev_v, v_k, t_k)) {
            die_on_step(walk_idx, k,
                        "edge (prev_v ← v_k @ t_k) not in graph",
                        g, slots, walk_len);
        }

        ++stats.edges_checked;
        prev_ts = t_k;
        prev_v  = v_k;
    }

    // 4. Trailing slots after walk_len must be sentinel-padded.
    for (int32_t k = walk_len; k < max_walk_len; ++k) {
        if (slots[k].v != tea::kWalkDeadSentinel) {
            die_on_step(walk_idx, k, "tail slot is not sentinel-padded",
                        g, slots, walk_len);
        }
    }

    ++stats.walks_checked;
    if (walk_len > stats.max_walk_len_seen) stats.max_walk_len_seen = walk_len;
}

void print_sample_walk(const tea::NodeStep* slots, int32_t walk_len,
                       int64_t walk_idx) {
    std::printf("\n  walk[%ld]  len=%d\n",
                static_cast<long>(walk_idx), walk_len);
    for (int32_t k = 0; k < walk_len; ++k) {
        if (k == 0) {
            std::printf("    [%d] v=%d  ts=SENTINEL\n", k, slots[k].v);
        } else {
            std::printf("    [%d] v=%d  ts=%ld\n",
                        k, slots[k].v, static_cast<long>(slots[k].t));
        }
    }
}

template <typename BiasT>
int run_non_node2vec(const tea::TemporalGraph& g,
                     const tea::CliArgs&       args,
                     const BiasT&              bias,
                     int32_t                   sample_walks_to_print) {
    using namespace tea;
    const char* disable_aux_env = std::getenv("TEA_DISABLE_AUX");
    const bool  disable_aux     = disable_aux_env && disable_aux_env[0] == '1';
    const std::size_t aux_budget = disable_aux ? 0 : kAuxIndexMaxBytes;

    Pat<BiasT>  pat;
    Hpat<BiasT> hpat;
    if (args.variant == Variant::Pat) pat.build(g, bias);
    else                              hpat.build(g, bias, aux_budget);

    const auto starts = make_all_nodes_starts(g, args.num_walks_per_node);
    const int64_t num_walks = static_cast<int64_t>(starts.size());
    std::vector<NodeStep> walks_out(num_walks * args.max_walk_len);
    std::vector<int32_t>  walk_lens_out(num_walks);

    WalkRunStats run_stats;
    if (args.variant == Variant::Hpat) {
        run_stats = run_walks_hpat(g, hpat, bias, starts.data(),
                                   static_cast<int32_t>(num_walks),
                                   args.max_walk_len, 0xc0ffee'd00d'd00dULL,
                                   walks_out.data(), walk_lens_out.data());
    } else {
        run_stats = run_walks_pat(g, pat, bias, starts.data(),
                                  static_cast<int32_t>(num_walks),
                                  args.max_walk_len, 0xc0ffee'd00d'd00dULL,
                                  walks_out.data(), walk_lens_out.data());
    }

    const double avg_len = (run_stats.num_walks > 0)
        ? static_cast<double>(run_stats.total_steps) /
              static_cast<double>(run_stats.num_walks)
        : 0.0;
    std::printf("Walks generated:    %ld  (avg_len=%.2f)\n",
                static_cast<long>(run_stats.num_walks), avg_len);

    // Verify every walk.
    VerifyStats stats;
    auto t0 = std::chrono::steady_clock::now();
    for (int64_t i = 0; i < num_walks; ++i) {
        const NodeStep* slots = walks_out.data() + i * args.max_walk_len;
        verify_walk(g, slots, walk_lens_out[i], args.max_walk_len, i, stats);
    }
    auto t1 = std::chrono::steady_clock::now();
    const double verify_sec = std::chrono::duration<double>(t1 - t0).count();

    std::printf("Verified:           %ld walks, %ld step-transitions  (%.2f s)\n",
                static_cast<long>(stats.walks_checked),
                static_cast<long>(stats.edges_checked),
                verify_sec);
    std::printf("Dead-at-start:      %ld\n", static_cast<long>(stats.dead_at_start));
    std::printf("Max walk length:    %ld\n",
                static_cast<long>(stats.max_walk_len_seen));

    // Sample walks for manual inspection — pick the longest few we have.
    if (sample_walks_to_print > 0) {
        std::printf("\nSample walks (longest %d for manual inspection):",
                    sample_walks_to_print);
        // collect (walk_len, walk_idx) and pick top-K longest
        std::vector<std::pair<int32_t, int64_t>> by_len;
        by_len.reserve(num_walks);
        for (int64_t i = 0; i < num_walks; ++i) {
            by_len.emplace_back(walk_lens_out[i], i);
        }
        const int32_t K = std::min<int32_t>(sample_walks_to_print, num_walks);
        std::partial_sort(by_len.begin(), by_len.begin() + K, by_len.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
        for (int32_t i = 0; i < K; ++i) {
            const int64_t idx = by_len[i].second;
            print_sample_walk(walks_out.data() + idx * args.max_walk_len,
                              walk_lens_out[idx], idx);
        }
    }

    std::printf("\nOK: all walks pass strict causal + edge-existence verification.\n");
    return 0;
}

int run_node2vec(const tea::TemporalGraph& g,
                 const tea::CliArgs&       args,
                 int32_t                   sample_walks_to_print) {
    using namespace tea;
    Node2VecBias bias;
    bias.p = 1.0;
    bias.q = 2.0;
    bias.timescale_bound = args.timescale_bound;

    NeighborSets neighbors;
    neighbors.build(g);

    const char* disable_aux_env = std::getenv("TEA_DISABLE_AUX");
    const bool  disable_aux     = disable_aux_env && disable_aux_env[0] == '1';
    const std::size_t aux_budget = disable_aux ? 0 : kAuxIndexMaxBytes;

    Pat<Node2VecBias>  pat;
    Hpat<Node2VecBias> hpat;
    if (args.variant == Variant::Pat) pat.build(g, bias);
    else                              hpat.build(g, bias, aux_budget);

    const auto starts = make_all_nodes_starts(g, args.num_walks_per_node);
    const int64_t num_walks = static_cast<int64_t>(starts.size());
    std::vector<NodeStep> walks_out(num_walks * args.max_walk_len);
    std::vector<int32_t>  walk_lens_out(num_walks);

    WalkRunStats run_stats;
    if (args.variant == Variant::Hpat) {
        run_stats = run_walks_hpat_node2vec(g, hpat, bias, neighbors,
                                            starts.data(),
                                            static_cast<int32_t>(num_walks),
                                            args.max_walk_len,
                                            0xc0ffee'd00d'd00dULL,
                                            walks_out.data(),
                                            walk_lens_out.data());
    } else {
        run_stats = run_walks_pat_node2vec(g, pat, bias, neighbors,
                                           starts.data(),
                                           static_cast<int32_t>(num_walks),
                                           args.max_walk_len,
                                           0xc0ffee'd00d'd00dULL,
                                           walks_out.data(),
                                           walk_lens_out.data());
    }

    const double avg_len = (run_stats.num_walks > 0)
        ? static_cast<double>(run_stats.total_steps) /
              static_cast<double>(run_stats.num_walks)
        : 0.0;
    std::printf("Walks generated:    %ld  (avg_len=%.2f)\n",
                static_cast<long>(run_stats.num_walks), avg_len);

    VerifyStats stats;
    auto t0 = std::chrono::steady_clock::now();
    for (int64_t i = 0; i < num_walks; ++i) {
        const NodeStep* slots = walks_out.data() + i * args.max_walk_len;
        verify_walk(g, slots, walk_lens_out[i], args.max_walk_len, i, stats);
    }
    auto t1 = std::chrono::steady_clock::now();
    const double verify_sec = std::chrono::duration<double>(t1 - t0).count();

    std::printf("Verified:           %ld walks, %ld step-transitions  (%.2f s)\n",
                static_cast<long>(stats.walks_checked),
                static_cast<long>(stats.edges_checked),
                verify_sec);
    std::printf("Dead-at-start:      %ld\n", static_cast<long>(stats.dead_at_start));
    std::printf("Max walk length:    %ld\n",
                static_cast<long>(stats.max_walk_len_seen));

    if (sample_walks_to_print > 0) {
        std::printf("\nSample walks (longest %d for manual inspection):",
                    sample_walks_to_print);
        std::vector<std::pair<int32_t, int64_t>> by_len;
        by_len.reserve(num_walks);
        for (int64_t i = 0; i < num_walks; ++i) {
            by_len.emplace_back(walk_lens_out[i], i);
        }
        const int32_t K = std::min<int32_t>(sample_walks_to_print, num_walks);
        std::partial_sort(by_len.begin(), by_len.begin() + K, by_len.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
        for (int32_t i = 0; i < K; ++i) {
            const int64_t idx = by_len[i].second;
            print_sample_walk(walks_out.data() + idx * args.max_walk_len,
                              walk_lens_out[idx], idx);
        }
    }

    std::printf("\nOK: all walks pass strict causal + edge-existence verification.\n");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto args = tea::parse_cli_args(argc, argv);
        // Read optional 8th positional: sample_walks_to_print (default 5).
        const int32_t sample_walks_to_print =
            (argc > 8) ? std::atoi(argv[8]) : 5;
        tea::print_config_banner(args);

        auto edges = tea::read_edges_csv(args.file_path);
        const auto stats = tea::compute_edge_stats(edges);
        std::printf("Edges loaded:       %zu\n", edges.size());

        tea::TemporalGraph graph;
        graph.build(std::move(edges),
                    /*num_vertices=*/stats.max_node_id + 1,
                    /*is_directed=*/args.is_directed);
        std::printf("Graph built:        %d vertices, %ld edges\n",
                    graph.num_vertices(),
                    static_cast<long>(graph.num_edges()));

        switch (args.picker) {
            case tea::Picker::Uniform:
                return run_non_node2vec(graph, args, tea::UniformBias{},
                                        sample_walks_to_print);
            case tea::Picker::Linear:
                return run_non_node2vec(graph, args, tea::LinearBias{},
                                        sample_walks_to_print);
            case tea::Picker::Exponential: {
                tea::ExponentialBias b;
                b.timescale_bound = args.timescale_bound;
                return run_non_node2vec(graph, args, b, sample_walks_to_print);
            }
            case tea::Picker::TemporalNode2Vec:
                return run_node2vec(graph, args, sample_walks_to_print);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "tea_verify_walks: %s\n", e.what());
        return 1;
    }
}
