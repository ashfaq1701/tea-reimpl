// tea_walk: CPU TEA temporal-walk reimplementation runner.
//
// Phase 6: CSV → TemporalGraph → PAT/HPAT (+ NeighborSets for Node2Vec)
//                              → parallel walks → stdout stats.
//
// Argv layout matches Tempest's ablation_streaming.cpp so the existing
// tempest-benchmarks/ablation_runner harness can drive both binaries with
// the same `common.py` regex parsing.

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

template <typename BiasT>
void run_non_node2vec(const tea::TemporalGraph& g,
                      const tea::CliArgs&       args,
                      const BiasT&              bias) {
    using namespace tea;

    // --- Build PAT or HPAT
    // TEA_DISABLE_AUX=1 turns off the precomputed AuxiliaryIndex (paper §3.4)
    // for HPAT, forcing the sampler to use on-the-fly binary decomposition.
    // Used for §8 ablation runs (aux-on vs aux-off contribution).
    const char* disable_aux_env = std::getenv("TEA_DISABLE_AUX");
    const bool  disable_aux     = disable_aux_env && disable_aux_env[0] == '1';
    const std::size_t aux_budget = disable_aux ? 0 : kAuxIndexMaxBytes;

    auto t0 = std::chrono::steady_clock::now();
    Pat<BiasT>  pat;
    Hpat<BiasT> hpat;
    if (args.variant == Variant::Pat) {
        pat.build(g, bias);
    } else {
        hpat.build(g, bias, aux_budget);
    }
    auto t1 = std::chrono::steady_clock::now();
    const double struct_secs = std::chrono::duration<double>(t1 - t0).count();
    const int64_t struct_mem =
        (args.variant == Variant::Hpat) ? hpat.memory_bytes() : pat.memory_bytes();
    const bool aux_active =
        (args.variant == Variant::Hpat) && hpat.aux().active();
    std::printf("%-12s built (%s):  %.1f MB  (%.2f s)  aux=%s\n",
                variant_to_str(args.variant), BiasT::name(),
                static_cast<double>(struct_mem) / 1e6, struct_secs,
                aux_active ? "on" : "off");

    // --- Wall-time bracket (matches Tempest's wall-time semantic).
    //     Tempest brackets EVERYTHING inside get_random_walks_and_times_*:
    //     start-vertex list construction (repeated_node_ids), output
    //     buffer cudaMalloc, walk kernels + sync, D2H copy.  TEA's
    //     analog includes the same kinds of work it has to do:
    //     make_all_nodes_starts, output vector<> allocation, the
    //     OpenMP walk loop.  TEA writes to host memory directly, so
    //     it has no D2H copy to include.
    const auto wall_t0 = std::chrono::steady_clock::now();

    // --- Build start-vertex list
    const auto starts = make_all_nodes_starts(g, args.num_walks_per_node);
    const int64_t num_walks = static_cast<int64_t>(starts.size());
    if (num_walks == 0) {
        std::printf("No starting vertices (all vertices have degree 0).\n");
        return;
    }
    const int64_t output_slots = num_walks * args.max_walk_len;

    // --- Allocate output buffers
    std::vector<NodeStep> walks_out(static_cast<std::size_t>(output_slots));
    std::vector<int32_t>  walk_lens_out(static_cast<std::size_t>(num_walks));

    // --- Run walks
    //     Inner bracket around just the run_walks_* call: pure walk-loop time
    //     (no start-list build, no output-buffer allocation). The outer wall-
    //     time bracket above is the Tempest-comparable figure; this inner
    //     bracket isolates the walker hot loop for ablation use.
    WalkRunStats stats;
    const auto walk_loop_t0 = std::chrono::steady_clock::now();
    if (args.variant == Variant::Hpat) {
        stats = run_walks_hpat(g, hpat, bias, starts.data(),
                                static_cast<int32_t>(num_walks),
                                args.max_walk_len,
                                /*global_seed=*/0xc0ffee'd00d'd00dULL,
                                walks_out.data(), walk_lens_out.data());
    } else {
        stats = run_walks_pat(g, pat, bias, starts.data(),
                               static_cast<int32_t>(num_walks),
                               args.max_walk_len,
                               /*global_seed=*/0xc0ffee'd00d'd00dULL,
                               walks_out.data(), walk_lens_out.data());
    }
    const auto walk_loop_t1 = std::chrono::steady_clock::now();
    const double walk_loop_sec =
        std::chrono::duration<double>(walk_loop_t1 - walk_loop_t0).count();

    const auto wall_t1 = std::chrono::steady_clock::now();
    const double wall_sec =
        std::chrono::duration<double>(wall_t1 - wall_t0).count();

    const double walks_per_sec = (wall_sec > 0.0)
        ? static_cast<double>(stats.num_walks) / wall_sec : 0.0;
    const double steps_per_sec = (wall_sec > 0.0)
        ? static_cast<double>(stats.total_steps) / wall_sec : 0.0;
    const double walk_loop_steps_per_sec = (walk_loop_sec > 0.0)
        ? static_cast<double>(stats.total_steps) / walk_loop_sec : 0.0;
    const double avg_len = (stats.num_walks > 0)
        ? static_cast<double>(stats.total_steps) /
              static_cast<double>(stats.num_walks)
        : 0.0;

    // --- Report — same format the harness has been parsing all along;
    //     only the time figure inside Walks done changed (now wall time
    //     around the whole walker invocation, matching Tempest).
    //     New "Walk loop ..." lines are additive; existing parsers anchored
    //     to "^Walks done:" / "^Steps/sec:" / "^Throughput:" are unaffected.
    std::printf("Walks scheduled:    %ld  (wpn=%d × active vertices)\n",
                static_cast<long>(num_walks), args.num_walks_per_node);
    std::printf("Walks done:         %ld  (%.2f s)\n",
                static_cast<long>(stats.num_walks), wall_sec);
    std::printf("Throughput:         %.3e walks/sec\n", walks_per_sec);
    std::printf("Steps/sec:          %.3e steps/sec\n", steps_per_sec);
    std::printf("Walk loop time:     %.2f s\n", walk_loop_sec);
    std::printf("Walk loop steps/sec: %.3e steps/sec\n",
                walk_loop_steps_per_sec);
    std::printf("Final avg walk length: %.2f\n",        avg_len);
    std::printf("Dead-at-start:      %ld\n",
                static_cast<long>(stats.dead_at_start));
}

void run_node2vec(const tea::TemporalGraph& g,
                  const tea::CliArgs&       args) {
    using namespace tea;

    Node2VecBias bias;
    bias.p = 1.0;
    bias.q = 2.0;
    bias.timescale_bound = args.timescale_bound;

    // --- Neighbor sets (Node2Vec-only — Phase 5 dep)
    auto t0 = std::chrono::steady_clock::now();
    NeighborSets neighbors;
    neighbors.build(g);
    auto t1 = std::chrono::steady_clock::now();
    std::printf("Neighbor sets:      %ld entries  (%.2f s, %.1f MB)\n",
                static_cast<long>(neighbors.total_neighbor_count()),
                std::chrono::duration<double>(t1 - t0).count(),
                static_cast<double>(neighbors.memory_bytes()) / 1e6);

    // --- Build PAT or HPAT (env knobs identical to non-node2vec path)
    const char* disable_aux_env = std::getenv("TEA_DISABLE_AUX");
    const bool  disable_aux     = disable_aux_env && disable_aux_env[0] == '1';
    const std::size_t aux_budget = disable_aux ? 0 : kAuxIndexMaxBytes;

    t0 = std::chrono::steady_clock::now();
    Pat<Node2VecBias>  pat;
    Hpat<Node2VecBias> hpat;
    if (args.variant == Variant::Pat) {
        pat.build(g, bias);
    } else {
        hpat.build(g, bias, aux_budget);
    }
    t1 = std::chrono::steady_clock::now();
    const int64_t struct_mem =
        (args.variant == Variant::Hpat) ? hpat.memory_bytes() : pat.memory_bytes();
    std::printf("%-12s built (node2vec): %.1f MB  (%.2f s)\n",
                variant_to_str(args.variant),
                static_cast<double>(struct_mem) / 1e6,
                std::chrono::duration<double>(t1 - t0).count());

    // --- Wall-time bracket (matches Tempest's wall-time semantic).
    const auto wall_t0 = std::chrono::steady_clock::now();

    // --- Starts + buffers
    const auto starts = make_all_nodes_starts(g, args.num_walks_per_node);
    const int64_t num_walks    = static_cast<int64_t>(starts.size());
    const int64_t output_slots = num_walks * args.max_walk_len;
    std::vector<NodeStep> walks_out(static_cast<std::size_t>(output_slots));
    std::vector<int32_t>  walk_lens_out(static_cast<std::size_t>(num_walks));

    // --- Run walks
    //     Inner bracket around just the run_walks_*_node2vec call: pure walk-
    //     loop time (no start-list build, no output-buffer allocation). The
    //     outer wall-time bracket above is the Tempest-comparable figure;
    //     this inner bracket isolates the walker hot loop for ablation use.
    WalkRunStats stats;
    const auto walk_loop_t0 = std::chrono::steady_clock::now();
    if (args.variant == Variant::Hpat) {
        stats = run_walks_hpat_node2vec(g, hpat, bias, neighbors,
                                         starts.data(),
                                         static_cast<int32_t>(num_walks),
                                         args.max_walk_len,
                                         0xc0ffee'd00d'd00dULL,
                                         walks_out.data(),
                                         walk_lens_out.data());
    } else {
        stats = run_walks_pat_node2vec(g, pat, bias, neighbors,
                                        starts.data(),
                                        static_cast<int32_t>(num_walks),
                                        args.max_walk_len,
                                        0xc0ffee'd00d'd00dULL,
                                        walks_out.data(),
                                        walk_lens_out.data());
    }
    const auto walk_loop_t1 = std::chrono::steady_clock::now();
    const double walk_loop_sec =
        std::chrono::duration<double>(walk_loop_t1 - walk_loop_t0).count();

    const auto wall_t1 = std::chrono::steady_clock::now();
    const double wall_sec =
        std::chrono::duration<double>(wall_t1 - wall_t0).count();

    const double walks_per_sec = (wall_sec > 0.0)
        ? static_cast<double>(stats.num_walks) / wall_sec : 0.0;
    const double steps_per_sec = (wall_sec > 0.0)
        ? static_cast<double>(stats.total_steps) / wall_sec : 0.0;
    const double walk_loop_steps_per_sec = (walk_loop_sec > 0.0)
        ? static_cast<double>(stats.total_steps) / walk_loop_sec : 0.0;
    const double avg_len = (stats.num_walks > 0)
        ? static_cast<double>(stats.total_steps) /
              static_cast<double>(stats.num_walks)
        : 0.0;

    std::printf("Walks scheduled:    %ld  (wpn=%d × active vertices)\n",
                static_cast<long>(num_walks), args.num_walks_per_node);
    std::printf("Walks done:         %ld  (%.2f s)\n",
                static_cast<long>(stats.num_walks), wall_sec);
    std::printf("Throughput:         %.3e walks/sec\n", walks_per_sec);
    std::printf("Steps/sec:          %.3e steps/sec\n", steps_per_sec);
    std::printf("Walk loop time:     %.2f s\n", walk_loop_sec);
    std::printf("Walk loop steps/sec: %.3e steps/sec\n",
                walk_loop_steps_per_sec);
    std::printf("Final avg walk length: %.2f\n",        avg_len);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto args = tea::parse_cli_args(argc, argv);
        tea::print_config_banner(args);

        // 1. Load CSV
        auto t0 = std::chrono::steady_clock::now();
        auto edges = tea::read_edges_csv(args.file_path);
        auto t1 = std::chrono::steady_clock::now();
        const auto stats = tea::compute_edge_stats(edges);
        std::printf("Edges loaded:       %zu  (%.2f s)\n", edges.size(),
                    std::chrono::duration<double>(t1 - t0).count());
        std::printf("Max node id:        %d  (=> %d vertices, dense)\n",
                    stats.max_node_id, stats.max_node_id + 1);
        std::printf("Timestamp range:    [%ld, %ld]\n",
                    static_cast<long>(stats.min_ts),
                    static_cast<long>(stats.max_ts));

        // 2. Build graph
        t0 = std::chrono::steady_clock::now();
        tea::TemporalGraph graph;
        graph.build(std::move(edges),
                    /*num_vertices=*/stats.max_node_id + 1,
                    /*is_directed=*/args.is_directed);
        t1 = std::chrono::steady_clock::now();
        std::printf("Graph built:        %d vertices, %ld edges  (%.2f s)\n",
                    graph.num_vertices(),
                    static_cast<long>(graph.num_edges()),
                    std::chrono::duration<double>(t1 - t0).count());

        // 3. Dispatch by bias type, run walks, print throughput.
        switch (args.picker) {
            case tea::Picker::Uniform:
                run_non_node2vec(graph, args, tea::UniformBias{}); break;
            case tea::Picker::Linear:
                run_non_node2vec(graph, args, tea::LinearBias{}); break;
            case tea::Picker::Exponential: {
                tea::ExponentialBias b;
                b.timescale_bound = args.timescale_bound;
                run_non_node2vec(graph, args, b);
                break;
            }
            case tea::Picker::TemporalNode2Vec:
                run_node2vec(graph, args);
                break;
        }

        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "tea_walk: %s\n\n", e.what());
        tea::print_usage(argv[0]);
        return 2;
    }
}
