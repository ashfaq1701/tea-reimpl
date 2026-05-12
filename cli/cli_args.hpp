// Argv parser for tea_walk.  Every accepted arg is read and used; nothing is
// parsed-then-discarded.  Picker and Variant strings are TEA-native — the
// paper's §2.3 biases (Linear, Exponential, Temporal Node2Vec) and the
// paper's §3.2/§3.3 data-structure variants (PAT, HPAT).

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "tea/config.hpp"

namespace tea {

// TEA paper §2.3 biases. TEA also supports uniform "by assigning uniform
// weights to all edges" (§2.3 preamble) so we keep it as an unbiased option.
enum class Picker {
    Uniform,
    Linear,            // §2.3.I — δ(e) = rank(e)
    Exponential,       // §2.3.II — δ(e) = exp(t_e)  (with static-weight cancellation)
    TemporalNode2Vec,  // §2.3.III — exp(t_e) × β rejection on previous-vertex distance
};

inline Picker parse_picker(const std::string& s) {
    if (s == "uniform")           return Picker::Uniform;
    if (s == "linear")            return Picker::Linear;
    if (s == "exponential")       return Picker::Exponential;
    if (s == "temporal_node2vec") return Picker::TemporalNode2Vec;
    throw std::runtime_error(
        "Invalid picker '" + s + "' — expected one of: "
        "uniform, linear, exponential, temporal_node2vec");
}

inline const char* picker_to_str(Picker p) {
    switch (p) {
        case Picker::Uniform:          return "uniform";
        case Picker::Linear:           return "linear";
        case Picker::Exponential:      return "exponential";
        case Picker::TemporalNode2Vec: return "temporal_node2vec";
    }
    return "?";
}

// TEA paper data-structure variants.
//   Pat  — §3.2: single-level √D trunks.
//   Hpat — §3.3 + §3.4: hierarchical 2^k trunks + precomputed Auxiliary Index.
enum class Variant {
    Pat,
    Hpat,
};

inline Variant parse_variant(const std::string& s) {
    if (s == "tea_pat")  return Variant::Pat;
    if (s == "tea_hpat") return Variant::Hpat;
    throw std::runtime_error(
        "Invalid kernel_launch_type '" + s + "' — expected one of: "
        "tea_pat, tea_hpat");
}

inline const char* variant_to_str(Variant v) {
    switch (v) {
        case Variant::Pat:  return "tea_pat";
        case Variant::Hpat: return "tea_hpat";
    }
    return "?";
}

struct CliArgs {
    std::string file_path;
    Picker      picker             = Picker::Exponential;
    Variant     variant            = Variant::Hpat;
    bool        is_directed        = true;
    int         num_walks_per_node = 20;
    int         max_walk_len       = 80;
    // -1 → strict-paper TEA: δ = exp(t_i − t_max_u), softmax-shift-invariant.
    // > 0 → Tempest-compat rescale: δ = exp((t_i − t_max_u) · timescale_bound / span_u).
    double      timescale_bound    = -1.0;
};

inline void print_usage(const char* progname) {
    std::fprintf(stderr,
        "Usage: %s <file_path>\n"
        "           [picker=exponential]\n"
        "           [kernel_launch_type=tea_hpat]\n"
        "           [is_directed=1]\n"
        "           [num_walks_per_node=20]\n"
        "           [max_walk_len=80]\n"
        "           [timescale_bound=-1]\n"
        "\n"
        "Pickers : uniform, linear, exponential, temporal_node2vec\n"
        "Variants: tea_pat, tea_hpat\n",
        progname);
}

inline CliArgs parse_cli_args(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        std::exit(1);
    }
    std::string a1 = argv[1];
    if (a1 == "--help" || a1 == "-h") {
        print_usage(argv[0]);
        std::exit(0);
    }

    auto try_int    = [&](int idx, int  def) {
        return (argc > idx) ? std::atoi(argv[idx]) : def;
    };
    auto try_double = [&](int idx, double def) {
        return (argc > idx) ? std::atof(argv[idx]) : def;
    };
    auto try_str    = [&](int idx, const char* def) -> std::string {
        return (argc > idx) ? std::string(argv[idx]) : std::string(def);
    };

    CliArgs a;
    a.file_path          = argv[1];
    a.picker             = parse_picker(try_str(2, "exponential"));
    a.variant            = parse_variant(try_str(3, "tea_hpat"));
    a.is_directed        = (try_int(4, 1) != 0);
    a.num_walks_per_node =  try_int(5, 20);
    a.max_walk_len       =  try_int(6, 80);
    a.timescale_bound    =  try_double(7, -1.0);

    if (a.num_walks_per_node <= 0) {
        throw std::runtime_error("num_walks_per_node must be > 0");
    }
    if (a.max_walk_len <= 0) {
        throw std::runtime_error("max_walk_len must be > 0");
    }
    return a;
}

inline void print_config_banner(const CliArgs& a) {
    std::printf(
        "=== tea_walk — TEA reimplementation (CPU) ===\n"
        "File:               %s\n"
        "Hop picker:         %s\n"
        "Sampler variant:    %s\n"
        "Directed graph:     %s\n"
        "Walks per node:     %d\n"
        "Max walk length:    %d\n"
        "Timescale bound:    %g\n",
        a.file_path.c_str(),
        picker_to_str(a.picker),
        variant_to_str(a.variant),
        a.is_directed ? "yes" : "no",
        a.num_walks_per_node,
        a.max_walk_len,
        a.timescale_bound);
}

}  // namespace tea
