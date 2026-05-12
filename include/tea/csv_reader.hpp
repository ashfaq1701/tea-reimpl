// CSV reader for the u,i,ts format Tempest's preprocess_other_datasets.py
// emits. Header line skipped; each row is `int32_u,int32_v,int64_ts`.
// Returns a vector<Edge>. Does not sort; the TemporalGraph builder does that.

#pragma once

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "tea/edge.hpp"

namespace tea {

// Reads the entire CSV into memory. Single-threaded; bottleneck is disk on
// large files. For our datasets (<= 1.5 GB CSV) this is fine.
//
// Throws on parse error or open failure.
inline std::vector<Edge> read_edges_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("could not open: " + path);
    }

    std::vector<Edge> out;
    // Heuristic reserve: most rows are ~25 bytes. Don't over-commit.
    out.reserve(1 << 20);

    std::string line;
    // skip header
    if (!std::getline(in, line)) {
        throw std::runtime_error("empty CSV: " + path);
    }

    size_t lineno = 1;
    while (std::getline(in, line)) {
        ++lineno;
        if (line.empty()) continue;

        // Parse three comma-separated ints. Use C string parsing for speed.
        // We accept floats too (some Tempest CSVs have ".0" suffixes) by
        // truncating at the decimal point — strtoll stops at non-digit.
        const char* p   = line.c_str();
        char*       end = nullptr;

        long long u  = std::strtoll(p, &end, 10);
        if (end == p || (*end != ',' && *end != '.')) {
            throw std::runtime_error("bad u at line " + std::to_string(lineno));
        }
        // skip optional ".0" then the comma
        while (*end && *end != ',') ++end;
        p = end + 1;

        long long v = std::strtoll(p, &end, 10);
        if (end == p || (*end != ',' && *end != '.')) {
            throw std::runtime_error("bad v at line " + std::to_string(lineno));
        }
        while (*end && *end != ',') ++end;
        p = end + 1;

        long long t = std::strtoll(p, &end, 10);
        if (end == p) {
            throw std::runtime_error("bad ts at line " + std::to_string(lineno));
        }

        out.push_back(Edge{
            static_cast<int32_t>(u),
            static_cast<int32_t>(v),
            static_cast<int64_t>(t),
        });
    }
    return out;
}

// Quick stats on a loaded edge list, useful for the run banner.
struct EdgeStats {
    int32_t max_node_id = -1;
    int64_t min_ts      = 0;
    int64_t max_ts      = 0;
};

inline EdgeStats compute_edge_stats(const std::vector<Edge>& edges) {
    EdgeStats s;
    if (edges.empty()) return s;
    s.min_ts = edges.front().t;
    s.max_ts = edges.front().t;
    for (const auto& e : edges) {
        if (e.u > s.max_node_id) s.max_node_id = e.u;
        if (e.v > s.max_node_id) s.max_node_id = e.v;
        if (e.t < s.min_ts)      s.min_ts      = e.t;
        if (e.t > s.max_ts)      s.max_ts      = e.t;
    }
    return s;
}

}  // namespace tea
