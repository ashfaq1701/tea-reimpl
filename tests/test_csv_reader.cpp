// Phase 1.1 sanity: round-trip a CSV through disk.

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <gtest/gtest.h>
#include <string>

#include "tea/csv_reader.hpp"

namespace {

std::string write_temp(const std::string& contents) {
    char path[] = "/tmp/tea_csv_test_XXXXXX";
    int fd = mkstemp(path);
    EXPECT_GE(fd, 0);
    if (fd < 0) return {};
    std::ofstream f(path);
    f << contents;
    f.close();
    close(fd);
    return path;
}

}  // namespace

TEST(CsvReaderTest, ParsesIntegerRows) {
    const auto path = write_temp(
        "u,i,ts\n"
        "0,1,100\n"
        "2,3,200\n"
        "4,5,300\n");
    auto edges = tea::read_edges_csv(path);
    std::remove(path.c_str());
    ASSERT_EQ(edges.size(), 3u);
    EXPECT_EQ(edges[0].u, 0); EXPECT_EQ(edges[0].v, 1); EXPECT_EQ(edges[0].t, 100);
    EXPECT_EQ(edges[1].u, 2); EXPECT_EQ(edges[1].v, 3); EXPECT_EQ(edges[1].t, 200);
    EXPECT_EQ(edges[2].u, 4); EXPECT_EQ(edges[2].v, 5); EXPECT_EQ(edges[2].t, 300);
}

TEST(CsvReaderTest, ToleratesFloatSuffixes) {
    // Some Tempest CSVs (ml_tgbl-coin.csv via the pkl→csv pipeline) write
    // "0.0,1.0,1648811421.0". The reader truncates at the decimal.
    const auto path = write_temp(
        "u,i,ts\n"
        "0.0,1.0,100.0\n"
        "42.0,99.0,1648811421.0\n");
    auto edges = tea::read_edges_csv(path);
    std::remove(path.c_str());
    ASSERT_EQ(edges.size(), 2u);
    EXPECT_EQ(edges[0].u, 0);  EXPECT_EQ(edges[0].v, 1); EXPECT_EQ(edges[0].t, 100);
    EXPECT_EQ(edges[1].u, 42); EXPECT_EQ(edges[1].v, 99);
    EXPECT_EQ(edges[1].t, 1648811421LL);
}

TEST(CsvReaderTest, ComputesEdgeStats) {
    std::vector<tea::Edge> edges = {
        {0, 5, 10}, {3, 1, 50}, {2, 7, 30},
    };
    auto s = tea::compute_edge_stats(edges);
    EXPECT_EQ(s.max_node_id, 7);
    EXPECT_EQ(s.min_ts,      10);
    EXPECT_EQ(s.max_ts,      50);
}

TEST(CsvReaderTest, EmptyVsHeaderOnly) {
    // header-only with no rows → empty edge list, not error
    const auto path = write_temp("u,i,ts\n");
    auto edges = tea::read_edges_csv(path);
    std::remove(path.c_str());
    EXPECT_TRUE(edges.empty());
}

TEST(CsvReaderTest, MissingFileThrows) {
    EXPECT_THROW(tea::read_edges_csv("/this/path/does/not/exist.csv"),
                 std::runtime_error);
}
