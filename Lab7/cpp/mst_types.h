#pragma once

#include <cstdint>
#include <vector>

struct Edge {
    int u = 0;
    int v = 0;
    int w = 0;
};

struct Graph {
    int n = 0;
    std::vector<Edge> edges;
};

struct MstResult {
    std::int64_t total_weight = 0;
    std::vector<Edge> edges;
};
