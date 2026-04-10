#include <algorithm>
#include <vector>

#include "algorithms.h"
#include "dsu.h"

MstResult kruskal_mst(const Graph& graph) {
    std::vector<Edge> edges = graph.edges;
    std::sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
        return a.w < b.w;
    });

    Dsu dsu(graph.n);
    MstResult result;
    result.edges.reserve(graph.n > 0 ? graph.n - 1 : 0);

    for (const Edge& edge : edges) {
        if (!dsu.unite(edge.u, edge.v)) {
            continue;
        }
        result.total_weight += edge.w;
        result.edges.push_back(edge);
        if (static_cast<int>(result.edges.size()) == graph.n - 1) {
            break;
        }
    }

    return result;
}
