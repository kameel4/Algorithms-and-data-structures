#include <unordered_map>

#include "algorithms.h"
#include "dsu.h"

MstResult boruvka_mst(const Graph& graph) {
    Dsu dsu(graph.n);
    MstResult result;
    result.edges.reserve(graph.n > 0 ? graph.n - 1 : 0);
    int components = graph.n;

    while (components > 1) {
        std::unordered_map<int, Edge> best;
        best.reserve(static_cast<std::size_t>(components) * 2U);

        for (const Edge& edge : graph.edges) {
            int ru = dsu.find(edge.u);
            int rv = dsu.find(edge.v);
            if (ru == rv) {
                continue;
            }

            auto it_u = best.find(ru);
            if (it_u == best.end() || edge.w < it_u->second.w) {
                best[ru] = edge;
            }
            auto it_v = best.find(rv);
            if (it_v == best.end() || edge.w < it_v->second.w) {
                best[rv] = edge;
            }
        }

        if (best.empty()) {
            break;
        }

        for (const auto& [component, edge] : best) {
            (void)component;
            if (!dsu.unite(edge.u, edge.v)) {
                continue;
            }
            result.total_weight += edge.w;
            result.edges.push_back(edge);
            --components;
            if (components == 1) {
                break;
            }
        }
    }

    return result;
}
