#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "algorithms.h"
#include "graph_io.h"

namespace {

MstResult run_algorithm(const std::string& name, const Graph& graph) {
    if (name == "kruskal") {
        return kruskal_mst(graph);
    }
    if (name == "prim_binary_heap") {
        return prim_binary_heap_mst(graph);
    }
    if (name == "prim_fibonacci_heap") {
        return prim_fibonacci_heap_mst(graph);
    }
    if (name == "boruvka") {
        return boruvka_mst(graph);
    }
    throw std::runtime_error("Unknown algorithm");
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "usage: mst_runner.exe <algorithm> <graph_path> [repeats]\n";
        return 1;
    }

    const std::string algorithm = argv[1];
    const std::string graph_path = argv[2];
    const int repeats = argc >= 4 ? std::max(1, std::atoi(argv[3])) : 1;
    const Graph graph = load_graph(graph_path);

    MstResult result;
    double total_ms = 0.0;

    for (int i = 0; i < repeats; ++i) {
        const auto start = std::chrono::high_resolution_clock::now();
        result = run_algorithm(algorithm, graph);
        const auto finish = std::chrono::high_resolution_clock::now();
        total_ms += std::chrono::duration<double, std::milli>(finish - start).count();
    }

    std::cout << '{'
              << "\"algorithm\":\"" << algorithm << "\","
              << "\"n\":" << graph.n << ','
              << "\"m\":" << graph.edges.size() << ','
              << "\"mst_weight\":" << result.total_weight << ','
              << "\"mst_edges\":" << result.edges.size() << ','
              << "\"time_ms\":" << std::fixed << std::setprecision(6) << (total_ms / repeats)
              << "}\n";
    return 0;
}
