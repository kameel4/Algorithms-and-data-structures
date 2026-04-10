#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "algorithms.h"
#include "graph_io.h"

namespace {

struct AlgorithmEntry {
    std::string name;
    std::function<MstResult(const Graph&)> run;
};

struct Row {
    std::string graph_name;
    int n = 0;
    int m = 0;
    std::string algorithm;
    std::int64_t mst_weight = 0;
    int mst_edges = 0;
    double avg_time_ms = 0.0;
};

const std::vector<int> kSizes = {10, 100, 250, 500, 750, 1000, 1500, 2000, 5000};
const std::vector<int> kPercents = {5, 15, 25, 35, 45, 55, 65, 75, 85};

std::vector<std::filesystem::path> get_graph_paths(const std::filesystem::path& directory, int selected_size) {
    std::vector<std::filesystem::path> paths;
    for (int size : kSizes) {
        if (selected_size != 0 && size != selected_size) {
            continue;
        }
        for (int percent : kPercents) {
            std::ostringstream name;
            name << "n" << size << "_p" << std::setw(2) << std::setfill('0') << percent << ".json";
            paths.push_back(directory / name.str());
        }
    }
    return paths;
}

void save_rows(const std::filesystem::path& path, const std::vector<Row>& rows) {
    std::ofstream output(path);
    output << "graph,n,m,algorithm,mst_weight,mst_edges,avg_time_ms\n";
    for (const Row& row : rows) {
        output << row.graph_name << ','
               << row.n << ','
               << row.m << ','
               << row.algorithm << ','
               << row.mst_weight << ','
               << row.mst_edges << ','
               << std::fixed << std::setprecision(6) << row.avg_time_ms << '\n';
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    std::filesystem::path directory = "generated_graphs";
    std::filesystem::path output_path = "benchmark_results_cpp.csv";
    int repeats = 5;
    int selected_size = 0;

    if (argc >= 2) {
        directory = argv[1];
    }
    if (argc >= 3) {
        repeats = std::stoi(argv[2]);
    }
    if (argc >= 4) {
        output_path = argv[3];
    }
    if (argc >= 5) {
        selected_size = std::stoi(argv[4]);
    }

    const std::vector<AlgorithmEntry> algorithms = {
        {"kruskal", kruskal_mst},
        {"prim_binary_heap", prim_binary_heap_mst},
        {"prim_fibonacci_heap", prim_fibonacci_heap_mst},
        {"boruvka", boruvka_mst},
    };

    std::vector<Row> rows;
    std::cout << std::left
              << std::setw(18) << "graph"
              << std::setw(22) << "algorithm"
              << std::right
              << std::setw(10) << "m"
              << std::setw(10) << "weight"
              << std::setw(14) << "time_ms"
              << '\n';

    for (const auto& path : get_graph_paths(directory, selected_size)) {
        Graph graph = load_graph(path);
        std::int64_t expected_weight = -1;

        for (const auto& algorithm : algorithms) {
            double total_ms = 0.0;
            MstResult result;

            for (int i = 0; i < repeats; ++i) {
                auto start = std::chrono::high_resolution_clock::now();
                result = algorithm.run(graph);
                auto finish = std::chrono::high_resolution_clock::now();
                total_ms += std::chrono::duration<double, std::milli>(finish - start).count();
            }

            if (expected_weight == -1) {
                expected_weight = result.total_weight;
            } else if (expected_weight != result.total_weight) {
                std::cerr << "MST weight mismatch for " << path.filename().string() << '\n';
                return 1;
            }

            Row row;
            row.graph_name = path.filename().string();
            row.n = graph.n;
            row.m = static_cast<int>(graph.edges.size());
            row.algorithm = algorithm.name;
            row.mst_weight = result.total_weight;
            row.mst_edges = static_cast<int>(result.edges.size());
            row.avg_time_ms = total_ms / repeats;
            rows.push_back(row);

            std::cout << std::left
                      << std::setw(18) << row.graph_name
                      << std::setw(22) << row.algorithm
                      << std::right
                      << std::setw(10) << row.m
                      << std::setw(10) << row.mst_weight
                      << std::setw(14) << std::fixed << std::setprecision(3) << row.avg_time_ms
                      << '\n';
        }
    }

    save_rows(output_path, rows);
    std::cout << output_path.lexically_normal().string() << '\n';
    return 0;
}
