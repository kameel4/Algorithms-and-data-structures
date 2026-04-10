#pragma once

#include <cctype>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "mst_types.h"

inline Graph load_graph(const std::filesystem::path& path) {
    std::ifstream input(path);
    std::string content((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    std::vector<int> values;
    values.reserve(content.size() / 3);

    bool reading = false;
    int sign = 1;
    int value = 0;

    for (char ch : content) {
        unsigned char uch = static_cast<unsigned char>(ch);
        if (std::isdigit(uch)) {
            if (!reading) {
                reading = true;
                sign = 1;
                value = 0;
            }
            value = value * 10 + (ch - '0');
        } else if (ch == '-' && !reading) {
            reading = true;
            sign = -1;
            value = 0;
        } else if (reading) {
            values.push_back(sign * value);
            reading = false;
            sign = 1;
            value = 0;
        }
    }

    if (reading) {
        values.push_back(sign * value);
    }

    Graph graph;
    if (values.empty()) {
        return graph;
    }

    graph.n = values[0];
    graph.edges.reserve((values.size() - 1) / 3);
    for (std::size_t i = 1; i + 2 < values.size(); i += 3) {
        graph.edges.push_back({values[i], values[i + 1], values[i + 2]});
    }
    return graph;
}
