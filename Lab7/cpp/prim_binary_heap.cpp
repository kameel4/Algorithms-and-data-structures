#include <limits>
#include <utility>
#include <vector>

#include "algorithms.h"

namespace {

class BinaryMinHeap {
public:
    explicit BinaryMinHeap(int n)
        : pos_(n, -1), key_(n, std::numeric_limits<int>::max()) {}

    bool empty() const {
        return heap_.empty();
    }

    bool contains(int vertex) const {
        return pos_[vertex] != -1;
    }

    int key(int vertex) const {
        return key_[vertex];
    }

    void push_or_decrease(int vertex, int key) {
        if (pos_[vertex] == -1) {
            key_[vertex] = key;
            pos_[vertex] = static_cast<int>(heap_.size());
            heap_.push_back(vertex);
            sift_up(pos_[vertex]);
            return;
        }
        if (key < key_[vertex]) {
            key_[vertex] = key;
            sift_up(pos_[vertex]);
        }
    }

    std::pair<int, int> pop() {
        int vertex = heap_.front();
        int key = key_[vertex];
        swap_nodes(0, static_cast<int>(heap_.size()) - 1);
        heap_.pop_back();
        pos_[vertex] = -1;
        if (!heap_.empty()) {
            sift_down(0);
        }
        return {vertex, key};
    }

private:
    void sift_up(int index) {
        while (index > 0) {
            int parent = (index - 1) / 2;
            if (key_[heap_[parent]] <= key_[heap_[index]]) {
                break;
            }
            swap_nodes(parent, index);
            index = parent;
        }
    }

    void sift_down(int index) {
        int size = static_cast<int>(heap_.size());
        while (true) {
            int left = index * 2 + 1;
            int right = left + 1;
            int smallest = index;

            if (left < size && key_[heap_[left]] < key_[heap_[smallest]]) {
                smallest = left;
            }
            if (right < size && key_[heap_[right]] < key_[heap_[smallest]]) {
                smallest = right;
            }
            if (smallest == index) {
                break;
            }
            swap_nodes(index, smallest);
            index = smallest;
        }
    }

    void swap_nodes(int a, int b) {
        std::swap(heap_[a], heap_[b]);
        pos_[heap_[a]] = a;
        pos_[heap_[b]] = b;
    }

    std::vector<int> heap_;
    std::vector<int> pos_;
    std::vector<int> key_;
};

}  // namespace

MstResult prim_binary_heap_mst(const Graph& graph) {
    if (graph.n == 0) {
        return {};
    }

    std::vector<std::vector<std::pair<int, int>>> adj(graph.n);
    for (const Edge& edge : graph.edges) {
        adj[edge.u].push_back({edge.v, edge.w});
        adj[edge.v].push_back({edge.u, edge.w});
    }

    std::vector<bool> used(graph.n, false);
    std::vector<int> parent(graph.n, -1);
    BinaryMinHeap heap(graph.n);
    MstResult result;
    result.edges.reserve(graph.n - 1);

    for (int start = 0; start < graph.n; ++start) {
        if (used[start]) {
            continue;
        }
        heap.push_or_decrease(start, 0);

        while (!heap.empty()) {
            auto [vertex, key] = heap.pop();
            if (used[vertex]) {
                continue;
            }

            used[vertex] = true;
            if (parent[vertex] != -1) {
                result.total_weight += key;
                result.edges.push_back({parent[vertex], vertex, key});
            }

            for (const auto& [to, weight] : adj[vertex]) {
                if (used[to]) {
                    continue;
                }
                if (!heap.contains(to) || weight < heap.key(to)) {
                    parent[to] = vertex;
                    heap.push_or_decrease(to, weight);
                }
            }
        }
    }

    return result;
}
