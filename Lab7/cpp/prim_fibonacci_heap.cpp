#include <cstddef>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "algorithms.h"

namespace {

struct FibNode {
    int key = 0;
    int vertex = 0;
    int degree = 0;
    bool mark = false;
    FibNode* parent = nullptr;
    FibNode* child = nullptr;
    FibNode* left = this;
    FibNode* right = this;
};

class FibHeap {
public:
    FibHeap() = default;

    bool empty() const {
        return min_ == nullptr;
    }

    FibNode* insert(int key, int vertex) {
        storage_.push_back(std::make_unique<FibNode>());
        FibNode* node = storage_.back().get();
        node->key = key;
        node->vertex = vertex;
        node->degree = 0;
        node->mark = false;
        node->parent = nullptr;
        node->child = nullptr;
        node->left = node;
        node->right = node;
        add_root(node);
        ++size_;
        return node;
    }

    FibNode* extract_min() {
        FibNode* node = min_;
        if (node == nullptr) {
            return nullptr;
        }

        if (node->child != nullptr) {
            std::vector<FibNode*> children = collect_cycle(node->child);
            for (FibNode* child : children) {
                child->parent = nullptr;
                child->mark = false;
                child->left = child;
                child->right = child;
                add_root(child);
            }
            node->child = nullptr;
        }

        if (node->right == node) {
            min_ = nullptr;
        } else {
            unlink(node);
            min_ = node->right;
            consolidate();
        }

        node->left = node;
        node->right = node;
        --size_;
        return node;
    }

    void decrease_key(FibNode* node, int new_key) {
        if (new_key >= node->key) {
            return;
        }
        node->key = new_key;
        FibNode* parent = node->parent;
        if (parent != nullptr && node->key < parent->key) {
            cut(node, parent);
            cascading_cut(parent);
        }
        if (min_ == nullptr || node->key < min_->key) {
            min_ = node;
        }
    }

private:
    static std::vector<FibNode*> collect_cycle(FibNode* start) {
        std::vector<FibNode*> nodes;
        if (start == nullptr) {
            return nodes;
        }

        FibNode* current = start;
        do {
            nodes.push_back(current);
            current = current->right;
        } while (current != start);
        return nodes;
    }

    static void unlink(FibNode* node) {
        node->left->right = node->right;
        node->right->left = node->left;
    }

    void add_root(FibNode* node) {
        node->parent = nullptr;
        if (min_ == nullptr) {
            node->left = node;
            node->right = node;
            min_ = node;
            return;
        }

        node->left = min_;
        node->right = min_->right;
        min_->right->left = node;
        min_->right = node;
        if (node->key < min_->key) {
            min_ = node;
        }
    }

    void link(FibNode* child, FibNode* parent) {
        child->parent = parent;
        child->mark = false;
        child->left = child;
        child->right = child;

        if (parent->child == nullptr) {
            parent->child = child;
        } else {
            child->left = parent->child;
            child->right = parent->child->right;
            parent->child->right->left = child;
            parent->child->right = child;
        }
        ++parent->degree;
    }

    void consolidate() {
        std::vector<FibNode*> roots = collect_cycle(min_);
        std::vector<FibNode*> trees(64, nullptr);
        min_ = nullptr;

        for (FibNode* node : roots) {
            node->left = node;
            node->right = node;
            int degree = node->degree;
            while (degree >= static_cast<int>(trees.size())) {
                trees.push_back(nullptr);
            }
            while (trees[degree] != nullptr) {
                FibNode* other = trees[degree];
                trees[degree] = nullptr;
                if (other->key < node->key) {
                    std::swap(node, other);
                }
                link(other, node);
                degree = node->degree;
                while (degree >= static_cast<int>(trees.size())) {
                    trees.push_back(nullptr);
                }
            }
            trees[degree] = node;
        }

        for (FibNode* node : trees) {
            if (node != nullptr) {
                add_root(node);
            }
        }
    }

    void cut(FibNode* node, FibNode* parent) {
        if (node->right == node) {
            parent->child = nullptr;
        } else {
            if (parent->child == node) {
                parent->child = node->right;
            }
            unlink(node);
        }
        --parent->degree;
        node->left = node;
        node->right = node;
        node->parent = nullptr;
        node->mark = false;
        add_root(node);
    }

    void cascading_cut(FibNode* node) {
        FibNode* parent = node->parent;
        if (parent == nullptr) {
            return;
        }
        if (!node->mark) {
            node->mark = true;
            return;
        }
        cut(node, parent);
        cascading_cut(parent);
    }

    FibNode* min_ = nullptr;
    std::size_t size_ = 0;
    std::vector<std::unique_ptr<FibNode>> storage_;
};

}  // namespace

MstResult prim_fibonacci_heap_mst(const Graph& graph) {
    if (graph.n == 0) {
        return {};
    }

    std::vector<std::vector<std::pair<int, int>>> adj(graph.n);
    for (const Edge& edge : graph.edges) {
        adj[edge.u].push_back({edge.v, edge.w});
        adj[edge.v].push_back({edge.u, edge.w});
    }

    FibHeap heap;
    std::vector<FibNode*> nodes(graph.n, nullptr);
    std::vector<int> parent(graph.n, -1);
    std::vector<bool> used(graph.n, false);
    MstResult result;
    result.edges.reserve(graph.n - 1);

    for (int start = 0; start < graph.n; ++start) {
        if (used[start]) {
            continue;
        }
        nodes[start] = heap.insert(0, start);

        while (!heap.empty()) {
            FibNode* node = heap.extract_min();
            if (node == nullptr) {
                break;
            }

            int vertex = node->vertex;
            if (used[vertex]) {
                continue;
            }

            used[vertex] = true;
            nodes[vertex] = nullptr;
            if (parent[vertex] != -1) {
                result.total_weight += node->key;
                result.edges.push_back({parent[vertex], vertex, node->key});
            }

            for (const auto& [to, weight] : adj[vertex]) {
                if (used[to]) {
                    continue;
                }
                if (nodes[to] == nullptr) {
                    parent[to] = vertex;
                    nodes[to] = heap.insert(weight, to);
                } else if (weight < nodes[to]->key) {
                    parent[to] = vertex;
                    heap.decrease_key(nodes[to], weight);
                }
            }
        }
    }

    return result;
}
