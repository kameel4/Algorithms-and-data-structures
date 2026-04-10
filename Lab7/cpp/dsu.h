#pragma once

#include <vector>

class Dsu {
public:
    explicit Dsu(int n) : parent_(n), size_(n, 1) {
        for (int i = 0; i < n; ++i) {
            parent_[i] = i;
        }
    }

    int find(int x) {
        while (parent_[x] != x) {
            parent_[x] = parent_[parent_[x]];
            x = parent_[x];
        }
        return x;
    }

    bool unite(int a, int b) {
        int ra = find(a);
        int rb = find(b);
        if (ra == rb) {
            return false;
        }
        if (size_[ra] < size_[rb]) {
            std::swap(ra, rb);
        }
        parent_[rb] = ra;
        size_[ra] += size_[rb];
        return true;
    }

private:
    std::vector<int> parent_;
    std::vector<int> size_;
};
