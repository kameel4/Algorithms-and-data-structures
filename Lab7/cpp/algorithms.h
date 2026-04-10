#pragma once

#include "mst_types.h"

MstResult kruskal_mst(const Graph& graph);
MstResult prim_binary_heap_mst(const Graph& graph);
MstResult prim_fibonacci_heap_mst(const Graph& graph);
MstResult boruvka_mst(const Graph& graph);
