from __future__ import annotations

import tracemalloc
from collections.abc import Callable
from typing import TypeVar

from models import SearchResult


StateT = TypeVar("StateT")


def measure_search_run(
    run: Callable[[], SearchResult[StateT]],
) -> SearchResult[StateT]:
    tracemalloc.start()
    try:
        result = run()
    finally:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    if result.metrics.peak_memory_kb <= 0:
        result.metrics.peak_memory_kb = peak / 1024
    return result
