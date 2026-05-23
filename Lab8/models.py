from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar


StateT = TypeVar("StateT")


@dataclass(slots=True)
class SearchMetrics:
    elapsed_ms: float = 0.0
    nodes_expanded: int = 0
    states_generated: int = 0
    pruned_states: int = 0
    max_frontier: int = 0
    max_depth: int = 0
    peak_memory_kb: float = 0.0


@dataclass(slots=True)
class SearchResult(Generic[StateT]):
    algorithm: str
    found: bool
    solution: StateT | None
    metrics: SearchMetrics = field(default_factory=SearchMetrics)
    message: str = ""
