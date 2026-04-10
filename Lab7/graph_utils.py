import json
from pathlib import Path


def load_graph(path: str | Path) -> tuple[int, list[tuple[int, int, int]]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    edges = [tuple(edge) for edge in data["edges"]]
    return data["n"], edges


def save_graph(path: str | Path, n: int, edges: list[tuple[int, int, int]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {"n": n, "edges": [list(edge) for edge in edges]}
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
