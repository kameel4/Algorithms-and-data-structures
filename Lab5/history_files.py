from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np

DEFAULT_HISTORY_DIR = Path(__file__).resolve().parent / "saved_histories"
DEFAULT_OBJECTIVE_NAME = "eggholder"
DEFAULT_REFERENCE_MIN = (512.0, 404.2319)


def _slugify(value):
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower())
    return slug.strip("_") or "history"


def _normalize_result(result):
    return {
        "best_point": np.asarray(result["best_point"], dtype=float).tolist(),
        "best_value": float(result["best_value"]),
        "history_positions": [np.asarray(points, dtype=float).tolist() for points in result["history_positions"]],
        "history_best": [float(value) for value in result["history_best"]],
        "history_best_point": [np.asarray(point, dtype=float).tolist() for point in result["history_best_point"]],
        "history_mean": [float(value) for value in result["history_mean"]],
        "name": str(result["name"]),
    }


def _short_algorithm_name(result_name):
    normalized_name = result_name.lower()
    if normalized_name.startswith("ga"):
        return "ga-bit" if "bitwise" in normalized_name else "ga"
    if normalized_name.startswith("pso"):
        return "pso-std" if "standard" in normalized_name else "pso"
    return _slugify(result_name)


def _next_available_path(directory, base_name):
    candidate = Path(directory) / f"{base_name}.json"
    suffix = 2
    while candidate.exists():
        candidate = Path(directory) / f"{base_name}_{suffix}.json"
        suffix += 1
    return candidate


def save_algorithm_history(
    result,
    *,
    bounds,
    output_dir=DEFAULT_HISTORY_DIR,
    objective_name=DEFAULT_OBJECTIVE_NAME,
    reference_min=DEFAULT_REFERENCE_MIN,
    run_label=None,
):
    normalized_result = _normalize_result(result)
    run_stamp = run_label or datetime.now().strftime("%y%m%d_%H%M%S")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    short_name = _short_algorithm_name(normalized_result["name"])
    file_path = _next_available_path(output_path, f"{run_stamp}_{short_name}")

    payload = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "objective_name": objective_name,
        "bounds": [float(bounds[0]), float(bounds[1])],
        "reference_min": np.asarray(reference_min, dtype=float).tolist(),
        "result": normalized_result,
    }

    with file_path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)

    return file_path


def save_run_histories(
    results,
    *,
    bounds,
    output_dir=DEFAULT_HISTORY_DIR,
    objective_name=DEFAULT_OBJECTIVE_NAME,
    reference_min=DEFAULT_REFERENCE_MIN,
):
    run_label = datetime.now().strftime("%y%m%d_%H%M%S")
    saved_paths = []

    for result in results:
        saved_paths.append(
            save_algorithm_history(
                result,
                bounds=bounds,
                output_dir=output_dir,
                objective_name=objective_name,
                reference_min=reference_min,
                run_label=run_label,
            )
        )

    return saved_paths


def load_algorithm_history(file_path):
    with Path(file_path).open("r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)

    if not isinstance(payload, dict) or "result" not in payload:
        raise ValueError(f"Invalid history file: {file_path}")

    result = payload["result"]
    payload["bounds"] = tuple(float(value) for value in payload["bounds"])
    payload["reference_min"] = np.asarray(payload["reference_min"], dtype=float)
    result["best_point"] = np.asarray(result["best_point"], dtype=float)
    result["history_positions"] = [np.asarray(points, dtype=float) for points in result["history_positions"]]
    result["history_best"] = [float(value) for value in result["history_best"]]
    result["history_best_point"] = [np.asarray(point, dtype=float) for point in result["history_best_point"]]
    result["history_mean"] = [float(value) for value in result["history_mean"]]
    result["best_value"] = float(result["best_value"])
    result["name"] = str(result["name"])

    return payload
