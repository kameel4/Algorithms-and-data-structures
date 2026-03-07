from pathlib import Path

import matplotlib.pyplot as plt

try:
    from .history_files import load_algorithm_history
except ImportError:
    from history_files import load_algorithm_history

HISTORY_FILES = [
    r"C:\Users\kameel\Repositories\Algorithms-and-data-structures\Lab5\saved_histories\260307_052013_pso-vmax005.json",
    r"C:\Users\kameel\Repositories\Algorithms-and-data-structures\Lab5\saved_histories\260307_052050_pso-vmax02.json",
    r"C:\Users\kameel\Repositories\Algorithms-and-data-structures\Lab5\saved_histories\260307_052136_pso-vmax05.json",
    r"C:\Users\kameel\Repositories\Algorithms-and-data-structures\Lab5\saved_histories\260307_052209_pso-vmax1.json",
]


def visualize_saved_histories(history_items):
    if not history_items:
        raise ValueError("Add at least one path to HISTORY_FILES before running this script.")

    fig, (ax_mean, ax_best) = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    for item in history_items:
        label = item["label"]
        result = item["payload"]["result"]
        mean_values = result["history_mean"]
        best_values = result["history_best"]

        ax_mean.plot(range(len(mean_values)), mean_values, label=label)
        ax_best.plot(range(len(best_values)), best_values, label=label)

    ax_mean.set_title("Mean fitness evolution")
    ax_mean.set_xlabel("Iteration / generation")
    ax_mean.set_ylabel("f(x, y)")
    ax_mean.grid(True, alpha=0.3)
    ax_mean.legend()

    ax_best.set_title("Best fitness evolution")
    ax_best.set_xlabel("Iteration / generation")
    ax_best.set_ylabel("f(x, y)")
    ax_best.grid(True, alpha=0.3)
    ax_best.legend()

    plt.tight_layout()
    plt.show()


def main():
    history_items = []
    for raw_path in HISTORY_FILES:
        file_path = Path(raw_path)
        history_items.append(
            {
                "label": file_path.stem,
                "payload": load_algorithm_history(file_path),
            }
        )

    visualize_saved_histories(history_items)


if __name__ == "__main__":
    main()
