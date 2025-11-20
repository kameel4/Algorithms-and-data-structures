#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Прогоняет несколько датасетов через:
  1) hier_simple.py  (иерархическая кластеризация)
  2) cluster_report.py (отчёт по кластерам)
  3) plot_3d_clusters.py (3D визуализация)

Для каждого исходного .xlsx создаёт отдельную папку с результатами.
Кластеризация идёт по признакам: price, delta_lon, departure_time
"""

import os
import sys
import subprocess

# === настроить под свои имена файлов, если нужно ===
DATASETS = [
    "passengers_dataset_small_processed.xlsx",
    "passengers_dataset_small_holes15.xlsx",
    "passengers_dataset_small_holes15_dropped.xlsx",
    "passengers_dataset_small_holes15_splineK2_filled.xlsx",
]

SHEET_NAME = "Sheet1"  # если лист называется иначе — поменяй здесь
FEATURES = "price,delta_lon,departure_time"


def run(cmd, cwd):
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    python_exe = sys.executable  # тот же интерпретатор, из которого запущен конвейер

    for fname in DATASETS:
        in_path = os.path.join(script_dir, fname)
        if not os.path.isfile(in_path):
            print(f"[SKIP] Файл не найден: {in_path}")
            continue

        base = os.path.splitext(os.path.basename(fname))[0]
        out_dir = os.path.join(script_dir, base + "_results")
        os.makedirs(out_dir, exist_ok=True)

        print("\n==============================")
        print("Обрабатываю датасет:", fname)
        print("Папка результатов:", out_dir)
        print("==============================\n")

        # 1) hier_simple.py — кластеризация
        clusters_xlsx = os.path.join(out_dir, "clusters_output.xlsx")
        sse_csv = os.path.join(out_dir, "clustering_sse_by_k.csv")
        sse_png = os.path.join(out_dir, "clustering_sse_by_k.png")
        dendro_png = os.path.join(out_dir, "dendrogram.png")

        run(
            [
                python_exe,
                "hier_simple.py",
                "--input", in_path,
                "--sheet", SHEET_NAME,
                "--features", FEATURES,
                "--out_excel", clusters_xlsx,
                "--out_sse_csv", sse_csv,
                "--out_sse_png", sse_png,
                "--out_dendrogram_png", dendro_png,
                "--k_star", "5",
            ],
            cwd=script_dir,
        )

        # 2) cluster_report.py — отчёт по кластерам
        cluster_summary_xlsx = os.path.join(out_dir, "cluster_summary.xlsx")

        run(
            [
                python_exe,
                "cluster_report.py",
                "--input", clusters_xlsx,
                "--sheet", SHEET_NAME,
                "--features", FEATURES,
                "--out-excel", cluster_summary_xlsx,
            ],
            cwd=script_dir,
        )

        # 3) plot_3d_clusters.py — 3D картинка (HTML)
        clusters_html = os.path.join(out_dir, "clusters_3d.html")

        run(
            [
                python_exe,
                "plot_3d_clusters.py",
                "--input", clusters_xlsx,
                "--sheet", SHEET_NAME,
                "--features", FEATURES,
                "--out-html", clusters_html,
                # если хочешь автозапуск браузера для каждого датасета, раскомментируй:
                # "--auto-open",
            ],
            cwd=script_dir,
        )

        print(f"\n[OK] Датасет {fname} обработан. Результаты в папке: {out_dir}\n")


if __name__ == "__main__":
    main()
