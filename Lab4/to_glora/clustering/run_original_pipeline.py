#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Пайплайн для passengers_dataset_small_processed_original.xlsx:

1) Прогон 1: кластеризация по ВСЕМ признакам.
2) Прогон 2: кластеризация по 4 признакам:
   price, departure_lon, arrival_lon, departure_time.

Для каждого прогона:
  - hier_simple.py
  - cluster_report.py
  - plot_3d_clusters.py

Результаты:
  passengers_dataset_small_processed_original_all_results/
  passengers_dataset_small_processed_original_4feat_results/
"""

import os
import sys
import subprocess
import pandas as pd


DATASET_NAME = "passengers_dataset_small_processed_original.xlsx"
SHEET_NAME = "Sheet1"   # у твоего файла как раз Sheet1


def run(cmd, cwd):
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    python_exe = sys.executable

    in_path = os.path.join(script_dir, DATASET_NAME)
    if not os.path.isfile(in_path):
        print(f"Файл не найден: {in_path}")
        return

    # --- читаем, чтобы узнать все признаки ---
    df = pd.read_excel(in_path, sheet_name=SHEET_NAME)
    # все колонки, кроме возможного cluster (на всякий случай)
    all_features = [c for c in df.columns if c != "cluster"]
    features_all_str = ",".join(all_features)

    # 4 признака для второй кластеризации
    features_4_str = "price,departure_lon,arrival_lon,departure_time"

    configs = [
        ("all_results", features_all_str, "ALL FEATURES"),
        ("4feat_results", features_4_str, "4 FEATURES (price, dep/arr lon, dep_time)"),
    ]

    for suffix, features_str, label in configs:
        out_dir = os.path.join(
            script_dir,
            os.path.splitext(DATASET_NAME)[0] + "_" + suffix,
        )
        os.makedirs(out_dir, exist_ok=True)

        print("\n====================================")
        print(f"Прогон: {label}")
        print("Папка результатов:", out_dir)
        print("Признаки:", features_str)
        print("====================================\n")

        # --- 1) hier_simple.py ---
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
                "--features", features_str,
                "--out_excel", clusters_xlsx,
                "--out_sse_csv", sse_csv,
                "--out_sse_png", sse_png,
                "--out_dendrogram_png", dendro_png,
            ],
            cwd=script_dir,
        )

        # --- 2) cluster_report.py ---
        cluster_summary_xlsx = os.path.join(out_dir, "cluster_summary.xlsx")

        run(
            [
                python_exe,
                "cluster_report.py",
                "--input", clusters_xlsx,
                "--sheet", "Sheet1",          # hier_simple всегда пишет Sheet1
                "--features", features_str,
                "--out-excel", cluster_summary_xlsx,
            ],
            cwd=script_dir,
        )

        # --- 3) plot_3d_clusters.py ---
        clusters_html = os.path.join(out_dir, "clusters_3d.html")

        run(
            [
                python_exe,
                "plot_3d_clusters.py",
                "--input", clusters_xlsx,
                "--sheet", "Sheet1",
                "--axes", "price,departure_lon,arrival_lon",
                "--size-feature", "departure_time",
                "--out-html", clusters_html,
                # если хочешь, чтобы сразу открывался браузер:
                # "--auto-open",
            ],
            cwd=script_dir,
        )

        print(f"\n[OK] Прогон '{label}' завершён. Результаты в: {out_dir}\n")


if __name__ == "__main__":
    main()
