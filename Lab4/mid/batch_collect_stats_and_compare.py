#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Батчевый сбор статистик по множеству датасетов и сравнение их с оригиналом.

Что делает скрипт:
1) Считает статистики (mean/median/mode) и распределения для ОРИГИНАЛЬНОГО датасета.
2) Для каждого другого .xlsx в папке делает то же самое (сохраняет в отдельную папку).
3) Для каждого датасета считает разницу статистик относительно оригинала
   и сохраняет в stats_diff_vs_original.xlsx.
4) Строит сводную таблицу средних модулей отличий (MAD) по каждому признаку.
5) (НОВОЕ) Строит наложенные гистограммы распределений для
   оригинала / dropped / spline по всем числовым признакам.

Пример запуска (для small, дырки 5%):

    python batch_collect_stats_and_compare.py \
        --original passengers_dataset_small_processed.xlsx \
        --orig-sheet processed \
        --rest-sheet data \
        --stats-root stats_small \
        --overlay-base passengers_dataset_small_holes5 \
        --skip-temp
"""

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MAX_BARS = 100  # максимум столбиков на категориальной диаграмме


def safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9А-Яа-я._-]+", "_", str(s))


def plot_numeric(series: pd.Series, colname: str, outdir: str):
    plt.figure()
    series.hist(bins=100)
    plt.title(f"Распределение: {colname}")
    plt.xlabel(colname)
    plt.ylabel("Частота")
    plt.grid(True, linestyle=":", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"distribution_{safe_name(colname)}.png"), dpi=150)
    plt.close()


def plot_categorical(series: pd.Series, colname: str, outdir: str, max_bars: int = MAX_BARS):
    counts = series.astype("object").value_counts(dropna=False)

    if len(counts) > max_bars:
        data = counts.iloc[:max_bars]
        n_bars = max_bars
    else:
        data = counts
        n_bars = len(counts)

    fig_width_at_max = 20
    fig_width_min = 6
    if n_bars <= 1:
        fig_w = fig_width_min
    else:
        frac = n_bars / float(max_bars)
        fig_w = fig_width_min + frac * (fig_width_at_max - fig_width_min)

    fig_h = 6 if n_bars <= 30 else 8

    plt.figure(figsize=(fig_w, fig_h))
    x = np.arange(n_bars)
    bar_width = 0.9 if n_bars < 10 else 0.8
    plt.bar(x, data.values, width=bar_width)
    plt.xticks(x, [str(idx) for idx in data.index], rotation=45, ha="right")

    plt.title(f"Распределение категорий: {colname}")
    plt.xlabel(colname)
    plt.ylabel("Частота")
    plt.grid(axis="y", linestyle=":", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"distribution_{safe_name(colname)}.png"), dpi=150)
    plt.close()


def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = pd.DataFrame({
        "mean": df.mean(numeric_only=True),
        "median": df.median(numeric_only=True),
        "mode": df.mode(numeric_only=True).iloc[0]
    })
    return stats


def load_excel(path: str, prefer_sheet: str | None = None) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    sheet = prefer_sheet if (prefer_sheet and prefer_sheet in xls.sheet_names) else xls.sheet_names[0]
    return pd.read_excel(path, sheet_name=sheet)


def process_one_file(path: str, out_root: str, prefer_sheet: str | None = None) -> pd.DataFrame:
    base = os.path.splitext(os.path.basename(path))[0]
    outdir = os.path.join(out_root, base)
    os.makedirs(outdir, exist_ok=True)

    print(f"[+] Обрабатываю {path} -> {outdir}")

    df = load_excel(path, prefer_sheet=prefer_sheet)

    stats = compute_stats(df)
    stats_path = os.path.join(outdir, "dataset_statistics.xlsx")
    stats.to_excel(stats_path)
    print(f"  - Сохранил числовые метрики: {stats_path}")

    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            plot_numeric(s, col, outdir)
        else:
            plot_categorical(s, col, outdir)

    return stats


def make_overlays(original_path: str,
                  original_sheet: str | None,
                  base_name: str,
                  rest_sheet: str | None,
                  stats_root: str):
    """
    Строит наложенные гистограммы для original / <base>_dropped / <base>_splineK2_filled.
    """
    candidates = {
        "original": (original_path, original_sheet),
        "dropped": (f"{base_name}_dropped.xlsx", rest_sheet),
        "spline": (f"{base_name}_splineK2_filled.xlsx", rest_sheet),
    }

    datasets = {}
    for label, (path, sheet) in candidates.items():
        if os.path.exists(path):
            print(f"[overlay] Нашёл {label}: {path}")
            datasets[label] = load_excel(path, prefer_sheet=sheet)
        else:
            print(f"[overlay] Файл для {label} не найден, пропускаю: {path}")

    if len(datasets) < 2:
        print("[overlay] Недостаточно датасетов для оверлеев (нужно хотя бы 2).")
        return

    # общие числовые столбцы
    common_numeric_cols = None
    for df in datasets.values():
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if common_numeric_cols is None:
            common_numeric_cols = set(num_cols)
        else:
            common_numeric_cols &= set(num_cols)

    if not common_numeric_cols:
        print("[overlay] Не нашёл общих числовых столбцов для оверлеев.")
        return

    overlay_dir = os.path.join(stats_root, f"overlays_{os.path.basename(base_name)}")
    os.makedirs(overlay_dir, exist_ok=True)
    print(f"[overlay] Строю оверлеи в {overlay_dir}")

    for col in sorted(common_numeric_cols):
        plt.figure(figsize=(7, 5))

        # общий диапазон для бинов
        all_vals = []
        for df in datasets.values():
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s) > 0:
                all_vals.append(s)
        if not all_vals:
            plt.close()
            continue
        concat_vals = pd.concat(all_vals)
        vmin, vmax = concat_vals.min(), concat_vals.max()
        bins = np.linspace(vmin, vmax, 60)

        for label, df in datasets.items():
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s) == 0:
                continue
            plt.hist(s, bins=bins, density=True, alpha=0.4, label=label, histtype="stepfilled")

        plt.title(f"Наложенное распределение: {col}")
        plt.xlabel(col)
        plt.ylabel("Плотность")
        plt.legend()
        plt.grid(True, linestyle=":", linewidth=0.5)
        plt.tight_layout()
        out_path = os.path.join(overlay_dir, f"overlay_{safe_name(col)}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  - Overlay для {col}: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Батчевый сбор статистик и сравнение с оригиналом.")
    parser.add_argument("--original", required=True,
                        help="Путь к ОРИГИНАЛЬНОМУ датасету (эталон).")
    parser.add_argument("--orig-sheet", default=None,
                        help="Имя листа в оригинальном файле (например 'processed'). Если не указано — первый лист.")
    parser.add_argument("--rest-sheet", default=None,
                        help="Предпочитаемый лист для восстановленных датасетов (например 'data'). Если нет — первый.")
    parser.add_argument("--stats-root", default="stats",
                        help="Корневая папка для результатов (подпапка на каждый датасет).")
    parser.add_argument("--pattern", default="*.xlsx",
                        help="Маска файлов для обработки (по умолчанию *.xlsx).")
    parser.add_argument("--skip-temp", action="store_true",
                        help="Пропускать временные файлы Excel (~$...).")
    parser.add_argument("--overlay-base", default=None,
                        help="Базовое имя дырявого датасета для наложения распределений "
                             "(например passengers_dataset_small_holes5). "
                             "Будут искаться файлы <base>_dropped.xlsx и <base>_splineK2_filled.xlsx.")

    args = parser.parse_args()

    os.makedirs(args.stats_root, exist_ok=True)

    # --- Эталон ---
    ref_path = args.original
    ref_stats = process_one_file(ref_path, args.stats_root, prefer_sheet=args.orig_sheet)
    print("[ref] Эталонные метрики посчитаны.")

    # --- Остальные файлы ---
    all_xlsx = glob.glob(args.pattern)
    files = []
    for p in all_xlsx:
        name = os.path.basename(p)
        if os.path.abspath(p) == os.path.abspath(ref_path):
            continue
        if args.skip_temp and name.startswith("~$"):
            continue
        files.append(p)

    print(f"Найдено {len(files)} файлов для обработки (кроме эталона).")

    diffs_summary = []

    for path in sorted(files):
        base = os.path.splitext(os.path.basename(path))[0]
        stats = process_one_file(path, args.stats_root, prefer_sheet=args.rest_sheet)

        # выравниваем индексы по названиям столбцов
        combined = ref_stats.join(stats, lsuffix="_orig", rsuffix="_rest", how="outer")

        orig_cols = [c for c in combined.columns if c.endswith("_orig")]
        rest_cols = [c for c in combined.columns if c.endswith("_rest")]

        diff_values = combined[rest_cols].values - combined[orig_cols].values
        diff_df = pd.DataFrame(
            diff_values,
            index=combined.index,
            columns=[c.replace("_rest", "") for c in rest_cols],
        )

        outdir = os.path.join(args.stats_root, base)
        diff_path = os.path.join(outdir, "stats_diff_vs_original.xlsx")
        diff_df.to_excel(diff_path)
        print(f"  - Сохранил отличие метрик от оригинала: {diff_path}")

        mean_abs_diff = diff_df.abs().mean()
        row = {"dataset": base}
        for col, val in mean_abs_diff.items():
            row[f"MAD_{col}"] = float(val)
        diffs_summary.append(row)

    if diffs_summary:
        summary_df = pd.DataFrame(diffs_summary)
        summary_path = os.path.join(args.stats_root, "stats_diff_summary_vs_original.xlsx")
        summary_df.to_excel(summary_path, index=False)
        print("[+] Сводная таблица отличий сохранена в:", summary_path)

    # --- Оверлеи распределений ---
    if args.overlay_base:
        make_overlays(
            original_path=args.original,
            original_sheet=args.orig_sheet,
            base_name=args.overlay_base,
            rest_sheet=args.rest_sheet,
            stats_root=args.stats_root,
        )

    print("Готово.")


if __name__ == "__main__":
    main()
