#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Простая иерархическая кластеризация:

- Берём только заданные признаки (по умолчанию: price, arrival_lon, departure_lon, departure_time)
- Импутируем пропуски (медиана)
- Стандартизируем
- Считаем linkage (complete, евклидово расстояние)
- Для k = [kmin..kmax]:
    * режем дендрограмму -> метки кластеров
    * считаем компактность SSE
- Строим:
    * дендрограмму (укороченную)
    * график SSE(k)

Выходы:
- Excel с меткой кластера для выбранного k (k_star)
- CSV с таблицей SSE по k
- PNG с дендрограммой
- PNG с графиком SSE(k)
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram


# ---------- утилиты ----------

def impute_simple(X: pd.DataFrame, how: str = "median") -> pd.DataFrame:
    """Простая импутация: медиана (или среднее) по столбцу."""
    X = X.copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        if how == "median":
            X[c] = X[c].fillna(X[c].median())
        else:
            X[c] = X[c].fillna(X[c].mean())
    return X


def standardize(X: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X.values.astype(float))
    return Xz, scaler


def compactness_sse(X: np.ndarray, labels: np.ndarray) -> float:
    """
    SSE = сумма квадратов отклонений точек от центров своих кластеров.
    """
    sse = 0.0
    for lab in np.unique(labels):
        M = X[labels == lab]
        if M.shape[0] == 0:
            continue
        mu = M.mean(axis=0, keepdims=True)
        dif = M - mu
        sse += float(np.sum(dif * dif))
    return sse


# ---------- основной пайплайн ----------

def main():
    ap = argparse.ArgumentParser(
        description="Чистая иерархическая кластеризация + дендрограмма + график SSE(k)."
    )
    ap.add_argument("--input", required=True, help="Входной Excel с данными")
    ap.add_argument("--sheet", default="data", help="Имя листа Excel (по умолчанию 'data')")
    ap.add_argument(
        "--features",
        default="price,arrival_lon,departure_lon,departure_time",
        help="Список признаков через запятую (по умолчанию: price,arrival_lon,departure_lon,departure_time)",
    )
    ap.add_argument("--impute", default="median", choices=["median", "mean"],
                    help="Способ импутации пропусков (median/mean)")
    ap.add_argument("--kmin", type=int, default=2, help="Минимальное число кластеров")
    ap.add_argument("--kmax", type=int, default=15, help="Максимальное число кластеров")
    ap.add_argument(
        "--k_star",
        type=int,
        default=10,
        help="Какое k использовать для финальных меток (по умолчанию 10, можно перегрузить)",
    )
    ap.add_argument("--out_excel", default="clusters_output.xlsx",
                    help="Excel с колонкой 'cluster' для k_star")
    ap.add_argument("--out_sse_csv", default="clustering_sse_by_k.csv",
                    help="CSV с SSE по k")
    ap.add_argument("--out_sse_png", default="clustering_sse_by_k.png",
                    help="PNG с графиком SSE(k)")
    ap.add_argument("--out_dendrogram_png", default="dendrogram.png",
                    help="PNG с дендрограммой (укороченной)")
    ap.add_argument("--truncate_p", type=int, default=50,
                    help="Сколько последних кластеров показывать на дендрограмме (truncate_mode='lastp')")
    args = ap.parse_args()

    # 1) читаем данные
    df = pd.read_excel(args.input, sheet_name=args.sheet)

    # 2) берём только нужные признаки
    feature_cols = [c.strip() for c in args.features.split(",") if c.strip()]
    for c in feature_cols:
        if c not in df.columns:
            raise ValueError(f"Столбец '{c}' не найден в данных.")
    X = df[feature_cols]
    X = impute_simple(X, how=args.impute)

    # 3) стандартизация
    Xz, scaler = standardize(X)

    # 4) linkage один раз
    print("Вычисляю linkage (complete, euclidean)...")
    Z = linkage(Xz, method="complete", metric="euclidean")
    print("Готово.")

    # 5) перебор k и расчёт SSE
    records = []
    for k in range(args.kmin, args.kmax + 1):
        labels_k = fcluster(Z, t=k, criterion="maxclust")
        sse_k = compactness_sse(Xz, labels_k)
        records.append({"k": k, "SSE": sse_k})
        print(f"k={k}: SSE={sse_k:.2f}")

    sse_table = pd.DataFrame(records)
    sse_table.to_csv(args.out_sse_csv, index=False, encoding="utf-8")
    print("Сохранил SSE по k в:", args.out_sse_csv)

    # 6) график SSE(k)
    plt.figure(figsize=(6, 4))
    plt.plot(sse_table["k"], sse_table["SSE"], marker="o")
    plt.xlabel("Число кластеров k")
    plt.ylabel("SSE (компактность)")
    plt.title("Зависимость компактности кластеров от k (иерархическая кластеризация)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out_sse_png, dpi=150)
    plt.close()
    print("Сохранил график SSE(k) в:", args.out_sse_png)

    # 7) дендрограмма (укороченная)
    plt.figure(figsize=(10, 6))
    dendrogram(
        Z,
        truncate_mode="lastp",
        p=args.truncate_p,
        show_leaf_counts=True,
        leaf_rotation=90.,
        leaf_font_size=8.,
        color_threshold=None,
    )
    plt.title("Иерархическая кластеризация (дендрограмма, последние p кластеров)")
    plt.xlabel("Кластеры")
    plt.ylabel("Расстояние")
    plt.tight_layout()
    plt.savefig(args.out_dendrogram_png, dpi=160)
    plt.close()
    print("Сохранил дендрограмму в:", args.out_dendrogram_png)

    # 8) финальные метки для k_star
    if not (args.kmin <= args.k_star <= args.kmax):
        print(f"Внимание: k_star={args.k_star} вне диапазона [{args.kmin}, {args.kmax}]. "
              f"Всё равно посчитаю, но SSE в таблице будет только для k_min..k_max.")

    labels_star = fcluster(Z, t=args.k_star, criterion="maxclust")
    out_df = df.copy()
    out_df["cluster"] = labels_star
    out_df.to_excel(args.out_excel, index=False)
    print("Сохранил финальные метки кластеров (k_star) в:", args.out_excel)


if __name__ == "__main__":
    main()
