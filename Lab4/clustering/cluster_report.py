#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт-репорт по результатам кластеризации.

Ожидает Excel с:
  - признаками (features),
  - колонкой с метками кластеров (по умолчанию 'cluster').

Считает для каждого кластера:
  - размер,
  - центр кластера (среднее по признакам в исходных единицах),
  - самого типичного представителя (ближайшая точка к центру в z-пространстве).
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


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


def main():
    ap = argparse.ArgumentParser(
        description="Отчёт по кластерам: размер, центр, типичный представитель."
    )
    ap.add_argument("--input", required=True,
                    help="Входной Excel с данными и колонкой кластеров")
    ap.add_argument("--sheet", default="data",
                    help="Имя листа Excel (по умолчанию 'data')")
    ap.add_argument("--features", required=True,
                    help="Список признаков через запятую "
                         "(те же, что использовались при кластеризации)")
    ap.add_argument("--cluster-col", default="cluster",
                    help="Имя колонки с метками кластеров (по умолчанию 'cluster')")
    ap.add_argument("--id-col", default=None,
                    help="Колонка-идентификатор объекта (если есть). "
                         "Если не указана, будет использоваться индекс строки.")
    ap.add_argument("--impute", choices=["median", "mean"], default="median",
                    help="Способ импутации пропусков (median/mean)")
    ap.add_argument("--out-excel", default="cluster_summary.xlsx",
                    help="Файл Excel для сохранения отчёта по кластерам")

    args = ap.parse_args()

    # 1) читаем данные
    df = pd.read_excel(args.input, sheet_name=args.sheet)

    # 2) проверяем наличие нужных колонок
    feature_cols = [c.strip() for c in args.features.split(",") if c.strip()]
    for c in feature_cols:
        if c not in df.columns:
            raise ValueError(f"Столбец признака '{c}' не найден в данных.")
    if args.cluster_col not in df.columns:
        raise ValueError(f"Столбец кластеров '{args.cluster_col}' не найден в данных.")

    # 3) подготавливаем матрицу признаков
    X_raw = df[feature_cols]
    X_imp = impute_simple(X_raw, how=args.impute)

    # 4) стандартизация (для поиска "типичного представителя")
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X_imp.values)

    # 5) метки кластеров
    labels = df[args.cluster_col].values
    unique_clusters = np.sort(pd.unique(labels))

    summary_records = []
    repr_rows = []

    for cl in unique_clusters:
        mask = labels == cl
        idx_cluster = df.index[mask]

        size = int(mask.sum())
        if size == 0:
            continue

        # центр кластера в исходных единицах
        X_cl_orig = X_imp.loc[mask, :]
        center_orig = X_cl_orig.mean(axis=0)  # pandas Series

        # центр в стандартизованном пространстве
        X_cl_z = Xz[mask, :]
        center_z = X_cl_z.mean(axis=0)

        # типичный представитель: минимальное расстояние до center_z
        diffs = X_cl_z - center_z
        dists = np.linalg.norm(diffs, axis=1)
        j = int(np.argmin(dists))
        repr_index = idx_cluster[j]

        if args.id_col is not None and args.id_col in df.columns:
            repr_id = df.loc[repr_index, args.id_col]
        else:
            repr_id = repr_index

        # запись в сводную таблицу
        rec = {
            "cluster": cl,
            "size": size,
            "repr_index": repr_index,
            "repr_id": repr_id,
        }
        for col in feature_cols:
            rec[f"center_{col}"] = center_orig[col]

        summary_records.append(rec)

        # полная строка представителя
        row_repr = df.loc[repr_index].copy()
        row_repr[args.cluster_col] = cl
        repr_rows.append(row_repr)

    summary_df = pd.DataFrame(summary_records).sort_values("cluster").reset_index(drop=True)
    repr_df = pd.DataFrame(repr_rows)

    # 6) сохраняем в Excel: два листа
    with pd.ExcelWriter(args.out_excel) as writer:
        summary_df.to_excel(writer, sheet_name="cluster_summary", index=False)
        repr_df.to_excel(writer, sheet_name="representatives", index=False)

    print("Готово. Сводка по кластерам сохранена в:", args.out_excel)


if __name__ == "__main__":
    main()
