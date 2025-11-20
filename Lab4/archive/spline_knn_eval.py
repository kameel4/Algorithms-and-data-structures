
"""
spline_knn_eval.py

Восстанавливает пропуски в "дырявом" датасете сплайн-интерполяцией по столбцам
(по атрибуту) с разным числом ближайших соседей K = 2, 4, ..., 2n и считает
суммарную относительную погрешность (в процентах) по формуле из задания:

Δ_M(K) = sum_i |a_i - â_i| / |a_i| * 100%

Где a_i — истинное (удалённое) значение из ПОЛНОГО датасета,
â_i — предсказанное.

Запуск:
    python spline_knn_eval.py --complete complete.xlsx --holes holes.xlsx --max-k 20

Опции:
    --complete   Путь к файлу с ПОЛНЫМ датасетом (без пропусков) — источник истинных значений.
    --holes      Путь к файлу с "дырявым" датасетом (там, где NaN будем восстанавливать).
    --sheet      Имя листа в Excel (по умолчанию Sheet1).
    --max-k      Максимальное чётное K (включительно). Будут перебраны 2,4,...,max_k.
    --columns    Список столбцов для интерполяции (через запятую). По умолчанию — все числовые.
    --save-preds Путь для сохранения предсказаний (CSV) для лучшего K. Необязательно.
    --plot       Сохранить график ошибки в PNG по пути (необязательно).

Зависимости: pandas, numpy, scipy, matplotlib (только если используете --plot).
"""
from __future__ import annotations
import argparse
import math
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.interpolate import UnivariateSpline, CubicSpline
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def even_ks(max_k: int) -> List[int]:
    max_k = int(max_k)
    if max_k < 2:
        raise ValueError("max_k должно быть >= 2")
    return [k for k in range(2, max_k + 1) if k % 2 == 0]


def pick_neighbors(idxs: np.ndarray, target_idx: int, k: int) -> np.ndarray:
    """
    Берёт k ближайших по индексу строк к target_idx из отсортированного массива idxs.
    """
    if idxs.size == 0:
        return idxs
    # расстояния по индексу
    d = np.abs(idxs - target_idx)
    order = np.argsort(d, kind="mergesort")
    return np.sort(idxs[order[:k]])


def predict_with_spline(x_obs: np.ndarray, y_obs: np.ndarray, x_target: float) -> float:
    """
    Предсказывает значение в x_target по наблюдениям (x_obs, y_obs).
    Использует сплайн третьего порядка, если точек достаточно; иначе — линейную интерполяцию.
    Ставит s=0 (интерполирующий сплайн). На вырожденных наборах — fallback к линейной.
    """
    # Удаляем дубликаты x (берём среднее по дублям)
    x_obs, uniq_idx = np.unique(x_obs, return_index=True)
    y_obs = y_obs[uniq_idx]

    # Нельзя интерполировать без >= 2 точек
    if len(x_obs) < 2:
        return np.nan

    k = min(3, len(x_obs) - 1)  # степень сплайна
    if _HAS_SCIPY and len(x_obs) >= 2:
        try:
            if k >= 2:
                # CubicSpline/QuadraticSpline элегантен и устойчив на небольших наборах
                cs = CubicSpline(x_obs, y_obs, bc_type='natural', extrapolate=False)
                y_pred = float(cs(x_target))
                if not np.isfinite(y_pred):
                    raise ValueError("CubicSpline produced non-finite")
                return y_pred
            else:
                # линейка
                return np.interp(x_target, x_obs, y_obs)
        except Exception:
            # Fallback: интерполируем линейно
            return float(np.interp(x_target, x_obs, y_obs))
    else:
        # Без SciPy — линейка
        return float(np.interp(x_target, x_obs, y_obs))


def evaluate_for_k(
    full_df: pd.DataFrame,
    holes_df: pd.DataFrame,
    columns: List[str],
    k: int,
) -> Tuple[float, pd.DataFrame]:
    """
    Восстанавливает пропуски в holes_df для заданных columns,
    используя k ближайших соседей по индексу в КАЖДОМ столбце.
    Возвращает (суммарная относительная ошибка %, таблица предсказаний).
    """
    # Где пропуски?
    mask = holes_df[columns].isna()
    rows, cols = np.where(mask.values)

    preds_records = []
    total_rel_err = 0.0

    index_arr = holes_df.index.to_numpy()

    for r, cpos in zip(rows, cols):
        col = columns[cpos]
        row_idx = index_arr[r]

        # Истинное значение (из полного датасета)
        true_val = full_df.loc[row_idx, col]
        if pd.isna(true_val) or true_val == 0:
            # Пропускаем некорректные или нулевые a_i для относительной ошибки
            continue

        # Наблюдаемые точки в этом столбце (где нет NaN)
        non_nan_mask = holes_df[col].notna().values
        obs_idxs = index_arr[non_nan_mask]
        obs_vals = holes_df.loc[obs_idxs, col].to_numpy(dtype=float)

        if obs_idxs.size < 2:
            continue

        # Берём k ближайших
        neigh_idxs = pick_neighbors(obs_idxs, row_idx, min(k, obs_idxs.size))
        neigh_vals = holes_df.loc[neigh_idxs, col].to_numpy(dtype=float)

        # Предсказываем сплайном (ось — индекс строки)
        y_pred = predict_with_spline(neigh_idxs.astype(float), neigh_vals.astype(float), float(row_idx))

        if not np.isfinite(y_pred):
            continue

        rel_err = abs(true_val - y_pred) / abs(true_val)
        total_rel_err += float(rel_err)

        preds_records.append({
            "row_index": row_idx,
            "column": col,
            "true": float(true_val),
            "pred": float(y_pred),
            "rel_err": float(rel_err),
            "k_used": int(len(neigh_idxs)),
        })

    total_percent = total_rel_err * 100.0
    preds_df = pd.DataFrame.from_records(preds_records)
    return total_percent, preds_df


def choose_numeric_columns(df: pd.DataFrame) -> List[str]:
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return num_cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--complete", required=True, help="Путь к ПОЛНОМУ датасету (истина)")
    ap.add_argument("--holes", required=True, help="Путь к датасету с пропусками")
    ap.add_argument("--sheet", default="Sheet1", help="Лист Excel")
    ap.add_argument("--max-k", type=int, default=20, help="Максимальное чётное K")
    ap.add_argument("--columns", default=None, help="Явный список столбцов через запятую")
    ap.add_argument("--save-preds", default=None, help="Сохранить предсказания для лучшего K в CSV")
    ap.add_argument("--plot", default=None, help="Сохранить график ошибки (PNG)")
    args = ap.parse_args()

    full_df = pd.read_excel(args.complete, sheet_name=args.sheet)
    holes_df = pd.read_excel(args.holes, sheet_name=args.sheet)

    if args.columns:
        columns = [c.strip() for c in args.columns.split(",") if c.strip()]
    else:
        columns = choose_numeric_columns(holes_df)

    # Убедимся, что индексы сопоставимы (используем позиционные индексы)
    if not full_df.index.equals(holes_df.index):
        # Попробуем выровнять по порядку строк
        if len(full_df) != len(holes_df):
            raise ValueError("complete и holes должны иметь одинаковое число строк или совпадающие индексы.")
        holes_df = holes_df.copy()
        holes_df.index = full_df.index

    ks = even_ks(args.max_k)
    results: List[Tuple[int, float]] = []
    best = (None, float("inf"), None)  # (k, err, preds_df)

    for k in ks:
        total_percent, preds_df = evaluate_for_k(full_df, holes_df, columns, k)
        results.append((k, total_percent))
        if total_percent < best[1]:
            best = (k, total_percent, preds_df)

    # Вывод
    res_df = pd.DataFrame(results, columns=["k", "total_relative_error_percent"]).sort_values("k")
    print("\n=== Ошибка по K ===")
    print(res_df.to_string(index=False))

    print(f"\nЛучшее K = {best[0]} с ошибкой {best[1]:.6f}% (чем меньше, тем лучше).")

    if args.save_preds and best[2] is not None:
        best[2].to_csv(args.save_preds, index=False)
        print(f"Сохранил предсказания для лучшего K в: {args.save_preds}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(res_df["k"], res_df["total_relative_error_percent"], marker="o")
            plt.xlabel("Число соседей K")
            plt.ylabel("Суммарная относительная ошибка, %")
            plt.title("Сплайн-интерполяция по столбцам: ошибка vs K")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(args.plot, dpi=160)
            print(f"График сохранён в: {args.plot}")
        except Exception as e:
            print(f"Не удалось построить график: {e}")


if __name__ == "__main__":
    main()
