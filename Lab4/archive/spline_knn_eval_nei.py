
"""
spline_knn_eval.py — версия с «скобками» (ближайший сверху и снизу) и ignore NaN цепочек.
"""
from __future__ import annotations
import argparse
from typing import List, Tuple
import numpy as np
import pandas as pd

try:
    from scipy.interpolate import CubicSpline
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
    ВЕРСИЯ С БАЛАНСОМ СТОРОН.
    Выбирает соседей так, чтобы при наличии точек слева и справа
    всегда взять ближайшие "сверху" (меньший индекс) и "снизу" (больший индекс),
    даже если между ними длинная цепочка NaN.
    Для k>2 добавляет пары симметрично: слева, справа, слева, справа...
    Если одна сторона закончилась, добираем с другой.
    Пустые строки (NaN) игнорируются, т.к. idxs — это уже индексы НЕ-NaN.
    """
    if idxs.size == 0:
        return idxs
    idxs = np.sort(idxs)
    left = idxs[idxs < target_idx]
    right = idxs[idxs > target_idx]
    chosen = []
    if left.size > 0:
        chosen.append(left[-1])
    if right.size > 0:
        chosen.append(right[0])
    li = left.size - 2
    ri = 1
    while len(chosen) < k and (li >= -left.size or ri < right.size):
        if len(chosen) < k and li >= -left.size:
            chosen.append(left[li])
            li -= 1
        if len(chosen) < k and ri < right.size:
            chosen.append(right[ri])
            ri += 1
    chosen = np.array(sorted(set(chosen)))
    return chosen


def predict_with_spline(x_obs: np.ndarray, y_obs: np.ndarray, x_target: float) -> float:
    """Сначала пробуем сплайн внутри диапазона.
    Если x_target вне диапазона наблюдений — возвращаем ближайшего соседа.
    """
    x_obs, uniq_idx = np.unique(x_obs, return_index=True)
    y_obs = y_obs[uniq_idx]
    if len(x_obs) < 1:
        return np.nan
    xmin, xmax = x_obs.min(), x_obs.max()
    if x_target <= xmin:
        return float(y_obs[x_obs.argmin()])
    if x_target >= xmax:
        return float(y_obs[x_obs.argmax()])
    if len(x_obs) == 1:
        return float(y_obs[0])
    if _HAS_SCIPY and len(x_obs) >= 2:
        try:
            cs = CubicSpline(x_obs, y_obs, bc_type='natural', extrapolate=False)
            y_pred = float(cs(x_target))
            if np.isfinite(y_pred):
                return y_pred
        except Exception:
            pass
    return float(np.interp(x_target, x_obs, y_obs))


def evaluate_for_k(full_df: pd.DataFrame, holes_df: pd.DataFrame,
                   columns: List[str], k: int) -> Tuple[float, pd.DataFrame]:
    mask = holes_df[columns].isna()
    rows, cols = np.where(mask.values)
    preds_records = []
    total_rel_err = 0.0
    index_arr = holes_df.index.to_numpy()

    for r, cpos in zip(rows, cols):
        col = columns[cpos]
        row_idx = index_arr[r]
        true_val = full_df.loc[row_idx, col]
        if pd.isna(true_val) or true_val == 0:
            continue
        non_nan_mask = holes_df[col].notna().values
        obs_idxs = index_arr[non_nan_mask]
        if obs_idxs.size < 1:
            continue
        neigh_idxs = pick_neighbors(obs_idxs, row_idx, min(k, obs_idxs.size))
        y_obs = holes_df.loc[neigh_idxs, col].to_numpy(dtype=float)
        y_pred = predict_with_spline(neigh_idxs.astype(float), y_obs, float(row_idx))
        if not np.isfinite(y_pred):
            continue
        rel_err = abs(true_val - y_pred) / abs(true_val)
        total_rel_err += float(rel_err)
        preds_records.append({
            "row_index": row_idx, "column": col,
            "true": float(true_val), "pred": float(y_pred),
            "rel_err": float(rel_err), "k_used": int(len(neigh_idxs)),
        })
    total_percent = total_rel_err * 100.0
    preds_df = pd.DataFrame.from_records(preds_records)
    return total_percent, preds_df


def choose_numeric_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--complete", required=True)
    ap.add_argument("--holes", required=True)
    ap.add_argument("--sheet", default="Sheet1")
    ap.add_argument("--max-k", type=int, default=20)
    ap.add_argument("--columns", default=None)
    ap.add_argument("--save-preds", default=None)
    ap.add_argument("--plot", default=None)
    args = ap.parse_args()

    full_df = pd.read_excel(args.complete, sheet_name=args.sheet)
    holes_df = pd.read_excel(args.holes, sheet_name=args.sheet)

    if args.columns:
        columns = [c.strip() for c in args.columns.split(",") if c.strip()]
    else:
        columns = choose_numeric_columns(holes_df)

    if not full_df.index.equals(holes_df.index):
        if len(full_df) != len(holes_df):
            raise ValueError("complete и holes должны иметь одинаковое число строк")
        holes_df = holes_df.copy(); holes_df.index = full_df.index

    ks = even_ks(args.max_k)
    results = []
    best = (None, float("inf"), None)
    for k in ks:
        total_percent, preds_df = evaluate_for_k(full_df, holes_df, columns, k)
        results.append((k, total_percent))
        if total_percent < best[1]:
            best = (k, total_percent, preds_df)

    res_df = pd.DataFrame(results, columns=["k", "total_relative_error_percent"]).sort_values("k")
    print("\\n=== Ошибка по K ===")
    print(res_df.to_string(index=False))
    print(f"\\nЛучшее K = {best[0]} с ошибкой {best[1]:.6f}%.")

    if args.save_preds and best[2] is not None:
        best[2].to_csv(args.save_preds, index=False)
        print(f"Сохранил предсказания: {args.save_preds}")

    if args.plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(res_df["k"], res_df["total_relative_error_percent"], marker="o")
        plt.xlabel("Число соседей K"); plt.ylabel("Суммарная относительная ошибка, %")
        plt.title("Ошибка vs K (скобочная выборка соседей)")
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(args.plot, dpi=160)
        print(f"График сохранён в: {args.plot}")


if __name__ == "__main__":
    main()
