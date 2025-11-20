#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Единая сплайн-импутация по датасету.

Логика:
1) trip_hours = arrival_time - departure_time (в ЧАСАХ).
2) Оцениваем коэффициенты наценки по train_number: median(price / trip_hours).
3) Алгебраически восстанавливаем один из {price, arrival_time, departure_time, train_number}, если известны остальные:
   - price = trip_hours * coef[group]
   - train_number: по markup = price / trip_hours берём ближайший coef -> группа
   - arrival_time = departure_time + price / coef[group]
   - departure_time = arrival_time - price / coef[group]
   Если недостающих полей слишком много — фоллбэк на сплайн/линейку по индексу.
4) Координаты: 2D-интерполяция (RBF) dep <- f(arr) и arr <- f(dep); фоллбэк nearest.
5) coach_number: внутри маршрута (тип + пара координат) сплайн coach~price с округлением; фоллбэк мода.
6) Остальные числовые столбцы — сплайн по индексу (заглушка), для целочисленных — округление и клип по наблюдаемому диапазону.

Сохраняет XLSX и печатает счётчик NaN ДО/ПОСЛЕ.

Требования: pandas, numpy; SciPy желательно (иначе всё равно работает, но с линейными/nearest фоллбэками).
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

# Пытаемся использовать SciPy (сплайны и RBF 2D); при отсутствии — мягкие фоллбэки
try:
    from scipy.interpolate import UnivariateSpline, Rbf
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# ----------------------------- Утилиты -----------------------------
def compute_trip_hours(df: pd.DataFrame) -> pd.Series:
    """arrival_time и departure_time уже в ЧАСАХ с 2023-01-01.
    Возвращает (arrival - departure) с фильтрацией некорректных значений."""
    arr = pd.to_numeric(df.get("arrival_time"), errors="coerce")
    dep = pd.to_numeric(df.get("departure_time"), errors="coerce")
    trip = arr - dep
    trip[(~np.isfinite(trip)) | (trip <= 0)] = np.nan
    return trip


def median_markup_by_group(df: pd.DataFrame) -> dict:
    """Оцениваем коэффициенты наценки по train_number: median(price / trip_hours)."""
    if "train_number" not in df.columns:
        return {}
    trip = compute_trip_hours(df)
    price = pd.to_numeric(df.get("price"), errors="coerce")
    markup = price / trip
    mark_df = pd.DataFrame({"g": pd.to_numeric(df["train_number"], errors="coerce"),
                            "mu": pd.to_numeric(markup, errors="coerce")}).dropna()
    if mark_df.empty:
        return {}
    return mark_df.groupby("g", dropna=True)["mu"].median().to_dict()


def nearest_group_from_coef(markup: float, coef_map: dict[int, float]) -> int | None:
    """Ищем группу с коэффициентом, ближайшим к заданному markup."""
    if markup is None or not np.isfinite(markup) or not coef_map:
        return None
    items = sorted(coef_map.items())
    groups = np.array([int(g) for g, _ in items])
    coefs = np.array([float(c) for _, c in items], dtype=float)
    idx = int(np.argmin(np.abs(coefs - float(markup))))
    return int(groups[idx])


def _interp_1d_series_by_index(s: pd.Series, method: str = "spline") -> pd.Series:
    """Сплайн/линейная по индексу (0..N-1). Безопасная заглушка."""
    s = pd.to_numeric(s, errors="coerce")
    idx = np.arange(len(s), dtype=float)
    known = s.notna().values
    if known.sum() < 2:
        return s
    x = idx[known]
    y = s.values[known].astype(float)
    out = s.copy()
    if _HAVE_SCIPY and method == "spline" and len(np.unique(x)) >= 4:
        try:
            spl = UnivariateSpline(x, y, s=0.0)
            out.iloc[~known] = spl(idx[~known])
            return out
        except Exception:
            pass
    # линейная интерполяция
    out.iloc[~known] = np.interp(idx[~known], x, y, left=y[0], right=y[-1])
    return out


def _mode_int(series: pd.Series) -> float | np.nan:
    vals = pd.to_numeric(series, errors="coerce").dropna().astype(int)
    if vals.empty:
        return np.nan
    return vals.mode().iloc[0]


# -------------------- Импутация формульного блока -------------------
def impute_via_formula(df: pd.DataFrame, max_passes: int = 2) -> pd.DataFrame:
    """Алгебраически восстанавливаем price / times / train_number (группу) из формулы.
    Несколько проходов, чтобы «дотянуть» зависимые значения. Остатки — по индексу."""
    out = df.copy()
    coef_map = median_markup_by_group(out)  # group -> coef

    for _ in range(max_passes):
        changed = False

        trip = compute_trip_hours(out)
        price = pd.to_numeric(out.get("price"), errors="coerce")
        g = pd.to_numeric(out.get("train_number"), errors="coerce")

        # 1) price, если есть g и trip
        mask_price = price.isna() & g.notna() & trip.notna()
        if mask_price.any():
            c = g[mask_price].map(coef_map)
            fill = trip[mask_price] * pd.to_numeric(c, errors="coerce")
            out.loc[mask_price, "price"] = fill.values
            changed = True

        # 2) train_number по markup (price/trip)
        trip = compute_trip_hours(out)
        price = pd.to_numeric(out.get("price"), errors="coerce")
        g = pd.to_numeric(out.get("train_number"), errors="coerce")
        mask_g = g.isna() & price.notna() & trip.notna()
        if mask_g.any() and coef_map:
            mu = (price[mask_g] / trip[mask_g]).astype(float)
            cand = mu.map(lambda v: nearest_group_from_coef(v, coef_map))
            out.loc[mask_g, "train_number"] = pd.to_numeric(cand, errors="coerce").astype("Int64")
            changed = True

        # 3) arrival_time = departure_time + price/coef
        dep = pd.to_numeric(out.get("departure_time"), errors="coerce")
        arr = pd.to_numeric(out.get("arrival_time"), errors="coerce")
        price = pd.to_numeric(out.get("price"), errors="coerce")
        g = pd.to_numeric(out.get("train_number"), errors="coerce")
        mask_arr = arr.isna() & dep.notna() & price.notna() & g.notna()
        if mask_arr.any():
            c = g[mask_arr].map(coef_map)
            out.loc[mask_arr, "arrival_time"] = dep[mask_arr].values + (
                price[mask_arr].values / pd.to_numeric(c, errors="coerce").values
            )
            changed = True

        # 4) departure_time = arrival_time - price/coef
        dep = pd.to_numeric(out.get("departure_time"), errors="coerce")
        arr = pd.to_numeric(out.get("arrival_time"), errors="coerce")
        price = pd.to_numeric(out.get("price"), errors="coerce")
        g = pd.to_numeric(out.get("train_number"), errors="coerce")
        mask_dep = dep.isna() & arr.notna() & price.notna() & g.notna()
        if mask_dep.any():
            c = g[mask_dep].map(coef_map)
            out.loc[mask_dep, "departure_time"] = arr[mask_dep].values - (
                price[mask_dep].values / pd.to_numeric(c, errors="coerce").values
            )
            changed = True

        if not changed:
            break

    # Фоллбэк по индексу (если остались NaN) + жёсткие ограничения
    # 3.1 train_number — только в допустимом диапазоне
    if "train_number" in out.columns and out["train_number"].isna().any():
        s = pd.to_numeric(out["train_number"], errors="coerce")
        filled = _interp_1d_series_by_index(s, method="spline")
        known = pd.to_numeric(out["train_number"], errors="coerce")
        if np.isfinite(known).any():
            lo = int(np.nanmin(known))
            hi = int(np.nanmax(known))
        else:
            lo = hi = 1
        filled_num = pd.to_numeric(filled, errors="coerce")
        filled_round = np.rint(filled_num)
        filled_clip = np.clip(filled_round, lo, hi)
        out["train_number"] = pd.Series(filled_clip, index=out.index).astype("Int64")

    # 3.2 price >= 0
    if "price" in out.columns and out["price"].isna().any():
        s = pd.to_numeric(out["price"], errors="coerce")
        filled = _interp_1d_series_by_index(s, method="spline")
        out["price"] = np.maximum(0, pd.to_numeric(filled, errors="coerce"))

    # 3.3 times >= 0 и arrival >= departure
    for tcol in ("departure_time", "arrival_time"):
        if tcol in out.columns and out[tcol].isna().any():
            s = pd.to_numeric(out[tcol], errors="coerce")
            filled = _interp_1d_series_by_index(s, method="spline")
            out[tcol] = np.maximum(0, pd.to_numeric(filled, errors="coerce"))

    if set(("departure_time", "arrival_time")).issubset(out.columns):
        dep = pd.to_numeric(out["departure_time"], errors="coerce")
        arr = pd.to_numeric(out["arrival_time"], errors="coerce")
        both = dep.notna() & arr.notna()
        arr[both] = np.maximum(arr[both], dep[both])  # enforce arrival >= departure
        out["arrival_time"] = arr

    return out


# --------------------- 2D интерполяция координат --------------------
def _impute_coords_pair(df: pd.DataFrame, target_lon: str, target_lat: str,
                        src_lon: str, src_lat: str) -> pd.DataFrame:
    """Заполняем target_(lon/lat) как f(src_lon, src_lat) (RBF сплайн; фоллбэк nearest)."""
    out = df.copy()

    need = out[target_lon].isna() | out[target_lat].isna()
    known = (
        out[target_lon].notna() & out[target_lat].notna() &
        out[src_lon].notna() & out[src_lat].notna()
    )
    if known.sum() < 3:
        return out  # мало опорных точек

    X_src = out.loc[known, [src_lon, src_lat]].values
    y_lon = out.loc[known, target_lon].values
    y_lat = out.loc[known, target_lat].values

    mask_q = need & out[src_lon].notna() & out[src_lat].notna()
    idx_q = out.index[mask_q]
    if len(idx_q) == 0:
        return out
    X_q = out.loc[idx_q, [src_lon, src_lat]].values

    try:
        if _HAVE_SCIPY:
            rbf_lon = Rbf(X_src[:, 0], X_src[:, 1], y_lon, function="thin_plate")
            rbf_lat = Rbf(X_src[:, 0], X_src[:, 1], y_lat, function="thin_plate")
            pred_lon = rbf_lon(X_q[:, 0], X_q[:, 1])
            pred_lat = rbf_lat(X_q[:, 0], X_q[:, 1])
        else:
            raise RuntimeError("SciPy unavailable")
    except Exception:
        # nearest по евклиду
        def nearest_predict(X_train, y_train, Xtest):
            outv = np.empty(len(Xtest), dtype=float)
            for i, (a, b) in enumerate(Xtest):
                d = np.square(X_train - np.array([a, b])).sum(axis=1)
                outv[i] = y_train[np.argmin(d)]
            return outv
        pred_lon = nearest_predict(X_src, y_lon, X_q)
        pred_lat = nearest_predict(X_src, y_lat, X_q)

    # заполняем только NaN среди idx_q
    out.loc[idx_q, target_lon] = out.loc[idx_q, target_lon].fillna(pd.Series(pred_lon, index=idx_q))
    out.loc[idx_q, target_lat] = out.loc[idx_q, target_lat].fillna(pd.Series(pred_lat, index=idx_q))
    return out


def impute_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = _impute_coords_pair(out, "departure_lon", "departure_lat", "arrival_lon", "arrival_lat")
    out = _impute_coords_pair(out, "arrival_lon", "arrival_lat", "departure_lon", "departure_lat")
    return out


# ---------------------- Импутация coach_number ----------------------
GROUP_ROUTE = ["train_number", "departure_lon", "departure_lat", "arrival_lon", "arrival_lat"]

def _coach_from_price_group(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    if "coach_number" not in g.columns or "price" not in g.columns:
        return g

    known_mask = g["coach_number"].notna() & g["price"].notna()
    if known_mask.sum() >= 4:
        x = pd.to_numeric(g.loc[known_mask, "price"], errors="coerce").values
        y = pd.to_numeric(g.loc[known_mask, "coach_number"], errors="coerce").values
        try:
            if _HAVE_SCIPY and len(np.unique(x)) >= 4:
                model = UnivariateSpline(x, y, s=0.0)
                pred_idx = g["coach_number"].isna() & g["price"].notna()
                if pred_idx.any():
                    yp = model(pd.to_numeric(g.loc[pred_idx, "price"], errors="coerce").values)
                    valid = pd.to_numeric(g.loc[known_mask, "coach_number"], errors="coerce").dropna().astype(int)
                    if len(valid):
                        lo, hi = int(valid.min()), int(valid.max())
                        yp = np.clip(np.rint(yp), lo, hi)
                    g.loc[pred_idx, "coach_number"] = yp.astype(int)
                return g
        except Exception:
            pass

    # fallback: условная мода по группе
    mode_val = _mode_int(g["coach_number"])
    g.loc[g["coach_number"].isna(), "coach_number"] = mode_val
    return g


def impute_coach(df: pd.DataFrame) -> pd.DataFrame:
    if not set(GROUP_ROUTE).issubset(df.columns):
        return df
    return (
        df.groupby(GROUP_ROUTE, dropna=False, group_keys=True)
          .apply(_coach_from_price_group)
          .reset_index(drop=True)
    )


# ----------------------- Заглушка по остальным ----------------------
def fallback_all_numeric_by_index(df: pd.DataFrame, skip_cols: set[str]) -> pd.DataFrame:
    out = df.copy()
    for col in out.select_dtypes(include=[np.number]).columns:
        if col in skip_cols:
            continue
        if out[col].isna().any():
            s = pd.to_numeric(out[col], errors="coerce")
            filled = _interp_1d_series_by_index(s, method="spline")

            # целочисленные поля: округление и клип по наблюдаемому диапазону
            if col in ("fio", "passport", "train_number", "coach_number", "seat_number"):
                known = pd.to_numeric(out[col], errors="coerce")
                if np.isfinite(known).any():
                    lo = int(np.nanmin(known))
                    hi = int(np.nanmax(known))
                else:
                    lo = 0
                    hi = 0
                if col in ("coach_number", "seat_number"):
                    lo = max(1, lo)
                filled_num = pd.to_numeric(filled, errors="coerce")
                filled_round = np.rint(filled_num)
                filled_clip = np.clip(filled_round, lo, hi)
                out[col] = pd.Series(filled_clip, index=out.index).astype("Int64")
            else:
                out[col] = filled
    return out


# ------------------------------ MAIN --------------------------------
def main():
    ap = argparse.ArgumentParser(description="Сплайн-импутация всего датасета по заданной логике.")
    ap.add_argument("input", help="Входной .xlsx/.csv (processed)")
    ap.add_argument("output", help="Выходной .xlsx")
    args = ap.parse_args()

    # чтение
    if args.input.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(args.input)
    else:
        df = pd.read_csv(args.input)

    # фиксируем исходный порядок строк
    df["_row_id_"] = np.arange(len(df))
    before_nans = df.isna().sum().sum()

    # 1) Формульный блок (price/arrival_time/departure_time/train_number)
    df = impute_via_formula(df)

    # 2) Координаты (2D сплайн RBF через противоположные координаты)
    df = impute_coordinates(df)

    # 3) coach_number (по цене внутри маршрута)
    if "coach_number" in df.columns:
        df = impute_coach(df)

    # 4) Заглушка: для остальных числовых колонок — сплайн по индексу
    skip = {

    }
    df = fallback_all_numeric_by_index(df, skip_cols=skip)

    # 5) Вернуть исходный порядок и привести типы/инварианты
    df = df.sort_values("_row_id_").drop(columns=["_row_id_"])

    for icol in ("fio", "passport", "train_number", "coach_number", "seat_number"):
        if icol in df.columns:
            df[icol] = pd.to_numeric(df[icol], errors="coerce").round().astype("Int64")

    for c in ("price", "departure_time", "arrival_time"):
        if c in df.columns:
            df[c] = np.maximum(0, pd.to_numeric(df[c], errors="coerce"))

    if set(("departure_time", "arrival_time")).issubset(df.columns):
        dep = pd.to_numeric(df["departure_time"], errors="coerce")
        arr = pd.to_numeric(df["arrival_time"], errors="coerce")
        both = dep.notna() & arr.notna()
        arr[both] = np.maximum(arr[both], dep[both])
        df["arrival_time"] = arr

    # сохранение XLSX
    with pd.ExcelWriter(args.output, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="processed")

    after_nans = df.isna().sum().sum()
    print(f"Готово: {args.output}")
    print(f"Всего NaN было: {before_nans}, стало: {after_nans}")


if __name__ == "__main__":
    main()
