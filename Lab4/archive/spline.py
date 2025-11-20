#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator  # сплайн PCHIP (shape-preserving)

# ---------- утилиты ----------

def to_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0088
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2*np.arcsin(np.sqrt(a))
    return R * c

def add_distance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["distance_km"] = np.nan
    mask = df[["departure_lon","departure_lat","arrival_lon","arrival_lat"]].notna().all(axis=1)
    if mask.any():
        df.loc[mask, "distance_km"] = haversine_km(
            df.loc[mask, "departure_lon"],
            df.loc[mask, "departure_lat"],
            df.loc[mask, "arrival_lon"],
            df.loc[mask, "arrival_lat"],
        )
    return df

def pchip_fit_predict(x_train, y_train, x_pred):
    """
    Строит shape-preserving сплайн (PCHIP) на обучающих точках (x_train, y_train)
    и возвращает предсказания на x_pred. 
    - Удаляет NaN
    - Сортирует по x
    - Склеивает дубликаты x (берёт медиану y по одинаковым x)
    - Требует >= 2 уникальных x
    - Без экстраполяции (вне диапазона -> NaN)
    - Фоллбэк: линейная интерполяция в пределах диапазона, если PCHIP не удаётся
    """
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    x_pred  = np.asarray(x_pred,  dtype=float)

    # 1) убрать нечисловые
    m = np.isfinite(x_train) & np.isfinite(y_train)
    x, y = x_train[m], y_train[m]
    if x.size == 0:
        return None

    # 2) сортировка по x
    order = np.argsort(x)
    x, y = x[order], y[order]

    # 3) агрегация дубликатов x -> медиана y
    #    (можно mean, но median робастнее)
    uniq_x, idx_start = np.unique(x, return_index=True)
    if uniq_x.size < x.size:
        # для каждого uniq_x берём медиану соответствующих y
        y_medians = []
        # найдём границы блоков одинаковых x
        # индексы начала каждого блока в idx_start, конец — следующий старт или конец массива
        idx_bounds = list(idx_start) + [len(x)]
        for i in range(len(idx_start)):
            a, b = idx_bounds[i], idx_bounds[i+1]
            y_medians.append(np.median(y[a:b]))
        x, y = uniq_x, np.asarray(y_medians)

    # 4) нужно >= 2 уникальных x
    if x.size < 2:
        return None

    # 5) PCHIP без экстраполяции
    try:
        f = PchipInterpolator(x, y, extrapolate=False)
        yhat = f(x_pred)
    except Exception as e:
        print(f"PCHIP failed ({e}); falling back to linear interpolation.")
        # линейная интерполяция в пределах [min(x), max(x)]
        yhat = np.full_like(x_pred, np.nan, dtype=float)
        xmin, xmax = x.min(), x.max()
        inside = (x_pred >= xmin) & (x_pred <= xmax) & np.isfinite(x_pred)
        if inside.any():
            yhat[inside] = np.interp(x_pred[inside], x, y)
    return yhat

def clip_by_known_quantiles(s: pd.Series, low=0.01, high=0.99) -> pd.Series:
    known = s.dropna()
    if len(known) == 0:
        return s
    lo, hi = known.quantile([low, high]).values
    return s.clip(lo, hi)

def ensure_nonnegative(s: pd.Series) -> pd.Series:
    return s.where(s >= 0, s.dropna().median() if s.dropna().size else 0.0)

def round_int_if_present(s: pd.Series) -> pd.Series:
    out = s.copy()
    m = out.notna()
    out.loc[m] = np.round(out.loc[m])
    return out.astype("Int64")

# ---------- основной пайплайн ----------

def impute_with_splines_distance_based(df_in: pd.DataFrame,
                                       group_for_price=("coach_number",),
                                       group_for_duration=("train_number",)) -> pd.DataFrame:
    """
    Восстановление ТОЛЬКО сплайном, но по смысловым зависимостям:
      duration = f(distance_km), price = g(distance_km)
    Сплайны строятся на известных точках (внутри групп), интерполяция — без экстраполяции.
    """
    df = to_numeric_df(df_in)
    df = add_distance(df)

    # --- 1) ДЛИТЕЛЬНОСТЬ ---
    if "departure_time" in df.columns and "arrival_time" in df.columns:
        df["duration"] = df["arrival_time"] - df["departure_time"]
    else:
        df["duration"] = np.nan

    # сперва пробуем строить сплайны внутри групп (например, по поездy)
    def fill_duration_in_group(g: pd.DataFrame) -> pd.DataFrame:
        # тренируем на строках, где distance и duration известны
        train_mask = g["distance_km"].notna() & g["duration"].notna()
        pred_mask  = g["distance_km"].notna() & g["duration"].isna()
        if train_mask.sum() >= 2 and pred_mask.any():
            yhat = pchip_fit_predict(g.loc[train_mask, "distance_km"],
                                     g.loc[train_mask, "duration"],
                                     g.loc[pred_mask,  "distance_km"].values)
            if yhat is not None:
                g.loc[pred_mask, "duration"] = yhat
        return g

    if len(group_for_duration) > 0 and all(c in df.columns for c in group_for_duration):
        df = df.groupby(list(group_for_duration), dropna=False, observed=True, group_keys=False).apply(fill_duration_in_group)
    else:
        df = fill_duration_in_group(df)

    # глобальный бэкап (если в группе не хватило точек)
    still = df["distance_km"].notna() & df["duration"].isna()
    if still.any():
        train_mask = df["distance_km"].notna() & df["duration"].notna()
        if train_mask.sum() >= 2:
            yhat = pchip_fit_predict(df.loc[train_mask, "distance_km"],
                                     df.loc[train_mask, "duration"],
                                     df.loc[still,  "distance_km"].values)
            if yhat is not None:
                df.loc[still, "duration"] = yhat

    # приводим длительность к разумным рамкам по СВОИМ же квантилям
    df["duration"] = clip_by_known_quantiles(df["duration"], 0.01, 0.99)
    df["duration"] = ensure_nonnegative(df["duration"])

    # восстанавливаем arrival/departure, если известно одно из времён
    dep_na, arr_na = df["departure_time"].isna(), df["arrival_time"].isna()
    dep_ok_arr_na = (~dep_na) & arr_na
    arr_ok_dep_na = (~arr_na) & dep_na
    df.loc[dep_ok_arr_na, "arrival_time"]  = df.loc[dep_ok_arr_na, "departure_time"] + df.loc[dep_ok_arr_na, "duration"]
    df.loc[arr_ok_dep_na, "departure_time"] = df.loc[arr_ok_dep_na, "arrival_time"]  - df.loc[arr_ok_dep_na, "duration"]

    # гарантия: arrival >= departure (минимальная длительность = 0)
    both = df["arrival_time"].notna() & df["departure_time"].notna()
    bad  = both & (df["arrival_time"] < df["departure_time"])
    if bad.any():
        df.loc[bad, "arrival_time"] = df.loc[bad, "departure_time"] + 0.0

    # --- 2) ЦЕНА ---
    def fill_price_in_group(g: pd.DataFrame) -> pd.DataFrame:
        train_mask = g["distance_km"].notna() & g["price"].notna()
        pred_mask  = g["distance_km"].notna() & g["price"].isna()
        if train_mask.sum() >= 2 and pred_mask.any():
            yhat = pchip_fit_predict(g.loc[train_mask, "distance_km"],
                                     g.loc[train_mask, "price"],
                                     g.loc[pred_mask,  "distance_km"].values)
            if yhat is not None:
                g.loc[pred_mask, "price"] = yhat
        return g

    if len(group_for_price) > 0 and all(c in df.columns for c in group_for_price):
        df = df.groupby(list(group_for_price), dropna=False, observed=True, group_keys=False).apply(fill_price_in_group)
    else:
        df = fill_price_in_group(df)

    # глобальный бэкап
    still = df["distance_km"].notna() & df["price"].isna()
    if still.any():
        train_mask = df["distance_km"].notna() & df["price"].notna()
        if train_mask.sum() >= 2:
            yhat = pchip_fit_predict(df.loc[train_mask, "distance_km"],
                                     df.loc[train_mask, "price"],
                                     df.loc[still,  "distance_km"].values)
            if yhat is not None:
                df.loc[still, "price"] = yhat

    # цена — неотрицательная, клип по собственным квантилям
    df["price"] = ensure_nonnegative(df["price"])
    df["price"] = clip_by_known_quantiles(df["price"], 0.01, 0.99)

    # --- 3) целочисленные поля: только приведение типа (НЕ «придумываем» новые значения сплайном)
    for c in ["train_number","coach_number","seat_number","fio","passport"]:
        if c in df.columns:
            df[c] = round_int_if_present(df[c])

    return df

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Spline-based imputation with meaningful x (distance).")
    ap.add_argument("--input", required=True, help="Excel с пропусками (напр., passengers_holes10.xlsx)")
    ap.add_argument("--sheet", default="Sheet1")
    ap.add_argument("--out", default="passengers_spline_distance.xlsx")
    ap.add_argument("--group_price", default="coach_number", help="Группировка для цены (через запятую, пусто — без групп)")
    ap.add_argument("--group_duration", default="train_number", help="Группировка для длительности (через запятую, пусто — без групп)")
    args = ap.parse_args()

    df = pd.read_excel(args.input, sheet_name=args.sheet)
    groups_price = tuple([g.strip() for g in args.group_price.split(",") if g.strip()]) if args.group_price else tuple()
    groups_dur   = tuple([g.strip() for g in args.group_duration.split(",") if g.strip()]) if args.group_duration else tuple()

    out = impute_with_splines_distance_based(df, group_for_price=groups_price, group_for_duration=groups_dur)
    out.to_excel(args.out, index=False)
    print(f"Saved to: {args.out}")

if __name__ == "__main__":
    main()
