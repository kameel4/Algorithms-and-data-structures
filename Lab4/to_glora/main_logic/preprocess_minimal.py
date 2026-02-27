from __future__ import annotations
import argparse
import re
import runpy
import json
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# ---- уникальные скорости по типам (для однозначного ранжирования групп) ----
TYPE_SPEEDS: Dict[str, float] = {
    "Туристические": 65.0,
    "Пассажирские сезонные": 70.0,
    "Пассажирские круглогодичные": 75.0,
    "Скорые сезонные": 80.0,
    "Скорые круглогодичные": 85.0,
    "Люкс (СВ)": 90.0,
    "Ласточка": 160.0,
    "Стриж": 180.0,
    "Сапсан": 230.0,
}

# ---- trains.py → номер поезда -> тип ----
def load_number_to_type(path: str) -> Dict[str, str]:
    d = runpy.run_path(path)
    td = d.get("trains_data", d.get("trains", {}))
    mapping: Dict[str, str] = {}
    trains_list = td.get("trains", []) if isinstance(td, dict) else td
    for block in trains_list or []:
        tname = block.get("name")
        routes = block.get("routes", {})
        for r in routes.values():
            num = r.get("number")
            if num and tname:
                mapping[str(num)] = str(tname)
    return mapping

# ---- словари имён для пола (только по спискам) ----
def load_name_sets(base="used_data"):
    sets = {"male": set(), "female": set()}
    try:
        m = runpy.run_path(f"{base}/male.py").get("male_names", {})
        f = runpy.run_path(f"{base}/female.py").get("female_names", {})
        for v in m.values():
            sets["male"].update(x.lower() for x in v)
        for v in f.values():
            sets["female"].update(x.lower() for x in v)
    except Exception:
        pass
    return sets

NAME_SETS = load_name_sets()

def gender_code(fio: str):
    if pd.isna(fio) or not str(fio).strip():
        return np.nan
    parts = str(fio).lower().split()
    if len(parts) < 2:
        return np.nan
    first = parts[1]
    pat = parts[2] if len(parts) > 2 else ""
    if pat in NAME_SETS["female"] or first in NAME_SETS["female"]:
        return 2
    if pat in NAME_SETS["male"] or first in NAME_SETS["male"]:
        return 1
    return np.nan

# ---- координаты городов (из твоего списка) ----
CITY_COORDS: Dict[str, Tuple[float, float]] = {
    "Адлер": (39.9173, 43.4300),
    "Анапа": (37.3167, 44.8947),
    "Архангельск": (40.5453, 64.5399),
    "Байкал": (107.0000, 53.5000),
    "Бологое": (34.0606, 57.8859),
    "Великий Новгород": (31.2690, 58.5215),
    "Великий Устюг": (46.3063, 60.7603),
    "Владивосток": (131.8869, 43.1155),
    "Дивеево": (43.2453, 55.0416),
    "Екатеринбург": (60.5975, 56.8389),
    "Золотое кольцо": (39.8938, 57.6266),
    "Иркутск": (104.2960, 52.2860),
    "Казань": (49.1064, 55.7963),
    "Калининград": (20.5000, 54.7167),
    "Карелия": (34.3469, 61.7849),
    "Красноярск": (92.8526, 56.0106),
    "Москва": (37.6173, 55.7558),
    "Мурманск": (33.0830, 68.9700),
    "Нижний Новгород": (44.0020, 56.3269),
    "Новороссийск": (37.7666, 44.7239),
    "Омск": (73.3686, 54.9914),
    "Рязань": (39.7399, 54.6095),
    "Самара": (50.1500, 53.1959),
    "Санкт-Петербург": (30.3351, 59.9343),
    "Симферополь": (34.1083, 44.9481),
    "Сочи": (39.7277, 43.5883),
    "Тверь": (35.9077, 56.8584),
    "Уфа": (56.0407, 54.7348),
    "Хабаровск": (135.0719, 48.4814),
    "Челябинск": (61.4026, 55.1600),
    "Ярославль": (39.8938, 57.6266),
}

# ---- утилиты ----
def passport_year(passport) -> float:
    """Год выдачи по 3–4 цифрам серии (YY → 19YY/20YY)."""
    if pd.isna(passport):
        return np.nan
    digits = re.sub(r"\D", "", str(passport))
    if len(digits) < 4:
        return np.nan
    yy = int(digits[2:4])
    return 1900 + yy if yy >= 90 else 2000 + yy

def to_hours_since_2023(dt_series: pd.Series) -> pd.Series:
    """Datetime -> часы с 2023-01-01 00:00."""
    dt = pd.to_datetime(dt_series, errors="coerce")
    base = pd.Timestamp("2023-01-01 00:00:00")
    delta = (dt - base).dt.total_seconds() / 3600.0
    return pd.to_numeric(delta, errors="coerce")

def build_type_to_group(unique_speeds: Dict[str, float]) -> Dict[str, int]:
    # сортируем типы по скорости (возрастание), выдаём группы 1..n (каждому типу своя)
    ordered = sorted(unique_speeds.items(), key=lambda kv: kv[1])
    return {t: i + 1 for i, (t, _) in enumerate(ordered)}

# ---- основной пайплайн ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="passengers_dataset.xlsx", help="исходный .xlsx/.csv")
    ap.add_argument("--output", default="passengers_processed.xlsx", help="файл результата .xlsx")
    ap.add_argument("--trains", default="used_data/trains.py", help="путь к trains.py")
    ap.add_argument("--markup-json", default="type_markup.json",
                    help="Куда сохранить оценённые коэффициенты наценки по типам (JSON)")
    args = ap.parse_args()

    # читаем датасет
    df = pd.read_excel(args.input) if args.input.endswith(("xlsx", "xls")) else pd.read_csv(args.input)

    # --- временные: часы с начала 2023 (а не дни)
    dep_hours = to_hours_since_2023(df["departure_time"])
    arr_hours = to_hours_since_2023(df["arrival_time"])
    trip_hours = arr_hours - dep_hours
    trip_hours[(~np.isfinite(trip_hours)) | (trip_hours <= 0)] = np.nan

    # --- сохранённые во фрейм времена (требование)
    df["departure_time"] = dep_hours
    df["arrival_time"] = arr_hours

    # --- gender: 1/2 по спискам
    df["fio"] = df["fio"].map(gender_code).astype("Int64")

    # --- passport: ГОД выдачи
    df["passport"] = df["passport"].map(passport_year).astype("Int64")

    # --- trains.py: номер -> тип, тип -> группа (каждому типу — свой номер по скорости)
    num2type = load_number_to_type(args.trains)
    # запомним тип для оценки наценки, прежде чем заменим train_number на группу
    train_type_tmp = df["train_number"].astype(str).map(num2type)

    type2group = build_type_to_group(TYPE_SPEEDS)
    df["train_number"] = train_type_tmp.map(type2group).astype("Int64")

    # --- города -> координаты (заменяем столбцы на lon/lat)
    for col, lon, lat in [("departure_city", "departure_lon", "departure_lat"),
                          ("arrival_city", "arrival_lon", "arrival_lat")]:
        coords = df[col].map(lambda x: CITY_COORDS.get(x, (np.nan, np.nan)))
        df[lon] = [c[0] for c in coords]
        df[lat] = [c[1] for c in coords]
        df.drop(columns=[col], inplace=True)

    # --- оценка коэффициентов наценки по типам (только JSON; в датасет колонки не добавляю)
    # markup = price / trip_hours
    price = pd.to_numeric(df["price"], errors="coerce")
    markup = price / trip_hours
    # сгруппируем по исходному типу (train_type_tmp)
    mark_df = pd.DataFrame({"type": train_type_tmp, "markup": markup}).dropna()
    type_markup = (
        mark_df.groupby("type", dropna=True)["markup"]
        .median()  # робастно к выбросам
        .to_dict()
    )
    # сохраним JSON
    with open(args.markup_json, "w", encoding="utf-8") as f:
        json.dump(type_markup, f, ensure_ascii=False, indent=2)

    # --- сохранить в XLSX
    with pd.ExcelWriter(args.output, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="processed")

    print("Готово:", args.output)
    print("Сохранён словарь коэффициентов:", args.markup_json)

if __name__ == "__main__":
    main()
