#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Рекурсивно обходит папку, где лежит сам скрипт, находит все .xlsx и
во всех листах, где есть departure_lon и arrival_lon,
заменяет их на delta_lon = |departure_lon - arrival_lon|.
"""

import os
import argparse
import pandas as pd


def process_excel_file(
    path,
    departure_col="departure_lon",
    arrival_col="arrival_lon",
    delta_col="delta_lon",
):
    """
    Обрабатывает один .xlsx-файл.
    Возвращает True, если файл был изменён.
    """
    try:
        sheets = pd.read_excel(path, sheet_name=None)
    except Exception as e:
        print(f"[SKIP] Не удалось прочитать '{path}': {e}")
        return False

    changed = False

    for sheet_name, df in sheets.items():
        if departure_col in df.columns and arrival_col in df.columns:
            print(f"  Лист '{sheet_name}': заменяю {departure_col}, {arrival_col} -> {delta_col}")
            # приводим к числам
            dep = pd.to_numeric(df[departure_col], errors="coerce")
            arr = pd.to_numeric(df[arrival_col], errors="coerce")

            # считаем |dep - arr|
            df[delta_col] = (dep - arr).abs()

            # удаляем исходные столбцы
            df.drop(columns=[departure_col, arrival_col], inplace=True)

            sheets[sheet_name] = df
            changed = True

    if changed:
        # перезаписываем файл (можно сделать backup при желании)
        try:
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                for sheet_name, df in sheets.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"[OK] Изменён файл: {path}")
        except Exception as e:
            print(f"[ERR] Не удалось записать '{path}': {e}")
            return False
    else:
        print(f"[NO CHANGE] В '{path}' нет нужных столбцов.")

    return changed


def main():
    ap = argparse.ArgumentParser(
        description="Рекурсивно заменить departure_lon/arrival_lon на delta_lon во всех .xlsx "
                    "в папке, где лежит скрипт, и во всех вложенных."
    )
    ap.add_argument(
        "--departure-col",
        default="departure_lon",
        help="Имя столбца с долготой отправления (по умолчанию 'departure_lon')",
    )
    ap.add_argument(
        "--arrival-col",
        default="arrival_lon",
        help="Имя столбца с долготой прибытия (по умолчанию 'arrival_lon')",
    )
    ap.add_argument(
        "--delta-col",
        default="delta_lon",
        help="Имя нового столбца с |departure_lon - arrival_lon| (по умолчанию 'delta_lon')",
    )

    args = ap.parse_args()

    # корень = папка, где лежит этот .py
    root = os.path.dirname(os.path.abspath(__file__))

    print(f"Старт обхода из папки скрипта: {root}")

    for dirpath, dirnames, filenames in os.walk(root):
        for fname in filenames:
            # пропускаем временные файлы Excel (~$...)
            if not fname.lower().endswith(".xlsx") or fname.startswith("~$"):
                continue

            full_path = os.path.join(dirpath, fname)
            print(f"\nФайл: {full_path}")
            process_excel_file(
                full_path,
                departure_col=args.departure_col,
                arrival_col=args.arrival_col,
                delta_col=args.delta_col,
            )


if __name__ == "__main__":
    main()
