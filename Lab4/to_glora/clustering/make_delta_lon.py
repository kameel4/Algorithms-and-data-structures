#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import numpy as np


def main():
    ap = argparse.ArgumentParser(
        description="Заменить departure_lon и arrival_lon на delta_lon = |departure_lon - arrival_lon|"
    )
    ap.add_argument("--input", required=True,
                    help="Входной Excel-файл с датасетом")
    ap.add_argument("--sheet", default=0,
                    help="Имя листа Excel (или номер, по умолчанию 0)")
    ap.add_argument("--departure-col", default="departure_lon",
                    help="Имя столбца с долготой отправления (по умолчанию 'departure_lon')")
    ap.add_argument("--arrival-col", default="arrival_lon",
                    help="Имя столбца с долготой прибытия (по умолчанию 'arrival_lon')")
    ap.add_argument("--delta-col", default="delta_lon",
                    help="Имя нового столбца с |departure_lon - arrival_lon| (по умолчанию 'delta_lon')")
    ap.add_argument("--out-excel", default="with_delta_lon.xlsx",
                    help="Имя выходного Excel-файла (по умолчанию 'with_delta_lon.xlsx')")

    args = ap.parse_args()

    # читаем датасет
    df = pd.read_excel(args.input, sheet_name=args.sheet)

    if args.departure_col not in df.columns:
        raise ValueError(f"Столбец '{args.departure_col}' не найден в данных")
    if args.arrival_col not in df.columns:
        raise ValueError(f"Столбец '{args.arrival_col}' не найден в данных")

    # приводим к числам
    dep = pd.to_numeric(df[args.departure_col], errors="coerce")
    arr = pd.to_numeric(df[args.arrival_col], errors="coerce")

    # считаем абсолютную разность
    df[args.delta_col] = (dep - arr).abs()

    # удаляем старые два столбца
    df = df.drop(columns=[args.departure_col, args.arrival_col])

    # сохраняем
    df.to_excel(args.out_excel, index=False)
    print(f"Готово. Результат сохранён в '{args.out_excel}'")


if __name__ == "__main__":
    main()
