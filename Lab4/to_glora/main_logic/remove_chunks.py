#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import pandas as pd
import random
from math import sqrt

def remove_chunks(df: pd.DataFrame, percent: float) -> pd.DataFrame:
    """Удаляет percent% данных из датафрейма кусками (матрицами)."""
    df = df.copy()
    total_values = df.size
    n_remove = int(total_values * percent / 100)

    # размеры возможных блоков (матриц)
    patterns = [(2, 2), (3, 3), (4, 2), (2, 3)]
    removed = 0
    rows, cols = df.shape

    while removed < n_remove:
        # случайный шаблон (размер блока)
        h, w = random.choice(patterns)
        # случайная позиция в пределах таблицы
        r = random.randint(0, max(0, rows - h))
        c = random.randint(0, max(0, cols - w))
        # применяем вырезание
        df.iloc[r:r + h, c:c + w] = np.nan
        removed += h * w

    print(f"Удалено примерно {removed} ячеек (~{removed / total_values * 100:.1f}%)")
    return df


def main():
    ap = argparse.ArgumentParser(description="Удаление данных кусками из processed_dataset")
    ap.add_argument("--input", help="путь к исходному файлу XLSX", default="passengers_processed.xlsx")
    ap.add_argument("--output", help="путь для сохранения результата XLSX", default=f"passengers_holes15.xlsx")
    ap.add_argument("--percent", type=float, default=15, help="процент удаляемых данных (3,5,10,20,30 и т.д.)")
    args = ap.parse_args()

    df = pd.read_excel(args.input)
    df_removed = remove_chunks(df, args.percent)
    df_removed.to_excel(args.output, index=False)
    print("Сохранено:", args.output)


if __name__ == "__main__":
    main()
