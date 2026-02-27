import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="Удалить строки с пустыми значениями из XLSX.")
    ap.add_argument("--input", default="passengers_holes.xlsx", help="Путь к входному .xlsx (processed c пропусками)")
    ap.add_argument("--output", default="passengers_dropped_rows.xlsx", help="Путь для сохранения результата .xlsx")
    ap.add_argument(
        "--subset",
        help="Список столбцов через запятую. Если не задан, проверяются все столбцы.",
        default=None,
    )
    args = ap.parse_args()

    # читаем
    df = pd.read_excel(args.input)

    before = len(df)
    subset_cols = None
    if args.subset:
        subset_cols = [c.strip() for c in args.subset.split(",") if c.strip()]

    # удаляем строки с NaN (в любом из заданных столбцов / во всех, если subset не задан)
    cleaned = df.dropna(subset=subset_cols, how="any")
    after = len(cleaned)

    # сохраняем корректно в xlsx
    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        cleaned.to_excel(writer, index=False, sheet_name="data")

    removed = before - after
    print(f"Готово: {args.output}")
    print(f"Строк было: {before}, осталось: {after}, удалено: {removed} ({removed / max(1, before) * 100:.2f}%).")


if __name__ == "__main__":
    main()
