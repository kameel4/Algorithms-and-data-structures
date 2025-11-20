import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- настройки ---
INPUT_XLSX = "small/passengers_dataset_small_processed.xlsx"   # оставил как у тебя
OUTDIR = "small/original_stats"
MAX_BARS = 100  # максимум столбиков на категориальной диаграмме

os.makedirs(OUTDIR, exist_ok=True)

def _safe_name(s: str) -> str:
    return re.sub(r'[^a-zA-Z0-9А-Яа-я._-]+', '_', str(s))

def _plot_numeric(series: pd.Series, colname: str):
    plt.figure()
    # как было: 100 бинов; если хочешь умнее: bins=min(100, max(10, int(np.sqrt(series.nunique()))))
    series.hist(bins=100)
    plt.title(f"Распределение: {colname}")
    plt.xlabel(colname)
    plt.ylabel("Частота")
    plt.grid(True, linestyle=":", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/distribution_{_safe_name(colname)}.png", dpi=150)
    plt.close()

def _plot_categorical(series: pd.Series, colname: str, max_bars: int = MAX_BARS):
    # считаем частоты
    counts = series.astype("object").value_counts(dropna=False)

    # если категорий > max_bars — оставляем ровно max_bars (топ по частоте)
    # иначе берём все категории
    if len(counts) > max_bars:
        data = counts.iloc[:max_bars]
        n_bars = max_bars
    else:
        data = counts
        n_bars = len(counts)

    # --- адаптивная ширина фигуры, чтобы столбики были "толще", когда категорий мало ---
    # при 100 столбиках картинка широкая, при малом числе — сильно сужаем полотно
    fig_width_at_max = 20  # ширина при 100 столбиках
    fig_width_min = 6      # минимальная ширина при очень малом числе столбиков
    if n_bars <= 1:
        fig_w = fig_width_min
    else:
        frac = n_bars / float(max_bars)  # 1.0 при n_bars == max_bars
        fig_w = fig_width_min + frac * (fig_width_at_max - fig_width_min)

    fig_h = 6 if n_bars <= 30 else 8

    # рисуем
    plt.figure(figsize=(fig_w, fig_h))
    x = np.arange(n_bars)
    bar_width = 0.9 if n_bars < 10 else 0.8  # чуть толще при совсем малом числе столбиков
    plt.bar(x, data.values, width=bar_width)
    plt.xticks(x, [str(idx) for idx in data.index], rotation=45, ha="right")

    plt.title(f"Распределение категорий: {colname}")
    plt.xlabel(colname)
    plt.ylabel("Частота")
    plt.grid(axis="y", linestyle=":", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/distribution_{_safe_name(colname)}.png", dpi=150)
    plt.close()

# --- читаем твой итоговый файл ---
df = pd.read_excel(INPUT_XLSX)

# --- метрики по числовым (как у тебя) ---
stats = pd.DataFrame({
    "mean": df.mean(numeric_only=True),
    "median": df.median(numeric_only=True),
    "mode": df.mode(numeric_only=True).iloc[0]
})
stats.to_excel(f"{OUTDIR}/dataset_statistics.xlsx")

# --- распределения для каждого столбца по отдельности ---
for col in df.columns:
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        _plot_numeric(s, col)
    else:
        _plot_categorical(s, col, max_bars=MAX_BARS)

print("Готово: сохранены метрики и графики распределений в", OUTDIR)
