import pandas as pd
import numpy as np
import os

# === НАСТРОЙКИ ===
original_path = "Lab2/passengers_dataset.xlsx"       # исходный файл
anonymized_path = "Lab2/anonymized_dataset.xlsx"     # обезличенный файл

# === ФУНКЦИЯ KLD ===
def kl_divergence(p, q):
    """Вычисление расстояния Кульбака–Лейблера между двумя распределениями."""
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    eps = 1e-10
    p = np.where(p == 0, eps, p)
    q = np.where(q == 0, eps, q)
    return np.sum(p * np.log2(p / q))

# === ЗАГРУЗКА ФАЙЛОВ ===
orig = pd.read_excel(original_path)
anon = pd.read_excel(anonymized_path)

print("✅ Файлы успешно загружены.")

# === СРАВНЕНИЕ ОБЩИХ ПОЛЕЙ ===
common_cols = set(orig.columns).intersection(set(anon.columns))
print(f"\n🔍 Общие столбцы для сравнения: {common_cols}\n")

results = {}

# --- Категориальные признаки (города)
for col in ['departure_city', 'arrival_city']:
    if col in common_cols:
        p_dist = orig[col].value_counts(normalize=True)
        q_dist = anon[col].value_counts(normalize=True)
        all_keys = set(p_dist.index).union(set(q_dist.index))
        p = np.array([p_dist.get(k, 0) for k in all_keys])
        q = np.array([q_dist.get(k, 0) for k in all_keys])
        results[col] = kl_divergence(p, q)

# --- Временные признаки
for col in ['departure_time', 'arrival_time']:
    if col in common_cols:
        p_dist = orig[col].value_counts(normalize=True)
        q_dist = anon[col].value_counts(normalize=True)
        all_keys = set(p_dist.index).union(set(q_dist.index))
        p = np.array([p_dist.get(k, 0) for k in all_keys])
        q = np.array([q_dist.get(k, 0) for k in all_keys])
        results[col] = kl_divergence(p, q)

# --- Цены (если есть)
if 'price' in orig.columns and 'price_range' in anon.columns:
    p_bins = pd.cut(orig['price'], bins=[0, 7000, np.inf], labels=['до 7000', 'больше 7000'])
    p_dist = p_bins.value_counts(normalize=True)
    q_dist = anon['price_range'].value_counts(normalize=True)
    all_keys = set(p_dist.index).union(set(q_dist.index))
    p = np.array([p_dist.get(k, 0) for k in all_keys])
    q = np.array([q_dist.get(k, 0) for k in all_keys])
    results['price_range'] = kl_divergence(p, q)

# === ВЫВОД РЕЗУЛЬТАТОВ ===
print("📊 Результаты расчёта KLD:")
for key, value in results.items():
    print(f"{key:20s}  ->  KLD = {value:.4f}")

# === ИНТЕРПРЕТАЦИЯ ===
print("\nℹ️ Чем меньше значение KLD, тем ближе распределения исходных и обезличенных данных.\n")
