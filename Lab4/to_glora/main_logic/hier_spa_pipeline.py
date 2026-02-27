import argparse
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster

# ---------- утилиты ----------

def select_feature_columns(df, exclude=("fio", "passport", "cluster")):
    """Берём только осмысленные числовые признаки для кластеризации."""
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols

def impute_simple(X: pd.DataFrame, how="median") -> pd.DataFrame:
    """Простая импутация: медиана (или среднее) по столбцу."""
    X = X.copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        if how == "median":
            X[c] = X[c].fillna(X[c].median())
        else:
            X[c] = X[c].fillna(X[c].mean())
    return X

def standardize(X: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X.values.astype(float))
    return Xz, scaler

def compactness_sse(X: np.ndarray, labels: np.ndarray) -> float:
    """SSE = сумма квадратов отклонений точек от центров своих кластеров."""
    sse = 0.0
    for lab in np.unique(labels):
        M = X[labels == lab]
        mu = M.mean(axis=0, keepdims=True)
        dif = M - mu
        sse += float(np.sum(dif * dif))  # квадраты евклидова расстояния
    return sse

def hclust_complete_labels(X: np.ndarray, k: int) -> np.ndarray:
    """Иерархическая кластеризация, furthest neighbor (complete linkage)."""
    Z = linkage(X, method="complete", metric="euclidean")
    labels = fcluster(Z, t=k, criterion="maxclust")
    return labels

# ---------- SPA: случайный поиск с адаптацией ----------

def spa_feature_selection(Xz: np.ndarray, feature_names: list[str],
                          K_spa: int = 5, iters: int = 120,
                          p0: float = 0.5, alpha: float = 0.12,
                          min_features: int = 3, random_state: int = 42):
    """
    SPA: поддерживаем вероятности включения p_j; на каждой итерации
    случайно формируем подмножество, кластеризуем при фиксированном K_spa,
    считаем SSE, адаптируем p.
    Возвращает: mask лучших признаков, журнал p, лучшую SSE и лучший mask.
    """
    rng = np.random.default_rng(random_state)
    m = Xz.shape[1]
    p = np.full(m, p0, dtype=float)

    best_score = np.inf
    best_mask = None

    for t in range(iters):
        S = rng.random(m) < p           # случайное подмножество
        if S.sum() < min_features:      # минимум признаков
            idx = rng.choice(m, size=min_features, replace=False)
            S[idx] = True

        Xs = Xz[:, S]
        labs = hclust_complete_labels(Xs, K_spa)
        score = compactness_sse(Xs, labs)

        # обновляем лучшее
        if score < best_score:
            best_score = score
            best_mask = S.copy()

        # адаптация вероятностей: усиливаем вошедшие, ослабляем невошедшие
        # (гребём в сторону «успешного» подмножества)
        if score <= best_score:  # greedy-усиление
            p[S] = p[S] + alpha * (1.0 - p[S])   # вверх к 1
            p[~S] = (1.0 - alpha) * p[~S]        # вниз к 0
        else:
            p[S] = (1.0 - 0.5*alpha) * p[S]      # мягкое ослабление

    # итоговый выбор: по порогу по p или best_mask (если мало)
    selected_mask = (p >= 0.6)
    if selected_mask.sum() < min_features:
        selected_mask = best_mask

    selected_features = [name for name, keep in zip(feature_names, selected_mask) if keep]
    return selected_mask, p, best_score, best_mask, selected_features

# ---------- выбор K по «локтю» (улучшение SSE) ----------

def choose_k_by_elbow(Xz_selected: np.ndarray, kmin=2, kmax=10, rel_drop_threshold=0.05):
    """
    Считаем SSE для всех k в [kmin, kmax], выбираем наименьший k,
    после которого относительное улучшение (ΔSSE / SSE_prev) < threshold.
    Возвращает: выбранный k, таблицу со значениями.
    """
    records = []
    prev_sse = None
    chosen_k = kmax
    for k in range(kmin, kmax+1):
        labels = hclust_complete_labels(Xz_selected, k)
        sse = compactness_sse(Xz_selected, labels)
        rel_drop = None
        if prev_sse is not None:
            rel_drop = (prev_sse - sse) / prev_sse
            if chosen_k == kmax and rel_drop is not None and rel_drop < rel_drop_threshold:
                chosen_k = k  # первый k, где улучшение «маленькое»
        records.append({"k": k, "SSE": sse, "rel_drop_from_prev": rel_drop})
        prev_sse = sse
    return chosen_k, pd.DataFrame(records)

# ---------- основной скрипт ----------

def main():
    ap = argparse.ArgumentParser(description="Иерархическая кластеризация с SPA-отбором признаков и выбором k по компактности.")
    ap.add_argument("--input", required=True, help="Файл Excel с данными")
    ap.add_argument("--sheet", default="Sheet1")
    ap.add_argument("--impute", default="median", choices=["median", "mean"])
    ap.add_argument("--kmin", type=int, default=2)
    ap.add_argument("--kmax", type=int, default=20)
    ap.add_argument("--k_spa", type=int, default=5, help="Фиксированный K для оценки подмножеств в SPA")
    ap.add_argument("--iters", type=int, default=120, help="Число итераций SPA")
    ap.add_argument("--out", default="clusters_output.xlsx")
    ap.add_argument("--save_report", default="clustering_report.csv")
    args = ap.parse_args()

    # 1) читаем
    df = pd.read_excel(args.input, sheet_name=args.sheet)

    # 2) выбираем осмысленные числовые признаки
    feature_names = select_feature_columns(df, exclude=("fio", "passport", "cluster"))
    X = impute_simple(df[feature_names], how=args.impute)

    # 3) стандартизация
    Xz, scaler = standardize(X)

    # 4) SPA-отбор признаков
    selected_mask, p, best_sse_subset, best_mask, selected_features = spa_feature_selection(
        Xz, feature_names, K_spa=args.k_spa, iters=args.iters, p0=0.5, alpha=0.12,
        min_features=3, random_state=42
    )

    print("Выбранные SPA-признаки:", selected_features)

    # 5) Перебор k и выбор по «локтю» (малой прибавке качества)
    Xz_sel = Xz[:, selected_mask]
    k_star, table_k = choose_k_by_elbow(Xz_sel, kmin=args.kmin, kmax=args.kmax, rel_drop_threshold=0.05)
    print("SSE по k:\n", table_k)
    print(f"Выбран k* = {k_star}")

    # 6) Финальная кластеризация
    labels = hclust_complete_labels(Xz_sel, k_star)
    sse_final = compactness_sse(Xz_sel, labels)
    print(f"Финальная компактность (SSE): {sse_final:.4f}")

    # 7) Сохраняем результаты
    out_df = df.copy()
    out_df["cluster"] = labels
    out_df.to_excel(args.out, index=False)

    report = table_k.copy()
    report["selected_feature"] = False
    report_path = args.save_report
    report.to_csv(report_path, index=False, encoding="utf-8")

    # Также сохраняем список признаков
    with open("spa_selected_features.txt", "w", encoding="utf-8") as f:
        for name in selected_features:
            f.write(name + "\n")

    print(f"Сохранено: {args.out}, {report_path}, spa_selected_features.txt")

if __name__ == "__main__":
    main()
