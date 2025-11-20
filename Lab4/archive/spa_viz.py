#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster

# ---------- utils ----------

def select_numeric_features(df, exclude=("fio", "passport", "cluster")):
    feats = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            feats.append(c)
    return feats

def simple_impute(X: pd.DataFrame, how="median") -> pd.DataFrame:
    X = X.copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].fillna(X[c].median() if how == "median" else X[c].mean())
    return X

def standardize(X: pd.DataFrame):
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X.values.astype(float))
    return Xz, scaler

def hclust_complete_labels(X: np.ndarray, k: int) -> np.ndarray:
    # Complete linkage (furthest neighbor), euclidean metric
    Z = linkage(X, method="complete", metric="euclidean")
    labels = fcluster(Z, t=k, criterion="maxclust")
    return labels

def compactness_sse(X: np.ndarray, labels: np.ndarray) -> float:
    sse = 0.0
    for lab in np.unique(labels):
        M = X[labels == lab]
        if M.shape[0] == 0:
            continue
        mu = M.mean(axis=0, keepdims=True)
        dif = M - mu
        sse += float(np.sum(dif * dif))  # sum of squared Euclidean distances
    return sse

# ---------- SPA core (with probability trace) ----------

def run_spa(Xz: np.ndarray, feature_names: list[str],
            K_spa=5, iters=120, p0=0.5, alpha=0.12,
            min_features=3, random_state=42):
    rng = np.random.default_rng(random_state)
    m = Xz.shape[1]
    p = np.full(m, p0, dtype=float)          # inclusion probabilities
    p_hist = np.zeros((iters, m), dtype=float)
    scores = []

    best_score = np.inf
    best_mask = None

    for t in range(iters):
        S = rng.random(m) < p                 # sample subset by Bernoulli(p)
        if S.sum() < min_features:
            idx = rng.choice(m, size=min_features, replace=False)
            S[idx] = True

        Xs = Xz[:, S]
        labs = hclust_complete_labels(Xs, K_spa)
        score = compactness_sse(Xs, labs)
        scores.append(score)

        improved = score < best_score
        if improved:
            best_score = score
            best_mask = S.copy()

        # Adapt probabilities (greedy reinforcement)
        if improved:
            p[S]  = p[S] + alpha * (1.0 - p[S])  # push up for used features
            p[~S] = (1.0 - alpha) * p[~S]        # push down for unused
        else:
            p[S]  = (1.0 - 0.5*alpha) * p[S]     # mild decay

        p_hist[t, :] = p

    # Final selection: threshold or best observed
    selected_mask = (p >= 0.6)
    if selected_mask.sum() < min_features and best_mask is not None:
        selected_mask = best_mask

    selected_features = [nm for nm, keep in zip(feature_names, selected_mask) if keep]

    return {
        "p_final": p,
        "p_history": p_hist,
        "scores": np.array(scores),
        "selected_mask": selected_mask,
        "selected_features": selected_features,
        "best_score": best_score,
    }

# ---------- plotting ----------

def plot_p_history(feature_names, p_hist, out_png=None):
    iters = p_hist.shape[0]
    xs = np.arange(1, iters+1)
    plt.figure(figsize=(10, 6))
    for j, name in enumerate(feature_names):
        plt.plot(xs, p_hist[:, j], label=name)
    plt.xlabel("Итерация SPA")
    plt.ylabel("Вероятность включения p_j")
    plt.title("Эволюция вероятностей включения признаков (SPA)")
    plt.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=150)
        plt.close()
    else:
        plt.show()

def plot_p_final(feature_names, p_final, out_png=None):
    ser = pd.Series(p_final, index=feature_names).sort_values(ascending=False)
    plt.figure(figsize=(10, 4))
    ser.plot(kind="bar")
    plt.xlabel("Признак")
    plt.ylabel("Итоговая вероятность включения p_j")
    plt.title("Итоговые p_j после SPA")
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=150)
        plt.close()
    else:
        plt.show()

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Визуализация эволюции SPA (вероятности включения признаков).")
    ap.add_argument("--input", required=True, help="Excel-файл с данными")
    ap.add_argument("--sheet", default="Sheet1")
    ap.add_argument("--exclude", default="fio,passport,cluster",
                    help="Через запятую: столбцы, которые не использовать в кластеризации")
    ap.add_argument("--impute", default="median", choices=["median", "mean"])
    ap.add_argument("--iters", type=int, default=100, help="Число итераций SPA")
    ap.add_argument("--k_spa", type=int, default=5, help="Число кластеров в SPA для оценки подмножеств")
    ap.add_argument("--alpha", type=float, default=0.12, help="Скорость адаптации вероятностей")
    ap.add_argument("--min_features", type=int, default=3, help="Мин. число признаков в подмножестве")
    ap.add_argument("--outdir", default="spa_viz_out")
    args = ap.parse_args()

    df = pd.read_excel(args.input)
    exclude = [x.strip() for x in args.exclude.split(",") if x.strip()]
    feats = select_numeric_features(df, exclude=tuple(exclude))
    X = simple_impute(df[feats], how=args.impute)
    Xz, _ = standardize(X)

    res = run_spa(Xz, feats,
                  K_spa=args.k_spa, iters=args.iters,
                  p0=0.5, alpha=args.alpha,
                  min_features=args.min_features, random_state=42)

    # save CSVs
    import os
    os.makedirs(args.outdir, exist_ok=True)
    pd.DataFrame(res["p_history"], columns=feats).assign(iter=np.arange(1, args.iters+1)) \
        .to_csv(f"{args.outdir}/spa_prob_history.csv", index=False)
    pd.DataFrame({"feature": feats, "p_final": res["p_final"],
                  "selected": res["selected_mask"]}).to_csv(f"{args.outdir}/spa_final_probs.csv", index=False)

    # plots
    plot_p_history(feats, res["p_history"], out_png=f"{args.outdir}/spa_p_history.png")
    plot_p_final(feats, res["p_final"], out_png=f"{args.outdir}/spa_p_final.png")

    # print summary
    print("Выбранные признаки:", ", ".join(res["selected_features"]))
    print(f"Лучшая компактность (SSE) на SPA: {res['best_score']:.4f}")
    print(f"Сохранено в папку: {args.outdir}")

if __name__ == "__main__":
    main()
