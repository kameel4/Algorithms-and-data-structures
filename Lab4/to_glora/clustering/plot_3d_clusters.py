#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import plotly.express as px


def main():
    ap = argparse.ArgumentParser(
        description="3D визуализация кластеров (оси = 3 признака, цвет = кластер)"
    )
    ap.add_argument("--input", required=True,
                    help="Excel с результатами кластеризации (clusters_output.xlsx)")
    ap.add_argument("--sheet", default="Sheet1",
                    help="Имя листа Excel (по умолчанию 'Sheet1')")
    ap.add_argument("--features", default="price,delta_lon,departure_time",
                    help="Список признаков через запятую "
                         "(по умолчанию: price,delta_lon,departure_time)")
    ap.add_argument("--cluster-col", default="cluster",
                    help="Имя столбца с метками кластеров (по умолчанию 'cluster')")
    ap.add_argument("--n-max", type=int, default=20000,
                    help="Максимум точек для визуализации (по умолчанию 10000)")
    ap.add_argument("--out-html", default="clusters_3d.html",
                    help="Имя выходного HTML-файла (по умолчанию clusters_3d.html)")
    ap.add_argument("--auto-open", action="store_true",
                    help="Открыть HTML сразу в браузере")

    args = ap.parse_args()

    feature_cols = [c.strip() for c in args.features.split(",") if c.strip()]

    # 1. читаем данные
    df = pd.read_excel(args.input, sheet_name=args.sheet)

    # 2. при необходимости подсэмплим
    if len(df) > args.n_max:
        df = df.sample(args.n_max, random_state=0)

    # 3. очищаем и стандартизируем признаки
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())

    scaler = StandardScaler()
    Xz = scaler.fit_transform(X.values)

    df_plot = df.copy()
    for i, col in enumerate(feature_cols):
        df_plot[f"z_{col}"] = Xz[:, i]

    # возьмём последний признак для размера точки (если их >=1)
    last_feat = feature_cols[-1]
    w = df_plot[f"z_{last_feat}"]
    w_norm = (w - w.min()) / (w.max() - w.min() + 1e-9)
    df_plot["marker_size"] = 4 + 10 * w_norm

    xcol = f"z_{feature_cols[0]}"
    ycol = f"z_{feature_cols[1]}"
    zcol = f"z_{feature_cols[2]}"

    fig = px.scatter_3d(
        df_plot,
        x=xcol,
        y=ycol,
        z=zcol,
        color=args.cluster_col,
        size="marker_size",
        size_max=14,
        opacity=0.8,
        title="3D визуализация кластеров",
    )

    fig.update_layout(
        scene=dict(
            xaxis_title=f"{feature_cols[0]} (z-score)",
            yaxis_title=f"{feature_cols[1]} (z-score)",
            zaxis_title=f"{feature_cols[2]} (z-score)",
        )
    )

    fig.write_html(args.out_html, auto_open=args.auto_open)
    print("Сохранено в", args.out_html)


if __name__ == "__main__":
    main()
