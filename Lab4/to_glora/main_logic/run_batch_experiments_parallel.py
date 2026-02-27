
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parallel batch experiments for small/mid/large datasets (multiprocessing + threads)."""
import os, argparse, subprocess, pandas as pd, numpy as np, runpy
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

SIZES = {
    "small":  "small/passengers_dataset_small.xlsx",
    "mid":    "mid/passengers_dataset_mid.xlsx",
    "large":  "large/passengers_dataset_large.xlsx",
}
PERCENTS = [5, 15, 30]
FEATURES_FOR_CLUSTERING = ["price", "arrival_lon", "departure_lon", "departure_time"]
INT_COLUMNS_DEFAULT = ["fio","passport","seat_number","coach_number","train_number"]

def run(cmd):
    print(">>", " ".join(map(str, cmd)))
    res = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if res.stdout: print(res.stdout.rstrip())
    if res.returncode != 0:
        if res.stderr: print(res.stderr.rstrip())
        raise RuntimeError(f"Command failed: {' '.join(map(str, cmd))}")
    return res

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def default_names(size_key: str):
    base = os.path.splitext(os.path.basename(SIZES[size_key]))[0]
    folder = os.path.dirname(SIZES[size_key])
    processed = os.path.join(folder, f"{base}_processed.xlsx")
    return folder, base, processed

def fill_with_spline_k2(complete_path: str, holes_path: str, out_path: str, int_columns=INT_COLUMNS_DEFAULT, sheet_complete="processed", sheet_holes="Sheet1"):
    mod = runpy.run_path("spline_knn_eval.py")
    pick_neighbors = mod["pick_neighbors"]; predict_with_spline = mod["predict_with_spline"]
    full_df = pd.read_excel(complete_path, sheet_name=sheet_complete)
    holes_df = pd.read_excel(holes_path, sheet_name=sheet_holes)
    if not full_df.index.equals(holes_df.index):
        holes_df = holes_df.copy(); holes_df.index = full_df.index
    num_cols = [c for c in holes_df.columns if pd.api.types.is_numeric_dtype(holes_df[c])]
    index_arr = holes_df.index.to_numpy(); out = holes_df.copy()
    for col in num_cols:
        missing_idx = np.where(out[col].isna().values)[0]
        if len(missing_idx) == 0: continue
        non_nan_mask = out[col].notna().values; obs_idxs = index_arr[non_nan_mask]
        if obs_idxs.size == 0: continue
        for rpos in missing_idx:
            row_idx = index_arr[rpos]
            neigh_idxs = pick_neighbors(obs_idxs, row_idx, k=min(2, obs_idxs.size))
            y_obs = out.loc[neigh_idxs, col].to_numpy(dtype=float)
            y_pred = predict_with_spline(neigh_idxs.astype(float), y_obs, float(row_idx))
            if pd.isna(y_pred): continue
            if col in int_columns: y_pred = float(np.rint(y_pred))
            out.iat[rpos, out.columns.get_loc(col)] = y_pred
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        out.to_excel(w, index=False, sheet_name="data")
    print("Saved spline-imputed:", out_path); return out_path

def cluster_fixed_features(input_xlsx: str, out_xlsx: str, report_csv: str, kmin=2, kmax=12, sheet="data"):
    mod = runpy.run_path("hier_spa_pipeline.py")
    standardize = mod["standardize"]; hclust_complete_labels = mod["hclust_complete_labels"]; choose_k_by_elbow = mod["choose_k_by_elbow"]
    df = pd.read_excel(input_xlsx, sheet_name=sheet); miss = [c for c in FEATURES_FOR_CLUSTERING if c not in df.columns]
    if miss: raise ValueError(f"Features not found in {input_xlsx}: {miss}")
    X = df[FEATURES_FOR_CLUSTERING].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce"); X[c] = X[c].fillna(X[c].median())
    Xz, _ = standardize(X); k_star, table_k = choose_k_by_elbow(Xz, kmin=kmin, kmax=kmax, rel_drop_threshold=0.05)
    print(f"[{os.path.basename(input_xlsx)}] chosen k* = {k_star}")
    labels = hclust_complete_labels(Xz, k_star); out_df = df.copy(); out_df["cluster"] = labels
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w: out_df.to_excel(w, index=False, sheet_name="data")
    table_k.to_csv(report_csv, index=False, encoding="utf-8")
    print("Saved clusters:", out_xlsx, "and report:", report_csv); return out_xlsx, report_csv

def process_percent(folder, base, processed, percent, skip_existing=False):
    holes = os.path.join(folder, f"{base}_holes{percent}.xlsx")
    dropped = os.path.join(folder, f"{base}_holes{percent}_dropped.xlsx")
    spline_filled = os.path.join(folder, f"{base}_holes{percent}_splineK2_filled.xlsx")
    if not (skip_existing and os.path.exists(holes)):
        run(["python", "remove_chunks.py", "--input", processed, "--output", holes, "--percent", str(percent)])
    else: print("[skip] remove_chunks ->", holes)
    if not (skip_existing and os.path.exists(dropped)):
        run(["python", "drop_missing_rows.py", "--input", holes, "--output", dropped])
    else: print("[skip] drop_missing_rows ->", dropped)
    if not (skip_existing and os.path.exists(spline_filled)):
        fill_with_spline_k2(processed, holes, spline_filled, int_columns=INT_COLUMNS_DEFAULT, sheet_complete="processed", sheet_holes="Sheet1")
    else: print("[skip] spline-impute ->", spline_filled)
    return {"holes": holes, "dropped": dropped, "spline": spline_filled}

def process_one_size(size, max_workers_holes=3, skip_existing=False):
    folder, base, processed = default_names(size); infile = SIZES[size]; ensure_dir(folder)
    markup_json = os.path.join(folder, f"type_markup_{size}.json")
    if not (skip_existing and os.path.exists(processed)):
        run(["python", "preprocess_minimal.py", "--input", infile, "--output", processed, "--trains", "used_data/trains.py", "--markup-json", markup_json])
    else: print("[skip] preprocess ->", processed)
    artifacts = {}
    with ThreadPoolExecutor(max_workers=max_workers_holes) as ex:
        futures = {ex.submit(process_percent, folder, base, processed, p, skip_existing): p for p in PERCENTS}
        for fut in as_completed(futures):
            p = futures[fut]; artifacts[p] = fut.result(); print(f"[{size}] finished percent {p}")
    p = 5; dropped_5 = artifacts[p]["dropped"]; spline_5 = artifacts[p]["spline"]
    cluster_fixed_features(dropped_5, out_xlsx=os.path.join(folder, f"{base}_holes{p}_dropped_clusters.xlsx"),
                           report_csv=os.path.join(folder, f"{base}_holes{p}_dropped_clustering_report.csv"), sheet="data")
    cluster_fixed_features(spline_5, out_xlsx=os.path.join(folder, f"{base}_holes{p}_splineK2_filled_clusters.xlsx"),
                           report_csv=os.path.join(folder, f"{base}_holes{p}_splineK2_clustering_report.csv"), sheet="data")
    return True

def main():
    parser = argparse.ArgumentParser(description="Parallel pipeline for datasets.")
    parser.add_argument("--sizes", nargs="*", default=list(SIZES.keys()), help="subset of sizes to run")
    parser.add_argument("--max-workers-size", type=int, default=None, help="processes across sizes")
    parser.add_argument("--max-workers-holes", type=int, default=None, help="threads per size for percents")
    parser.add_argument("--skip-existing", action="store_true", help="skip steps if output exists")
    args = parser.parse_args()
    sizes = [s for s in args.sizes if s in SIZES]
    if not sizes: print("Nothing to do."); return
    max_workers_size = args.max_workers_size or min(len(sizes), os.cpu_count() or 2)
    max_workers_holes = args.max_workers_holes or min(len(PERCENTS), max(1, (os.cpu_count() or 4)//2))
    print(f"Running sizes {sizes} with {max_workers_size} processes; per-size holes threads = {max_workers_holes}")
    with ProcessPoolExecutor(max_workers=max_workers_size) as ex:
        futures = {ex.submit(process_one_size, s, max_workers_holes, args.skip_existing): s for s in sizes}
        for fut in as_completed(futures):
            s = futures[fut]
            try: fut.result(); print(f"[{s}] COMPLETED")
            except Exception as e: print(f"[{s}] FAILED: {e}")
if __name__ == "__main__": main()
