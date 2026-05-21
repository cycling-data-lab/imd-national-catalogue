"""
d40_paris_spectral_3axes.py — Recover Paris R^2_spectral by dropping
the degenerate axis.

Table tab:gsp reports Paris R^2_spectral as "n/a" because one IMD axis
is constant within the V\'elib station-info GBFS export, making the
4-D projection ill-conditioned.  We identify the degenerate axis,
recompute the spectral upper bound on the 3 non-degenerate axes,
and check whether the missing axis is recoverable from OSM/Overpass.

Output:
  outputs/d40_paris_spectral.json
"""
from __future__ import annotations
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
IMD = ROOT / "data_collection" / "imd_international"
OUT = ROOT / "experiments" / "outputs"

FEATS = ["gtfs_heavy_stops_300m", "infra_cyclable_features_300m",
         "elevation_m", "topography_roughness_index"]


def build_knn_laplacian(coords, k=10, sigma=None):
    n = len(coords)
    nn = NearestNeighbors(n_neighbors=k+1).fit(coords)
    d, idx = nn.kneighbors(coords)
    d, idx = d[:, 1:], idx[:, 1:]
    if sigma is None: sigma = float(np.median(d))
    rows = np.repeat(np.arange(n), k)
    cols = idx.flatten()
    w = np.exp(-(d.flatten() ** 2) / (2 * sigma ** 2))
    W = csr_matrix((w, (rows, cols)), shape=(n, n))
    W = (W + W.T) / 2
    L = laplacian(W, normed=True)
    return L.toarray()


def main():
    paris_path = IMD / "world_fr_v_lib_metropole.parquet"
    if not paris_path.exists():
        print(f"✗ {paris_path} missing"); return
    df = pd.read_parquet(paris_path)
    df["station_id"] = df["station_id"].astype(str)
    print(f"Paris IMD : {len(df)} stations")

    # Inspect distributional summary of each axis
    print("\n=== Distributional check ===")
    degen = []
    for f in FEATS:
        if f not in df.columns:
            print(f"  {f:35s} : MISSING column"); degen.append(f); continue
        col = df[f].dropna()
        std = col.std()
        nunique = col.nunique()
        print(f"  {f:35s}  N_nan={df[f].isna().sum():>5d}  "
              f"std={std:>8.3f}  nunique={nunique}")
        if std < 1e-6 or nunique <= 1:
            print(f"     ↳ DEGENERATE")
            degen.append(f)
    print(f"\nDegenerate axes : {degen if degen else 'none — recompute with all 4 should work'}")
    avail = [f for f in FEATS if f not in degen and f in df.columns]
    print(f"Non-degenerate axes : {avail}")

    # Need station coordinates
    if not {"lat", "lng"}.issubset(df.columns):
        print("✗ no lat/lng in IMD parquet"); return
    sub = df.dropna(subset=avail + ["lat", "lng"]).reset_index(drop=True)
    print(f"\nFinal panel : {len(sub)} stations")
    if len(sub) < 100:
        print("✗ too small to compute Laplacian spectrum"); return

    # Try the panel WITHOUT the demand signal first — just spectral concentration
    # We use station-mean from d20 if available, else skip
    pred = None
    for p in [OUT / "d10_Paris_predictions.parquet",
              OUT / "d20_lso_tier2_paris_per_station.csv"]:
        if p.exists():
            if p.suffix == ".parquet":
                d = pd.read_parquet(p)
                d["station_id"] = d["station_id"].astype(str)
                d["y_true"] = np.expm1(d["y_true_log"])
                pred = d.groupby("station_id")["y_true"].mean()
            else:
                d = pd.read_csv(p)
                d["station_id"] = d["station_id"].astype(str)
                pred = d.set_index("station_id")["true_mean"]
            break
    if pred is None:
        print("✗ no Paris predictions parquet"); return
    sub["y"] = sub["station_id"].map(pred.to_dict())
    sub = sub.dropna(subset=["y"]).reset_index(drop=True)
    N = len(sub)
    print(f"Stations with demand : {N}")
    if N < 100:
        print("✗ too few stations with matched demand"); return

    # Build Laplacian on lat/lng
    coords = sub[["lat", "lng"]].values
    L = build_knn_laplacian(coords, k=10)
    eigval, eigvec = eigh(L)

    # Demand signal
    y = sub["y"].values
    y_z = (y - y.mean()) / (y.std() + 1e-12)
    y_hat = eigvec.T @ y_z
    smoothness = float(y_z @ L @ y_z / (y_z @ y_z))
    cum = np.cumsum(y_hat ** 2) / (y_hat @ y_hat)
    k90 = int(np.searchsorted(cum, 0.90)) + 1

    # IMD spectral upper bound (3-axis or 4-axis)
    X = sub[avail].astype(float).values
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
    Q, _ = np.linalg.qr(Xs)
    proj = Q @ (Q.T @ y_z)
    R2_spec = float((proj ** 2).sum() / (y_z @ y_z))
    print(f"\n=== Paris with {len(avail)} non-degenerate axes ===")
    print(f"  Smoothness y^T L y / y^T y = {smoothness:.3f}")
    print(f"  K_90% / N = {k90}/{N} = {k90/N*100:.1f}%")
    print(f"  R^2_spectral = {R2_spec:.3f}  (vs n/a in tab:gsp)")

    out = {
        "city": "Vélib Paris",
        "N": int(N),
        "n_axes_used": len(avail),
        "axes_used": avail,
        "degenerate_axes": degen,
        "smoothness": smoothness,
        "K_90pct": k90,
        "K_90pct_frac": float(k90 / N),
        "R2_spectral": R2_spec,
    }
    with open(OUT / "d40_paris_spectral.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n✓ Saved.")


if __name__ == "__main__":
    main()
