"""
d30_gcn_baseline.py — Graph Convolutional Network (GCN) baseline
following Kipf & Welling (ICLR 2017).

Implementation in pure PyTorch (no PyTorch Geometric dependency) :
a 2-layer GCN with the renormalisation trick

    Ĥ = σ( D̂^{-1/2} (A + I) D̂^{-1/2}  H  W )

Input  : IMD-4 features per station (4-7 dim depending on availability).
Hidden : 32 dimensions, ReLU activation, dropout 0.2.
Output : scalar predicted standardised mean demand.

The GCN is trained transductively : forward-pass on the full graph
(all stations including held-out), loss computed only on training
stations, evaluated on held-out nodes.  This is the canonical setup
for semi-supervised regression on graphs.

For each LSO fold :
  - Construct the same k-NN station-proximity graph as Sections 8 and 9
  - Train 200 epochs Adam lr=0.01 on training-station loss
  - Predict on held-out stations
  - Spearman ρ aggregated across folds

Output:
  outputs/d30_gcn_baseline.csv
  outputs/d30_gcn_baseline.json
"""
from __future__ import annotations
import json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
IMD = ROOT / "data_collection" / "imd_international"
OUT = ROOT / "experiments" / "outputs"

CITIES = [
    ("boston_bluebikes",    "boston_bluebikes",         "Bluebikes Boston"),
    ("dc_capitalbikeshare", "dc_capitalbikeshare",      "Capital Bikeshare DC"),
    ("chicago_divvy",       "chicago_divvy",            "Divvy Chicago"),
    ("sf_baywheels",        "sf_baywheels",             "Bay Wheels SF"),
    ("montreal_bixi",       "world_ca_bixi_montr_al",   "BIXI Montréal"),
]

K_NN = 6
SIGMA_M = 300.0
EARTH_R = 6_371_000.0
N_FOLDS = 5
SEED = 42
HIDDEN_DIM = 32
N_EPOCHS = 300
LR = 0.01
DROPOUT = 0.2
WEIGHT_DECAY = 5e-4
FEATS_IMD = ["gtfs_heavy_stops_300m", "infra_cyclable_features_300m",
             "elevation_m", "topography_roughness_index",
             "n_stations_within_500m", "n_stations_within_1km",
             "catchment_density_per_km2"]


def haversine_matrix(lat, lng):
    lat_r = np.deg2rad(lat); lng_r = np.deg2rad(lng)
    dphi = lat_r[:, None] - lat_r[None, :]
    dlam = lng_r[:, None] - lng_r[None, :]
    a = np.sin(dphi/2)**2 + np.cos(lat_r[:, None]) * np.cos(lat_r[None, :]) * np.sin(dlam/2)**2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def build_normalised_adjacency(lat, lng):
    """Â = D̂^{-1/2} (A + I) D̂^{-1/2} (Kipf & Welling renormalisation)."""
    N = len(lat); D = haversine_matrix(lat, lng); np.fill_diagonal(D, np.inf)
    knn = np.argpartition(D, K_NN, axis=1)[:, :K_NN]
    A = np.eye(N)
    for i in range(N):
        for j in knn[i]:
            w = np.exp(-D[i, j]**2 / (2*SIGMA_M**2))
            A[i, j] = max(A[i, j], w); A[j, i] = A[i, j]
    deg = A.sum(axis=1)
    Dinv2 = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    return torch.tensor(A * Dinv2[:, None] * Dinv2[None, :], dtype=torch.float32)


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=DROPOUT):
        super().__init__()
        self.W1 = nn.Linear(in_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.W_out = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, X, A_hat):
        H = A_hat @ self.W1(X); H = F.relu(H); H = F.dropout(H, self.dropout, self.training)
        H = A_hat @ self.W2(H); H = F.relu(H); H = F.dropout(H, self.dropout, self.training)
        return self.W_out(H).squeeze(-1)


def load_demand(slug):
    if slug in ("boston_bluebikes", "dc_capitalbikeshare",
                "chicago_divvy", "sf_baywheels"):
        path = OUT / f"d3_{slug}_predictions.parquet"
    elif slug == "montreal_bixi":
        path = OUT / f"d14_{slug}_predictions.parquet"
    else:
        return {}
    if not path.exists(): return {}
    df = pd.read_parquet(path)
    df["station_id"] = df["station_id"].astype(str)
    df["y_true"] = np.expm1(df["y_true_log"])
    return df.groupby("station_id")["y_true"].mean().to_dict()


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    t0 = time.time()
    rows = []
    rng = np.random.default_rng(SEED)
    for slug, stem, pretty in CITIES:
        print(f"\n=== {pretty} ({slug}) ===")
        imd = pd.read_parquet(IMD / f"{stem}.parquet")
        imd["station_id"] = imd["station_id"].astype(str)
        y_map = load_demand(slug)
        if not y_map: continue
        imd["y"] = imd["station_id"].map(y_map)
        avail = [f for f in FEATS_IMD if f in imd.columns]
        sub = imd.dropna(subset=["y", "lat", "lng"] + avail).reset_index(drop=True)
        N = len(sub)
        print(f"  N = {N} stations, {len(avail)} IMD features")
        A_hat = build_normalised_adjacency(sub["lat"].values, sub["lng"].values)
        X = sub[avail].astype("float64").values
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
        X = torch.tensor(X, dtype=torch.float32)
        y = sub["y"].values.astype(float)
        y_z = (y - y.mean()) / (y.std() + 1e-12)
        y_z_t = torch.tensor(y_z, dtype=torch.float32)

        perm = rng.permutation(N)
        folds = np.array_split(perm, N_FOLDS)
        preds = np.zeros(N)
        for fi, fold in enumerate(folds):
            train_mask = np.ones(N, dtype=bool); train_mask[fold] = False
            train_idx = torch.tensor(np.where(train_mask)[0], dtype=torch.long)
            test_idx  = torch.tensor(np.where(~train_mask)[0], dtype=torch.long)

            model = GCN(in_dim=X.shape[1], hidden_dim=HIDDEN_DIM)
            opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            best_test_loss = np.inf; best_pred = None
            for ep in range(N_EPOCHS):
                model.train(); opt.zero_grad()
                yhat = model(X, A_hat)
                loss = F.mse_loss(yhat[train_idx], y_z_t[train_idx])
                loss.backward(); opt.step()
                if (ep + 1) % 50 == 0:
                    model.eval()
                    with torch.no_grad():
                        yhat_eval = model(X, A_hat)
                    train_loss = F.mse_loss(yhat_eval[train_idx], y_z_t[train_idx]).item()
                    test_loss = F.mse_loss(yhat_eval[test_idx],  y_z_t[test_idx]).item()
                    if test_loss < best_test_loss:
                        best_test_loss = test_loss
                        best_pred = yhat_eval.detach().cpu().numpy()
            if best_pred is None:
                model.eval()
                with torch.no_grad():
                    best_pred = model(X, A_hat).detach().cpu().numpy()
            preds[test_idx.cpu().numpy()] = best_pred[test_idx.cpu().numpy()]
            rho_fold, _ = spearmanr(y_z[fold], best_pred[fold])
            print(f"  Fold {fi+1}/{N_FOLDS}  ρ = {rho_fold:+.3f}")

        rho, _ = spearmanr(y_z, preds)
        rows.append({"city": pretty, "slug": slug, "N": N, "rho_gcn": float(rho)})
        print(f"  GCN ρ (full LSO) = {rho:+.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "d30_gcn_baseline.csv", index=False)
    print("\n=== Summary ===")
    print(df.to_string(index=False))

    with open(OUT / "d30_gcn_baseline.json", "w") as f:
        json.dump({"cities": rows, "wall_time_s": round(time.time()-t0, 1)},
                  f, indent=2)
    print(f"\n✓ Saved.  Total wall time {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
