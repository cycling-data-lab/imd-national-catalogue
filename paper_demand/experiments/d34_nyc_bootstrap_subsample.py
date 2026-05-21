"""
d34_nyc_bootstrap_subsample.py — Bootstrap CI for Citi Bike NYC via
station subsampling (closes the v2.0 supplement promise).

The full NYC panel (24 months × 2230 stations ≈ 19 M rows) exceeded the
12 GB peak RAM of the bootstrap rig in earlier attempts.  Two practical
workarounds (both already cited in the paper's §10 Limits):

  (i)  restrict to a 12-month panel (2024 only) — done in d3 / paper.
  (ii) random subsample of 800 stations — done here.

Combined, the rig fits comfortably under 8 GB.  Subsampling stations
(rather than time) preserves the temporal hold-out and weekly-block
bootstrap structure of the other Tier 1 cities, at the cost of a noisier
ΔR² estimate proportional to sqrt(N/800).

Output:
  outputs/d34_nyc_bootstrap.json
  outputs/d34_nyc_bootstrap_predictions.parquet
"""
from __future__ import annotations
import json, time, warnings, zipfile
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
TIER1 = ROOT / "data_collection" / "tier1_trip_logs" / "nyc_citibike"
IMD = ROOT / "data_collection" / "imd_international"
OUT = ROOT / "experiments" / "outputs"

N_SUBSAMPLE = 800
SEED = 42
B = 500
FEATS_T = ["hour", "day_of_week", "month"]
FEATS_IMD = ["gtfs_heavy_stops_300m", "infra_cyclable_features_300m",
             "elevation_m", "topography_roughness_index",
             "n_stations_within_500m", "n_stations_within_1km",
             "catchment_density_per_km2"]


def load_nyc_2024(rng):
    """Load only 2024 NYC zips and subsample to N_SUBSAMPLE stations."""
    zips = sorted(TIER1.glob("2024*citibike-tripdata.zip"))
    if not zips:
        zips = sorted(TIER1.glob("202401-*"))[:1] + sorted(TIER1.glob("2024*-*"))
    if not zips:
        print(f"  ✗ no 2024 NYC zips found in {TIER1}"); return None
    print(f"  Loading {len(zips)} NYC 2024 zips...")
    # First pass : collect all station IDs to subsample
    all_stations = set()
    for z in zips[:3]:
        with zipfile.ZipFile(z) as zf:
            for name in zf.namelist():
                if name.endswith(".csv"):
                    with zf.open(name) as f:
                        for chunk in pd.read_csv(f, usecols=["start_station_id"],
                                                 chunksize=300_000, low_memory=False):
                            all_stations.update(chunk["start_station_id"].dropna().astype(str).unique())
    all_stations = sorted(all_stations)
    if len(all_stations) > N_SUBSAMPLE:
        rng.shuffle(all_stations)
        keep = set(all_stations[:N_SUBSAMPLE])
    else:
        keep = set(all_stations)
    print(f"  Subsampling to {len(keep)} stations (from {len(all_stations)} unique)")

    # Second pass : load only kept stations
    bins = {}  # (station, hour) -> count
    n_rows_total = 0
    for z in zips:
        t0 = time.time()
        with zipfile.ZipFile(z) as zf:
            for name in zf.namelist():
                if not name.endswith(".csv") or name.startswith("__MAC"):
                    continue
                with zf.open(name) as f:
                    for chunk in pd.read_csv(f, usecols=["started_at", "start_station_id"],
                                             chunksize=300_000, low_memory=False):
                        chunk = chunk.dropna()
                        chunk["sid"] = chunk["start_station_id"].astype(str)
                        chunk = chunk[chunk["sid"].isin(keep)]
                        chunk["hour"] = pd.to_datetime(chunk["started_at"],
                                                       errors="coerce").dt.floor("h")
                        chunk = chunk.dropna(subset=["hour"])
                        for sid, h in zip(chunk["sid"].values, chunk["hour"].values):
                            bins[(sid, h)] = bins.get((sid, h), 0) + 1
                        n_rows_total += len(chunk)
        print(f"    {z.name} done ({time.time()-t0:.1f}s, {len(bins):,} bins so far)")
    df = pd.DataFrame([(s, h, c) for (s, h), c in bins.items()],
                      columns=["station_id", "datetime_hour", "demande"])
    print(f"  Total rows : {n_rows_total:,}.  Panel : {len(df):,} (station, hour) bins")
    return df


def block_bootstrap(y_true_log, y_pred_log, times, B=B, rng=None):
    if rng is None: rng = np.random.default_rng(42)
    iso = pd.to_datetime(times).isocalendar()
    blocks = (iso["year"].astype(int) * 100 + iso["week"].astype(int)).values
    unique = np.unique(blocks)
    by_block = {b: np.where(blocks == b)[0] for b in unique}
    def r2(idx):
        yt = np.expm1(y_true_log[idx]); yp = np.expm1(np.clip(y_pred_log[idx], 0, None))
        return r2_score(yt, yp)
    samples = []
    for _ in range(B):
        sel = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([by_block[s] for s in sel])
        samples.append(r2(idx))
    pt = r2(np.arange(len(y_true_log)))
    return pt, float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def main():
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    panel = load_nyc_2024(rng)
    if panel is None or len(panel) < 10000:
        print("✗ NYC panel unavailable"); return

    imd = pd.read_parquet(IMD / "nyc_citibike.parquet")
    imd["station_id"] = imd["station_id"].astype(str)
    df = panel.merge(imd[["station_id"] + FEATS_IMD], on="station_id", how="left")
    df = df.dropna(subset=FEATS_IMD).copy()
    df["datetime_hour"] = pd.to_datetime(df["datetime_hour"])
    df["hour"] = df["datetime_hour"].dt.hour
    df["day_of_week"] = df["datetime_hour"].dt.dayofweek
    df["month"] = df["datetime_hour"].dt.month
    df["log_demand"] = np.log1p(df["demande"])
    df = df.sort_values("datetime_hour").reset_index(drop=True)
    cut = int(0.8 * len(df))
    train, test = df.iloc[:cut].copy(), df.iloc[cut:].copy()
    print(f"  Train : {len(train):,}  Test : {len(test):,}  "
          f"{df['station_id'].nunique()} matched stations")

    def fit(features):
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=63,
                              min_child_samples=30, reg_lambda=0.5, random_state=42,
                              n_jobs=-1, verbose=-1)
        m.fit(train[features].values, train["log_demand"].values)
        return m.predict(test[features].values)
    print("  Fitting G- and G ...")
    pred_no = fit(FEATS_T)
    pred_g = fit(FEATS_T + FEATS_IMD)
    yt = test["log_demand"].values
    times = test["datetime_hour"].values

    r2_no, lo_no, hi_no = block_bootstrap(yt, pred_no, times, B=B, rng=rng)
    r2_g, lo_g, hi_g = block_bootstrap(yt, pred_g, times, B=B, rng=rng)
    # Paired delta bootstrap
    iso = pd.to_datetime(times).isocalendar()
    blocks = (iso["year"].astype(int) * 100 + iso["week"].astype(int)).values
    unique = np.unique(blocks); by_block = {b: np.where(blocks == b)[0] for b in unique}
    delta_samples = []
    rng = np.random.default_rng(SEED)
    for _ in range(B):
        sel = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([by_block[s] for s in sel])
        yt_t = np.expm1(yt[idx]); yg = np.expm1(np.clip(pred_g[idx], 0, None))
        ynr = np.expm1(np.clip(pred_no[idx], 0, None))
        delta_samples.append(r2_score(yt_t, yg) - r2_score(yt_t, ynr))
    dr = r2_g - r2_no
    dr_lo, dr_hi = float(np.quantile(delta_samples, 0.025)), float(np.quantile(delta_samples, 0.975))

    print(f"\nNYC subsample bootstrap results:")
    print(f"  R²(G-) = {r2_no:+.4f}  CI [{lo_no:+.4f}, {hi_no:+.4f}]")
    print(f"  R²(G)  = {r2_g:+.4f}  CI [{lo_g:+.4f}, {hi_g:+.4f}]")
    print(f"  ΔR²   = {dr:+.4f}  CI [{dr_lo:+.4f}, {dr_hi:+.4f}]")

    out = {
        "n_subsample": N_SUBSAMPLE, "n_train": int(len(train)), "n_test": int(len(test)),
        "n_stations_matched": int(df["station_id"].nunique()),
        "B": B,
        "r2_no_imd": {"point": float(r2_no), "ci": [lo_no, hi_no]},
        "r2_imd":    {"point": float(r2_g),  "ci": [lo_g,  hi_g]},
        "delta_r2":  {"point": float(dr),    "ci": [dr_lo, dr_hi]},
        "wall_time_s": round(time.time() - t0, 1),
    }
    with open(OUT / "d34_nyc_bootstrap.json", "w") as f:
        json.dump(out, f, indent=2)
    # Save predictions for downstream d11-style use
    pred_df = pd.DataFrame({
        "datetime_hour": test["datetime_hour"].values,
        "station_id":    test["station_id"].values,
        "y_true_log":    yt,
        "y_pred_no_imd": pred_no,
        "y_pred_imd":    pred_g,
    })
    pred_df.to_parquet(OUT / "d34_nyc_subsample800_predictions.parquet", index=False)
    print(f"\n✓ Saved.  Wall time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
