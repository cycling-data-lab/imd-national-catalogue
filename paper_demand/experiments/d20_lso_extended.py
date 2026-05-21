"""
d20_lso_extended.py — Rigorous LSO on Tier 1 networks with custom loaders
(TfL Santander Cycles, BIXI Montréal) and Tier 2 GBFS pseudo-flow networks
(Paris Vélib, Lyon Vélo'v, Toulouse VéLÔ).

Same LSO machinery as d19 (G^- / G_FE / G models, paired station-bootstrap CIs,
random folds, hypergeometric null), but with city-specific panel loaders.

Usage:
    python3 d20_lso_extended.py --source tfl
    python3 d20_lso_extended.py --source bixi
    python3 d20_lso_extended.py --source tier2_paris
    python3 d20_lso_extended.py --source tier2_lyon
    python3 d20_lso_extended.py --source tier2_toulouse
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from sklearn.neighbors import BallTree
from scipy.stats import spearmanr, kendalltau, hypergeom

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
TIER1_DIR = ROOT / "data_collection" / "tier1_trip_logs"
IMD_INTL_DIR = ROOT / "data_collection" / "imd_international"
SNAPSHOTS = Path(
    os.environ.get(
        "GBFS_SNAPSHOTS",
        str(ROOT.parents[1] / "bikeshare-data-explorer" / "data" / "status_snapshots"),
    )
)
OUT = ROOT / "experiments" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

FEATS_T = ["hour", "day_of_week", "month"]
FEATS_IMD = [
    "gtfs_heavy_stops_300m", "infra_cyclable_features_300m",
    "elevation_m", "topography_roughness_index",
    "n_stations_within_500m", "n_stations_within_1km",
    "catchment_density_per_km2",
]
LGB_PARAMS = dict(
    n_estimators=300, learning_rate=0.05, num_leaves=63,
    min_child_samples=30, reg_lambda=0.5, random_state=42,
    n_jobs=-1, verbose=-1,
)
N_BOOT = 1000
SEED = 42


# ── Panel loaders ─────────────────────────────────────────────────────────────
def load_tfl(imd: pd.DataFrame) -> pd.DataFrame:
    """Stream TfL weekly CSVs (London) and aggregate to (station, hour)."""
    folder = TIER1_DIR / "london_tfl"
    csvs = sorted(folder.glob("*.csv"))
    print(f"  TfL: {len(csvs)} weekly CSVs")
    counts = {}
    for csv in csvs:
        for chunk in pd.read_csv(csv, chunksize=300_000,
                                 usecols=["Start date", "Start station number"],
                                 low_memory=False):
            chunk = chunk.dropna(subset=["Start date", "Start station number"])
            chunk["sid"] = chunk["Start station number"].astype(str).str.zfill(6)
            ts = pd.to_datetime(chunk["Start date"], errors="coerce")
            chunk["hour"] = ts.dt.floor("h")
            chunk = chunk.dropna(subset=["hour"])
            grouped = chunk.groupby(["sid", "hour"]).size().reset_index(name="d")
            for s, h, c in grouped.itertuples(index=False):
                counts[(s, h)] = counts.get((s, h), 0) + int(c)
    df = pd.DataFrame([(s, h, c) for (s, h), c in counts.items()],
                      columns=["station_id", "datetime_hour", "demande"])
    return df


def load_bixi(imd: pd.DataFrame) -> pd.DataFrame:
    """Stream BIXI 2024 CSV (Montréal), map coords to IMD via BallTree,
    apply seasonal (Apr-Nov) filter to align with year-round Tier 1 panels."""
    folder = TIER1_DIR / "montreal_bixi"
    zips = sorted(folder.glob("*.zip"))
    if not zips:
        print("  ✗ no BIXI zip"); return pd.DataFrame()
    print(f"  BIXI: {zips[0].name}")
    pts = np.deg2rad(imd[["lat", "lng"]].values)
    tree = BallTree(pts, metric="haversine")
    EARTH_R = 6_371_000.0
    counts = {}
    with zipfile.ZipFile(zips[0]) as z:
        name = z.namelist()[0]
        with z.open(name) as f:
            for chunk in pd.read_csv(f, chunksize=400_000,
                                     usecols=["STARTSTATIONLATITUDE",
                                              "STARTSTATIONLONGITUDE",
                                              "STARTTIMEMS"],
                                     low_memory=False):
                chunk = chunk.dropna()
                coords = np.deg2rad(chunk[["STARTSTATIONLATITUDE",
                                            "STARTSTATIONLONGITUDE"]].values)
                dist_rad, idx = tree.query(coords, k=1)
                dist_m = dist_rad[:, 0] * EARTH_R
                ok = dist_m <= 50.0
                ts = pd.to_datetime(chunk["STARTTIMEMS"].values[ok], unit="ms")
                hours = ts.floor("h")
                sids = imd["station_id"].values[idx[:, 0][ok]]
                for s, h in zip(sids.astype(str), hours):
                    counts[(s, h)] = counts.get((s, h), 0) + 1
    df = pd.DataFrame([(s, h, c) for (s, h), c in counts.items()],
                      columns=["station_id", "datetime_hour", "demande"])
    df["datetime_hour"] = pd.to_datetime(df["datetime_hour"])
    df = df[df["datetime_hour"].dt.month.between(4, 11)].copy()
    return df


def load_tier2(slug: str, imd: pd.DataFrame) -> pd.DataFrame:
    """Compute pseudo-flow demand from GBFS station-status polling parquets."""
    poll_dir = SNAPSHOTS / slug
    parquets = sorted([p for p in poll_dir.glob("*.parquet")
                       if p.name != "station_info.parquet"])
    if not parquets:
        return pd.DataFrame()
    print(f"  Tier 2 {slug}: {len(parquets)} polling parquets")
    frames = [pd.read_parquet(p) for p in parquets]
    df = pd.concat(frames, ignore_index=True)
    df["fetched_at"] = pd.to_datetime(df["fetched_at"], utc=True)
    df = df.sort_values(["station_id", "fetched_at"]).reset_index(drop=True)
    df["bikes_prev"] = df.groupby("station_id")["num_bikes_available"].shift(1)
    df["delta"] = df["num_bikes_available"] - df["bikes_prev"]
    df["time_prev"] = df.groupby("station_id")["fetched_at"].shift(1)
    df["gap_min"] = (df["fetched_at"] - df["time_prev"]).dt.total_seconds() / 60
    valid = df["gap_min"].between(0, 60, inclusive="right")
    df["p"] = np.where(valid & (df["delta"] < 0), -df["delta"].astype("Int64"), 0)
    df["datetime_hour"] = df["fetched_at"].dt.floor("h").dt.tz_convert(None)
    out = (df.groupby(["station_id", "datetime_hour"], as_index=False)["p"].sum()
             .rename(columns={"p": "demande"}))
    out["station_id"] = out["station_id"].astype(str)
    out["demande"] = out["demande"].astype(int)
    return out


SOURCES = {
    "tfl":        dict(imd="london_tfl",                   loader=load_tfl,
                       pretty="london_tfl"),
    "bixi":       dict(imd="world_ca_bixi_montr_al",       loader=load_bixi,
                       pretty="montreal_bixi"),
    "tier2_paris":      dict(imd="world_fr_v_lib_metropole",
                              loader=lambda imd: load_tier2("Paris", imd),
                              pretty="tier2_paris"),
    "tier2_lyon":       dict(imd="world_fr_v_lo_v",
                              loader=lambda imd: load_tier2("lyon", imd),
                              pretty="tier2_lyon"),
    "tier2_toulouse":   dict(imd="world_fr_v_l_toulouse",
                              loader=lambda imd: load_tier2("toulouse", imd),
                              pretty="tier2_toulouse"),
}


# ── LSO core (identical to d19) ───────────────────────────────────────────────
def precision_at_k(true_s, pred_s, k):
    return len(set(true_s[:k]) & set(pred_s[:k])) / k


def hypergeom_null_pk(N, K):
    rv = hypergeom(N, K, K)
    return float(rv.mean()/K), float(rv.std()/K)


def station_bootstrap(ps, rng, B=N_BOOT):
    n = len(ps)
    rho_g, rho_fe, rho_no, dr_g_vs_fe, dr_g_vs_no = [], [], [], [], []
    pk = {K: {"g": [], "fe": [], "no": []} for K in (5, 10, 20, 50) if K <= n}
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        sub = ps.iloc[idx]
        r_g,  _ = spearmanr(sub["true_mean"], sub["pred_imd_mean"])
        r_fe, _ = spearmanr(sub["true_mean"], sub["pred_fe_mean"])
        r_no, _ = spearmanr(sub["true_mean"], sub["pred_no_imd_mean"])
        rho_g.append(r_g); rho_fe.append(r_fe); rho_no.append(r_no)
        dr_g_vs_fe.append(r_g - r_fe)
        dr_g_vs_no.append(r_g - r_no)
        for K in pk:
            ord_obs = sub.sort_values("true_mean",        ascending=False)["station_id"].tolist()
            ord_g   = sub.sort_values("pred_imd_mean",    ascending=False)["station_id"].tolist()
            ord_fe  = sub.sort_values("pred_fe_mean",     ascending=False)["station_id"].tolist()
            ord_no  = sub.sort_values("pred_no_imd_mean", ascending=False)["station_id"].tolist()
            pk[K]["g"].append(precision_at_k(ord_obs, ord_g, K))
            pk[K]["fe"].append(precision_at_k(ord_obs, ord_fe, K))
            pk[K]["no"].append(precision_at_k(ord_obs, ord_no, K))

    def ci(s): return float(np.quantile(s, 0.025)), float(np.quantile(s, 0.975))
    out = {
        "rho_imd": dict(mean=float(np.mean(rho_g)), ci=ci(rho_g)),
        "rho_fe":  dict(mean=float(np.mean(rho_fe)), ci=ci(rho_fe)),
        "rho_no":  dict(mean=float(np.mean(rho_no)), ci=ci(rho_no)),
        "delta_rho_imd_vs_fe": dict(mean=float(np.mean(dr_g_vs_fe)), ci=ci(dr_g_vs_fe)),
        "delta_rho_imd_vs_no": dict(mean=float(np.mean(dr_g_vs_no)), ci=ci(dr_g_vs_no)),
        "precision_at_k": {f"K{K}": dict(
            imd=dict(mean=float(np.mean(pk[K]["g"])), ci=ci(pk[K]["g"])),
            fe=dict(mean=float(np.mean(pk[K]["fe"])), ci=ci(pk[K]["fe"])),
            no_imd=dict(mean=float(np.mean(pk[K]["no"])), ci=ci(pk[K]["no"])),
        ) for K in pk},
    }
    return out


def run(source: str, n_folds: int = 5, seed: int = SEED):
    t_start = time.time()
    rng = np.random.default_rng(seed)
    cfg = SOURCES[source]
    pretty = cfg["pretty"]
    imd = pd.read_parquet(IMD_INTL_DIR / f"{cfg['imd']}.parquet")
    imd["station_id"] = imd["station_id"].astype(str)
    if "lat" not in imd.columns or "lng" not in imd.columns:
        raise SystemExit("IMD missing lat/lng")
    print(f"IMD : {len(imd)} stations")
    if pretty.startswith("london_tfl"):
        imd["station_id"] = imd["station_id"].str.zfill(6)
    panel = cfg["loader"](imd)
    if panel.empty:
        print("✗ Empty panel"); return
    print(f"Panel : {len(panel):,} bins  {panel['demande'].sum():,} trips/pseudo-trips")

    df = panel.merge(imd[["station_id", "lat", "lng"] + FEATS_IMD],
                     on="station_id", how="left")
    df = df.dropna(subset=FEATS_IMD).copy()
    df["datetime_hour"] = pd.to_datetime(df["datetime_hour"])
    df["hour"] = df["datetime_hour"].dt.hour
    df["day_of_week"] = df["datetime_hour"].dt.dayofweek
    df["month"] = df["datetime_hour"].dt.month
    df["log_demand"] = np.log1p(df["demande"])
    # Tier 1 networks: require >=6 months activity per station;
    # Tier 2 polling = ~8 days total so the filter is skipped there.
    if not pretty.startswith("tier2_"):
        span = df.groupby("station_id")["datetime_hour"].agg(["min", "max"])
        span["months"] = (span["max"] - span["min"]).dt.days / 30
        keep = span[span["months"] >= 6].index.tolist()
        df = df[df["station_id"].isin(keep)].copy()
    stations = sorted(df["station_id"].unique())
    n = len(stations)
    print(f"After IMD merge / activity filter: {len(df):,} bins  {n} stations")

    perm = list(stations); rng.shuffle(perm)
    folds = [list(a) for a in np.array_split(perm, n_folds)]
    print(f"Folds: {[len(f) for f in folds]}")
    station_id_to_code = {s: i for i, s in enumerate(stations)}
    df["station_code"] = df["station_id"].map(station_id_to_code)

    per_station = []; fold_summary = []
    for fi, fold in enumerate(folds):
        fold_set = set(fold)
        train = df[~df["station_id"].isin(fold_set)]
        test = df[df["station_id"].isin(fold_set)].copy()
        if len(train) == 0 or len(test) == 0:
            continue
        print(f"\n=== Fold {fi+1}/{n_folds}: "
              f"{train['station_id'].nunique()} train, {len(fold_set)} holdout, "
              f"n_test={len(test):,} ===")

        def fit(features, name, cat=None):
            m = lgb.LGBMRegressor(**LGB_PARAMS)
            X_train = train[features].copy()
            X_test = test[features].copy()
            kwargs = {}
            if cat:
                for c in cat:
                    X_train[c] = X_train[c].astype("category")
                    X_test[c] = X_test[c].astype("category")
                kwargs["categorical_feature"] = cat
            else:
                X_train = X_train.astype("float64")
                X_test = X_test.astype("float64")
            t0 = time.time()
            m.fit(X_train, train["log_demand"].values, **kwargs)
            ys = m.predict(X_test)
            yt = test["log_demand"].values
            yt_trips = np.expm1(yt); yp_trips = np.expm1(np.clip(ys, 0, None))
            return ys, yp_trips, time.time() - t0

        ys_no,  yp_no,  ts_no  = fit(FEATS_T,                "Gminus")
        ys_fe,  yp_fe,  ts_fe  = fit(FEATS_T + ["station_code"], "GFE", cat=["station_code"])
        ys_imd, yp_imd, ts_imd = fit(FEATS_T + FEATS_IMD,    "G")

        test["pred_no_imd"] = yp_no
        test["pred_fe"] = yp_fe
        test["pred_imd"] = yp_imd
        agg = (test.groupby("station_id")
                  .agg(n_bins=("demande", "size"),
                       true_mean=("demande", "mean"),
                       pred_no_imd_mean=("pred_no_imd", "mean"),
                       pred_fe_mean=("pred_fe", "mean"),
                       pred_imd_mean=("pred_imd", "mean"))
                  .reset_index())
        agg["fold"] = fi + 1
        per_station.append(agg)

        yt_trips = np.expm1(test["log_demand"].values)
        r2_no  = r2_score(yt_trips, yp_no)
        r2_fe  = r2_score(yt_trips, yp_fe)
        r2_imd = r2_score(yt_trips, yp_imd)
        print(f"  R²: G^-={r2_no:+.4f}  G_FE={r2_fe:+.4f}  G={r2_imd:+.4f}")
        fold_summary.append(dict(fold=fi+1, n_holdout=len(fold_set),
                                  n_test=int(len(test)),
                                  r2_hourly_no_imd=r2_no,
                                  r2_hourly_fe=r2_fe,
                                  r2_hourly_imd=r2_imd))

    ps = pd.concat(per_station, ignore_index=True)
    ps.to_csv(OUT / f"d20_lso_{pretty}_per_station.csv", index=False)
    N = len(ps)
    rho_g_pt,  _ = spearmanr(ps["true_mean"], ps["pred_imd_mean"])
    rho_fe_pt, _ = spearmanr(ps["true_mean"], ps["pred_fe_mean"])
    rho_no_pt, _ = spearmanr(ps["true_mean"], ps["pred_no_imd_mean"])
    print(f"\n=== Point estimates ===")
    print(f"Spearman ρ:  G={rho_g_pt:+.3f}  G_FE={rho_fe_pt:+.3f}  G^-={rho_no_pt:+.3f}")

    pk_null = {K: hypergeom_null_pk(N, K) for K in (5, 10, 20, 50) if K <= N}
    print(f"\nRunning station-bootstrap (B={N_BOOT})...")
    boot = station_bootstrap(ps, rng, B=N_BOOT)
    print(f"Δρ (IMD - G_FE):  mean={boot['delta_rho_imd_vs_fe']['mean']:+.3f}, "
          f"95% CI = [{boot['delta_rho_imd_vs_fe']['ci'][0]:+.3f}, "
          f"{boot['delta_rho_imd_vs_fe']['ci'][1]:+.3f}]")
    for K, (mu, sd) in pk_null.items():
        if f"K{K}" in boot["precision_at_k"]:
            print(f"P@{K}: G={boot['precision_at_k'][f'K{K}']['imd']['mean']:.3f}  "
                  f"G_FE={boot['precision_at_k'][f'K{K}']['fe']['mean']:.3f}  "
                  f"null={mu:.3f}±{sd:.3f}")

    r2_vals_no  = [f["r2_hourly_no_imd"] for f in fold_summary]
    r2_vals_fe  = [f["r2_hourly_fe"]     for f in fold_summary]
    r2_vals_imd = [f["r2_hourly_imd"]    for f in fold_summary]
    metrics = {
        "source": source, "city": pretty,
        "n_stations_evaluated": int(N),
        "n_folds": n_folds,
        "wall_time_s": round(time.time() - t_start, 1),
        "r2_hourly": {
            "no_imd_mean": float(np.mean(r2_vals_no)),
            "fe_mean":     float(np.mean(r2_vals_fe)),
            "imd_mean":    float(np.mean(r2_vals_imd)),
        },
        "rho_point": {"imd": float(rho_g_pt), "fe": float(rho_fe_pt), "no_imd": float(rho_no_pt)},
        "bootstrap": boot,
        "hypergeom_null": {f"K{K}": dict(mean=mu, std=sd) for K, (mu, sd) in pk_null.items()},
        "per_fold": fold_summary,
    }
    with open(OUT / f"d20_lso_{pretty}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Saved {OUT / f'd20_lso_{pretty}.json'}")
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, choices=list(SOURCES))
    p.add_argument("--folds", type=int, default=5)
    args = p.parse_args()
    run(args.source, args.folds)


if __name__ == "__main__":
    main()
