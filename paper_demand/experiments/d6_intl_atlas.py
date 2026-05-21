"""
d6_intl_atlas.py — Worldwide IMD-4 ranking on the 183 international cities.

Applies the French-calibrated IMD-4 weights (w_M=0.78, w_I=0.07,
w_T=0.08, w_D=0.06) to every city for which we have an IMD-international
parquet.  Outputs:

  outputs/d6_world_imd_ranking.csv  — sorted by city-mean IMD score
  outputs/d6_world_imd_summary.csv  — per-country aggregate
  outputs/d6_world_imd_top30.txt    — pretty top-30 ranking

Runs in seconds, no heavy compute.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/home/rohanfosse/Bureau/Recherche/imd-national-catalogue")
IMD_DIR = REPO / "paper_demand/data_collection/imd_international"
OUT_DIR = REPO / "paper_demand/experiments/outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Weights from Bayesian calibration on the French 59-city panel
W = {"M": 0.78, "I": 0.07, "T": 0.08, "D": 0.06}

# Per-axis column names produced by imd_international.py
M_COL = "gtfs_heavy_stops_300m"
I_COL = "infra_cyclable_features_300m"
T_COL = "topography_roughness_index"   # roughness from SRTM
D_COL = "n_stations_within_1km"          # intra-system density proxy

def zscore(s: pd.Series) -> pd.Series:
    s = s.astype(float).replace([np.inf, -np.inf], np.nan)
    mu, sd = s.mean(skipna=True), s.std(skipna=True)
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd

def main():
    files = sorted(IMD_DIR.glob("world_*.parquet"))
    if not files:
        print("No world_*.parquet files found.  Did d6 download anything?")
        return

    # Pool all stations into a single dataframe so z-scores are computed
    # against the WORLD distribution (otherwise city-mean IMD is mechanically 0).
    pooled = []
    for f in files:
        try:
            df = pd.read_parquet(f)
        except Exception as e:
            print(f"  ✗ {f.name}: {e}")
            continue
        cslug = f.stem
        country = cslug.split("_")[1].upper() if "_" in cslug else "??"
        name = "_".join(cslug.split("_")[2:])
        for col in (M_COL, I_COL, T_COL, D_COL):
            if col not in df.columns:
                df[col] = np.nan
        df = df[[M_COL, I_COL, T_COL, D_COL]].copy()
        df["city_slug"] = cslug
        df["country"] = country
        df["city"] = name
        pooled.append(df)

    big = pd.concat(pooled, ignore_index=True)
    big["Mz"] = zscore(big[M_COL])
    big["Iz"] = zscore(big[I_COL])
    big["Tz"] = zscore(big[T_COL])
    big["Dz"] = zscore(big[D_COL])
    big["imd4"] = W["M"]*big["Mz"] + W["I"]*big["Iz"] + W["T"]*big["Tz"] + W["D"]*big["Dz"]

    out = (big.groupby(["city_slug", "country", "city"], sort=False)
              .agg(n_stations=("imd4", "size"),
                   mean_M=(M_COL, "mean"),
                   mean_I=(I_COL, "mean"),
                   mean_T=(T_COL, "mean"),
                   mean_D=(D_COL, "mean"),
                   imd4_mean=("imd4", "mean"),
                   imd4_p10=("imd4", lambda s: s.quantile(0.10)),
                   imd4_p50=("imd4", "median"),
                   imd4_p90=("imd4", lambda s: s.quantile(0.90)))
              .reset_index()
              .sort_values("imd4_mean", ascending=False))
    out.to_csv(OUT_DIR / "d6_world_imd_ranking.csv", index=False)

    by_country = (out.groupby("country")
                    .agg(n_cities=("city", "count"),
                         n_stations=("n_stations", "sum"),
                         imd4_mean=("imd4_mean", "mean"))
                    .sort_values("imd4_mean", ascending=False))
    by_country.to_csv(OUT_DIR / "d6_world_imd_summary.csv")

    # Pretty top-30 text
    with open(OUT_DIR / "d6_world_imd_top30.txt", "w") as f:
        f.write(f"WORLD IMD-4 RANKING (French-calibrated weights)\n")
        f.write(f"{len(out)} cities, {int(out['n_stations'].sum())} stations total\n\n")
        f.write(f"{'rk':>3} {'IMD-4':>8} {'cn':>3} {'stns':>5}  city\n")
        f.write("-" * 60 + "\n")
        for i, row in enumerate(out.head(30).itertuples(), 1):
            f.write(f"{i:>3} {row.imd4_mean:+8.3f} {row.country:>3} "
                    f"{row.n_stations:>5}  {row.city}\n")
        f.write(f"\nBOTTOM 10:\n")
        for i, row in enumerate(out.tail(10).itertuples(), 1):
            f.write(f"{len(out)-10+i:>3} {row.imd4_mean:+8.3f} {row.country:>3} "
                    f"{row.n_stations:>5}  {row.city}\n")

    print(f"✓ Wrote {OUT_DIR / 'd6_world_imd_ranking.csv'}")
    print(f"✓ Wrote {OUT_DIR / 'd6_world_imd_summary.csv'}")
    print(f"✓ Wrote {OUT_DIR / 'd6_world_imd_top30.txt'}")
    print(f"\nTotal cities ranked: {len(out)} from {out['country'].nunique()} countries")
    print(f"Total stations: {int(out['n_stations'].sum()):,}")

if __name__ == "__main__":
    main()
