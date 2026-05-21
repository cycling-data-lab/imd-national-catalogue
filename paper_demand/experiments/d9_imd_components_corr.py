"""
d9_imd_components_corr.py — Inter-component correlation analysis on
the 183 international cities.

Two analyses :
 1. Per-city aggregate values of M, I, T, D — distribution per axis,
    Pearson + Spearman pairwise correlation across cities.
 2. By-country aggregation : which countries have the most heterogeneous
    cycling-environment profiles?

Output :
  outputs/d9_components_by_city.csv        — per-city aggregates
  outputs/d9_components_correlation.csv    — correlation matrix
  outputs/d9_components_by_country.csv     — country aggregates
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

REPO = Path("/home/rohanfosse/Bureau/Recherche/imd-national-catalogue")
IMD_DIR = REPO / "paper_demand/data_collection/imd_international"
OUT = REPO / "paper_demand/experiments/outputs"
OUT.mkdir(parents=True, exist_ok=True)

AXES = {
    "M": "gtfs_heavy_stops_300m",
    "I": "infra_cyclable_features_300m",
    "T": "topography_roughness_index",
    "D": "n_stations_within_1km",
}

def main():
    files = sorted(IMD_DIR.glob("world_*.parquet"))
    if not files:
        print("No world_*.parquet files."); return

    rows = []
    for f in files:
        try:
            df = pd.read_parquet(f)
        except Exception as e:
            print(f"  ✗ {f.name}: {e}"); continue
        cslug = f.stem
        country = cslug.split("_")[1].upper() if "_" in cslug else "??"
        row = {"city_slug": cslug, "country": country, "n_stations": len(df)}
        for axis, col in AXES.items():
            if col in df.columns:
                row[f"{axis}_mean"] = df[col].mean(skipna=True)
                row[f"{axis}_std"] = df[col].std(skipna=True)
            else:
                row[f"{axis}_mean"] = np.nan
                row[f"{axis}_std"] = np.nan
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "d9_components_by_city.csv", index=False)
    print(f"✓ Wrote per-city : {len(out)} cities, {out['n_stations'].sum():,} stations")

    # Correlation matrix across cities
    cols = [f"{a}_mean" for a in AXES]
    sub = out[cols].dropna()
    print(f"\n=== Pearson correlation between IMD axes ({len(sub)} cities) ===")
    pcorr = sub.corr(method="pearson")
    print(pcorr.round(3).to_string())
    pcorr.to_csv(OUT / "d9_components_correlation.csv")

    print(f"\n=== Spearman correlation ===")
    scorr = sub.corr(method="spearman")
    print(scorr.round(3).to_string())

    # By-country summary
    bc = out.groupby("country").agg(
        n_cities=("city_slug", "count"),
        n_stations=("n_stations", "sum"),
        M_mean=("M_mean", "mean"),
        I_mean=("I_mean", "mean"),
        T_mean=("T_mean", "mean"),
        D_mean=("D_mean", "mean"),
    ).sort_values("n_cities", ascending=False)
    bc.to_csv(OUT / "d9_components_by_country.csv")
    print(f"\n✓ Wrote by-country : {len(bc)} countries")
    print(bc.head(15).round(2).to_string())

    # Top heterogeneity countries (high std across cities)
    bc_std = out.groupby("country").agg(
        M_std=("M_mean", "std"),
        I_std=("I_mean", "std"),
        D_std=("D_mean", "std"),
    ).fillna(0)
    bc_std["heterogeneity"] = bc_std.sum(axis=1)
    top_hetero = bc_std.nlargest(10, "heterogeneity")
    print(f"\n=== Top 10 most heterogeneous countries (inter-city variation) ===")
    print(top_hetero.round(2).to_string())

if __name__ == "__main__":
    main()
