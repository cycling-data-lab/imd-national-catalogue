"""
d8_polling_quality.py — Quality / completeness report on the GBFS
station-status polling snapshots accumulated during the vacation.

Per city :
  - number of daily parquet files
  - time span covered (first poll → last poll)
  - mean poll interval (target was 60 s)
  - station count
  - pseudo-flow yield (= sum of negative Δ available_bikes)
  - bytes total / station / hour

Useful to identify which cities have enough polling data for the
Tier 2 multi-city benchmark, and which would need additional polling.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

SNAPSHOTS = Path("/home/rohanfosse/Bureau/Recherche/bikeshare-data-explorer/data/status_snapshots")
OUT = Path("/home/rohanfosse/Bureau/Recherche/imd-national-catalogue/paper_demand/experiments/outputs")
OUT.mkdir(parents=True, exist_ok=True)

def analyse_city(city_dir: Path) -> dict | None:
    parquets = sorted([p for p in city_dir.glob("*.parquet")
                       if p.name != "station_info.parquet"])
    if not parquets:
        return None
    frames = []
    for p in parquets:
        try:
            frames.append(pd.read_parquet(p))
        except Exception:
            continue
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df["fetched_at"] = pd.to_datetime(df["fetched_at"], utc=True)
    df = df.sort_values(["station_id", "fetched_at"]).reset_index(drop=True)

    total_bytes = sum(p.stat().st_size for p in parquets)
    span_h = (df["fetched_at"].max() - df["fetched_at"].min()).total_seconds() / 3600
    n_stations = df["station_id"].nunique()
    n_rows = len(df)

    # pseudo-flow yield = sum of negative deltas (bikes leaving)
    df["bikes_prev"] = df.groupby("station_id")["num_bikes_available"].shift(1)
    df["delta"] = df["num_bikes_available"] - df["bikes_prev"]
    df["time_prev"] = df.groupby("station_id")["fetched_at"].shift(1)
    df["gap_min"] = (df["fetched_at"] - df["time_prev"]).dt.total_seconds() / 60
    valid = df["gap_min"] < 60
    pseudo_trips = int(((df["delta"] < 0) & valid).sum())
    # Mean inter-poll gap (only valid ones)
    mean_gap_s = float(df.loc[valid, "gap_min"].mean() * 60) if valid.any() else None

    return {
        "city": city_dir.name,
        "n_daily_parquets": len(parquets),
        "n_stations": n_stations,
        "n_rows": n_rows,
        "span_hours": round(span_h, 1),
        "mean_poll_gap_sec": round(mean_gap_s, 1) if mean_gap_s else None,
        "pseudo_trips_total": pseudo_trips,
        "bytes": total_bytes,
        "kb_per_station_hour": round(total_bytes / max(n_stations * span_h, 1) / 1024, 2),
    }

def main():
    cities = sorted([d for d in SNAPSHOTS.iterdir() if d.is_dir()])
    print(f"=== Analyzing {len(cities)} polling directories ===")
    out = []
    for c in cities:
        try:
            r = analyse_city(c)
        except Exception as e:
            print(f"  ✗ {c.name}: {e}")
            continue
        if r is None:
            continue
        out.append(r)
        print(f"  {r['city']:25s}  {r['n_stations']:4d} stns  "
              f"{r['span_hours']:>5.1f}h  {r['pseudo_trips_total']:>6d} pseudo-trips  "
              f"poll_gap={r['mean_poll_gap_sec']}s")

    df = pd.DataFrame(out).sort_values("pseudo_trips_total", ascending=False)
    df.to_csv(OUT / "d8_polling_quality.csv", index=False)
    print(f"\n✓ Wrote {OUT / 'd8_polling_quality.csv'}")

    # Filter cities with enough data for a Tier 2 benchmark
    usable = df[df["pseudo_trips_total"] >= 100]
    usable[["city","n_stations","span_hours","pseudo_trips_total"]].to_csv(
        OUT / "d8_polling_usable.csv", index=False)
    print(f"\nCities with ≥100 pseudo-trips (usable for Tier 2 benchmark) : {len(usable)}")
    print(f"Total pseudo-trips across all cities                          : {int(df['pseudo_trips_total'].sum())}")

if __name__ == "__main__":
    main()
