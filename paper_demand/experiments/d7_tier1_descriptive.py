"""
d7_tier1_descriptive.py — Descriptive statistics on the 5 Tier 1
North American trip-log archives.

Per city :
  - total trips, total active months
  - mean trips per day, std
  - peak hour weekday, peak hour weekend
  - weekday/weekend ratio
  - member vs casual split (Lyft schema field)
  - rideable_type split (electric vs classic)

Streams CSVs from zips so RAM stays under ~500 MB even on NYC.
"""
from __future__ import annotations
import zipfile
from pathlib import Path
import pandas as pd
import numpy as np

REPO = Path("/home/rohanfosse/Bureau/Recherche/imd-national-catalogue")
TIER1 = REPO / "paper_demand/data_collection/tier1_trip_logs"
OUT = REPO / "paper_demand/experiments/outputs"
OUT.mkdir(parents=True, exist_ok=True)

CITIES = ["dc_capitalbikeshare", "chicago_divvy", "boston_bluebikes",
          "sf_baywheels", "nyc_citibike"]

def scan_zip(zpath: Path, rows: dict):
    """Stream the trip CSVs from one monthly zip, accumulate stats in rows[]"""
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():
            if name.startswith("__MACOSX") or not name.endswith(".csv"):
                continue
            with z.open(name) as f:
                cols = ["started_at", "member_casual", "rideable_type"]
                for chunk in pd.read_csv(f, usecols=cols, chunksize=400_000,
                                         low_memory=False):
                    chunk["started_at"] = pd.to_datetime(chunk["started_at"],
                                                          errors="coerce")
                    chunk = chunk.dropna(subset=["started_at"])
                    rows["total"] += len(chunk)
                    chunk["hour"] = chunk["started_at"].dt.hour
                    chunk["dow"] = chunk["started_at"].dt.dayofweek
                    chunk["is_we"] = chunk["dow"] >= 5
                    rows["weekday"] += int((~chunk["is_we"]).sum())
                    rows["weekend"] += int(chunk["is_we"].sum())
                    rows["hour_wd"] = rows["hour_wd"].add(
                        chunk.loc[~chunk["is_we"], "hour"].value_counts(),
                        fill_value=0)
                    rows["hour_we"] = rows["hour_we"].add(
                        chunk.loc[chunk["is_we"], "hour"].value_counts(),
                        fill_value=0)
                    if "member_casual" in chunk.columns:
                        v = chunk["member_casual"].value_counts()
                        rows["member"] += int(v.get("member", 0))
                        rows["casual"] += int(v.get("casual", 0))
                    if "rideable_type" in chunk.columns:
                        v = chunk["rideable_type"].value_counts()
                        for k in v.index:
                            rows["rideable"][k] = rows["rideable"].get(k, 0) + int(v[k])
                    rows["days"].update(chunk["started_at"].dt.date.unique())

def init_rows():
    return {
        "total": 0,
        "weekday": 0, "weekend": 0,
        "hour_wd": pd.Series(dtype=float),
        "hour_we": pd.Series(dtype=float),
        "member": 0, "casual": 0,
        "rideable": {},
        "days": set(),
    }

def main():
    summary = []
    for city in CITIES:
        cdir = TIER1 / city
        if not cdir.is_dir():
            print(f"  - skip {city} (no dir)"); continue
        zips = sorted(cdir.glob("*.zip"))
        if not zips:
            print(f"  - skip {city} (no zips)"); continue
        print(f"=== {city} : {len(zips)} months ===")
        rows = init_rows()
        for i, z in enumerate(zips, 1):
            try:
                scan_zip(z, rows)
                if i % 6 == 0:
                    print(f"  {city}: {i}/{len(zips)} months, {rows['total']:,} trips so far")
            except Exception as e:
                print(f"  ✗ {z.name}: {e}")

        n_days = len(rows["days"])
        trips_per_day = rows["total"] / n_days if n_days else 0
        peak_wd = int(rows["hour_wd"].idxmax()) if len(rows["hour_wd"]) else -1
        peak_we = int(rows["hour_we"].idxmax()) if len(rows["hour_we"]) else -1
        ratio_we_wd = (rows["weekend"] / max(rows["weekday"], 1))

        summary.append({
            "city": city,
            "n_months": len(zips),
            "n_days": n_days,
            "n_trips_total": rows["total"],
            "trips_per_day_mean": trips_per_day,
            "peak_hour_weekday": peak_wd,
            "peak_hour_weekend": peak_we,
            "weekend_to_weekday_ratio": ratio_we_wd,
            "n_member": rows["member"],
            "n_casual": rows["casual"],
            "pct_member": 100*rows["member"]/max(rows["member"]+rows["casual"],1),
            "rideable_split": dict(rows["rideable"]),
        })
        print(f"  → {rows['total']:,} trips, {trips_per_day:,.0f}/day, "
              f"peak {peak_wd}h (wd) {peak_we}h (we)")

    out = pd.DataFrame(summary)
    out.to_csv(OUT / "d7_tier1_descriptive.csv", index=False)
    print(f"\n✓ Wrote {OUT / 'd7_tier1_descriptive.csv'}")
    print("\n=== TOTAL ACROSS 5 CITIES ===")
    print(f"  trips         : {int(out['n_trips_total'].sum()):,}")
    print(f"  months        : {int(out['n_months'].sum())}")
    print(f"  days          : {int(out['n_days'].sum()):,}")
    print(f"  avg pct member: {out['pct_member'].mean():.1f} %")

if __name__ == "__main__":
    main()
