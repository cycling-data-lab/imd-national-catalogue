"""
d15_tfl_setup.py — Generate IMD-4 features for TfL Santander Cycles
(London) by querying the TfL BikePoint API for station coordinates
and feeding them to imd_international.compute_imd_for_city.

Writes paper_demand/data_collection/imd_international/london_tfl.parquet.
"""
from __future__ import annotations
import sys
import urllib.request
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_collection"))
from imd_international import compute_imd_for_city, OUT_DIR  # noqa: E402

API = "https://api.tfl.gov.uk/BikePoint"


def fetch_stations() -> pd.DataFrame:
    print(f"Fetching {API}...")
    req = urllib.request.Request(API, headers={"User-Agent": "Mozilla/5.0 (imd-research)"})
    raw = urllib.request.urlopen(req, timeout=60).read()
    data = json.loads(raw)
    rows = []
    for d in data:
        terminal = None
        for p in d.get("additionalProperties", []):
            if p.get("key") == "TerminalName":
                terminal = p.get("value")
                break
        if not terminal:
            continue
        rows.append({
            "station_id": terminal,
            "name": d.get("commonName", ""),
            "lat": float(d["lat"]),
            "lng": float(d["lon"]),
            "city": "London",
        })
    df = pd.DataFrame(rows)
    print(f"  {len(df)} BikePoints with TerminalName + lat/lng")
    print(f"  Bbox : lat [{df['lat'].min():.4f}, {df['lat'].max():.4f}]")
    print(f"         lng [{df['lng'].min():.4f}, {df['lng'].max():.4f}]")
    return df


def main():
    stations = fetch_stations()
    imd = compute_imd_for_city(stations, "london_tfl",
                                skip_osm=False, skip_elevation=False)
    out_path = OUT_DIR / "london_tfl.parquet"
    imd.to_parquet(out_path, index=False)
    print(f"\n✓ Wrote {out_path} ({len(imd)} stations × {len(imd.columns)} cols)")


if __name__ == "__main__":
    main()
