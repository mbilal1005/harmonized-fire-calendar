"""Harmonized Fire Calendar - data pipeline.

Loads every FIRMS CSV in ./data/raw, filters, aggregates to a 0.5 deg grid x
month x sensor, derives VIIRS->MODIS conversion factors, validates them and
writes small outputs to ./data/agg.

Usage:  python pipeline.py [report|all]
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

RAW_DIR = "data/raw"
AGG_DIR = "data/agg"
GRID = 0.5
OVERLAP_START = pd.Period("2018-04", "M")
TODAY = pd.Timestamp.today().normalize()
LATEST_FULL_MONTH = (TODAY.to_period("M") - 1)

# Region = bounding box over grid cells (lat_min, lat_max, lon_min, lon_max).
# Data is aggregated region-independently, so adding a region is one line here.
REGIONS = {
    "California": (32.5, 42.1, -124.5, -114.0),
    "Greece": (34.5, 42.0, 19.0, 29.7),
}

DEDUP_KEYS = ["latitude", "longitude", "acq_date", "acq_time", "satellite"]


def detect_sensor(df, path):
    if "instrument" in df.columns and df["instrument"].notna().any():
        return str(df["instrument"].dropna().iloc[0]).upper()
    name = os.path.basename(path).upper()
    if "M-C61" in name or "MODIS" in name:
        return "MODIS"
    if "V-C2" in name or "VIIRS" in name or "SV-C2" in name:
        return "VIIRS"
    raise ValueError(f"Cannot identify sensor for {path}")


def load_raw(verbose=True):
    """Load + filter all CSVs. Returns (hotspots, per-file stats)."""
    frames, stats = [], []
    for path in sorted(glob.glob(os.path.join(RAW_DIR, "**", "*.csv"), recursive=True)):
        df = pd.read_csv(path, dtype={"confidence": str, "acq_time": str})
        sensor = detect_sensor(df, path)
        n0 = len(df)
        has_type = "type" in df.columns
        if has_type:
            df = df[df["type"] == 0]
        n_type = len(df)
        if sensor == "MODIS":
            conf = pd.to_numeric(df["confidence"], errors="coerce")
            df = df[conf >= 30]
        else:
            df = df[df["confidence"].str.lower().str[0] != "l"]
        df = df.assign(sensor=sensor, source=os.path.basename(path),
                       is_nrt="nrt" in os.path.basename(path).lower())
        stats.append(dict(file=os.path.basename(path), sensor=sensor, rows_raw=n0,
                          has_type=has_type, after_type=n_type, after_conf=len(df)))
        frames.append(df[["latitude", "longitude", "acq_date", "acq_time", "satellite",
                          "sensor", "confidence", "frp", "daynight", "source", "is_nrt"]])
    hs = pd.concat(frames, ignore_index=True)
    hs["acq_time"] = hs["acq_time"].str.zfill(4)
    # Archive wins over NRT on overlaps (archive is sorted first after this sort).
    n_before = len(hs)
    hs = hs.sort_values("is_nrt").drop_duplicates(DEDUP_KEYS, keep="first")
    hs["acq_date"] = pd.to_datetime(hs["acq_date"])
    hs["month"] = hs["acq_date"].dt.to_period("M")
    stats_df = pd.DataFrame(stats)
    stats_df.attrs["duplicates_removed"] = n_before - len(hs)
    return hs.reset_index(drop=True), stats_df


def assign_region_points(df, lat="latitude", lon="longitude"):
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    for name, (a, b, c, d) in REGIONS.items():
        m = df[lat].between(a, b) & df[lon].between(c, d) & out.isna()
        out[m] = name
    return out


def step1_report():
    hs, stats = load_raw()
    print("=== Rows per file ===")
    print(stats.to_string(index=False))
    print(f"Duplicates removed (archive/NRT overlap): {stats.attrs['duplicates_removed']}")
    hs["region"] = assign_region_points(hs)
    print(f"Hotspots kept: {len(hs)}  (unassigned to a region: {hs.region.isna().sum()})")
    m = hs.groupby(["region", "sensor", "month"]).size().rename("n").reset_index()
    print("\n=== Top 8 months per region x sensor ===")
    for (r, s), g in m.groupby(["region", "sensor"]):
        top = g.nlargest(8, "n")
        print(f"{r:11s} {s:5s}: " + ", ".join(f"{p}={n}" for p, n in zip(top.month, top.n)))
    print("\n=== Sanity-check months ===")
    for r, p in [("California", "2018-11"), ("California", "2020-08"), ("Greece", "2023-08")]:
        for s in ["MODIS", "VIIRS"]:
            g = m[(m.region == r) & (m.sensor == s)].set_index("month")["n"]
            v = g.get(pd.Period(p, "M"), 0)
            rank = int((g > v).sum()) + 1
            print(f"{r:11s} {p} {s}: {v:6d} hotspots, rank {rank}/{len(g)} months, "
                  f"median month {int(g.median())}")
    wide = m.pivot_table(index=["region", "month"], columns="sensor", values="n", fill_value=0)
    os.makedirs("data/tmp", exist_ok=True)
    wide.to_csv("data/tmp/monthly_counts.csv")
    print("\nFull monthly table written to data/tmp/monthly_counts.csv")
    return hs


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "report"
    if stage == "report":
        step1_report()
