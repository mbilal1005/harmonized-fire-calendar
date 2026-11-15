"""Harmonized Fire Calendar - data pipeline.

Loads every FIRMS CSV under ./data/raw, filters, aggregates to a 0.5 deg grid x
month x sensor, derives VIIRS -> MODIS-equivalent conversion factors per
5 deg tile, validates them and writes small outputs to ./data/agg.

Regions are bounding boxes in regions.json [lat_min, lat_max, lon_min, lon_max];
adding a region (e.g. USA, Europe) needs no code change.

Usage:  python pipeline.py [report|agg|all]
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

RAW_DIR = "data/raw"
AGG_DIR = "data/agg"
GRID = 0.5            # deg, aggregation cell
TILE = 5.0            # deg, conversion-factor tile
OVERLAP_START = "2018-04"
MIN_CALMONTH_MODIS = 50    # min MODIS hotspots for a tile x calendar-month factor
MIN_TILE_MODIS = 200       # min MODIS hotspots in overlap for a tile's own factor
MIN_MAPE_COUNT = 10        # months with fewer actual MODIS hotspots are skipped in MdAPE
MASK_DEG = 0.01            # NRT static-source mask resolution

with open("regions.json") as fh:
    REGIONS = {k: tuple(v) for k, v in json.load(fh).items()}

DEDUP_KEYS = ["latitude", "longitude", "acq_date", "acq_time", "satellite"]


# ----------------------------------------------------------------- loading
def detect_sensor(df, path):
    if "instrument" in df.columns and df["instrument"].notna().any():
        return str(df["instrument"].dropna().iloc[0]).upper()
    name = os.path.basename(path).upper()
    if "M-C61" in name or "MODIS" in name:
        return "MODIS"
    if "V-C2" in name or "VIIRS" in name:
        return "VIIRS"
    raise ValueError(f"Cannot identify sensor for {path}")


def region_of(lat, lon):
    """First matching region name for point/cell coordinates (for reporting)."""
    out = pd.Series("(none)", index=lat.index, dtype="object")
    for name, (a, b, c, d) in reversed(list(REGIONS.items())):
        out[lat.between(a, b) & lon.between(c, d)] = name
    return out


def load_raw():
    """Load + filter all CSVs.

    Filters: type == 0 where the column exists (archive); for files without a
    type column (NRT) drop points at 0.01 deg locations where any archive file
    has type != 0 (static sources: industry, volcanoes, offshore);
    MODIS confidence >= 30; VIIRS confidence != low.
    Returns (hotspots, per-file stats, NRT mask report).
    """
    frames, stats, static = [], [], []
    for path in sorted(glob.glob(os.path.join(RAW_DIR, "**", "*.csv"), recursive=True)):
        df = pd.read_csv(path, dtype={"confidence": str, "acq_time": str})
        sensor = detect_sensor(df, path)
        has_type = "type" in df.columns
        df["sensor"] = sensor
        df["source"] = os.path.basename(path)
        df["is_nrt"] = not has_type
        df["ky"] = np.round(df["latitude"] / MASK_DEG).astype("int64")
        df["kx"] = np.round(df["longitude"] / MASK_DEG).astype("int64")
        n0 = len(df)
        if has_type:
            static.append(df.loc[df["type"] != 0, ["ky", "kx"]].drop_duplicates())
            df = df[df["type"] == 0]
        stats.append(dict(file=os.path.basename(path), sensor=sensor, rows_raw=n0,
                          has_type=has_type, after_type=len(df)))
        frames.append(df[["latitude", "longitude", "acq_date", "acq_time", "satellite",
                          "sensor", "confidence", "frp", "daynight", "source", "is_nrt",
                          "ky", "kx"]])
    hs = pd.concat(frames, ignore_index=True)

    # NRT mask
    mask_report = []
    if static:
        static = pd.concat(static).drop_duplicates().assign(static=True)
        hs = hs.merge(static, on=["ky", "kx"], how="left")
        hs["static"] = hs["static"].notna()
        nrt = hs[hs["is_nrt"]]
        if len(nrt):
            rep = nrt.assign(region=region_of(nrt["latitude"], nrt["longitude"])) \
                .groupby(["region", "sensor"])["static"].agg(nrt_points="size", masked="sum")
            rep["masked_pct"] = (100 * rep["masked"] / rep["nrt_points"]).round(1)
            mask_report = rep.reset_index().to_dict("records")
        hs = hs[~(hs["is_nrt"] & hs["static"])]
    after_mask = hs.groupby("source").size()

    # confidence
    is_modis = hs["sensor"] == "MODIS"
    conf_num = pd.to_numeric(hs["confidence"], errors="coerce")
    keep = np.where(is_modis, conf_num >= 30, hs["confidence"].str.lower().str[0] != "l")
    hs = hs[keep]
    after_conf = hs.groupby("source").size()

    # dedup: archive wins over NRT
    n_before = len(hs)
    hs = hs.sort_values("is_nrt", kind="stable").drop_duplicates(DEDUP_KEYS, keep="first")
    hs["acq_time"] = hs["acq_time"].str.zfill(4)
    hs["acq_date"] = pd.to_datetime(hs["acq_date"])
    hs["month"] = hs["acq_date"].dt.strftime("%Y-%m")

    stats = pd.DataFrame(stats)
    stats["after_nrt_mask"] = stats["file"].map(after_mask).fillna(0).astype(int)
    stats["after_conf"] = stats["file"].map(after_conf).fillna(0).astype(int)
    stats.attrs["duplicates_removed"] = n_before - len(hs)
    return hs.drop(columns=["ky", "kx", "static"], errors="ignore").reset_index(drop=True), \
        stats, mask_report


def data_calendar(hs):
    """Key months derived from the data itself (no hard-coded dates)."""
    last_day = hs["acq_date"].max()
    last_month = last_day.to_period("M")
    full = last_month if last_day == last_month.end_time.normalize() else last_month - 1
    nrt = hs.loc[hs["is_nrt"], "month"]
    nrt_start = nrt.min() if len(nrt) else None
    # factors are fitted / validated on archive-quality months only
    fit_end = str(pd.Period(nrt_start, "M") - 1) if nrt_start else str(full)
    return dict(first_month=hs["month"].min(), last_month=str(last_month),
                last_data_day=str(last_day.date()), latest_full_month=str(full),
                overlap_start=OVERLAP_START, nrt_start=nrt_start, fit_end=fit_end)


# ------------------------------------------------------------- aggregation
def aggregate(hs):
    g = hs.assign(
        cell_lat=np.floor(hs["latitude"] / GRID) * GRID,
        cell_lon=np.floor(hs["longitude"] / GRID) * GRID,
        night=(hs["daynight"] == "N").astype("int32"),
    ).groupby(["cell_lat", "cell_lon", "month", "sensor"], as_index=False).agg(
        n=("frp", "size"), n_night=("night", "sum"), frp_sum=("frp", "sum"))
    g["n"] = g["n"].astype("int32")
    g["frp_sum"] = g["frp_sum"].round(1).astype("float32")
    g["cell_lat"] = g["cell_lat"].astype("float32")
    g["cell_lon"] = g["cell_lon"].astype("float32")
    return g


def write_json(name, obj):
    os.makedirs(AGG_DIR, exist_ok=True)
    with open(os.path.join(AGG_DIR, name), "w") as fh:
        json.dump(obj, fh, indent=2, default=float)


def step_agg():
    hs, stats, mask_report = load_raw()
    cal = data_calendar(hs)
    print("=== Rows per file ===")
    print(stats.to_string(index=False))
    print(f"Duplicates removed: {stats.attrs['duplicates_removed']}   kept: {len(hs)}")
    print("\n=== NRT mask (points at 0.01 deg locations with archive type != 0) ===")
    print(pd.DataFrame(mask_report).to_string(index=False))
    print("\n=== Calendar ===")
    print(cal)
    grid = aggregate(hs)
    os.makedirs(AGG_DIR, exist_ok=True)
    path = os.path.join(AGG_DIR, "grid_monthly.parquet")
    grid.to_parquet(path, index=False, compression="zstd")
    none = (region_of(grid["cell_lat"] + GRID / 2, grid["cell_lon"] + GRID / 2) == "(none)")
    print(f"\ngrid_monthly.parquet: {len(grid)} rows, {os.path.getsize(path)/1e3:.0f} kB, "
          f"{grid[['cell_lat','cell_lon']].drop_duplicates().shape[0]} cells; "
          f"hotspots outside all regions: {int(grid.loc[none, 'n'].sum())}")
    meta = dict(calendar=cal, regions=REGIONS, grid_deg=GRID, tile_deg=TILE,
                filters=dict(type="== 0 (archive)", modis_confidence=">= 30",
                             viirs_confidence="nominal or high",
                             nrt="masked at 0.01 deg locations where archive type != 0"),
                rows_per_file=stats.to_dict("records"), nrt_mask=mask_report,
                hotspots_kept=int(len(hs)))
    write_json("meta.json", meta)
    return grid, meta


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage == "agg":
        step_agg()
