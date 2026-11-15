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


# ---------------------------------------------------------------- modelling
MODELS = ["pooled", "calendar_month", "power_law"]
FORWARD_TRAIN_END = "2022-12"


def tile_month_table(grid, region, cal):
    """Zero-filled tile x month table of MODIS / VIIRS counts for one region."""
    a, b, c, d = REGIONS[region]
    clat, clon = grid["cell_lat"] + GRID / 2, grid["cell_lon"] + GRID / 2
    g = grid[clat.between(a, b) & clon.between(c, d)]
    g = g.assign(tile_lat=np.floor(g["cell_lat"] / TILE) * TILE,
                 tile_lon=np.floor(g["cell_lon"] / TILE) * TILE)
    w = g.pivot_table(index=["tile_lat", "tile_lon", "month"], columns="sensor",
                      values="n", aggfunc="sum", fill_value=0)
    for s in ("MODIS", "VIIRS"):
        if s not in w:
            w[s] = 0
    months = [str(p) for p in pd.period_range(cal["first_month"], cal["last_month"], freq="M")]
    tiles = w.index.droplevel("month").unique()
    idx = pd.MultiIndex.from_tuples([(tl, to, m) for tl, to in tiles for m in months],
                                    names=["tile_lat", "tile_lon", "month"])
    w = w.reindex(idx, fill_value=0).reset_index()
    w.columns.name = None
    w["year"] = w["month"].str[:4].astype(int)
    w["cm"] = w["month"].str[5:].astype(int)
    return w


def fit_power(d):
    """MODIS = a * VIIRS^b, least squares in log space on months with both > 0."""
    d = d[(d["MODIS"] > 0) & (d["VIIRS"] > 0)]
    if len(d) < 12 or d["VIIRS"].nunique() < 3:
        return None
    b, loga = np.polyfit(np.log(d["VIIRS"]), np.log(d["MODIS"]), 1)
    return float(np.exp(loga)), float(b)


def fit_models(train):
    """Fit all three models on tile x month rows.

    Hierarchy: tile x calendar-month factor (>= MIN_CALMONTH_MODIS MODIS hotspots)
    -> tile pooled factor (>= MIN_TILE_MODIS) -> parent region pooled factor.
    Power law: per tile if the tile qualifies, else the region's power law.
    """
    M, V = train["MODIS"].sum(), train["VIIRS"].sum()
    rp = M / V if V > 0 else np.nan
    rpow = fit_power(train) or (rp, 1.0)
    P = dict(region_pooled=rp, region_power=rpow, tiles={})
    for (tl, to), t in train.groupby(["tile_lat", "tile_lon"]):
        tm, tv = t["MODIS"].sum(), t["VIIRS"].sum()
        own = bool(tm >= MIN_TILE_MODIS and tv > 0)
        pooled = tm / tv if own else rp
        cs = t.groupby("cm")[["MODIS", "VIIRS"]].sum()
        cal = {int(c): (r["MODIS"] / r["VIIRS"]
                        if r["MODIS"] >= MIN_CALMONTH_MODIS and r["VIIRS"] > 0 else pooled)
               for c, r in cs.iterrows()}
        power = (fit_power(t) if own else None) or rpow
        P["tiles"][(float(tl), float(to))] = dict(own=own, pooled=pooled, cal=cal,
                                                  power=power, modis=int(tm), viirs=int(tv))
    return P


def predict(P, df):
    """MODIS-equivalent prediction per tile x month row, for every model."""
    out = pd.DataFrame(0.0, index=df.index, columns=MODELS)
    for (tl, to), t in df.groupby(["tile_lat", "tile_lon"]):
        tp = P["tiles"].get((float(tl), float(to))) or dict(
            pooled=P["region_pooled"], cal={}, power=P["region_power"])
        v = t["VIIRS"].to_numpy(float)
        calf = t["cm"].map(lambda c: tp["cal"].get(int(c), tp["pooled"])).to_numpy(float)
        a, b = tp["power"]
        out.loc[t.index, "pooled"] = v * tp["pooled"]
        out.loc[t.index, "calendar_month"] = v * calf
        out.loc[t.index, "power_law"] = np.where(v > 0, a * np.power(np.maximum(v, 1e-9), b), 0.0)
    return out


def _corr(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and np.std(x) > 0 and np.std(y) > 0 else None


def metrics(d, col):
    """Errors of predicted vs actual MODIS, summed over the rows' tiles."""
    m = d.groupby("month")[["MODIS", col]].sum()
    big = m[m["MODIS"] >= MIN_MAPE_COUNT]
    ape = (big[col] - big["MODIS"]).abs() / big["MODIS"]
    y = d.groupby("year")[["MODIS", col]].sum()
    y = y[y["MODIS"] > 0]
    yerr = 100 * (y[col] - y["MODIS"]) / y["MODIS"]
    tot = m["MODIS"].sum()
    return dict(
        mdape_pct=round(float(100 * ape.median()), 1) if len(big) else None,
        n_months=int(len(big)), n_months_total=int(len(m)),
        wape_pct=round(float(100 * (m[col] - m["MODIS"]).abs().sum() / tot), 1) if tot else None,
        bias_pct=round(float(100 * (m[col].sum() / tot - 1)), 1) if tot else None,
        r=_corr(m[col], m["MODIS"]),
        r_log=_corr(np.log1p(m[col]), np.log1p(m["MODIS"])),
        annual_abs_err_median_pct=round(float(yerr.abs().median()), 1) if len(y) else None,
        annual_abs_err_max_pct=round(float(yerr.abs().max()), 1) if len(y) else None,
        annual_err_pct={int(k): round(float(v), 1) for k, v in yerr.items()},
    )


def evaluate(pred):
    """Region-level and per-tile metrics for each model."""
    res = dict(region={m: metrics(pred, m) for m in MODELS}, tiles=[])
    for (tl, to), t in pred.groupby(["tile_lat", "tile_lon"]):
        row = dict(tile_lat=float(tl), tile_lon=float(to), modis=int(t["MODIS"].sum()),
                   viirs=int(t["VIIRS"].sum()))
        n_ok = int((t.groupby("month")["MODIS"].sum() >= MIN_MAPE_COUNT).sum())
        row["n_months"] = n_ok
        if n_ok >= 12:
            for m in MODELS:
                mm = metrics(t, m)
                row[m] = dict(mdape_pct=mm["mdape_pct"], wape_pct=mm["wape_pct"],
                              annual_abs_err_median_pct=mm["annual_abs_err_median_pct"])
        res["tiles"].append(row)
    scored = [t for t in res["tiles"] if "pooled" in t]
    res["tile_summary"] = dict(
        n_tiles=len(res["tiles"]), n_tiles_scored=len(scored),
        median_tile_mdape_pct={m: round(float(np.median([t[m]["mdape_pct"] for t in scored])), 1)
                               for m in MODELS} if scored else None)
    return res


def validate_region(tab, cal):
    fit = tab[(tab["month"] >= cal["overlap_start"]) & (tab["month"] <= cal["fit_end"])]
    # leave-one-year-out
    parts = []
    for y in sorted(fit["year"].unique()):
        test = fit[fit["year"] == y]
        parts.append(pd.concat([test, predict(fit_models(fit[fit["year"] != y]), test)], axis=1))
    loyo = evaluate(pd.concat(parts))
    # forward test: fit on the early years, predict the later ones
    out = dict(loyo=loyo, forward=None)
    train, test = fit[fit["month"] <= FORWARD_TRAIN_END], fit[fit["month"] > FORWARD_TRAIN_END]
    if len(train) and len(test) and train["VIIRS"].sum() > 0:
        fwd = evaluate(pd.concat([test, predict(fit_models(train), test)], axis=1))
        fwd["train"] = f"{cal['overlap_start']}..{FORWARD_TRAIN_END}"
        fwd["test"] = f"{test['month'].min()}..{cal['fit_end']}"
        out["forward"] = fwd
    return out


def full_years(months):
    y = pd.Series(months).str[:4].astype(int).value_counts()
    return sorted(int(k) for k, v in y.items() if v == 12)


def headline(series, region, cal, model):
    s = series.copy()
    s["year"] = s["month"].str[:4].astype(int)
    pre_y = full_years(s.loc[s["month"] < cal["overlap_start"], "month"])
    post_y = full_years(s.loc[(s["month"] >= cal["overlap_start"]) & (s["month"] <= cal["fit_end"]), "month"])
    if not pre_y or not post_y:
        return dict(region=region, note="not enough full years for a before/after comparison")
    ann = s.groupby("year")[["modis", "viirs", "harmonized"]].sum()
    pre, post = ann.loc[pre_y], ann.loc[post_y]
    h = dict(
        region=region, model=model,
        pre_years=f"{pre_y[0]}-{pre_y[-1]}", post_years=f"{post_y[0]}-{post_y[-1]}",
        modis_per_year_pre=round(float(pre["modis"].mean())),
        modis_per_year_post=round(float(post["modis"].mean())),
        viirs_per_year_post=round(float(post["viirs"].mean())),
        harmonized_per_year_post=round(float(post["harmonized"].mean())),
    )
    h["naive_change_x"] = round(h["viirs_per_year_post"] / h["modis_per_year_pre"], 2)
    h["modis_change_x"] = round(h["modis_per_year_post"] / h["modis_per_year_pre"], 2)
    h["harmonized_change_x"] = round(h["harmonized_per_year_post"] / h["modis_per_year_pre"], 2)
    h["naive_inflation_x"] = round(h["viirs_per_year_post"] / h["modis_per_year_post"], 2)
    h["median_year_modis_pre"] = round(float(pre["modis"].median()))
    h["median_year_modis_post"] = round(float(post["modis"].median()))
    h["median_year_viirs_post"] = round(float(post["viirs"].median()))
    h["sentence"] = (
        f"{region}: splicing raw VIIRS onto MODIS shows {h['naive_change_x']}x the hotspots per year in "
        f"{h['post_years']} compared with {h['pre_years']}; MODIS itself shows {h['modis_change_x']}x. "
        f"The naive splice overstates post-2018 activity by a factor of {h['naive_inflation_x']}; "
        f"the harmonized series shows {h['harmonized_change_x']}x.")
    return h


def step_model(grid, meta):
    cal = meta["calendar"]
    tabs = {r: tile_month_table(grid, r, cal) for r in REGIONS}
    tabs = {r: t for r, t in tabs.items() if len(t) and
            t.loc[(t["month"] >= cal["overlap_start"]) & (t["month"] <= cal["fit_end"]), "VIIRS"].sum() > 0}
    val = {r: validate_region(t, cal) for r, t in tabs.items()}

    # model selection: lowest leave-one-year-out MdAPE on region-month counts, averaged over regions
    score = {m: float(np.mean([val[r]["loyo"]["region"][m]["mdape_pct"] for r in val])) for m in MODELS}
    fscore = {m: float(np.mean([val[r]["forward"]["region"][m]["mdape_pct"] for r in val if val[r]["forward"]]))
              for m in MODELS}
    chosen = min(score, key=score.get)

    fac_parts, ser_parts, heads, params = [], [], [], {}
    for r, tab in tabs.items():
        fit = tab[(tab["month"] >= cal["overlap_start"]) & (tab["month"] <= cal["fit_end"])]
        P = fit_models(fit)
        params[r] = dict(
            region_pooled_factor=round(P["region_pooled"], 4),
            region_viirs_per_modis=round(1 / P["region_pooled"], 2),
            region_power_law=dict(a=round(P["region_power"][0], 4), b=round(P["region_power"][1], 3)),
            tiles=[dict(tile_lat=k[0], tile_lon=k[1], own_factor=v["own"], modis_overlap=v["modis"],
                        viirs_overlap=v["viirs"], pooled=round(v["pooled"], 4),
                        calendar_month={c: round(f, 4) for c, f in v["cal"].items()},
                        power_law=dict(a=round(v["power"][0], 4), b=round(v["power"][1], 3)))
                   for k, v in P["tiles"].items()])
        ov = tab[tab["month"] >= cal["overlap_start"]]
        pred = predict(P, ov)
        f = ov[["tile_lat", "tile_lon", "month", "MODIS", "VIIRS"]].copy()
        pooled_f = ov.apply(lambda x: P["tiles"][(x["tile_lat"], x["tile_lon"])]["pooled"], axis=1)
        for m in MODELS:   # effective factor; where VIIRS == 0 the factor is unused -> tile pooled
            f["f_" + m] = np.where(ov["VIIRS"] > 0, pred[m] / ov["VIIRS"].replace(0, np.nan), pooled_f)
        f["factor"] = f["f_" + chosen]
        f.insert(0, "region", r)
        fac_parts.append(f)

        s = tab.groupby("month")[["MODIS", "VIIRS"]].sum().rename(columns=str.lower)
        hp = pd.concat([ov[["month"]], pred], axis=1).groupby("month").sum()
        is_ov = s.index >= cal["overlap_start"]
        s["naive"] = np.where(is_ov, s["viirs"], s["modis"])
        for m in MODELS:
            s["harm_" + m] = np.where(is_ov, hp[m].reindex(s.index).fillna(0), s["modis"])
        s["harmonized"] = s["harm_" + chosen]
        s["in_overlap"] = is_ov
        s["in_fit"] = is_ov & (s.index <= cal["fit_end"])
        s["is_nrt"] = (s.index >= cal["nrt_start"]) if cal["nrt_start"] else False
        s["is_partial"] = s.index > cal["latest_full_month"]
        s = s.reset_index()
        s.insert(0, "region", r)
        ser_parts.append(s)
        heads.append(headline(s, r, cal, chosen))

    factors = pd.concat(fac_parts, ignore_index=True)
    for c in factors.columns:
        if c.startswith("f_") or c == "factor":
            factors[c] = factors[c].astype("float32")
    factors.to_parquet(os.path.join(AGG_DIR, "factors.parquet"), index=False, compression="zstd")
    series = pd.concat(ser_parts, ignore_index=True)
    series.to_parquet(os.path.join(AGG_DIR, "region_monthly.parquet"), index=False, compression="zstd")

    write_json("validation.json", dict(
        fit_period=f"{cal['overlap_start']}..{cal['fit_end']}",
        note="NRT months are excluded from fitting and validation. MdAPE uses months with "
             f">= {MIN_MAPE_COUNT} actual MODIS hotspots; WAPE uses all months.",
        thresholds=dict(min_calmonth_modis=MIN_CALMONTH_MODIS, min_tile_modis=MIN_TILE_MODIS,
                        tile_deg=TILE),
        models=MODELS, chosen_model=chosen,
        selection_rule="lowest leave-one-year-out MdAPE on region-month counts, mean over regions",
        selection_scores_loyo_mdape=score, forward_mdape=fscore,
        regions=val, fitted=params))
    write_json("headline.json", dict(model=chosen, regions=heads))

    # ------------------------------------------------------------ report
    print(f"\nFit period {cal['overlap_start']}..{cal['fit_end']} (NRT months excluded)")
    rows = []
    for r in val:
        for test in ("loyo", "forward"):
            if not val[r][test]:
                continue
            for m in MODELS:
                x = val[r][test]["region"][m]
                rows.append(dict(region=r, test=test, model=m, MdAPE=x["mdape_pct"], WAPE=x["wape_pct"],
                                 bias=x["bias_pct"], annual_med=x["annual_abs_err_median_pct"],
                                 annual_max=x["annual_abs_err_max_pct"], r=round(x["r"], 3),
                                 r_log=round(x["r_log"], 3), months=x["n_months"],
                                 tile_MdAPE=(val[r][test]["tile_summary"]["median_tile_mdape_pct"] or {}).get(m)))
    print("\n=== Validation, region-month level (percent) ===")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n=== Per-tile LOYO MdAPE ===")
    trows = []
    for r in val:
        own = {(t["tile_lat"], t["tile_lon"]): t["own_factor"] for t in params[r]["tiles"]}
        for t in val[r]["loyo"]["tiles"]:
            trows.append(dict(region=r, tile=f"{t['tile_lat']:.0f},{t['tile_lon']:.0f}", modis=t["modis"],
                              viirs=t["viirs"], own_factor=own.get((t["tile_lat"], t["tile_lon"])),
                              months=t["n_months"],
                              **{m: (t[m]["mdape_pct"] if m in t else None) for m in MODELS}))
    print(pd.DataFrame(trows).to_string(index=False))
    print(f"\nMean LOYO MdAPE: {({k: round(v, 1) for k, v in score.items()})}")
    print(f"Mean forward MdAPE: {({k: round(v, 1) for k, v in fscore.items()})}")
    print(f"Chosen model: {chosen}")
    print("\n=== Fitted (full fit period) ===")
    for r, p in params.items():
        print(f"{r}: pooled {p['region_viirs_per_modis']} VIIRS per MODIS; power law {p['region_power_law']}")
    print("\n=== Headline ===")
    for h in heads:
        print(json.dumps({k: v for k, v in h.items() if k != "sentence"}))
        print(h.get("sentence", h.get("note")))
    for fn in sorted(os.listdir(AGG_DIR)):
        print(f"  {fn}: {os.path.getsize(os.path.join(AGG_DIR, fn))/1e3:.0f} kB")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage == "agg":
        step_agg()
    else:
        step_model(*step_agg())
