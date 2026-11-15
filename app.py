"""Harmonized Fire Calendar - Streamlit app.

Reads only the small aggregates in ./data/agg (built by pipeline.py).
Run:  streamlit run app.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

AGG = "data/agg"
RAW = "data/raw"
C_HARM, C_NAIVE, C_MODIS = "#2a78d6", "#eb6834", "#199e70"   # validated colorblind-safe trio, >= 3:1 on white
INK, MUTED, GRIDC = "#1f1f1e", "#6b6a63", "#e6e5df"
MIN_ANOMALY_COUNT = 10
# Known large fires, labelled on the main chart (month of peak hotspot activity).
EVENTS = {
    "Greece": [("2007-08", "Peloponnese 2007"), ("2023-08", "Evros 2023")],
    "California": [("2020-08", "August Complex 2020"), ("2021-08", "Dixie 2021")],
}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

st.set_page_config(page_title="Harmonized Fire Calendar", page_icon="🔥", layout="wide")


# ------------------------------------------------------------------ data
@st.cache_data
def load_agg():
    grid = pd.read_parquet(os.path.join(AGG, "grid_monthly.parquet"))
    mtd = pd.read_parquet(os.path.join(AGG, "mtd_monthly.parquet"))
    factors = pd.read_parquet(os.path.join(AGG, "factors.parquet"))
    out = {}
    for name in ("meta", "validation", "headline"):
        with open(os.path.join(AGG, f"{name}.json")) as fh:
            out[name] = json.load(fh)
    return grid, mtd, factors, out["meta"], out["validation"], out["headline"]


grid, mtd_grid, factors, meta, val, head = load_agg()
cal = meta["calendar"]
REGIONS = {k: tuple(v) for k, v in meta["regions"].items()}
GRID, TILE = meta["grid_deg"], meta["tile_deg"]
_fit = factors[factors["month"] <= cal["fit_end"]]
GLOBAL_POOLED = float(_fit["MODIS"].sum() / _fit["VIIRS"].sum())
ALL_MONTHS = [str(p) for p in pd.period_range(cal["first_month"], cal["last_month"], freq="M")]


def regions_for_box(box):
    """Preset regions whose box overlaps the selection."""
    a, b, c, d = box
    return [n for n, (ra, rb, rc, rd) in REGIONS.items() if a < rb and b > ra and c < rd and d > rc]


def _convert(g, box, prefer):
    """Per-month MODIS, VIIRS and converted-VIIRS sums of a cell table inside a box.

    Each VIIRS cell-month is converted with the factor of its 5 deg tile from the
    preset region that contains the cell; cells outside every preset region use
    the pooled factor over all regions.
    """
    a, b, c, d = box
    clat, clon = g["cell_lat"] + GRID / 2, g["cell_lon"] + GRID / 2
    g = g[clat.between(a, b) & clon.between(c, d)]
    modis = g[g["sensor"] == "MODIS"].groupby("month")["n"].sum()
    v = g[g["sensor"] == "VIIRS"].copy()
    v["clat"], v["clon"] = v["cell_lat"] + GRID / 2, v["cell_lon"] + GRID / 2
    v["tile_lat"] = np.floor(v["cell_lat"].astype(float) / TILE) * TILE
    v["tile_lon"] = np.floor(v["cell_lon"].astype(float) / TILE) * TILE
    v["factor"] = np.nan
    order = ([prefer] if prefer in REGIONS else []) + [r for r in REGIONS if r != prefer]
    for name in order:
        ra, rb, rc, rd = REGIONS[name]
        m = v["factor"].isna() & v["clat"].between(ra, rb) & v["clon"].between(rc, rd)
        if m.any():
            f = factors.loc[factors["region"] == name, ["tile_lat", "tile_lon", "month", "factor"]]
            got = v.loc[m, ["tile_lat", "tile_lon", "month"]].merge(
                f, how="left", on=["tile_lat", "tile_lon", "month"])
            v.loc[m, "factor"] = got["factor"].to_numpy()
    fallback = v["factor"].isna()
    fallback_share = float(v.loc[fallback, "n"].sum() / v["n"].sum()) if v["n"].sum() else 0.0
    v["conv"] = v["n"] * v["factor"].fillna(GLOBAL_POOLED)
    return modis, v.groupby("month")["n"].sum(), v.groupby("month")["conv"].sum(), fallback_share


@st.cache_data
def month_to_date(box, prefer=None):
    """Harmonized hotspots for the current month's day range (e.g. 1-15 Nov), per year."""
    modis, _, conv, _ = _convert(mtd_grid, box, prefer)
    cm = cal["last_month"][5:]
    idx = [f"{y}-{cm}" for y in range(int(cal["first_month"][:4]), int(cal["last_month"][:4]) + 1)]
    idx = [m for m in idx if m >= cal["first_month"]]
    out = pd.Series([float(conv.get(m, 0.0)) if m >= cal["overlap_start"] else float(modis.get(m, 0))
                     for m in idx], index=idx)
    return out


@st.cache_data
def box_series(box, prefer=None):
    """Monthly MODIS / VIIRS / naive / harmonized counts for a lat-lon box."""
    modis, viirs, conv, fallback_share = _convert(grid, box, prefer)
    s = pd.DataFrame(index=pd.Index(ALL_MONTHS, name="month"))
    s["modis"] = modis.reindex(s.index).fillna(0)
    s["viirs"] = viirs.reindex(s.index).fillna(0)
    conv = conv.reindex(s.index).fillna(0)
    ov = s.index >= cal["overlap_start"]
    s["naive"] = np.where(ov, s["viirs"], s["modis"])
    s["harmonized"] = np.where(ov, conv, s["modis"])
    s["in_overlap"] = ov
    s["is_nrt"] = (s.index >= cal["nrt_start"]) if cal["nrt_start"] else False
    s["is_partial"] = s.index > cal["latest_full_month"]
    s["year"] = s.index.str[:4].astype(int)
    s["cm"] = s.index.str[5:].astype(int)
    return s, fallback_share


def anomalies(s, col, log=True):
    """Month > median + 2 x scaled MAD of the same calendar month (full months only).

    Computed on log1p(counts) by default: hotspot counts are strongly right-skewed,
    so on raw counts a quiet calendar month has a tiny MAD and flags too easily.
    """
    x = np.log1p(s[col]) if log else s[col]
    full = x[~s["is_partial"]]
    cm = s.loc[full.index, "cm"]
    med = full.groupby(cm).median()
    mad = full.groupby(cm).apply(lambda v: (v - v.median()).abs().median()) * 1.4826
    thr = (med + 2 * mad).reindex(s["cm"]).to_numpy()
    return (x > thr) & (s[col] >= MIN_ANOMALY_COUNT) & ~s["is_partial"], med


def full_years(months):
    y = pd.Series(months).str[:4].astype(int).value_counts()
    return sorted(int(k) for k, n in y.items() if n == 12)


def headline_numbers(s):
    pre_y = full_years(s.index[s.index < cal["overlap_start"]])
    post_y = full_years(s.index[(s.index >= cal["overlap_start"]) & (s.index <= cal["fit_end"])])
    ann = s.groupby("year")[["modis", "viirs", "harmonized"]].sum()
    pre, post = ann.loc[pre_y], ann.loc[post_y]
    out = dict(pre=f"{pre_y[0]}–{pre_y[-1]}", post=f"{post_y[0]}–{post_y[-1]}")
    for name, fn in (("mean", "mean"), ("median", "median")):
        base = getattr(pre["modis"], fn)()
        out[name] = dict(
            modis_pre=base, modis_post=getattr(post["modis"], fn)(),
            viirs_post=getattr(post["viirs"], fn)(), harm_post=getattr(post["harmonized"], fn)())
        for k in ("modis_post", "viirs_post", "harm_post"):
            out[name][k + "_x"] = out[name][k] / base if base > 0 else np.nan
    out["inflation"] = (post["viirs"].sum() / post["modis"].sum()) if post["modis"].sum() > 0 else np.nan
    return out


def fmt_x(x):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.2f}×"


def base_layout(fig, height=380, **kw):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff", font=dict(color=INK, size=13), hovermode="x unified",
        legend=dict(orientation="h", y=1.08, x=0, font=dict(color=INK)), **kw)
    fig.update_xaxes(showgrid=False, linecolor=GRIDC, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRIDC, zeroline=False, tickfont=dict(color=MUTED), rangemode="tozero")
    return fig


# --------------------------------------------------------------- sidebar
st.sidebar.header("Area")
presets = list(REGIONS)
default = presets.index("Greece") if "Greece" in presets else 0
choice = st.sidebar.selectbox("Region", presets + ["Custom box"], index=default)
if choice == "Custom box":
    ra, rb, rc, rd = REGIONS[presets[default]]
    c1, c2 = st.sidebar.columns(2)
    lat0 = c1.number_input("Lat min", -90.0, 90.0, 37.5, 0.5)
    lat1 = c2.number_input("Lat max", -90.0, 90.0, 39.0, 0.5)
    lon0 = c1.number_input("Lon min", -180.0, 180.0, 22.5, 0.5)
    lon1 = c2.number_input("Lon max", -180.0, 180.0, 24.5, 0.5)
    if lat1 <= lat0 or lon1 <= lon0:
        st.sidebar.error("Max must be larger than min.")
        st.stop()
    box, prefer = (lat0, lat1, lon0, lon1), None
    st.sidebar.caption(f"Data are on a {GRID}° grid: a cell counts if its centre is inside the box.")
else:
    box, prefer = REGIONS[choice], choice

y_min, y_max = int(cal["first_month"][:4]), int(cal["last_month"][:4])
st.sidebar.header("Period")
yr = st.sidebar.slider("Years", y_min, y_max, (y_min, y_max))

s_all, fallback_share = box_series(box, prefer)
used_regions = [choice] if prefer else regions_for_box(box)
title_area = choice if prefer else f"box {box[0]:g}–{box[1]:g}°N, {box[2]:g}–{box[3]:g}°E"

# ----------------------------------------------------------------- header
st.title("Harmonized Fire Calendar")
st.markdown("**More fires, or just a better satellite?** One consistent burning-activity record "
            "from MODIS (1 km, 2000–) and VIIRS NOAA-20 (375 m, 2018–) active-fire hotspots.")

if s_all[["modis", "viirs"]].to_numpy().sum() == 0:
    st.warning("No hotspots in this area. Data are currently loaded for: " + ", ".join(presets) + ".")
    st.stop()
if not prefer and fallback_share > 0:
    st.info(f"{fallback_share:.0%} of VIIRS hotspots in this box lie outside the preset regions and use "
            f"the pooled conversion factor ({1/GLOBAL_POOLED:.2f} VIIRS per MODIS hotspot).")

# ---------------------------------------------------------------- metrics
h = headline_numbers(s_all)
cur = s_all.iloc[-1]
cur_month = s_all.index[-1]
mtd = month_to_date(box, prefer)          # same day range in every year
cur_mtd, hist = mtd.iloc[-1], mtd.iloc[:-1]
lastfull = s_all.loc[cal["latest_full_month"]]
hist_full = s_all[(s_all["cm"] == lastfull["cm"]) & (s_all.index < cal["latest_full_month"])]["harmonized"]


def delta_vs(v, med):
    return f"{100 * (v / med - 1):+.0f}% vs median {med:,.0f}" if med > 0 else f"median {med:,.0f}"


m3, m4, m1, m2 = st.columns(4)   # story first: headline metrics, then current month
day = pd.Timestamp(cal["last_data_day"])
rng = f"1–{day.day} {MONTHS[day.month-1]}"
m1.metric(f"{rng} {int(cur['year'])} (month so far)",
          f"{cur_mtd:,.0f}", delta_vs(cur_mtd, hist.median()), delta_color="inverse",
          help=f"Harmonized (MODIS-equivalent) hotspots for {rng}, against the median of the same days "
               f"({rng}) in {hist.index[0][:4]}–{hist.index[-1][:4]}. The latest day may be incomplete.")
m2.metric(f"{MONTHS[int(lastfull['cm'])-1]} {int(lastfull['year'])} (last full month)",
          f"{lastfull['harmonized']:,.0f}", delta_vs(lastfull["harmonized"], hist_full.median()),
          delta_color="inverse", help="Harmonized hotspots against the median for that calendar month.")
m3.metric("Naive splice overstates by", fmt_x(h["inflation"]),
          help=f"Raw VIIRS hotspots divided by actual MODIS hotspots, {h['post']}.")
m4.metric(f"Actual MODIS change, {h['post']} vs {h['pre']}", fmt_x(h["mean"]["modis_post_x"]),
          help="Hotspots per year on average. See the table below for the median year.")

# -------------------------------------------------------- season briefing
def peak_run(medcal, cover=0.75):
    """Shortest run of consecutive months (may wrap across December) holding >= cover
    of the typical year (sum of calendar-month medians). Ties: the run with the larger share."""
    v = np.array([float(medcal.get(m, 0.0)) for m in range(1, 13)])
    for length in range(1, 13):
        shares = [(v[[(st0 + k) % 12 for k in range(length)]].sum() / v.sum(), st0) for st0 in range(12)]
        best, st0 = max(shares)
        if best >= cover:
            return [(st0 + k) % 12 + 1 for k in range(length)], best


def season_briefing(s, h):
    """Template text filled from computed numbers only."""
    full = s[~s["is_partial"]]
    medcal = full.groupby("cm")["harmonized"].median()
    if medcal.sum() <= 0:
        return None
    peak, share = peak_run(medcal)
    span = MONTHS[peak[0] - 1] if len(peak) == 1 else f"{MONTHS[peak[0] - 1]}–{MONTHS[peak[-1] - 1]}"
    wraps = peak[-1] < peak[0]
    f = full[full["cm"].isin(peak)].copy()
    # a season that wraps across December belongs to the year it starts in
    f["season"] = np.where(wraps & (f["cm"] < peak[0]), f["year"] - 1, f["year"])
    g = f.groupby("season")
    seas = g["harmonized"].sum()[g.size() == len(peak)]
    lines = [f"- **Peak months: {span}.** In a typical year they hold {share:.0%} of the year's hotspots "
             "(the shortest run of consecutive months covering at least 75% of the calendar-month medians)."]
    if len(seas) >= 3:
        ly, lv, hist = int(seas.index[-1]), float(seas.iloc[-1]), seas.iloc[:-1]
        med = float(hist.median())
        rank = int((seas > lv).sum()) + 1
        rel = (f"{abs(100 * (lv / med - 1)):.0f}% {'above' if lv >= med else 'below'} the median season "
               f"({med:,.0f}, {int(hist.index[0])}–{int(hist.index[-1])})") if med > 0 else "no median to compare"
        last = f[f["season"] == ly]
        caveat = bool(last["is_nrt"].any() or (last.index > cal["fit_end"]).any())
        label = f"{span} {ly}" if not wraps else f"{MONTHS[peak[0] - 1]} {ly}–{MONTHS[peak[-1] - 1]} {ly + 1}"
        lines.append(f"- **Last complete season ({label}): {lv:,.0f} hotspots**, {rel}. "
                     f"It ranks {rank} of {len(seas)} seasons."
                     + (" (includes near-real-time data; see 2026 note)" if caveat else ""))
    md = h["median"]
    if np.isfinite(md["viirs_post_x"]):
        lines.append(f"- **Trap to avoid:** comparing raw VIIRS counts with pre-2018 MODIS counts makes the "
                     f"median year of {h['post']} look like {md['viirs_post_x']:.2f}× the {h['pre']} level. "
                     f"Like for like it is {md['harm_post_x']:.2f}× (actual MODIS: {md['modis_post_x']:.2f}×).")
    return "\n".join(lines)


brief = season_briefing(s_all, h)
if brief:
    with st.container(border=True):
        st.markdown(f"**Season briefing for {title_area}** · for planning seasonal staffing · "
                    "harmonized (MODIS-equivalent) hotspots\n\n" + brief)

# ------------------------------------------------- naive vs harmonized chart
st.subheader(f"Naive splice vs harmonized vs actual MODIS — {title_area}")
view = st.radio("View", ["Yearly totals", "Monthly"], horizontal=True, label_visibility="collapsed")
s = s_all[(s_all["year"] >= yr[0]) & (s_all["year"] <= yr[1])]

fig = go.Figure()
if view == "Yearly totals":
    fy = set(full_years(s_all.index))
    a = s[s["year"].isin(fy)].groupby("year")[["modis", "naive", "harmonized"]].sum()
    start = int(cal["overlap_start"][:4]) - 1
    x_ov = a.index[a.index >= start]
    fig.add_trace(go.Scatter(x=x_ov, y=a.loc[x_ov, "naive"], name="Naive splice (raw VIIRS after 2018-04)",
                             mode="lines+markers", line=dict(color=C_NAIVE, width=2, dash="dash"),
                             marker=dict(size=8, symbol="diamond"), hovertemplate="%{y:,.0f}"))
    fig.add_trace(go.Scatter(x=a.index, y=a["harmonized"], name="Harmonized (MODIS-equivalent)",
                             mode="lines+markers", line=dict(color=C_HARM, width=2),
                             marker=dict(size=8), hovertemplate="%{y:,.0f}"))
    fig.add_trace(go.Scatter(x=x_ov, y=a.loc[x_ov, "modis"], name="Actual MODIS",
                             mode="lines+markers", line=dict(color=C_MODIS, width=2, dash="dot"),
                             marker=dict(size=8, symbol="square"), hovertemplate="%{y:,.0f}"))
    xline, table = int(cal["overlap_start"][:4]) - 0.25, a
    fig.update_xaxes(dtick=2)
else:
    x = pd.to_datetime(s.index)
    ov = s["in_overlap"].to_numpy()
    fig.add_trace(go.Scatter(x=x[ov], y=s.loc[ov, "naive"], name="Naive splice (raw VIIRS after 2018-04)",
                             mode="lines", line=dict(color=C_NAIVE, width=2, dash="dash"),
                             hovertemplate="%{y:,.0f}"))
    fig.add_trace(go.Scatter(x=x, y=s["harmonized"], name="Harmonized (MODIS-equivalent)",
                             mode="lines", line=dict(color=C_HARM, width=2), hovertemplate="%{y:,.0f}"))
    fig.add_trace(go.Scatter(x=x[ov], y=s.loc[ov, "modis"], name="Actual MODIS",
                             mode="lines", line=dict(color=C_MODIS, width=2, dash="dot"),
                             hovertemplate="%{y:,.0f}"))
    xline, table = pd.Timestamp(cal["overlap_start"] + "-01"), s[["modis", "viirs", "naive", "harmonized"]]
if yr[0] <= int(cal["overlap_start"][:4]) <= yr[1]:
    fig.add_vline(x=xline, line=dict(color=MUTED, width=1, dash="dot"))
    fig.add_annotation(x=xline, y=1, yref="paper", text="VIIRS NOAA-20 starts", showarrow=False,
                       xanchor="right", xshift=-6, font=dict(color=MUTED, size=12))
for k, (ev_month, ev_label) in enumerate(EVENTS.get(prefer, [])):
    ev_x = int(ev_month[:4]) if view == "Yearly totals" else ev_month
    if ev_x not in table.index:
        continue
    ev_y = float(table.loc[ev_x, ["naive", "harmonized", "modis"]].max())
    fig.add_annotation(x=ev_x if view == "Yearly totals" else pd.Timestamp(ev_month + "-01"), y=ev_y,
                       text=ev_label, showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor=MUTED,
                       ax=-45 if k % 2 == 0 else 45, ay=-28, font=dict(color=INK, size=12),
                       bgcolor="rgba(255,255,255,0.85)")
base_layout(fig, 420, yaxis_title="Hotspots")
fig.update_layout(margin=dict(t=60))
st.plotly_chart(fig, width="stretch")
st.caption("Before April 2018 all three lines are the same MODIS record. Yearly totals show complete "
           "years only. The gap between the dashed and solid lines is the sensor change, not fire.")
with st.expander("Show these numbers as a table"):
    st.dataframe(table.round(0), width="stretch")

# -------------------------------------------------- headline comparison
st.markdown(f"**Change in hotspots per year, {h['post']} compared with {h['pre']}**")
cmp = pd.DataFrame({
    "Series": ["Naive splice (raw VIIRS vs earlier MODIS)", "Actual MODIS", "Harmonized"],
    "Per year on average": [fmt_x(h["mean"][k]) for k in ("viirs_post_x", "modis_post_x", "harm_post_x")],
    "Median year": [fmt_x(h["median"][k]) for k in ("viirs_post_x", "modis_post_x", "harm_post_x")],
    f"Hotspots per year {h['post']} (mean)": [f"{h['mean'][k]:,.0f}" for k in ("viirs_post", "modis_post", "harm_post")],
    f"Hotspots per year {h['post']} (median)": [f"{h['median'][k]:,.0f}" for k in ("viirs_post", "modis_post", "harm_post")],
})
st.dataframe(cmp, hide_index=True, width="stretch")
st.caption(f"Baseline: MODIS {h['pre']}, {h['mean']['modis_pre']:,.0f} hotspots per year on average, "
           f"{h['median']['modis_pre']:,.0f} in the median year. Averages are pulled up by a few extreme "
           "fire years; the median year is not. These windows are fixed and ignore the year slider.")

drift = val.get("modis_drift_check", {}).get("regions", {})
for r, d in drift.items():
    if r not in used_regions:
        continue
    gap = d["modis_below_prediction_pct"]
    worst = val["regions"][r]["loyo"]["region"][val["chosen_model"]]["annual_abs_err_max_pct"]
    period = d["period"].replace("..", " to ")
    tail = (" Conversion factors are fitted on 2018–2025 only, so harmonized 2026 values show what MODIS "
            "would have reported at its 2018–2025 behaviour. Data from "
            f"{cal['nrt_start']} on are near-real-time and less strictly filtered.")
    if gap > worst:
        st.warning(f"**2026 values in {r}: MODIS is reporting fewer hotspots than expected.** For {period}, "
                   f"actual MODIS ({d['modis_actual']:,} hotspots) is {gap:.0f}% below the harmonized "
                   f"prediction. That is well outside the normal error: the largest annual error in any "
                   f"held-out year 2018–2025 is {worst:.0f}%. The cause is unverified (MODIS is near end "
                   "of life)." + tail)
    elif gap > 0:
        st.info(f"**2026 values in {r}: possible MODIS shortfall, not conclusive.** For {period}, actual "
                f"MODIS ({d['modis_actual']:,} hotspots) is {gap:.0f}% below the harmonized prediction, but "
                f"that is within this region's normal year-to-year error (up to {worst:.0f}% in held-out "
                "years 2018–2025) and rests on few hotspots." + tail)

# ------------------------------------------------------------- calendar
st.subheader("Burning-activity calendar")
mode = st.radio("Series", ["Harmonized", "Raw MODIS", "Raw VIIRS"], horizontal=True,
                label_visibility="collapsed")
col = {"Harmonized": "harmonized", "Raw MODIS": "modis", "Raw VIIRS": "viirs"}[mode]
sc = s_all.copy()
if col == "viirs":
    sc = sc[sc["in_overlap"]]
anom, _ = anomalies(sc, col)
sc = sc.assign(anom=anom)
sc = sc[(sc["year"] >= yr[0]) & (sc["year"] <= yr[1])]
if sc.empty:
    st.info("No data for this series in the selected years.")
else:
    years = list(range(int(sc["year"].min()), int(sc["year"].max()) + 1))
    z = sc.pivot(index="year", columns="cm", values=col).reindex(index=years, columns=range(1, 13))
    an = sc.pivot(index="year", columns="cm", values="anom").reindex(index=years, columns=range(1, 13))
    pa = sc.pivot(index="year", columns="cm", values="is_partial").reindex(index=years, columns=range(1, 13))
    zmax = max(float(np.nanmax(z.to_numpy())), 10)
    ticks = [t for t in (0, 1, 2, 3, 4, 5, 6) if 10 ** t <= zmax * 3]
    heat = go.Figure(go.Heatmap(
        z=np.log10(1 + z.to_numpy(float)), x=MONTHS, y=[str(y) for y in years],
        customdata=z.to_numpy(float), colorscale="Inferno", xgap=2, ygap=2,
        zmin=0, zmax=np.log10(1 + zmax), hoverongaps=False,
        hovertemplate="%{x} %{y}<br>%{customdata:,.0f} hotspots<extra></extra>",
        colorbar=dict(title="Hotspots", tickvals=[np.log10(1 + 10 ** t) for t in ticks],
                      ticktext=[f"{10 ** t:,}" for t in ticks], thickness=12, len=0.8)))
    for yi, y in enumerate(years):
        for mi in range(12):
            if an.iloc[yi, mi] is True or an.iloc[yi, mi] == 1:
                for color, w, pad in (("#0b1c24", 6, 0.44), ("#35e6ff", 3.5, 0.44)):
                    heat.add_shape(type="rect", x0=mi - pad, x1=mi + pad, y0=yi - pad, y1=yi + pad,
                                   line=dict(color=color, width=w), fillcolor="rgba(0,0,0,0)")
            if pa.iloc[yi, mi] is True or pa.iloc[yi, mi] == 1:
                bright = np.log10(1 + (z.iloc[yi, mi] or 0)) > 0.6 * np.log10(1 + zmax)
                heat.add_annotation(x=mi, y=yi, text="partial", showarrow=False,
                                    font=dict(size=9, color="#111111" if bright else "#ffffff"))
    heat.update_layout(height=max(260, 24 * len(years) + 80), margin=dict(l=10, r=10, t=10, b=10),
                       paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
                       font=dict(color=INK, size=12))
    heat.update_yaxes(autorange="reversed", type="category", tickfont=dict(color=MUTED))
    heat.update_xaxes(side="top", tickfont=dict(color=MUTED))
    st.plotly_chart(heat, width="stretch")
    n_an = int(sc["anom"].sum())
    st.caption(f"Colour is on a log scale. **Cells outlined in cyan are anomalies** ({n_an} shown): above the "
               "median + 2 × scaled MAD of the same calendar month across all years, computed on "
               "log(1 + count), and at least "
               f"{MIN_ANOMALY_COUNT} hotspots. Raw VIIRS counts are not comparable with MODIS counts.")
    if n_an:
        with st.expander("List anomalous months"):
            st.dataframe(sc.loc[sc["anom"], [col]].round(0).rename(columns={col: "hotspots"})
                         .sort_values("hotspots", ascending=False), width="stretch")

# ------------------------------------------------------------ validation
st.subheader("Validation")
if not used_regions:
    st.info("This box is outside the validated regions; the pooled factor is unvalidated here.")
for r in used_regions:
    v = val["regions"].get(r)
    if not v:
        continue
    rows = []
    for test, label in (("loyo", "Leave one year out"), ("forward", "Forward")):
        if not v.get(test):
            continue
        if test == "forward":
            label = f"Forward (fit {v[test]['train'].replace('..', ' to ')}, test {v[test]['test'].replace('..', ' to ')})"
        for m in val["models"]:
            x = v[test]["region"][m]
            rows.append({
                "Test": label, "Model": m.replace("_", " ") + (" ✓ used" if m == val["chosen_model"] else ""),
                "Median monthly error %": x["mdape_pct"], "Median annual error %": x["annual_abs_err_median_pct"],
                "Worst annual error %": x["annual_abs_err_max_pct"], "Bias %": x["bias_pct"],
                "Correlation": round(x["r"], 3), "Months": x["n_months"]})
    st.markdown(f"**{r}** — converted VIIRS compared with actual MODIS on data the factors never saw "
                f"(fit period {val['fit_period'].replace('..', ' to ')})")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    ts = v["loyo"]["tile_summary"]
    if ts["median_tile_mdape_pct"]:
        st.caption(f"At the level of single {TILE:g}° tiles the median monthly error of the model used is "
                   f"{ts['median_tile_mdape_pct'][val['chosen_model']]}% ({ts['n_tiles_scored']} tiles with "
                   "enough data), so small boxes are less reliable than whole regions.")
st.caption("Monthly error = median absolute percentage error over months with at least 10 MODIS hotspots. "
           "The model used was picked for the lowest leave-one-year-out monthly error across regions.")

# ------------------------------------------------------------- downloads
st.subheader("Downloads")
d1, d2 = st.columns(2)
out = s[["modis", "viirs", "naive", "harmonized", "in_overlap", "is_nrt", "is_partial"]].copy()
out["harmonized"] = out["harmonized"].round(1)
slug = (choice if prefer else "custom_box").lower().replace(" ", "_")
d1.download_button("Harmonized monthly CSV", out.to_csv().encode(), f"harmonized_monthly_{slug}.csv",
                   "text/csv")
raw_files = glob.glob(os.path.join(RAW, "**", "*.csv"), recursive=True)
if raw_files:
    @st.cache_data(show_spinner="Loading raw hotspots…")
    def raw_hotspots():
        import pipeline
        return pipeline.load_raw()[0]

    if d2.button("Prepare hotspot-level CSV"):
        hs = raw_hotspots()
        sel = hs[hs["latitude"].between(box[0], box[1]) & hs["longitude"].between(box[2], box[3])
                 & hs["acq_date"].dt.year.between(yr[0], yr[1])]
        d2.download_button(f"Download {len(sel):,} hotspots", sel.to_csv(index=False).encode(),
                           f"hotspots_{slug}.csv", "text/csv")
else:
    d2.caption("Hotspot-level download needs the raw FIRMS files in ./data/raw (local runs only).")

# --------------------------------------------------------- method & limits
with st.expander("Method and limits"):
    f = meta["filters"]
    st.markdown(f"""
**Method.** NASA FIRMS hotspots from MODIS Collection 6.1 (Terra + Aqua) and VIIRS NOAA-20 375 m are
filtered (vegetation-fire type only; MODIS confidence {f['modis_confidence']}; VIIRS confidence
{f['viirs_confidence']}) and counted per {GRID}° cell and month. For each {TILE:g}° tile and calendar month,
a conversion factor = MODIS hotspots / VIIRS hotspots over {val['fit_period'].replace('..', ' to ')}.
Tiles or months with too few MODIS hotspots fall back to the tile's or region's pooled factor.
The harmonized series is MODIS before {cal['overlap_start']} and converted VIIRS after.

**Limits.**
- Hotspots are pixels, not fire events. One large fire produces many pixels, so a few extreme months dominate totals.
- The factor assumes a stable relationship between the sensors. It was validated at region and month scale; small boxes have larger errors.
- The series is "Terra + Aqua equivalent" as observed in 2018–2025. Cloud cover, overpass times, satellite orbit drift and the larger night-time share of VIIRS detections are absorbed into the factor, not modelled.
- In 2026 MODIS reports fewer hotspots relative to VIIRS than before (cause unverified), so 2026 is excluded from the factor fit.
- Near-real-time data (from {cal['nrt_start']}) have no fire-type classification. Known static heat sources are masked at 0.01°, which removes most but not all non-vegetation detections.
- Tested on California and Greece only. Regions mixing forest and cropland burning may need finer tiles.
""")
st.caption(f"Data: NASA FIRMS (MODIS C6.1, VIIRS NOAA-20 C2). Last data day {cal['last_data_day']}. "
           f"{meta['hotspots_kept']:,} hotspots after filtering.")
