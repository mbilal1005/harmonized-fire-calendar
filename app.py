"""Harmonized Fire Calendar - Streamlit app.

Reads only the small aggregates in ./data/agg (built by pipeline.py).
Run:  streamlit run app.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import folium
import plotly.graph_objects as go
import streamlit as st
from folium.plugins import Draw
from plotly.colors import sample_colorscale
from streamlit_folium import st_folium

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
    fpath = os.path.join(AGG, "forecast.json")
    out["meta"]["forecast"] = json.load(open(fpath)) if os.path.exists(fpath) else None
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


def _with_factor(v, prefer):
    """Add the conversion factor and converted count to VIIRS cell-month rows."""
    v = v.copy()
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
    return v, fallback_share


@st.cache_data
def cell_density(prefer, y0, y1):
    """Harmonized hotspots per 0.5 deg cell, summed over the selected years (all loaded cells)."""
    g = grid[grid["month"].str[:4].astype(int).between(y0, y1)]
    m = g[(g["sensor"] == "MODIS") & (g["month"] < cal["overlap_start"])]
    v, _ = _with_factor(g[g["sensor"] == "VIIRS"], prefer)
    tot = m.groupby(["cell_lat", "cell_lon"])["n"].sum().add(
        v.groupby(["cell_lat", "cell_lon"])["conv"].sum(), fill_value=0)
    return tot.rename("n").reset_index()


def map_view(box, width_px=900, height_px=520):
    """Centre and zoom level that show the whole box (explicit, so the view always follows the selection)."""
    lat_span, lon_span = max(box[1] - box[0], 0.2), max(box[3] - box[2], 0.2)
    coslat = np.cos(np.radians((box[0] + box[1]) / 2))      # Web Mercator stretches latitude
    zoom = min(np.log2(360 * width_px / 256 / lon_span), np.log2(360 * coslat * height_px / 256 / lat_span))
    return ((box[0] + box[1]) / 2, (box[2] + box[3]) / 2), int(np.clip(np.floor(zoom), 2, 11))


def density_map(dens, box):
    """Light basemap with one coloured square per cell and a rectangle-only draw tool."""
    esri = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_{}/MapServer/tile/{{z}}/{{y}}/{{x}}"
    fmap = folium.Map(location=list(map_view(box)[0]), zoom_start=map_view(box)[1], tiles=None,
                      control_scale=True, prefer_canvas=True)
    folium.TileLayer(esri.format("Base"), attr="Esri, HERE, Garmin, © OpenStreetMap contributors",
                     name="Light basemap", control=False).add_to(fmap)
    dens = dens[dens["n"] >= 1]
    if len(dens):
        top = np.log10(1 + dens["n"].max())
        cols = sample_colorscale("Inferno", list(np.clip(np.log10(1 + dens["n"]) / top, 0, 1)), colortype="rgb")
        for (lat, lon, n), c in zip(dens[["cell_lat", "cell_lon", "n"]].itertuples(index=False), cols):
            folium.Rectangle([[float(lat), float(lon)], [float(lat) + GRID, float(lon) + GRID]], stroke=False,
                             fill=True, fill_color=c, fill_opacity=0.62,
                             tooltip=f"{n:,.0f} fire detections").add_to(fmap)
    folium.TileLayer(esri.format("Reference"), attr="Esri", name="Place names", control=False,
                     overlay=True).add_to(fmap)
    folium.Rectangle([[box[0], box[2]], [box[1], box[3]]], color="#2a78d6", weight=2.5, fill=False,
                     tooltip="Selected area").add_to(fmap)
    Draw(draw_options=dict(polyline=False, polygon=False, circle=False, marker=False, circlemarker=False,
                           rectangle=dict(shapeOptions=dict(color="#2a78d6", weight=2))),
         edit_options=dict(edit=False, remove=False)).add_to(fmap)
    return fmap


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
    v, fallback_share = _with_factor(g[g["sensor"] == "VIIRS"], prefer)
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


MONTHS_LONG = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
               "October", "November", "December"]


def peak_run(medcal, cover=0.75):
    """Shortest run of consecutive months (may wrap across December) holding >= cover
    of the typical year (sum of calendar-month medians). Ties: the run with the larger share."""
    v = np.array([float(medcal.get(m, 0.0)) for m in range(1, 13)])
    for length in range(1, 13):
        shares = [(v[[(st0 + k) % 12 for k in range(length)]].sum() / v.sum(), st0) for st0 in range(12)]
        best, st0 = max(shares)
        if best >= cover:
            return [(st0 + k) % 12 + 1 for k in range(length)], best


def season_facts(s):
    """Peak window and last complete season, computed from the harmonized series."""
    full = s[~s["is_partial"]]
    medcal = full.groupby("cm")["harmonized"].median()
    if medcal.sum() <= 0:
        return None
    peak, share = peak_run(medcal)
    one = len(peak) == 1
    f = dict(peak=peak, share=share, wraps=peak[-1] < peak[0],
             span=MONTHS[peak[0] - 1] if one else f"{MONTHS[peak[0] - 1]}–{MONTHS[peak[-1] - 1]}",
             span_long=MONTHS_LONG[peak[0] - 1] if one else
             f"{MONTHS_LONG[peak[0] - 1]} to {MONTHS_LONG[peak[-1] - 1]}",
             month_share={m: float(medcal.get(m, 0.0) / medcal.sum()) for m in peak},
             busiest=int(medcal.idxmax()), last=None)
    d = full[full["cm"].isin(peak)].copy()
    # a season that wraps across December belongs to the year it starts in
    d["season"] = np.where(f["wraps"] & (d["cm"] < peak[0]), d["year"] - 1, d["year"])
    g = d.groupby("season")
    seas = g["harmonized"].sum()[g.size() == len(peak)]
    if len(seas) >= 3:
        ly, lv, hist = int(seas.index[-1]), float(seas.iloc[-1]), seas.iloc[:-1]
        last = d[d["season"] == ly]
        f["last"] = dict(
            year=ly, value=lv, normal=float(hist.median()), rank=int((seas > lv).sum()) + 1, n=len(seas),
            hist=f"{int(hist.index[0])}–{int(hist.index[-1])}",
            label=f"{f['span']} {ly}" if not f["wraps"] else
            f"{MONTHS[peak[0] - 1]} {ly}–{MONTHS[peak[-1] - 1]} {ly + 1}",
            caveat=bool(last["is_nrt"].any() or (last.index > cal["fit_end"]).any()))
    return f


@st.cache_data
def month_character(box):
    """Per calendar month: mean fire power per detection and night share (MODIS, whole record)."""
    a, b, c, d = box
    clat, clon = grid["cell_lat"] + GRID / 2, grid["cell_lon"] + GRID / 2
    g = grid[clat.between(a, b) & clon.between(c, d) & (grid["sensor"] == "MODIS")]
    t = g.groupby(g["month"].str[5:].astype(int))[["n", "n_night", "frp_sum"]].sum()
    return pd.DataFrame({"frp_mean": t["frp_sum"] / t["n"], "night": t["n_night"] / t["n"], "n": t["n"]})


def join_months(ms):
    names = [MONTHS_LONG[m - 1] for m in ms]
    if len(ms) > 1 and all((b - a) % 12 == 1 for a, b in zip(ms, ms[1:])):
        return f"{names[0]}–{names[-1]}" if len(ms) > 2 else f"{names[0]} and {names[1]}"
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def small_fire_note(box, peak):
    """Sentence about peak-window months dominated by small daytime fires, if the data shows it.

    Small-fire month: mean fire power per detection below a third of the most intense peak month
    and under 10% of detections at night. Largest-wildfire months: at least two thirds of it.
    """
    mc = month_character(box).reindex(peak).dropna()
    mc = mc[mc["n"] >= 100]
    if len(mc) < 2:
        return ""
    top = mc["frp_mean"].max()
    small = [m for m in peak if m in mc.index and mc.loc[m, "frp_mean"] < top / 3 and mc.loc[m, "night"] < 0.10]
    big = [m for m in peak if m in mc.index and mc.loc[m, "frp_mean"] >= 2 * top / 3]
    if not small or not big:
        return ""
    verb = "has" if len(small) == 1 else "have"
    return (f" {join_months(small)} {verb} many small fires (likely agricultural burning); the largest "
            f"wildfires happen in {join_months(big)}.")


def vs_normal(v, normal):
    """Plain-language comparison, e.g. '43% quieter than normal'."""
    if normal <= 0:
        return "no normal level to compare with"
    p = 100 * (v / normal - 1)
    if abs(p) < 5:
        return "about normal"
    return f"{abs(p):.0f}% {'busier' if p > 0 else 'quieter'} than normal"


def ordinal(n):
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def calendar_figure(sc, col, yr):
    """Year x month heatmap with anomalous months outlined. Returns (figure, shown rows)."""
    anom, _ = anomalies(sc, col)
    sc = sc.assign(anom=anom)
    sc = sc[(sc["year"] >= yr[0]) & (sc["year"] <= yr[1])]
    if sc.empty:
        return None, sc
    years = list(range(int(sc["year"].min()), int(sc["year"].max()) + 1))
    piv = lambda c: sc.pivot(index="year", columns="cm", values=c).reindex(index=years, columns=range(1, 13))
    z, an, pa = piv(col), piv("anom"), piv("is_partial")
    zmax = max(float(np.nanmax(z.to_numpy())), 10)
    ticks = [t for t in (0, 1, 2, 3, 4, 5, 6) if 10 ** t <= zmax * 3]
    heat = go.Figure(go.Heatmap(
        z=np.log10(1 + z.to_numpy(float)), x=MONTHS, y=[str(y) for y in years],
        customdata=z.to_numpy(float), colorscale="Inferno", xgap=2, ygap=2,
        zmin=0, zmax=np.log10(1 + zmax), hoverongaps=False,
        hovertemplate="%{x} %{y}<br>%{customdata:,.0f} fire detections<extra></extra>",
        colorbar=dict(title="Fire detections", tickvals=[np.log10(1 + 10 ** t) for t in ticks],
                      ticktext=[f"{10 ** t:,}" for t in ticks], thickness=12, len=0.8)))
    for yi in range(len(years)):
        for mi in range(12):
            if an.iloc[yi, mi] is True or an.iloc[yi, mi] == 1:
                for color, w, pad in (("#0b1c24", 6, 0.44), ("#35e6ff", 3.5, 0.44)):
                    heat.add_shape(type="rect", x0=mi - pad, x1=mi + pad, y0=yi - pad, y1=yi + pad,
                                   line=dict(color=color, width=w), fillcolor="rgba(0,0,0,0)")
            if pa.iloc[yi, mi] is True or pa.iloc[yi, mi] == 1:
                bright = np.log10(1 + (z.iloc[yi, mi] or 0)) > 0.6 * np.log10(1 + zmax)
                heat.add_annotation(x=mi, y=yi, text="so far", showarrow=False,
                                    font=dict(size=9, color="#111111" if bright else "#ffffff"))
    heat.update_layout(height=max(260, 24 * len(years) + 80), margin=dict(l=10, r=10, t=10, b=10),
                       paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", font=dict(color=INK, size=12))
    heat.update_yaxes(autorange="reversed", type="category", tickfont=dict(color=MUTED))
    heat.update_xaxes(side="top", tickfont=dict(color=MUTED))
    return heat, sc


# --------------------------------------------------------------- sidebar
# A rectangle drawn on the map becomes the custom box (applied before the widgets are built).
for _k, _v in (("lat0", 37.5), ("lat1", 39.0), ("lon0", 22.5), ("lon1", 24.5)):
    st.session_state.setdefault(_k, _v)
if "pending_box" in st.session_state:
    _b = st.session_state.pop("pending_box")
    st.session_state.update(region_choice="Custom box", lat0=_b[0], lat1=_b[1], lon0=_b[2], lon1=_b[3])

st.sidebar.header("Area")
presets = list(REGIONS)
default = presets.index("Greece") if "Greece" in presets else 0
choice = st.sidebar.selectbox("Region", presets + ["Custom box"], index=default, key="region_choice")
if choice == "Custom box":
    with st.sidebar.expander("Box coordinates", expanded=True):
        c1, c2 = st.columns(2)
        lat0 = c1.number_input("Lat min", min_value=-90.0, max_value=90.0, step=0.5, key="lat0")
        lat1 = c2.number_input("Lat max", min_value=-90.0, max_value=90.0, step=0.5, key="lat1")
        lon0 = c1.number_input("Lon min", min_value=-180.0, max_value=180.0, step=0.5, key="lon0")
        lon1 = c2.number_input("Lon max", min_value=-180.0, max_value=180.0, step=0.5, key="lon1")
        st.caption(f"The map is divided into {GRID}° squares. A square counts if its centre is inside the box.")
    if lat1 <= lat0 or lon1 <= lon0:
        st.sidebar.error("Max must be larger than min.")
        st.stop()
    box, prefer = (lat0, lat1, lon0, lon1), None
else:
    box, prefer = REGIONS[choice], choice

y_min, y_max = int(cal["first_month"][:4]), int(cal["last_month"][:4])
st.sidebar.header("Period")
yr = st.sidebar.slider("Years", y_min, y_max, (y_min, y_max))

s_all, fallback_share = box_series(box, prefer)
used_regions = [choice] if prefer else regions_for_box(box)
title_area = choice if prefer else f"box {box[0]:g}–{box[1]:g}°N, {box[2]:g}–{box[3]:g}°E"
area_name = choice if prefer else "the selected box"
s = s_all[(s_all["year"] >= yr[0]) & (s_all["year"] <= yr[1])]

st.title("Harmonized Fire Calendar")
if s_all[["modis", "viirs"]].to_numpy().sum() == 0:
    st.warning("No fire detections in this area. Data are currently loaded for: " + ", ".join(presets) + ".")
    st.stop()

h = headline_numbers(s_all)
facts = season_facts(s_all)
day = pd.Timestamp(cal["last_data_day"])
tab1, tab2 = st.tabs(["Fire season check", "Method & validation"])

# =================================================== TAB 1: fire season check
with tab1:
    st.markdown("*For fire planners deciding when and how many seasonal crews to hire*")

    # ---- the answer
    if facts:
        st.markdown(f"## {area_name[0].upper() + area_name[1:]}: the fire season runs "
                    f"{facts['span_long']}")
        ans = (f"That window holds **{facts['share']:.0%} of a normal year's fire activity**. "
               f"The busiest month is {MONTHS_LONG[facts['busiest'] - 1]}.")
        if facts["last"]:
            L = facts["last"]
            ans += (f" The last full season ({L['label']}) was "
                    f"**{vs_normal(L['value'], L['normal'])}**.")
        ans += small_fire_note(box, facts["peak"])
        st.markdown(ans)

        # ---- season briefing
        with st.container(border=True):
            lines = ["**Season briefing**", ""]
            shares = ", ".join(f"{MONTHS[m - 1]} {p:.0%}" for m, p in facts["month_share"].items())
            lines.append(f"- **When to staff up: {facts['span']}.** Share of a normal year's fire activity by "
                         f"month: {shares}. Together {facts['share']:.0%}.")
            if facts["last"]:
                L = facts["last"]
                lines.append(
                    f"- **Last full season ({L['label']}): {L['value']:,.0f} fire detections**, "
                    f"{vs_normal(L['value'], L['normal'])} ({L['normal']:,.0f} in a normal season, "
                    f"{L['hist']}). It was the {ordinal(L['rank'])} busiest of {L['n']} seasons."
                    + (" (includes preliminary recent data; see the 2026 note under Method & validation)"
                       if L["caveat"] else ""))
            md = h["median"]
            if np.isfinite(md["viirs_post_x"]) and np.isfinite(h["inflation"]):
                lines.append(
                    f"- **Why not just count raw satellite detections?** A sharper satellite joined in 2018 and "
                    f"reports about {h['inflation']:.1f} times as many detections for the same fires. Raw counts "
                    f"make a normal year in {h['post']} look like {md['viirs_post_x']:.2f}× the {h['pre']} level. "
                    f"Counted the same way throughout, it is {md['harm_post_x']:.2f}×. All numbers on this page "
                    "are counted the same way throughout.")
            st.markdown("\n".join(lines))
        st.caption("Normal = the middle value across all earlier years. A fire detection is one satellite "
                   "pixel seen burning; a large fire produces many detections.")

    # ---- map
    st.subheader("Where it burns")
    st.caption(f"Fire activity per map square, {yr[0]}–{yr[1]}; brighter means more. **To check a smaller "
               "area, use the square tool on the left of the map and draw a rectangle.** Everything on this "
               "page then updates to that area. You can also type coordinates in the sidebar.")
    out_map = st_folium(density_map(cell_density(prefer, yr[0], yr[1]), box), height=520,
                        use_container_width=True, returned_objects=["last_active_drawing"],
                        center=map_view(box)[0], zoom=map_view(box)[1],
                        key=f"map_{choice}_{box}_{yr}")
    drawn = (out_map or {}).get("last_active_drawing")
    if drawn and drawn.get("geometry", {}).get("type") == "Polygon":
        pts = drawn["geometry"]["coordinates"][0]
        lats, lons = [p[1] for p in pts], [p[0] for p in pts]
        new_box = (round(max(min(lats), -90.0), 2), round(min(max(lats), 90.0), 2),
                   round(max(min(lons), -180.0), 2), round(min(max(lons), 180.0), 2))
        if new_box[1] > new_box[0] and new_box[3] > new_box[2] and st.session_state.get("last_draw") != new_box:
            st.session_state["last_draw"] = new_box
            st.session_state["pending_box"] = new_box
            st.rerun()

    # ---- calendar
    st.subheader("Fire activity calendar")
    heat, shown = calendar_figure(s_all, "harmonized", yr)
    if heat is None:
        st.info("No data in the selected years.")
    else:
        st.plotly_chart(heat, width="stretch", key="cal_main")
        n_an = int(shown["anom"].sum())
        st.caption("Each square is one month; brighter means more fire activity (each colour step is ten "
                   f"times more). **Squares outlined in light blue were unusually active for that time of "
                   f"year** ({n_an} shown).")
        if n_an:
            with st.expander("List the unusually active months"):
                st.dataframe(shown.loc[shown["anom"], ["harmonized"]].round(0)
                             .rename(columns={"harmonized": "fire detections"})
                             .sort_values("fire detections", ascending=False), width="stretch")

    # ---- current month vs normal
    st.subheader("Right now compared with normal")
    mtd = month_to_date(box, prefer)          # same day range in every year
    cur_mtd, hist = mtd.iloc[-1], mtd.iloc[:-1]
    lastfull = s_all.loc[cal["latest_full_month"]]
    hist_full = s_all[(s_all["cm"] == lastfull["cm"]) & (s_all.index < cal["latest_full_month"])]["harmonized"]

    def delta_vs(v, normal):
        return f"{100 * (v / normal - 1):+.0f}% vs normal ({normal:,.0f})" if normal > 0 else f"normal: {normal:,.0f}"

    rng = f"1–{day.day} {MONTHS[day.month - 1]}"
    m1, m2 = st.columns(2)
    m1.metric(f"This month so far ({rng} {day.year})", f"{cur_mtd:,.0f}", delta_vs(cur_mtd, hist.median()),
              delta_color="inverse",
              help=f"Fire detections for {rng}, compared with the same days in "
                   f"{hist.index[0][:4]}–{hist.index[-1][:4]}. The latest day may be incomplete.")
    m2.metric(f"Last full month ({MONTHS[int(lastfull['cm']) - 1]} {int(lastfull['year'])})",
              f"{lastfull['harmonized']:,.0f}", delta_vs(lastfull["harmonized"], hist_full.median()),
              delta_color="inverse", help="Compared with the same month in earlier years.")

    # ---- downloads
    st.subheader("Downloads")
    d1, d2 = st.columns(2)
    out = s[["modis", "viirs", "naive", "harmonized", "in_overlap", "is_nrt", "is_partial"]].copy()
    out["harmonized"] = out["harmonized"].round(1)
    slug = (choice if prefer else "custom_box").lower().replace(" ", "_")
    d1.download_button("Monthly fire activity (CSV)", out.to_csv().encode(),
                       f"harmonized_monthly_{slug}.csv", "text/csv")
    raw_files = glob.glob(os.path.join(RAW, "**", "*.csv"), recursive=True)
    if raw_files:
        @st.cache_data(show_spinner="Loading individual detections…")
        def raw_hotspots():
            import pipeline
            return pipeline.load_raw()[0]

        if d2.button("Prepare individual detections (CSV)"):
            hs = raw_hotspots()
            sel = hs[hs["latitude"].between(box[0], box[1]) & hs["longitude"].between(box[2], box[3])
                     & hs["acq_date"].dt.year.between(yr[0], yr[1])]
            d2.download_button(f"Download {len(sel):,} detections", sel.to_csv(index=False).encode(),
                               f"hotspots_{slug}.csv", "text/csv")
    else:
        d2.caption("The list of individual detections is only available when the app runs locally "
                   "with the raw data files.")

# ================================================ TAB 2: method & validation
with tab2:
    st.markdown("**More fires, or just a better satellite?** One consistent burning-activity record "
                "from MODIS (1 km, 2000–) and VIIRS NOAA-20 (375 m, 2018–) active-fire hotspots.")
    if not prefer and fallback_share > 0:
        st.info(f"{fallback_share:.0%} of VIIRS hotspots in this box lie outside the preset regions and use "
                f"the pooled conversion factor ({1/GLOBAL_POOLED:.2f} VIIRS per MODIS hotspot).")

    m3, m4, m5 = st.columns(3)
    m3.metric("Naive splice overstates by", fmt_x(h["inflation"]),
              help=f"Raw VIIRS hotspots divided by actual MODIS hotspots, {h['post']}.")
    m4.metric(f"Actual MODIS change, {h['post']} vs {h['pre']}", fmt_x(h["mean"]["modis_post_x"]),
              help="Hotspots per year on average. See the table below for the median year.")
    m5.metric("Naive splice change, same years", fmt_x(h["mean"]["viirs_post_x"]),
              help="Raw VIIRS hotspots per year against MODIS hotspots per year before 2018.")

    # ---- naive vs harmonized chart
    st.subheader(f"Naive splice vs harmonized vs actual MODIS — {title_area}")
    view = st.radio("View", ["Yearly totals", "Monthly"], horizontal=True, label_visibility="collapsed")
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
    st.plotly_chart(fig, width="stretch", key="main_chart")
    st.caption("Before April 2018 all three lines are the same MODIS record. Yearly totals show complete "
               "years only. The gap between the dashed and solid lines is the sensor change, not fire.")
    with st.expander("Show these numbers as a table"):
        st.dataframe(table.round(0), width="stretch")

    # ---- headline comparison
    st.markdown(f"**Change in hotspots per year, {h['post']} compared with {h['pre']}**")
    keys = ("viirs_post", "modis_post", "harm_post")
    cmp = pd.DataFrame({
        "Series": ["Naive splice (raw VIIRS vs earlier MODIS)", "Actual MODIS", "Harmonized"],
        "Per year on average": [fmt_x(h["mean"][k + "_x"]) for k in keys],
        "Median year": [fmt_x(h["median"][k + "_x"]) for k in keys],
        f"Hotspots per year {h['post']} (mean)": [f"{h['mean'][k]:,.0f}" for k in keys],
        f"Hotspots per year {h['post']} (median)": [f"{h['median'][k]:,.0f}" for k in keys],
    })
    st.dataframe(cmp, hide_index=True, width="stretch")
    st.caption(f"Baseline: MODIS {h['pre']}, {h['mean']['modis_pre']:,.0f} hotspots per year on average, "
               f"{h['median']['modis_pre']:,.0f} in the median year. Averages are pulled up by a few extreme "
               "fire years; the median year is not. These windows are fixed and ignore the year slider.")

    # ---- 2026 note
    st.subheader("2026 note")
    drift = val.get("modis_drift_check", {}).get("regions", {})
    shown_note = False
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
            shown_note = True
            st.warning(f"**2026 values in {r}: MODIS is reporting fewer hotspots than expected.** For {period}, "
                       f"actual MODIS ({d['modis_actual']:,} hotspots) is {gap:.0f}% below the harmonized "
                       f"prediction. That is well outside the normal error: the largest annual error in any "
                       f"held-out year 2018–2025 is {worst:.0f}%. The cause is unverified (MODIS is near end "
                       "of life)." + tail)
        elif gap > 0:
            shown_note = True
            st.info(f"**2026 values in {r}: possible MODIS shortfall, not conclusive.** For {period}, actual "
                    f"MODIS ({d['modis_actual']:,} hotspots) is {gap:.0f}% below the harmonized prediction, but "
                    f"that is within this region's normal year-to-year error (up to {worst:.0f}% in held-out "
                    "years 2018–2025) and rests on few hotspots." + tail)
    if not shown_note:
        st.caption(f"Data from {cal['nrt_start']} on are near-real-time and less strictly filtered.")

    # ---- validation
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
                label = (f"Forward (fit {v[test]['train'].replace('..', ' to ')}, "
                         f"test {v[test]['test'].replace('..', ' to ')})")
            for m in val["models"]:
                x = v[test]["region"][m]
                rows.append({
                    "Test": label,
                    "Model": m.replace("_", " ") + (" ✓ used" if m == val["chosen_model"] else ""),
                    "Median monthly error %": x["mdape_pct"],
                    "Median annual error %": x["annual_abs_err_median_pct"],
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

    # ---- ML forecast test
    fc = meta.get("forecast")
    fc_regions = [r for r in used_regions if fc and r in fc["regions"]]
    if fc_regions:
        any_win = any(fc["regions"][r]["ml_beats_baseline"] for r in fc_regions)
        st.subheader("Forecast test: machine learning against the seasonal baseline" if any_win else
                     "We tested ML; the seasonal baseline was as good or better")
        st.markdown(
            "We tested whether gradient boosting can predict harmonized monthly hotspots 1 to 3 months ahead "
            "better than the seasonal baseline (the median of the same calendar month in earlier years). "
            "Inputs: calendar month, the last three months' anomalies against the same-month median, and "
            "the same month one year earlier. Test: for every month the model is trained only on data up to "
            "that month and predicts the next three.")
        rows = []
        for r in fc_regions:
            for hz, x in fc["regions"][r]["metrics"].items():
                rows.append({"Region": r, "Months ahead": int(hz),
                             "ML monthly error %": x["ml_mdape_pct"],
                             "Seasonal baseline monthly error %": x["baseline_mdape_pct"],
                             "ML log error": x["ml_log_mae"], "Baseline log error": x["baseline_log_mae"],
                             "Share of months ML closer": x["ml_better_share"], "Test months": x["n"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        periods = "; ".join(f"{r} {fc['regions'][r]['test_period'].replace('..', ' to ')}" for r in fc_regions)
        st.caption(f"Monthly error = median absolute percentage error. Test months: {periods}. Model settings "
                   "were fixed in advance and not tuned on the test. "
                   + ("" if any_win else "Because ML did not beat the baseline, no forecast is shown in the "
                      "Fire season check tab. ")
                   + "Both errors are large: last season's activity says little about a given month ahead, "
                   "so the calendar of normal months remains the planning basis.")

    # ---- raw per-sensor calendar
    with st.expander("Calendar per sensor (raw counts)"):
        mode = st.radio("Series", ["Harmonized", "Raw MODIS", "Raw VIIRS"], horizontal=True,
                        label_visibility="collapsed")
        col = {"Harmonized": "harmonized", "Raw MODIS": "modis", "Raw VIIRS": "viirs"}[mode]
        heat2, _ = calendar_figure(s_all[s_all["in_overlap"]] if col == "viirs" else s_all, col, yr)
        if heat2 is None:
            st.info("No data for this series in the selected years.")
        else:
            st.plotly_chart(heat2, width="stretch", key="cal_raw")
        st.caption("Raw VIIRS counts are not comparable with MODIS counts. Outlined cells are anomalies of "
                   "the series shown.")

    # ---- method & limits
    with st.expander("Method and limits", expanded=True):
        f = meta["filters"]
        st.markdown(f"""
**Method.** NASA FIRMS hotspots from MODIS Collection 6.1 (Terra + Aqua) and VIIRS NOAA-20 375 m are
filtered (vegetation-fire type only; MODIS confidence {f['modis_confidence']}; VIIRS confidence
{f['viirs_confidence']}) and counted per {GRID}° cell and month. For each {TILE:g}° tile and calendar month,
a conversion factor = MODIS hotspots / VIIRS hotspots over {val['fit_period'].replace('..', ' to ')}.
Tiles or months with too few MODIS hotspots fall back to the tile's or region's pooled factor.
The harmonized series is MODIS before {cal['overlap_start']} and converted VIIRS after.

**Anomalies.** A month is outlined when log(1 + count) exceeds the median + 2 × scaled MAD of the same
calendar month across all years, and it has at least {MIN_ANOMALY_COUNT} hotspots.

**Small-fire months.** Within the fire season, a month is described as "many small fires" when its mean
fire radiative power per MODIS detection is below a third of the most intense season month and under 10%
of its detections are at night. "Likely agricultural burning" is an inference from low intensity, daytime
timing and wide spread; land cover was not checked.

**Fire season.** The shortest run of consecutive months that holds at least 75% of the sum of
calendar-month medians of the harmonized series.

**Limits.**
- Hotspots are pixels, not fire events. One large fire produces many pixels, so a few extreme months dominate totals.
- The factor assumes a stable relationship between the sensors. It was validated at region and month scale; small boxes have larger errors.
- The series is "Terra + Aqua equivalent" as observed in 2018–2025. Cloud cover, overpass times, satellite orbit drift and the larger night-time share of VIIRS detections are absorbed into the factor, not modelled.
- In 2026 MODIS reports fewer hotspots relative to VIIRS than before (cause unverified), so 2026 is excluded from the factor fit.
- Near-real-time data (from {cal['nrt_start']}) have no fire-type classification. Known static heat sources are masked at 0.01°, which removes most but not all non-vegetation detections.
- Tested on California and Greece only. Regions mixing forest and cropland burning may need finer tiles.
""")

st.caption(f"Data: NASA FIRMS (MODIS C6.1, VIIRS NOAA-20 C2). Last data day {cal['last_data_day']}. "
           f"{meta['hotspots_kept']:,} fire detections after filtering.")
