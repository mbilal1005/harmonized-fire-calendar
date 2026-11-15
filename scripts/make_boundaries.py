"""Build small boundary files for the map from Natural Earth (public domain, 1:50m).

Usage:  python scripts/make_boundaries.py <dir with ne_50m_admin_0_countries.geojson and
        ne_50m_admin_1_states_provinces.geojson> data/geo
Needs shapely (not needed by the app). Output: one GeoJSON of boundary lines per preset region.
"""
import json
import os
import sys

from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union

src, out = sys.argv[1], sys.argv[2]
countries = json.load(open(os.path.join(src, "ne_50m_admin_0_countries.geojson")))["features"]
states = json.load(open(os.path.join(src, "ne_50m_admin_1_states_provinces.geojson")))["features"]
EUROPE_BOX = box(-11.0, 35.0, 40.0, 71.0)


def write(name, geoms, tol):
    feats = []
    for g in geoms:
        g = g.simplify(tol, preserve_topology=True)
        if g.is_empty:
            continue
        feats.append({"type": "Feature", "properties": {}, "geometry": mapping(g)})
    txt = json.dumps({"type": "FeatureCollection", "features": feats}, separators=(",", ":"))
    # 3 decimals (about 100 m) is plenty for an outline
    import re
    txt = re.sub(r"(-?\d+\.\d{3})\d+", r"\1", txt)
    path = os.path.join(out, name + ".geojson")
    open(path, "w").write(txt)
    print(f"{path}: {len(feats)} features, {os.path.getsize(path) / 1e3:.0f} kB")


greece = [shape(f["geometry"]).boundary for f in countries if f["properties"]["ADMIN"] == "Greece"]
write("greece", greece, 0.005)

us = [f for f in states if f["properties"].get("iso_a2") == "US"]
write("california", [shape(f["geometry"]).boundary for f in us if f["properties"]["name"] == "California"], 0.005)
lower48 = unary_union([shape(f["geometry"]).buffer(0) for f in us
                       if f["properties"]["name"] not in ("Alaska", "Hawaii")])
write("usa", [lower48.boundary], 0.02)

eu = []
for f in countries:
    g = shape(f["geometry"]).buffer(0)
    if g.intersects(EUROPE_BOX):
        line = g.boundary.intersection(EUROPE_BOX)      # borders inside the box only
        if not line.is_empty:
            eu.append(line)
write("europe", [unary_union(eu)], 0.02)
