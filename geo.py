"""Geometry and soil-classification helpers (need geopandas)."""
import geopandas as gpd
import pandas as pd
from shapely import LineString


def extract_line_segments(gdf):
    """Split (Multi)LineStrings into straight 2-point LineString segments."""
    rows = []
    for _, row in gdf.iterrows():
        geom = row["geometry"]
        if geom is None:
            continue
        if geom.geom_type == "MultiLineString":
            lines = list(geom.geoms)
        elif geom.geom_type == "LineString":
            lines = [geom]
        else:
            lines = []
        for line in lines:
            coords = list(line.coords)
            for i in range(len(coords) - 1):
                r = row.copy()
                r["geometry"] = LineString(coords[i:i + 2])
                rows.append(r)
    return gpd.GeoDataFrame(rows, crs=gdf.crs).reset_index(drop=True)


def simplify_soil(genesis):
    """Classify Alberta Geological Survey genesis codes into broad soil types."""
    if pd.isna(genesis):
        return "Unknown"
    g = str(genesis).lower()
    if any(k in g for k in ["till", "glacial", "mudflow"]):
        return "Glacial_Till"
    if any(k in g for k in ["fluvial", "channel", "river"]):
        return "Fluvial"
    if any(k in g for k in ["lacustrine", "lake", "pond"]):
        return "Lacustrine"
    if "eolian" in g:
        return "Eolian"
    if "bedrock" in g:
        return "Bedrock"
    if any(k in g for k in ["colluvial", "slope"]):
        return "Colluvial"
    if any(k in g for k in ["organic", "peat"]):
        return "Organic"
    if any(k in g for k in ["alluvial", "flood"]):
        return "Alluvial"
    return "Other"
