#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
p5_figura_tfg.py
================
Etapa G del pipeline: generación de figura final y mapa HTML interactivo.

Genera dos salidas:
  1. PNG para LaTeX (matplotlib + contextily)
     - Fondo cartográfico de València
     - Tres rutas con convenciones de color del TFG
     - Marcadores de inicio y destino con nombres físicos
     - Caja de métricas (L_km, T_min, E) por ruta
     - 300 dpi, apto para insertar en LaTeX

  2. HTML interactivo (folium)
     - Mismas tres rutas con tooltips de métricas
     - Control de capas
     - Marcadores de inicio y destino

Convenciones de color:
  Naranja  (#ff7f0e) → mínimo tiempo
  Azul     (#1f77b4) → mínima exposición IDW (campo observado)
  Turquesa (#17becf) → mínima exposición LSTM+IDW (dinámica)

Uso:
  python scripts/p5_figura_tfg.py
    --route-time  <route_min_time_NO2_20230515_0800.geojson>
    --route-idw   <route_min_expo_idw_NO2_20230515_0800.geojson>
    --route-lstm  <route_min_expo_lstm_NO2_20230515_0800.geojson>
    --metrics-csv <routes_metrics_NO2_20230515_0800.csv>
    --origin-name "Campus de Tarongers"
    --dest-name   "Cabanyal-Canyamelar"
    --origin-coords 39.4856445 -0.3494007
    --dest-coords   39.4736445 -0.3254007
    --outdir      <dir>
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import folium

try:
    import contextily as ctx
    CTX_OK = True
except ImportError:
    CTX_OK = False

try:
    import geopandas as gpd
    from shapely.geometry import shape
    GPD_OK = True
except ImportError:
    GPD_OK = False


# ---------------------------------------------------------------------------
# Colores del TFG
# ---------------------------------------------------------------------------
COLOR_TIME = "#ff7f0e"   # naranja  — mínimo tiempo
COLOR_IDW  = "#1f77b4"   # azul     — mínima exposición IDW
COLOR_LSTM = "#17becf"   # turquesa — mínima exposición LSTM+IDW


# ---------------------------------------------------------------------------
# Carga de GeoJSON
# ---------------------------------------------------------------------------

def load_linestring_coords(path: Path) -> list[tuple[float, float]]:
    """Devuelve lista de (lon, lat) desde un GeoJSON LineString."""
    with open(path, "r", encoding="utf-8") as f:
        gj = json.load(f)

    def harvest(geom):
        if geom is None:
            return []
        t = geom.get("type", "")
        if t == "LineString":
            return geom.get("coordinates", [])
        if t == "MultiLineString":
            merged = []
            for seg in geom.get("coordinates", []):
                merged.extend(seg)
            return merged
        return []

    coords = []
    if gj.get("type") == "FeatureCollection":
        for ft in gj.get("features", []):
            c = harvest(ft.get("geometry"))
            if c:
                coords = c
                break
    elif gj.get("type") == "Feature":
        coords = harvest(gj.get("geometry"))
    else:
        coords = harvest(gj)

    if not coords:
        raise SystemExit(f"[FATAL] No se encontraron coordenadas en {path}")
    return coords  # lista de [lon, lat]


# ---------------------------------------------------------------------------
# Figura PNG para LaTeX
# ---------------------------------------------------------------------------

def make_png(
    coords_time, coords_idw, coords_lstm,
    metrics: pd.DataFrame,
    origin_coords, dest_coords,
    origin_name: str, dest_name: str,
    out_path: Path,
    time_label: str = "2023-05-15 08:00",
    dpi: int = 300,
):
    fig, ax = plt.subplots(figsize=(9, 9))

    # --- Dibujar rutas
    def draw_route(coords, color, label, lw=2.5, zorder=4):
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        ax.plot(lons, lats, color=color, linewidth=lw, label=label,
                zorder=zorder, solid_capstyle="round")

    draw_route(coords_time, COLOR_TIME, "Mínimo tiempo",           lw=2.5, zorder=4)
    draw_route(coords_idw,  COLOR_IDW,  "Mín. exposición IDW",     lw=2.5, zorder=5)
    draw_route(coords_lstm, COLOR_LSTM, "Mín. exposición LSTM+IDW", lw=2.5, zorder=6)

    # --- Marcadores inicio / destino
    ax.scatter(
        [origin_coords[1]], [origin_coords[0]],
        s=180, color="#2ca02c", zorder=10, marker="o",
        edgecolors="white", linewidths=1.5,
    )
    ax.scatter(
        [dest_coords[1]], [dest_coords[0]],
        s=180, color="#d62728", zorder=10, marker="o",
        edgecolors="white", linewidths=1.5,
    )
    ax.annotate(
        origin_name,
        xy=(origin_coords[1], origin_coords[0]),
        xytext=(8, 8), textcoords="offset points",
        fontsize=8, fontweight="bold", color="#2ca02c",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"),
        zorder=11,
    )
    ax.annotate(
        dest_name,
        xy=(dest_coords[1], dest_coords[0]),
        xytext=(8, -14), textcoords="offset points",
        fontsize=8, fontweight="bold", color="#d62728",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"),
        zorder=11,
    )

    # --- Fondo cartográfico con contextily
    if CTX_OK and GPD_OK:
        try:
            import geopandas as gpd
            from shapely.geometry import box

            # bbox con margen
            all_lons = (
                [c[0] for c in coords_time]
                + [c[0] for c in coords_idw]
                + [c[0] for c in coords_lstm]
            )
            all_lats = (
                [c[1] for c in coords_time]
                + [c[1] for c in coords_idw]
                + [c[1] for c in coords_lstm]
            )
            margin_lon = (max(all_lons) - min(all_lons)) * 0.15 + 0.003
            margin_lat = (max(all_lats) - min(all_lats)) * 0.15 + 0.003
            xmin = min(all_lons) - margin_lon
            xmax = max(all_lons) + margin_lon
            ymin = min(all_lats) - margin_lat
            ymax = max(all_lats) + margin_lat

            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

            ctx.add_basemap(
                ax,
                crs="EPSG:4326",
                source=ctx.providers.CartoDB.Positron,
                zoom="auto",
                attribution_size=6,
            )
        except Exception as e:
            print(f"[WARN] contextily no pudo cargar el fondo: {e}")
    else:
        print("[WARN] contextily o geopandas no disponibles; figura sin fondo cartográfico.")

    # --- Leyenda de rutas
    handles = [
        mlines.Line2D([], [], color=COLOR_TIME, linewidth=2.5,
                      label="Mínimo tiempo"),
        mlines.Line2D([], [], color=COLOR_IDW,  linewidth=2.5,
                      label="Mín. exposición IDW"),
        mlines.Line2D([], [], color=COLOR_LSTM, linewidth=2.5,
                      label="Mín. exposición LSTM+IDW"),
        mlines.Line2D([], [], marker="o", color="w", markerfacecolor="#2ca02c",
                      markersize=9, label=f"Inicio: {origin_name}"),
        mlines.Line2D([], [], marker="o", color="w", markerfacecolor="#d62728",
                      markersize=9, label=f"Destino: {dest_name}"),
    ]
    legend = ax.legend(
        handles=handles,
        loc="upper left",
        fontsize=9,
        framealpha=0.92,
        edgecolor="#cccccc",
        title="Rutas",
        title_fontsize=9,
    )

    # --- Caja de métricas
    def _row(mode):
        r = metrics.loc[metrics["mode"] == mode]
        if r.empty:
            return "—"
        r = r.iloc[0]
        return f"L={r['L_km']:.2f} km  T={r['T_min']:.1f} min  E={r['E']:.1f}"

    metric_text = (
        f"Mínimo tiempo:       {_row('min_time')}\n"
        f"Mín. expo. IDW:      {_row('min_expo_idw')}\n"
        f"Mín. expo. LSTM+IDW: {_row('min_expo_lstm')}"
    )
    ax.text(
        0.01, 0.01, metric_text,
        transform=ax.transAxes,
        fontsize=7.5,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.88, edgecolor="#aaaaaa"),
        family="monospace",
        zorder=12,
    )

    ax.set_xlabel("Longitud", fontsize=9)
    ax.set_ylabel("Latitud", fontsize=9)
    ax.set_title(
        f"Comparación de rutas peatonales — NO₂ — València\n"
        f"t₀ = {time_label}  (hora local)",
        fontsize=10,
    )
    ax.tick_params(labelsize=7)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] PNG -> {out_path}")


# ---------------------------------------------------------------------------
# Mapa HTML interactivo (folium)
# ---------------------------------------------------------------------------

def make_html(
    coords_time, coords_idw, coords_lstm,
    metrics: pd.DataFrame,
    origin_coords, dest_coords,
    origin_name: str, dest_name: str,
    out_path: Path,
):
    center_lat = (origin_coords[0] + dest_coords[0]) / 2.0
    center_lon = (origin_coords[1] + dest_coords[1]) / 2.0

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles="CartoDB positron",
        control_scale=True,
    )

    def _tooltip(mode_label, mode_key):
        r = metrics.loc[metrics["mode"] == mode_key]
        if r.empty:
            return mode_label
        r = r.iloc[0]
        return (
            f"{mode_label}<br>"
            f"L = {r['L_km']:.2f} km<br>"
            f"T = {r['T_min']:.1f} min<br>"
            f"E = {r['E']:.1f}"
        )

    def add_route(coords, color, layer_name, mode_key):
        fg = folium.FeatureGroup(name=layer_name)
        latlons = [(c[1], c[0]) for c in coords]
        folium.PolyLine(
            latlons, color=color, weight=5, opacity=0.9,
            tooltip=folium.Tooltip(_tooltip(layer_name, mode_key)),
        ).add_to(fg)
        fg.add_to(m)

    add_route(coords_time, COLOR_TIME, "Mínimo tiempo",            "min_time")
    add_route(coords_idw,  COLOR_IDW,  "Mín. exposición IDW",      "min_expo_idw")
    add_route(coords_lstm, COLOR_LSTM, "Mín. exposición LSTM+IDW", "min_expo_lstm")

    # Marcadores
    folium.Marker(
        location=[origin_coords[0], origin_coords[1]],
        tooltip=origin_name,
        icon=folium.Icon(color="green", icon="play", prefix="fa"),
    ).add_to(m)
    folium.Marker(
        location=[dest_coords[0], dest_coords[1]],
        tooltip=dest_name,
        icon=folium.Icon(color="red", icon="stop", prefix="fa"),
    ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path))
    print(f"[OK] HTML -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="p5_figura_tfg: genera PNG para LaTeX y mapa HTML interactivo"
    )
    ap.add_argument("--route-time",  required=True)
    ap.add_argument("--route-idw",   required=True)
    ap.add_argument("--route-lstm",  required=True)
    ap.add_argument("--metrics-csv", required=True)
    ap.add_argument("--origin-name", required=True)
    ap.add_argument("--dest-name",   required=True)
    ap.add_argument("--origin-coords", nargs=2, type=float, required=True,
                    metavar=("LAT", "LON"))
    ap.add_argument("--dest-coords",   nargs=2, type=float, required=True,
                    metavar=("LAT", "LON"))
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--var", default="NO2")
    ap.add_argument("--time-tag", default="20230515_0800",
                    help="Tag para nombres de fichero (default: 20230515_0800)")
    return ap.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    coords_time = load_linestring_coords(Path(args.route_time))
    coords_idw  = load_linestring_coords(Path(args.route_idw))
    coords_lstm = load_linestring_coords(Path(args.route_lstm))

    metrics = pd.read_csv(args.metrics_csv)
    for col in ("mode", "L_km", "T_min", "E"):
        if col not in metrics.columns:
            raise SystemExit(f"[FATAL] metrics_csv sin columna '{col}'")

    origin_coords = (args.origin_coords[0], args.origin_coords[1])
    dest_coords   = (args.dest_coords[0],   args.dest_coords[1])

    tag = args.time_tag
    var = args.var

    out_png  = outdir / f"figura_rutas_{var}_{tag}.png"
    out_html = outdir / f"mapa_rutas_{var}_{tag}.html"

    make_png(
        coords_time, coords_idw, coords_lstm,
        metrics,
        origin_coords, dest_coords,
        args.origin_name, args.dest_name,
        out_png,
        time_label=args.time_tag.replace("_", " "),
        dpi=args.dpi,
    )

    make_html(
        coords_time, coords_idw, coords_lstm,
        metrics,
        origin_coords, dest_coords,
        args.origin_name, args.dest_name,
        out_html,
    )

    print("[OK] p5_figura_tfg completado.")


if __name__ == "__main__":
    main()
