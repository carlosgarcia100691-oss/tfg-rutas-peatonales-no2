#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

"""
animate_lstm_gan_route.py  (v4 — heatmap PNG matplotlib + comparacion GAN vs LSTM)

Animacion HTML con:
  - Heatmap de campo GAN (NO2 escenario estres) interpolado por IDW,
    generado como imagen PNG con matplotlib y mostrado con L.imageOverlay.
    Cambia en cada paso temporal.
  - Circulos de estaciones coloreados por valor GAN; tooltip muestra
    tambien el valor LSTM base para comparar.
  - 3 marcadores animados sincronizados (min_time | min_expo_idw | min_expo_lstm).

Uso:
  python scripts/animate_lstm_gan_route.py ^
    --route-time  resultados/batch_gan/case_01/route_min_time_NO2_20230515_0800.geojson ^
    --route-idw   resultados/batch_gan/case_01/route_min_expo_idw_NO2_20230515_0800.geojson ^
    --route-lstm  resultados/batch_gan/case_01/route_min_expo_lstm_NO2_20230515_0800.geojson ^
    --gan-csv     resultados/batch_gan/case_01/gan_temporal_NO2_20230515_0800.csv ^
    --speed-kmh 4.8 --duration-min 100 --step-min 10 ^
    --out-dir resultados/animaciones
"""

import argparse
import base64
import io
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Lectura de archivos
# ─────────────────────────────────────────────

def _read_csv_any(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {path}")
    for sep in [",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(path, sep=sep, low_memory=False)
            if len(df.columns) > 1:
                return df
        except Exception:
            pass
    return pd.read_csv(path, low_memory=False)


def load_route(route_file):
    """Carga ruta GeoJSON o CSV. Devuelve DataFrame [lat, lon]."""
    path = Path(route_file)
    if not path.exists():
        raise FileNotFoundError(f"Ruta no encontrada: {path}")
    if path.suffix.lower() in (".geojson", ".json"):
        with open(path, "r", encoding="utf-8") as f:
            gj = json.load(f)
        coords = []
        for feat in gj.get("features", []):
            geom = feat.get("geometry", {})
            gtype = geom.get("type", "")
            if gtype == "LineString":
                for lon, lat in geom["coordinates"]:
                    coords.append({"lat": lat, "lon": lon})
            elif gtype == "MultiLineString":
                for seg in geom["coordinates"]:
                    for lon, lat in seg:
                        coords.append({"lat": lat, "lon": lon})
        if not coords:
            raise ValueError("GeoJSON sin coordenadas")
        return pd.DataFrame(coords)
    df = _read_csv_any(path)
    cl = {c.lower(): c for c in df.columns}
    lat_col = cl.get("lat") or cl.get("latitude") or cl.get("y")
    lon_col = cl.get("lon") or cl.get("longitude") or cl.get("x")
    if not (lat_col and lon_col):
        raise ValueError(f"Sin columnas lat/lon en {path}: {list(df.columns)}")
    out = df[[lat_col, lon_col]].rename(columns={lat_col: "lat", lon_col: "lon"})
    return out.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)


def load_gan_comparison(gan_csv):
    """
    Carga el CSV GAN temporal.
    Formato esperado: time, estacion, lat, lon, NO2, NO2_lstm_base
      NO2          = escenario GAN (estres)
      NO2_lstm_base = prediccion LSTM base (comparacion)
    Devuelve DataFrame [time, estacion, lat, lon, NO2_gan, NO2_lstm].
    """
    path = Path(gan_csv)
    if not path.exists():
        raise FileNotFoundError(f"CSV GAN no encontrado: {path}")
    df = _read_csv_any(path)
    cl = {c.lower(): c for c in df.columns}

    time_col = cl.get("time") or cl.get("fecha") or cl.get("timestamp")
    est_col  = cl.get("estacion") or cl.get("station")
    lat_col  = cl.get("lat") or cl.get("latitude")
    lon_col  = cl.get("lon") or cl.get("longitude")

    # Columna GAN: NO2 o NO2_gan
    gan_col  = cl.get("no2_gan") or cl.get("no2")
    # Columna LSTM base: NO2_lstm_base, lstm_base, no2_lstm
    lstm_col = (cl.get("no2_lstm_base") or cl.get("lstm_base")
                or cl.get("no2_lstm") or cl.get("no2_base"))

    if not (time_col and est_col and gan_col):
        raise ValueError(
            f"CSV GAN sin columnas esperadas. "
            f"Encontradas: {list(df.columns)}\n"
            f"Necesarias: time, estacion, NO2 [, NO2_lstm_base]")

    out = pd.DataFrame()
    out["time"]     = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    out["estacion"] = df[est_col].astype(str).str.strip()
    out["lat"]      = pd.to_numeric(df[lat_col], errors="coerce") if lat_col else np.nan
    out["lon"]      = pd.to_numeric(df[lon_col], errors="coerce") if lon_col else np.nan
    out["NO2_gan"]  = pd.to_numeric(df[gan_col], errors="coerce")
    out["NO2_lstm"] = pd.to_numeric(df[lstm_col], errors="coerce") if lstm_col else np.nan

    return out.dropna(subset=["time", "NO2_gan"]).reset_index(drop=True)


# ─────────────────────────────────────────────
# Utilidades geometricas
# ─────────────────────────────────────────────

def _haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat, dlon = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlon/2)**2
    return 2*R*math.asin(math.sqrt(max(0.0, min(1.0, a))))


def route_cumulative_km(route_df):
    lats, lons = route_df["lat"].to_numpy(), route_df["lon"].to_numpy()
    d = [0.0]
    for i in range(1, len(lats)):
        d.append(d[-1] + _haversine(lats[i-1], lons[i-1], lats[i], lons[i]))
    return np.array(d)


def walker_position(route_df, elapsed_min, speed_kmh):
    dist_km = min((elapsed_min / 60.0) * speed_kmh,
                  route_cumulative_km(route_df)[-1])
    cum = route_cumulative_km(route_df)
    if cum[-1] == 0:
        return float(route_df["lat"].iloc[0]), float(route_df["lon"].iloc[0])
    idx = min(int(np.searchsorted(cum, dist_km)), len(cum)-1)
    if idx == 0:
        return float(route_df["lat"].iloc[0]), float(route_df["lon"].iloc[0])
    d0, d1 = cum[idx-1], cum[idx]
    t = (dist_km - d0) / (d1 - d0) if d1 > d0 else 0.0
    lat = route_df["lat"].iloc[idx-1] + t*(route_df["lat"].iloc[idx]-route_df["lat"].iloc[idx-1])
    lon = route_df["lon"].iloc[idx-1] + t*(route_df["lon"].iloc[idx]-route_df["lon"].iloc[idx-1])
    return float(lat), float(lon)


def closest_time(df, t, delta_min=5):
    delta = pd.Timedelta(minutes=delta_min)
    sub = df[(df["time"] >= t - delta) & (df["time"] <= t + delta)]
    if sub.empty:
        nearest = df.loc[(df["time"] - t).abs().idxmin(), "time"]
        sub = df[df["time"] == nearest]
    return sub


# ─────────────────────────────────────────────
# IDW en malla
# ─────────────────────────────────────────────

def compute_idw_grid(st_lats, st_lons, st_vals, bbox,
                     n_grid=40, k=5, p=2.0):
    """
    IDW en malla n_grid x n_grid sobre bbox.
    bbox = (lat_min, lat_max, lon_min, lon_max)
    Devuelve array 2D (n_grid, n_grid) de valores interpolados.
    """
    valid = [(la, lo, v) for la, lo, v in zip(st_lats, st_lons, st_vals)
             if la is not None and lo is not None and v is not None
             and np.isfinite(la) and np.isfinite(lo) and np.isfinite(v)]
    if not valid:
        return None

    la_s = np.array([x[0] for x in valid])
    lo_s = np.array([x[1] for x in valid])
    v_s  = np.array([x[2] for x in valid])

    lat_min, lat_max, lon_min, lon_max = bbox
    gla, glo = np.meshgrid(
        np.linspace(lat_min, lat_max, n_grid),
        np.linspace(lon_min, lon_max, n_grid),
    )
    gla_flat = gla.flatten()
    glo_flat = glo.flatten()

    R = 6371.0
    lat1 = np.radians(gla_flat[:, None])
    lat2 = np.radians(la_s[None, :])
    dlat = lat2 - lat1
    dlon = np.radians(lo_s[None, :]) - np.radians(glo_flat[:, None])
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    dist_km = 2*R*np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    dist_km = np.where(dist_km < 1e-6, 1e-6, dist_km)

    w = 1.0 / dist_km**p
    k_eff = min(k, len(valid))
    if k_eff < len(valid):
        knn_idx = np.argsort(dist_km, axis=1)[:, :k_eff]
        w_knn = np.zeros_like(w)
        np.put_along_axis(w_knn, knn_idx,
                          np.take_along_axis(w, knn_idx, axis=1), axis=1)
        w = w_knn

    wsum = w.sum(axis=1, keepdims=True)
    idw_vals = (w / np.maximum(wsum, 1e-12) * v_s[None, :]).sum(axis=1)

    # Reshape: meshgrid es (n_lon, n_lat) → devolver (n_lat, n_lon) para imagen
    grid = idw_vals.reshape(n_grid, n_grid).T  # (n_lat, n_lon)
    return grid


# ─────────────────────────────────────────────
# Grid → PNG base64 (matplotlib)
# ─────────────────────────────────────────────

def grid_to_png_b64(grid_2d, vmin, vmax):
    """
    Convierte un array 2D de valores NO2 a imagen PNG transparente en base64.
    Usa matplotlib con colormap verde->amarillo->naranja->rojo.
    Retorna data URI 'data:image/png;base64,...' o None si matplotlib no disponible.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        return None

    if grid_2d is None:
        return None

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "no2", ["#00e400", "#ffff00", "#ff7e00", "#ff0000"])

    # Normalizar a [0, 1]
    vrange = max(vmax - vmin, 1.0)
    norm_grid = np.clip((grid_2d - vmin) / vrange, 0.0, 1.0)

    # Imagen RGBA: alpha dinamico — zonas limpias casi transparentes, sucias muy opacas
    rgba = cmap(norm_grid)                           # (H, W, 4)
    rgba[..., 3] = np.clip(norm_grid ** 1.3 * 0.92, 0.0, 0.92)

    # Invertir eje lat para que lat_max quede arriba (origen imagen = esquina superior)
    rgba = rgba[::-1, :, :]

    fig, ax = plt.subplots(figsize=(2, 2), dpi=120)
    ax.imshow(rgba, origin="upper", aspect="auto", interpolation="bilinear")
    ax.axis("off")
    fig.patch.set_alpha(0)
    plt.subplots_adjust(0, 0, 1, 1)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0,
                transparent=True, dpi=120)
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")


# ─────────────────────────────────────────────
# Construccion de frames
# ─────────────────────────────────────────────

def build_frames(times, gan_df,
                 route_time_df, route_idw_df, route_lstm_df,
                 speed_kmh, t0, bbox, vmin, vmax,
                 n_grid=40, k_idw=5, p_idw=2.0, step_min=10):
    """
    Para cada instante t:
      - Selecciona datos GAN del paso (NO2_gan + NO2_lstm)
      - Computa IDW en malla para el campo GAN
      - Genera imagen PNG del heatmap
      - Prepara circulos de estaciones con ambos valores
      - Calcula posicion de los 3 peatones
    """
    delta_search = max(step_min // 2, 1)
    frames = []

    for t in times:
        elapsed_min = (t - t0).total_seconds() / 60.0

        sub = closest_time(gan_df, t, delta_min=delta_search)
        agg = sub.groupby("estacion", as_index=False).agg(
            lat=("lat", "first"),
            lon=("lon", "first"),
            NO2_gan=("NO2_gan", "median"),
            NO2_lstm=("NO2_lstm", "median"),
        )

        st_lats, st_lons = [], []
        st_vals_gan, st_vals_lstm, st_names = [], [], []

        for _, row in agg.iterrows():
            la, lo = float(row["lat"]), float(row["lon"])
            if not (np.isfinite(la) and np.isfinite(lo)):
                continue
            v_gan  = float(row["NO2_gan"])  if np.isfinite(row["NO2_gan"])  else None
            v_lstm = float(row["NO2_lstm"]) if np.isfinite(row["NO2_lstm"]) else None
            st_lats.append(la)
            st_lons.append(lo)
            st_vals_gan.append(v_gan)
            st_vals_lstm.append(v_lstm)
            st_names.append(str(row["estacion"]))

        # IDW en malla para el campo GAN
        grid = compute_idw_grid(st_lats, st_lons, st_vals_gan, bbox,
                                n_grid=n_grid, k=k_idw, p=p_idw)
        img_b64 = grid_to_png_b64(grid, vmin, vmax)

        def wp(rdf):
            if rdf is None: return None
            la, lo = walker_position(rdf, elapsed_min, speed_kmh)
            return [round(la, 6), round(lo, 6)]

        frames.append({
            "time_str":    t.strftime("%Y-%m-%d %H:%M UTC"),
            "elapsed_min": int(elapsed_min),
            "img":  img_b64,
            "st": [
                {"la": la, "lo": lo,
                 "vg": round(vg, 1) if vg is not None else None,
                 "vl": round(vl, 1) if vl is not None else None,
                 "n":  n}
                for la, lo, vg, vl, n
                in zip(st_lats, st_lons, st_vals_gan, st_vals_lstm, st_names)
                if vg is not None
            ],
            "wt": wp(route_time_df),
            "wi": wp(route_idw_df),
            "wl": wp(route_lstm_df),
        })

    return frames


# ─────────────────────────────────────────────
# HTML Leaflet v4
# ─────────────────────────────────────────────

def build_html(frames, route_time_df, route_idw_df, route_lstm_df,
               out_path, start_time_str, vmin, vmax,
               duration_min, step_min, speed_kmh, bbox, has_gan=True):

    all_lats, all_lons = [], []
    for rdf in (route_time_df, route_idw_df, route_lstm_df):
        if rdf is not None:
            all_lats.extend(rdf["lat"].tolist())
            all_lons.extend(rdf["lon"].tolist())
    center_lat = float(np.mean(all_lats)) if all_lats else 39.47
    center_lon = float(np.mean(all_lons)) if all_lons else -0.37

    def to_js(rdf):
        if rdf is None: return "null"
        return json.dumps(
            [[round(float(r.lat), 6), round(float(r.lon), 6)] for _, r in rdf.iterrows()])

    r_time_js = to_js(route_time_df)
    r_idw_js  = to_js(route_idw_df)
    r_lstm_js = to_js(route_lstm_df)

    # Bounds del imageOverlay: [[lat_min, lon_min], [lat_max, lon_max]]
    lat_min, lat_max, lon_min, lon_max = bbox
    img_bounds_js = json.dumps([[lat_min, lon_min], [lat_max, lon_max]])

    # Labels segun modo
    date_str = start_time_str[:10] if len(start_time_str) >= 10 else start_time_str
    has_gan_js = "true" if has_gan else "false"
    if has_gan:
        page_title   = f"Animacion GAN vs LSTM NO2 | TFG Valencia"
        h2_text      = f"Escenario GAN de NO&#8322; vs prediccion LSTM | Valencia {date_str}"
        heatmap_desc = "Heatmap = campo GAN (escenario estres)"
        legend_label = "NO&#8322; GAN (&#xb5;g/m&#xb3;)"
    else:
        page_title   = f"Predicciones LSTM temporales NO2 | TFG Valencia"
        h2_text      = f"Predicciones LSTM temporales de NO&#8322; | Valencia {date_str}"
        heatmap_desc = "Heatmap = predicciones LSTM temporales"
        legend_label = "NO&#8322; LSTM pred. (&#xb5;g/m&#xb3;)"

    # Frames JSON (sin la imagen para no duplicar — la imagen va separada)
    # Separamos imgs del resto para que el JSON sea mas compacto
    imgs_js    = json.dumps([f["img"] if f["img"] else "" for f in frames])
    frames_js  = json.dumps([{
        "t":  f["time_str"],
        "em": f["elapsed_min"],
        "st": f["st"],
        "wt": f["wt"],
        "wi": f["wi"],
        "wl": f["wl"],
    } for f in frames])

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{page_title}</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:Arial,sans-serif;background:#111;display:flex;flex-direction:column;height:100vh}}
#header{{background:#12122a;color:#fff;padding:7px 18px;flex-shrink:0}}
#header h2{{font-size:14px;font-weight:bold;margin-bottom:2px}}
#header p{{font-size:11px;color:#aaa}}
#map{{flex:1;min-height:0}}
#controls{{background:#12122a;border-top:2px solid #e07b00;padding:9px 18px;flex-shrink:0}}
.ctrl-row{{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:5px}}
#time-display{{font-size:13px;font-weight:bold;color:#ffb347;min-width:240px}}
#slider{{flex:1;min-width:150px;height:6px;cursor:pointer;accent-color:#e07b00}}
#btn-play{{padding:5px 16px;background:#e07b00;color:#fff;border:none;border-radius:5px;
           font-size:14px;cursor:pointer;font-weight:bold}}
#btn-play:hover{{background:#b85e00}}
.legend{{display:flex;align-items:center;gap:10px;font-size:11px;flex-wrap:wrap}}
.leg-grad{{width:130px;height:12px;border-radius:3px;border:1px solid #555;
           background:linear-gradient(to right,#00e400,#ffff00,#ff7e00,#ff0000)}}
.leg-label{{color:#ccc}}
.leg-sep{{width:1px;height:16px;background:#444}}
.leg-item{{display:flex;align-items:center;gap:4px;color:#ccc}}
.leg-dot{{width:11px;height:11px;border-radius:50%;display:inline-block;
          border:2px solid rgba(255,255,255,0.4)}}
.leg-line{{width:20px;height:4px;border-radius:2px;display:inline-block}}
.leg-dash{{width:20px;height:0;border-top:3px dashed;display:inline-block}}
.leaflet-time-control{{
  background:rgba(0,0,0,0.78);color:#fff;border-radius:8px;padding:8px 14px;
  text-align:center;min-width:165px;border:1px solid rgba(255,180,0,0.3);
  backdrop-filter:blur(4px);pointer-events:none}}
.tc-time{{font-size:17px;font-weight:bold;color:#ffb347;letter-spacing:1px}}
.tc-elapsed{{font-size:11px;color:#aaa;margin-top:2px}}
.tc-label{{font-size:10px;color:#666;margin-bottom:3px;text-transform:uppercase;letter-spacing:1px}}
.leaflet-tooltip.no2-label{{
  background:none!important;border:none!important;box-shadow:none!important;
  font-size:10px;font-weight:bold;color:#fff;
  text-shadow:0 0 3px #000,0 0 3px #000;
  white-space:nowrap;padding:0}}
.tc-label{{font-size:10px;color:#666;margin-bottom:3px;
  text-transform:uppercase;letter-spacing:1px}}
</style>
</head>
<body>
<div id="header">
  <h2>{h2_text}</h2>
  <p>Inicio: {start_time_str} &nbsp;|&nbsp; {duration_min} min, paso {step_min} min
     &nbsp;|&nbsp; {speed_kmh} km/h &nbsp;|&nbsp;
     <span style="color:#ff7e00">&#9632;</span> {heatmap_desc}
     &nbsp;|&nbsp;
     <span style="color:#7ec8e3">&#9679;</span> Circulos = estaciones NO&#8322;</p>
</div>
<div id="map"></div>
<div id="controls">
  <div class="ctrl-row">
    <button id="btn-play">&#9654; Play</button>
    <input type="range" id="slider" min="0" max="{len(frames)-1}" value="0">
    <div id="time-display">Cargando...</div>
  </div>
  <div class="legend">
    <span class="leg-label"><b>{legend_label}:</b></span>
    <span class="leg-label">{vmin:.0f}</span>
    <div class="leg-grad"></div>
    <span class="leg-label">{vmax:.0f}</span>
    <div class="leg-sep"></div>
    <div class="leg-item"><span class="leg-dash" style="border-color:#ff7f0e"></span>Min. tiempo</div>
    <div class="leg-item"><span class="leg-line" style="background:#1f78b4"></span>Min. expo IDW</div>
    <div class="leg-item"><span class="leg-line" style="background:#39ff14"></span>Min. expo LSTM</div>
    <div class="leg-sep"></div>
    <div class="leg-item"><span class="leg-dot" style="background:#ff7f0e"></span>Peaton tiempo</div>
    <div class="leg-item"><span class="leg-dot" style="background:#1f78b4"></span>Peaton IDW</div>
    <div class="leg-item"><span class="leg-dot" style="background:#39ff14"></span>Peaton LSTM</div>
  </div>
</div>

<script>
var FRAMES    = {frames_js};
var IMGS      = {imgs_js};
var IMG_BOUNDS = {img_bounds_js};
var R_TIME = {r_time_js};
var R_IDW  = {r_idw_js};
var R_LSTM = {r_lstm_js};
var VMIN = {vmin:.2f}, VMAX = {vmax:.2f};
var HAS_GAN = {has_gan_js};

/* ── NO2 a color hex ── */
function no2Color(v) {{
  var t = Math.max(0, Math.min(1, (v - VMIN) / Math.max(VMAX - VMIN, 1)));
  var r, g, b;
  if (t < 0.35) {{ r = Math.round(t/0.35*255); g = 228; b = 0; }}
  else if (t < 0.65) {{ r = 255; g = Math.round(228*(1-(t-0.35)/0.30)); b = 0; }}
  else {{ r = 255; g = 0; b = 0; }}
  return 'rgb('+r+','+g+','+b+')';
}}

/* ── MAPA ── */
var map = L.map('map').setView([{center_lat:.5f}, {center_lon:.5f}], 14);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
  attribution: '&copy; OSM &copy; CARTO', subdomains: 'abcd', maxZoom: 19
}}).addTo(map);

/* ── RELOJ ── */
var TC = L.Control.extend({{
  onAdd: function() {{
    var d = L.DomUtil.create('div', 'leaflet-time-control');
    d.innerHTML = '<div class="tc-label">Tiempo simulado</div>' +
                  '<div class="tc-time" id="tc-time">--:--</div>' +
                  '<div class="tc-elapsed" id="tc-elapsed">t = 0 min</div>';
    return d;
  }}
}});
new TC({{position:'topright'}}).addTo(map);

/* ── HEATMAP IMAGEOVERLAY (PNG matplotlib) ── */
var heatOverlay = L.imageOverlay(
  IMGS[0] || '', IMG_BOUNDS, {{opacity: 0.82, interactive: false}}
).addTo(map);

/* ── RUTAS ESTATICAS ── */
function addRoute(pts, color, dashed, tip) {{
  if (!pts) return;
  var opt = {{color:color, weight:4.5, opacity:0.85}};
  if (dashed) opt.dashArray = '9 6';
  L.polyline(pts, opt).bindTooltip(tip).addTo(map);
}}
function addOD(pts, color) {{
  if (!pts || pts.length < 2) return;
  L.circleMarker(pts[0], {{radius:8,color:color,fillColor:'#fff',fillOpacity:1,weight:2.5}}).bindTooltip('Origen').addTo(map);
  L.circleMarker(pts[pts.length-1], {{radius:8,color:'#ff4444',fillColor:'#fff',fillOpacity:1,weight:2.5}}).bindTooltip('Destino').addTo(map);
}}
addRoute(R_TIME,'#ff7f0e',true, 'Ruta minimo tiempo');
addRoute(R_IDW, '#1f78b4',false,'Ruta minima exposicion IDW');
addRoute(R_LSTM,'#39ff14',false,'Ruta minima exposicion LSTM');
addOD(R_LSTM || R_IDW || R_TIME, '#aaa');

/* ── MARCADORES PEATONES ── */
function makeWalker(initPt, color, tip) {{
  if (!initPt) return null;
  return L.circleMarker(initPt, {{
    radius:12, color:'white', weight:2.5, fillColor:color, fillOpacity:0.95
  }}).bindTooltip(tip).addTo(map);
}}
var walkerTime = makeWalker(R_TIME ? R_TIME[0] : null,'#ff7f0e','Peaton min. tiempo');
var walkerIdw  = makeWalker(R_IDW  ? R_IDW[0]  : null,'#1f78b4','Peaton min. expo IDW');
var walkerLstm = makeWalker(R_LSTM ? R_LSTM[0]  : null,'#39ff14','Peaton min. expo LSTM');

/* ── CIRCULOS DE ESTACIONES ── */
var stLayer = L.layerGroup().addTo(map);
var curFrame = 0;
var prevVals = {{}};

function showFrame(idx) {{
  var f = FRAMES[idx];

  /* 1) Actualizar imagen del heatmap GAN */
  if (IMGS[idx]) heatOverlay.setUrl(IMGS[idx]);

  /* 2) Circulos de estaciones: color segun NO2, etiqueta permanente + flecha tendencia */
  stLayer.clearLayers();
  var newVals = {{}};
  (f.st || []).forEach(function(s) {{
    var ganColor = no2Color(s.vg);
    var prev = prevVals[s.n];
    var arrow = '';
    if (prev !== undefined) {{
      if (s.vg > prev + 0.15)      arrow = ' <span style="color:#ff5555">&#9650;</span>';
      else if (s.vg < prev - 0.15) arrow = ' <span style="color:#55ff55">&#9660;</span>';
      else                          arrow = ' <span style="color:#aaa">&#9644;</span>';
    }}
    newVals[s.n] = s.vg;
    var shortName = s.n.replace(/A[0-9]+_/,'');
    /* Etiqueta permanente: nombre + valor + tendencia */
    var permLabel = '<b style="color:#fff">'+s.vg.toFixed(1)+'</b>' + arrow;
    /* Tooltip al hover: detalle completo */
    var hoverTip =
      '<b>'+shortName+'</b><br>' +
      (HAS_GAN
        ? ('<span style="color:#ff9a4a">GAN: <b>'+s.vg.toFixed(1)+' &#xb5;g/m&#xb3;</b></span><br>'+
           '<span style="color:#7ec8e3">LSTM: <b>'+(s.vl!==null?s.vl.toFixed(1):'--')+' &#xb5;g/m&#xb3;</b></span>')
        : '<span style="color:#7ec8e3">LSTM pred.: <b>'+s.vg.toFixed(1)+' &#xb5;g/m&#xb3;</b></span>');
    var cm = L.circleMarker([s.la, s.lo], {{
      radius: 8,
      fillColor: ganColor,
      color: 'white',
      weight: 1.8,
      fillOpacity: 0.95
    }});
    cm.bindTooltip(permLabel, {{permanent:true, direction:'top', className:'no2-label', offset:[0,-4]}});
    cm.bindPopup(hoverTip);
    cm.addTo(stLayer);
  }});
  prevVals = newVals;

  /* 3) Peatones sincronizados */
  if (walkerTime && f.wt) walkerTime.setLatLng(f.wt);
  if (walkerIdw  && f.wi) walkerIdw.setLatLng(f.wi);
  if (walkerLstm && f.wl) walkerLstm.setLatLng(f.wl);

  /* 4) Reloj */
  var el = document.getElementById('tc-time');
  var ee = document.getElementById('tc-elapsed');
  if (el) el.textContent = f.t.replace(' UTC','') + ' UTC';
  if (ee) ee.textContent = 't = +' + f.em + ' min';
  document.getElementById('time-display').textContent = f.t + '  (+' + f.em + ' min)';
  document.getElementById('slider').value = idx;
  curFrame = idx;
}}

/* ── CONTROLES ── */
document.getElementById('slider').addEventListener('input', function() {{
  clearInterval(timer); playing = false;
  document.getElementById('btn-play').innerHTML = '&#9654; Play';
  showFrame(parseInt(this.value));
}});
var playing = false, timer = null;
document.getElementById('btn-play').addEventListener('click', function() {{
  if (playing) {{
    clearInterval(timer); playing = false;
    this.innerHTML = '&#9654; Play';
  }} else {{
    if (curFrame >= FRAMES.length - 1) curFrame = 0;
    playing = true;
    this.innerHTML = '&#9646;&#9646; Pausa';
    timer = setInterval(function() {{
      curFrame++;
      if (curFrame >= FRAMES.length) {{
        curFrame = FRAMES.length - 1;
        clearInterval(timer); playing = false;
        document.getElementById('btn-play').innerHTML = '&#9654; Play';
      }}
      showFrame(curFrame);
    }}, 1300);
  }}
}});

/* ── INICIO ── */
showFrame(0);
var allPts = [];
[R_TIME,R_IDW,R_LSTM].forEach(function(r){{ if(r) allPts=allPts.concat(r); }});
if (allPts.length) map.fitBounds(L.polyline(allPts).getBounds(), {{padding:[30,30]}});
</script>
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")
    print(f"[OK] HTML -> {out_path.resolve()}")
    return out_path


# ─────────────────────────────────────────────
# Argumentos
# ─────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description="Animacion GAN vs LSTM: heatmap PNG matplotlib + 3 marcadores sincronizados"
    )
    ap.add_argument("--route-lstm",  default=None, help="GeoJSON ruta min. expo LSTM (verde)")
    ap.add_argument("--route-idw",   default=None, help="GeoJSON ruta min. expo IDW (azul)")
    ap.add_argument("--route-time",  default=None, help="GeoJSON ruta min. tiempo (naranja)")
    # Alias v2/v3
    ap.add_argument("--route-file",  default=None, help="(alias --route-lstm)")
    ap.add_argument("--route-file2", default=None, help="(alias --route-time)")
    # Datos GAN (nuevo argumento principal)
    ap.add_argument("--gan-csv", default=None,
                    help="CSV GAN temporal con columnas: time,estacion,lat,lon,NO2,NO2_lstm_base")
    # Datos LSTM standalone (retrocompatibilidad, ignorado si --gan-csv presente)
    ap.add_argument("--lstm-file",    default=None, help="CSV LSTM solo (si no hay --gan-csv)")
    ap.add_argument("--stations-csv", default=None, help="CSV estaciones para coordenadas")
    ap.add_argument("--gan-file",     default=None, help="(ignorado en v4, usar --gan-csv)")
    # Tiempo
    ap.add_argument("--start-time",   default=None)
    ap.add_argument("--duration-min", type=int,   default=100)
    ap.add_argument("--step-min",     type=int,   default=10)
    # Fisicos
    ap.add_argument("--speed-kmh",    type=float, default=4.8)
    # IDW / grid
    ap.add_argument("--n-grid",  type=int,   default=40)
    ap.add_argument("--k-idw",   type=int,   default=5)
    ap.add_argument("--p-idw",   type=float, default=2.0)
    # Salida
    ap.add_argument("--out-dir",  default="resultados/animaciones")
    ap.add_argument("--out-name", default=None,
                    help="Nombre del archivo HTML de salida (defecto: ruta_lstm_gan.html)")
    # Obsoleto, silenciado
    ap.add_argument("--alpha",    type=float, default=0.5, help=argparse.SUPPRESS)
    ap.add_argument("--gan-pick", default="p90",           help=argparse.SUPPRESS)
    return ap.parse_args()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sep = "=" * 60
    print(f"\n{sep}\n  animate_lstm_gan_route.py v4 | TFG NO2 Valencia\n{sep}\n")

    route_lstm_path = args.route_lstm or args.route_file
    route_time_path = args.route_time or args.route_file2
    route_idw_path  = args.route_idw

    # 1) Rutas
    def try_load(path, label):
        if path is None: return None
        try:
            rdf = load_route(path)
            print(f"  [OK] {label}: {len(rdf)} nodos")
            return rdf
        except Exception as e:
            print(f"  [WARN] {label}: {e}")
            return None

    print("[1/4] Cargando rutas...")
    route_time_df = try_load(route_time_path, "min_time (naranja)")
    route_idw_df  = try_load(route_idw_path,  "min_expo_idw (azul)")
    route_lstm_df = try_load(route_lstm_path, "min_expo_lstm (verde)")

    # 2) Datos GAN con comparacion LSTM
    gan_df = None
    if args.gan_csv:
        print(f"[2/4] Cargando CSV GAN: {args.gan_csv}")
        gan_df = load_gan_comparison(args.gan_csv)
        ts_gan = sorted(gan_df["time"].dropna().unique())
        has_lstm_col = gan_df["NO2_lstm"].notna().any()
        print(f"  -> {len(gan_df)} filas | {gan_df['estacion'].nunique()} estaciones")
        print(f"  -> NO2_gan  [{gan_df['NO2_gan'].min():.1f}, {gan_df['NO2_gan'].max():.1f}] ug/m3")
        if has_lstm_col:
            print(f"  -> NO2_lstm [{gan_df['NO2_lstm'].min():.1f}, {gan_df['NO2_lstm'].max():.1f}] ug/m3")
        print(f"  -> Timestamps: {ts_gan[0]} ... {ts_gan[-1]} ({len(ts_gan)} pasos)")
    elif args.lstm_file:
        # Retrocompatibilidad: cargar LSTM y usarlo como campo (sin comparacion GAN)
        print(f"[2/4] Modo LSTM standalone: {args.lstm_file}")
        from animate_lstm_gan_route import load_predictions  # self-import
        lstm_raw = load_predictions(args.lstm_file, "LSTM")
        # Convertir al formato esperado por build_frames
        lstm_raw["NO2_gan"]  = lstm_raw["NO2"]
        lstm_raw["NO2_lstm"] = np.nan
        gan_df = lstm_raw.rename(columns={"NO2": "_NO2_drop"})[
            ["time","estacion","lat","lon","NO2_gan","NO2_lstm"]]
        ts_gan = sorted(gan_df["time"].dropna().unique())
        print(f"  -> {len(gan_df)} filas | {gan_df['estacion'].nunique()} estaciones")
    else:
        raise ValueError("Necesitas --gan-csv (o --lstm-file para modo compatibilidad)")

    # Detectar modo: GAN+LSTM (comparacion) vs LSTM solo (predicciones)
    has_gan_mode = gan_df["NO2_lstm"].notna().any() if gan_df is not None else False
    print(f"  -> Modo: {'GAN + LSTM comparacion' if has_gan_mode else 'LSTM predicciones temporales'}")

    # 3) Linea de tiempo
    ts_gan = sorted(gan_df["time"].dropna().unique())
    if args.start_time:
        try:    t0 = pd.to_datetime(args.start_time, utc=True)
        except: t0 = pd.to_datetime(args.start_time).tz_localize("Europe/Madrid").tz_convert("UTC")
    else:
        t0 = ts_gan[0]
        print(f"[INFO] start_time auto: {t0}")

    times = [t0 + pd.Timedelta(minutes=m)
             for m in range(0, args.duration_min + 1, args.step_min)]

    # Escala de color para el heatmap (p90 en lugar de p98 para más saturacion)
    all_gan = gan_df["NO2_gan"].dropna().tolist()
    if all_gan:
        vmin = 0.0
        vmax = float(np.nanpercentile(all_gan, 90))  # p90: satura el top 10% a rojo
        if vmax <= vmin: vmax = vmin + 20.0
    else:
        vmin, vmax = 0.0, 150.0
    print(f"  -> Escala NO2: [{vmin:.1f}, {vmax:.1f}] ug/m3 (0 - p90)")

    # Bbox para la malla IDW (union de rutas + estaciones)
    bbox_lats, bbox_lons = [], []
    for rdf in (route_time_df, route_idw_df, route_lstm_df):
        if rdf is not None:
            bbox_lats.extend(rdf["lat"].tolist())
            bbox_lons.extend(rdf["lon"].tolist())
    bbox_lats.extend(gan_df.loc[gan_df["lat"].notna(), "lat"].tolist())
    bbox_lons.extend(gan_df.loc[gan_df["lon"].notna(), "lon"].tolist())

    if bbox_lats:
        pad = 0.008
        bbox = (min(bbox_lats)-pad, max(bbox_lats)+pad,
                min(bbox_lons)-pad, max(bbox_lons)+pad)
    else:
        bbox = (39.40, 39.52, -0.45, -0.30)
        print("[WARN] Sin coordenadas, usando bbox Valencia por defecto")

    print(f"[3/4] Construyendo {len(times)} frames "
          f"(IDW {args.n_grid}x{args.n_grid}, PNG por frame)...")

    frames = build_frames(
        times, gan_df,
        route_time_df, route_idw_df, route_lstm_df,
        args.speed_kmh, t0, bbox, vmin, vmax,
        n_grid=args.n_grid, k_idw=args.k_idw, p_idw=args.p_idw,
        step_min=args.step_min,
    )

    n_with_img = sum(1 for f in frames if f["img"])
    print(f"  -> {len(frames)} frames | imagenes PNG: {n_with_img}/{len(frames)}")

    # 4) HTML
    print("[4/4] Generando HTML...")
    out_name = args.out_name or ("ruta_lstm_gan.html" if has_gan_mode else "ruta_lstm_predicciones.html")
    html_path = out_dir / out_name
    build_html(
        frames, route_time_df, route_idw_df, route_lstm_df,
        html_path,
        start_time_str=str(t0),
        vmin=vmin, vmax=vmax,
        duration_min=args.duration_min,
        step_min=args.step_min,
        speed_kmh=args.speed_kmh,
        bbox=bbox,
        has_gan=has_gan_mode,
    )

    print(f"\n{sep}")
    print(f"  COMPLETADO")
    print(f"  Abre: {html_path.resolve()}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
