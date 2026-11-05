#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
p6_batch_no_gan.py
==================
Batch de 10 casos SIN GAN.

Para cada caso:
  1. Extrae snapshot observado en t0 (S3)
  2. Genera predicciones LSTM (S2)
  3. Calcula las tres rutas (S4)
  4. Genera PNG y HTML (S5)

Reutiliza aire_60min_raw.csv, aire_10min_raw.csv y stations_all.csv
ya generados por S1. No rehace S1.

Salidas:
  resultados/batch_no_gan/case_XX/  (por cada caso)
  resultados/batch_no_gan/batch_summary.csv

Uso:
  python scripts/p6_batch_no_gan.py
    --pipeline-dir  <dir con CSVs de S1>
    --outdir        <dir raiz batch>
    --python-exe    <ruta explícita al intérprete>
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Definición de los 10 casos
# ---------------------------------------------------------------------------
CASES = [
    dict(
        case_id="case_01",
        time_local="2023-05-15 08:00",
        origin_name="Olivereta",
        dest_name="Cabanyal",
        origin_lat=39.4692, origin_lon=-0.4059,
        dest_lat=39.4736,   dest_lon=-0.3254,
    ),
    dict(
        case_id="case_02",
        time_local="2023-05-15 18:00",
        origin_name="Olivereta",
        dest_name="Cabanyal",
        origin_lat=39.4692, origin_lon=-0.4059,
        dest_lat=39.4736,   dest_lon=-0.3254,
    ),
    dict(
        case_id="case_03",
        time_local="2023-01-16 08:00",
        origin_name="Olivereta",
        dest_name="Politecnic",
        origin_lat=39.4692, origin_lon=-0.4059,
        dest_lat=39.4796,   dest_lon=-0.3374,
    ),
    dict(
        case_id="case_04",
        time_local="2023-01-16 18:00",
        origin_name="Patraix",
        dest_name="Cabanyal",
        origin_lat=39.4592, origin_lon=-0.4101,
        dest_lat=39.4736,   dest_lon=-0.3254,
    ),
    dict(
        case_id="case_05",
        time_local="2023-03-21 08:00",
        origin_name="Patraix",
        dest_name="Politecnic",
        origin_lat=39.4592, origin_lon=-0.4101,
        dest_lat=39.4796,   dest_lon=-0.3374,
    ),
    dict(
        case_id="case_06",
        time_local="2023-07-10 08:00",
        origin_name="Olivereta",
        dest_name="Cabanyal",
        origin_lat=39.4692, origin_lon=-0.4059,
        dest_lat=39.4736,   dest_lon=-0.3254,
    ),
    dict(
        case_id="case_07",
        time_local="2023-07-10 20:00",
        origin_name="Patraix",
        dest_name="Cabanyal",
        origin_lat=39.4592, origin_lon=-0.4101,
        dest_lat=39.4736,   dest_lon=-0.3254,
    ),
    dict(
        case_id="case_08",
        time_local="2023-10-05 08:00",
        origin_name="Olivereta",
        dest_name="Politecnic",
        origin_lat=39.4692, origin_lon=-0.4059,
        dest_lat=39.4796,   dest_lon=-0.3374,
    ),
    dict(
        case_id="case_09",
        time_local="2023-10-05 18:00",
        origin_name="Patraix",
        dest_name="Cabanyal",
        origin_lat=39.4592, origin_lon=-0.4101,
        dest_lat=39.4736,   dest_lon=-0.3254,
    ),
    dict(
        case_id="case_10",
        time_local="2023-12-11 08:00",
        origin_name="Patraix",
        dest_name="Politecnic",
        origin_lat=39.4592, origin_lon=-0.4101,
        dest_lat=39.4796,   dest_lon=-0.3374,
    ),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tag(time_local: str) -> str:
    return pd.to_datetime(time_local).strftime("%Y%m%d_%H%M")


def _run(cmd: list[str], label: str) -> bool:
    print(f"  [{label}] {' '.join(str(c) for c in cmd[:4])} ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [ERROR] {label} falló:")
        print(result.stderr[-2000:] if result.stderr else "(sin stderr)")
        return False
    # Mostrar últimas líneas de stdout
    lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
    for l in lines[-4:]:
        print(f"    {l}")
    return True


def _geojson_first_last(path: Path):
    """Devuelve (primer_vertice, ultimo_vertice) de un GeoJSON LineString."""
    with open(path, encoding="utf-8") as f:
        gj = json.load(f)
    coords = gj["features"][0]["geometry"]["coordinates"]
    return coords[0], coords[-1]  # [lon, lat]


def _routes_identical(case_dir: Path, tag: str) -> bool:
    paths = [
        case_dir / f"route_min_time_NO2_{tag}.geojson",
        case_dir / f"route_min_expo_idw_NO2_{tag}.geojson",
        case_dir / f"route_min_expo_lstm_NO2_{tag}.geojson",
    ]
    if not all(p.exists() for p in paths):
        return True  # si falta alguno, marcar como problema
    coords = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            gj = json.load(f)
        coords.append(gj["features"][0]["geometry"]["coordinates"])
    return coords[0] == coords[1] == coords[2]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline-dir", required=True,
                    help="Directorio con aire_60min_raw.csv, aire_10min_raw.csv, stations_all.csv")
    ap.add_argument("--outdir", required=True,
                    help="Directorio raíz de salida del batch")
    ap.add_argument("--python-exe", required=True,
                    help="Ruta explícita al intérprete Python")
    ap.add_argument("--scripts-dir", required=True,
                    help="Directorio con los scripts p2-p5")
    ap.add_argument("--var", default="NO2")
    ap.add_argument("--lookback", type=int, default=144)
    ap.add_argument("--train-extra", type=int, default=432)
    ap.add_argument("--horizon-steps", type=int, default=6)
    ap.add_argument("--r-idw-km", type=float, default=10.0,
                    help="Radio IDW en km (default: 10.0 — definitivo TFG)")
    ap.add_argument("--k-idw", type=int, default=5,
                    help="Vecinos IDW (default: 5 — definitivo TFG)")
    ap.add_argument("--p-idw", type=float, default=2.0,
                    help="Exponente IDW (default: 2.0 — definitivo TFG)")
    ap.add_argument("--buffer-km", type=float, default=3.0)
    ap.add_argument("--speed-kmh", type=float, default=4.8,
                    help="Velocidad peatonal km/h (default: 4.8)")
    ap.add_argument("--dpi", type=int, default=300)
    return ap.parse_args()


def run_case(case: dict, args, outdir_root: Path) -> dict:
    py = args.python_exe
    sd = args.scripts_dir
    pd_dir = args.pipeline_dir
    var = args.var

    cid = case["case_id"]
    t0 = case["time_local"]
    tag = _tag(t0)
    oname = case["origin_name"]
    dname = case["dest_name"]
    olat, olon = case["origin_lat"], case["origin_lon"]
    dlat, dlon = case["dest_lat"],   case["dest_lon"]

    case_dir = outdir_root / cid
    case_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[CASE] {cid}  t0={t0}  {oname} -> {dname}")
    print(f"{'='*60}")

    result = {
        "case_id": cid,
        "time_local": t0,
        "origin_name": oname,
        "dest_name": dname,
        "origin_lat": olat, "origin_lon": olon,
        "dest_lat": dlat,   "dest_lon": dlon,
        "status": "ok",
        "same_geometry_flag": False,
        "html_path": "", "png_path": "",
        "L_km_min_time": None, "T_min_min_time": None, "E_min_time": None,
        "L_km_idw": None, "T_min_idw": None, "E_idw": None,
        "L_km_lstm": None, "T_min_lstm": None, "E_lstm": None,
    }

    # --- S3: snapshot observado
    snap_csv = case_dir / f"snapshot_obs_{var}_{tag}.csv"
    ok = _run([
        py, f"{sd}/p3_snapshot_observed.py",
        "--air-csv",      f"{pd_dir}/aire_60min_raw.csv",
        "--stations-csv", f"{pd_dir}/stations_all.csv",
        "--var", var, "--time", t0,
        "--out", str(snap_csv),
    ], "S3-snapshot")
    if not ok:
        result["status"] = "error_s3"; return result

    # --- S2: predicciones LSTM
    lstm_csv = case_dir / f"lstm_preds_{var}_{tag}.csv"
    ok = _run([
        py, f"{sd}/p2_lstm_predict.py",
        "--air-csv",      f"{pd_dir}/aire_10min_raw.csv",
        "--stations-csv", f"{pd_dir}/stations_all.csv",
        "--var", var, "--time", t0,
        "--lookback",      str(args.lookback),
        "--train-extra",   str(args.train_extra),
        "--horizon-steps", str(args.horizon_steps),
        "--allow-fallback",
        "--out", str(lstm_csv),
    ], "S2-lstm")
    if not ok:
        result["status"] = "error_s2"; return result

    # --- S4: rutas
    ok = _run([
        py, f"{sd}/p4_compute_routes.py",
        "--snapshot-csv", str(snap_csv),
        "--lstm-csv",     str(lstm_csv),
        "--var", var, "--time", t0,
        "--origin", str(olat), str(olon),
        "--dest",   str(dlat), str(dlon),
        "--r-idw-km",  str(args.r_idw_km),
        "--k-idw",     str(args.k_idw),
        "--p-idw",     str(args.p_idw),
        "--buffer-km", str(args.buffer_km),
        "--speed-kmh", str(args.speed_kmh),
        "--outdir",    str(case_dir),
    ], "S4-routes")
    if not ok:
        result["status"] = "error_s4"; return result

    # Leer métricas
    met_csv = case_dir / f"routes_metrics_{var}_{tag}.csv"
    if met_csv.exists():
        met = pd.read_csv(met_csv)
        for _, row in met.iterrows():
            m = row["mode"]
            if m == "min_time":
                result["L_km_min_time"] = row["L_km"]
                result["T_min_min_time"] = row["T_min"]
                result["E_min_time"]     = row["E"]
            elif m == "min_expo_idw":
                result["L_km_idw"] = row["L_km"]
                result["T_min_idw"] = row["T_min"]
                result["E_idw"]     = row["E"]
            elif m == "min_expo_lstm":
                result["L_km_lstm"] = row["L_km"]
                result["T_min_lstm"] = row["T_min"]
                result["E_lstm"]     = row["E"]

    # Verificar si las rutas son idénticas
    result["same_geometry_flag"] = _routes_identical(case_dir, tag)

    # Extraer primer y último vértice de la ruta de mínimo tiempo
    # para colocar marcadores exactamente sobre la geometría
    route_time_gj = case_dir / f"route_min_time_{var}_{tag}.geojson"
    if route_time_gj.exists():
        first_coord, last_coord = _geojson_first_last(route_time_gj)
        # first_coord = [lon, lat], last_coord = [lon, lat]
        real_origin_lat = first_coord[1]
        real_origin_lon = first_coord[0]
        real_dest_lat   = last_coord[1]
        real_dest_lon   = last_coord[0]
    else:
        real_origin_lat, real_origin_lon = olat, olon
        real_dest_lat,   real_dest_lon   = dlat, dlon

    # --- S5: figura
    out_png  = case_dir / f"figura_rutas_{var}_{tag}.png"
    out_html = case_dir / f"mapa_rutas_{var}_{tag}.html"
    ok = _run([
        py, f"{sd}/p5_figura_tfg.py",
        "--route-time",  str(case_dir / f"route_min_time_{var}_{tag}.geojson"),
        "--route-idw",   str(case_dir / f"route_min_expo_idw_{var}_{tag}.geojson"),
        "--route-lstm",  str(case_dir / f"route_min_expo_lstm_{var}_{tag}.geojson"),
        "--metrics-csv", str(met_csv),
        "--origin-name", oname,
        "--dest-name",   dname,
        "--origin-coords", str(real_origin_lat), str(real_origin_lon),
        "--dest-coords",   str(real_dest_lat),   str(real_dest_lon),
        "--outdir",      str(case_dir),
        "--dpi",         str(args.dpi),
        "--time-tag",    tag,
    ], "S5-figura")
    if not ok:
        result["status"] = "error_s5"
    else:
        result["html_path"] = str(out_html)
        result["png_path"]  = str(out_png)

    return result


def main():
    args = parse_args()
    outdir_root = Path(args.outdir)
    outdir_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for case in CASES:
        row = run_case(case, args, outdir_root)
        summary_rows.append(row)
        status = row["status"]
        same = row["same_geometry_flag"]
        print(f"  -> {case['case_id']}: {status}  same_geom={same}")

    summary = pd.DataFrame(summary_rows)
    summary_path = outdir_root / "batch_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\n[OK] batch_summary.csv -> {summary_path}")

    ok_count = (summary["status"] == "ok").sum()
    print(f"[INFO] Casos completados: {ok_count}/{len(CASES)}")
    same_count = summary["same_geometry_flag"].sum()
    if same_count > 0:
        print(f"[WARN] {same_count} casos con rutas idénticas (same_geometry_flag=True)")


if __name__ == "__main__":
    main()
