#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
p7_batch_gan.py
===============
Batch GAN temporal de 10 casos.

Flujo por caso:
  1. Localiza snapshot_obs y lstm_preds del batch sin GAN (ya generados).
  2. Ejecuta p6_gan_temporal.py -> gan_temporal_NO2_*.csv
  3. Ejecuta p4_compute_routes.py con --lstm-csv = gan_temporal
  4. Ejecuta p5_figura_tfg.py -> PNG + HTML

Salidas en:
  resultados/batch_gan/case_XX/
  resultados/batch_gan/summary_gan_temporal.csv

Uso:
  python scripts/p7_batch_gan.py
    --no-gan-dir   <resultados/batch_no_gan>
    --outdir       <resultados/batch_gan>
    --python-exe   <ruta explicita al interprete>
    --scripts-dir  <scripts/>
    --model        <models/gan_no2_vectors_G.keras>
    --air-csv      <resultados/pipeline/aire_60min_raw.csv>
    --stations-csv <resultados/pipeline/stations_all.csv>
"""
from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Mismos 10 casos que el batch sin GAN
# ---------------------------------------------------------------------------
CASES = [
    dict(case_id="case_01", time_local="2023-05-15 08:00",
         origin_name="Olivereta", dest_name="Cabanyal",
         origin_lat=39.4692, origin_lon=-0.4059,
         dest_lat=39.4736,   dest_lon=-0.3254),
    dict(case_id="case_02", time_local="2023-05-15 18:00",
         origin_name="Olivereta", dest_name="Cabanyal",
         origin_lat=39.4692, origin_lon=-0.4059,
         dest_lat=39.4736,   dest_lon=-0.3254),
    dict(case_id="case_03", time_local="2023-01-16 08:00",
         origin_name="Olivereta", dest_name="Politecnic",
         origin_lat=39.4692, origin_lon=-0.4059,
         dest_lat=39.4796,   dest_lon=-0.3374),
    dict(case_id="case_04", time_local="2023-01-16 18:00",
         origin_name="Patraix", dest_name="Cabanyal",
         origin_lat=39.4592, origin_lon=-0.4101,
         dest_lat=39.4736,   dest_lon=-0.3254),
    dict(case_id="case_05", time_local="2023-03-21 08:00",
         origin_name="Patraix", dest_name="Politecnic",
         origin_lat=39.4592, origin_lon=-0.4101,
         dest_lat=39.4796,   dest_lon=-0.3374),
    dict(case_id="case_06", time_local="2023-07-10 08:00",
         origin_name="Olivereta", dest_name="Cabanyal",
         origin_lat=39.4692, origin_lon=-0.4059,
         dest_lat=39.4736,   dest_lon=-0.3254),
    dict(case_id="case_07", time_local="2023-07-10 20:00",
         origin_name="Patraix", dest_name="Cabanyal",
         origin_lat=39.4592, origin_lon=-0.4101,
         dest_lat=39.4736,   dest_lon=-0.3254),
    dict(case_id="case_08", time_local="2023-10-05 08:00",
         origin_name="Olivereta", dest_name="Politecnic",
         origin_lat=39.4692, origin_lon=-0.4059,
         dest_lat=39.4796,   dest_lon=-0.3374),
    dict(case_id="case_09", time_local="2023-10-05 18:00",
         origin_name="Patraix", dest_name="Cabanyal",
         origin_lat=39.4592, origin_lon=-0.4101,
         dest_lat=39.4736,   dest_lon=-0.3254),
    dict(case_id="case_10", time_local="2023-12-11 08:00",
         origin_name="Patraix", dest_name="Politecnic",
         origin_lat=39.4592, origin_lon=-0.4101,
         dest_lat=39.4796,   dest_lon=-0.3374),
]

VAR = "NO2"


def _tag(time_local: str) -> str:
    return pd.to_datetime(time_local).strftime("%Y%m%d_%H%M")


def _run(cmd: list, label: str) -> tuple[bool, str]:
    import os
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
    for l in lines[-5:]:
        print(f"    {l}")
    if result.returncode != 0:
        print(f"  [ERROR] {label}:")
        print(result.stderr[-1500:] if result.stderr else "(sin stderr)")
        return False, result.stderr
    return True, ""


def _geojson_first_last(path: Path):
    with open(path, encoding="utf-8") as f:
        gj = json.load(f)
    coords = gj["features"][0]["geometry"]["coordinates"]
    return coords[0], coords[-1]


def _routes_identical(case_dir: Path, tag: str) -> bool:
    paths = [
        case_dir / f"route_min_time_{VAR}_{tag}.geojson",
        case_dir / f"route_min_expo_idw_{VAR}_{tag}.geojson",
        case_dir / f"route_min_expo_lstm_{VAR}_{tag}.geojson",
    ]
    if not all(p.exists() for p in paths):
        return True
    coords = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            gj = json.load(f)
        coords.append(gj["features"][0]["geometry"]["coordinates"])
    return coords[0] == coords[1] == coords[2]


def run_case(case: dict, args, outdir_root: Path) -> dict:
    py  = args.python_exe
    sd  = args.scripts_dir
    cid = case["case_id"]
    t0  = case["time_local"]
    tag = _tag(t0)
    oname = case["origin_name"]
    dname = case["dest_name"]
    olat, olon = case["origin_lat"], case["origin_lon"]
    dlat, dlon = case["dest_lat"],   case["dest_lon"]

    case_dir = outdir_root / cid
    case_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[GAN BATCH] {cid}  t0={t0}  {oname} -> {dname}")
    print(f"{'='*60}")

    result = {
        "case_id": cid, "time": t0,
        "origin_name": oname, "dest_name": dname,
        "origin_lat": olat, "origin_lon": olon,
        "dest_lat": dlat,   "dest_lon": dlon,
        "status": "ok",
        "same_geometry_flag": False,
        "L_min_time_km": None, "T_min_time_min": None, "E_min_time": None,
        "L_min_expo_idw_km": None, "T_min_expo_idw_min": None, "E_min_expo_idw": None,
        "L_min_expo_lstm_km": None, "T_min_expo_lstm_min": None, "E_min_expo_lstm": None,
        "png_path": "", "html_path": "", "gan_csv_path": "",
    }

    # --- Localizar artefactos del batch sin GAN
    no_gan_case = Path(args.no_gan_dir) / cid
    snapshot_csv = no_gan_case / f"snapshot_obs_{VAR}_{tag}.csv"
    lstm_csv     = no_gan_case / f"lstm_preds_{VAR}_{tag}.csv"

    if not snapshot_csv.exists():
        print(f"  [ERROR] No existe snapshot: {snapshot_csv}")
        result["status"] = "error_missing_snapshot"; return result
    if not lstm_csv.exists():
        print(f"  [ERROR] No existe lstm_preds: {lstm_csv}")
        result["status"] = "error_missing_lstm"; return result

    # --- S6: GAN temporal
    gan_csv = case_dir / f"gan_temporal_{VAR}_{tag}.csv"
    ok, _ = _run([
        py, f"{sd}/p6_gan_temporal.py",
        "--lstm-csv",     str(lstm_csv),
        "--model",        args.model,
        "--air-csv",      args.air_csv,
        "--stations-csv", args.stations_csv,
        "--var",          VAR,
        "--stress-factor", str(args.stress_factor),
        "--noise-scale",   str(args.noise_scale),
        "--seed",          str(args.seed),
        "--out",           str(gan_csv),
    ], "S6-gan-temporal")
    if not ok:
        result["status"] = "error_s6"; return result
    result["gan_csv_path"] = str(gan_csv)

    # Extraer snapshot GAN en t0 (primer paso temporal) para usarlo como
    # campo de referencia en p4. En el escenario GAN, el "mundo real" es
    # el campo GAN, no el snapshot observado.
    import pandas as _pd
    gan_df = _pd.read_csv(gan_csv)
    time_col_gan = next((c for c in ("time","fecha","timestamp") if c in gan_df.columns), None)
    gan_df[time_col_gan] = _pd.to_datetime(gan_df[time_col_gan], format="mixed", errors="coerce")
    t0_gan = gan_df[time_col_gan].dropna().min()
    gan_snapshot = gan_df.loc[gan_df[time_col_gan] == t0_gan, ["estacion","lat","lon",VAR]].copy()
    # Añadir columna fecha para compatibilidad con p4
    gan_snapshot.insert(0, "fecha", str(t0_gan))
    gan_snapshot_csv = case_dir / f"gan_snapshot_t0_{VAR}_{tag}.csv"
    gan_snapshot.to_csv(gan_snapshot_csv, index=False)

    # --- S4: rutas con escenario GAN temporal
    # --snapshot-csv = campo GAN en t0 (no el observado real)
    # --lstm-csv     = serie GAN temporal completa
    ok, _ = _run([
        py, f"{sd}/p4_compute_routes.py",
        "--snapshot-csv", str(gan_snapshot_csv),  # campo GAN en t0 como referencia
        "--lstm-csv",     str(gan_csv),            # serie GAN temporal completa
        "--var",          VAR,
        "--time",         t0,
        "--origin",       str(olat), str(olon),
        "--dest",         str(dlat), str(dlon),
        "--r-idw-km",     str(args.r_idw_km),
        "--k-idw",        str(args.k_idw),
        "--p-idw",        str(args.p_idw),
        "--buffer-km",    str(args.buffer_km),
        "--speed-kmh",    str(args.speed_kmh),
        "--outdir",       str(case_dir),
        "--eval-dynamic",  # evaluar con campo GAN temporal real, no con t0 fijo
    ], "S4-routes-gan")
    if not ok:
        result["status"] = "error_s4"; return result

    # Leer metricas
    met_csv = case_dir / f"routes_metrics_{VAR}_{tag}.csv"
    if met_csv.exists():
        met = pd.read_csv(met_csv)
        for _, row in met.iterrows():
            m = row["mode"]
            if m == "min_time":
                result["L_min_time_km"]   = row["L_km"]
                result["T_min_time_min"]  = row["T_min"]
                result["E_min_time"]      = row["E"]
            elif m == "min_expo_idw":
                result["L_min_expo_idw_km"]  = row["L_km"]
                result["T_min_expo_idw_min"] = row["T_min"]
                result["E_min_expo_idw"]     = row["E"]
            elif m == "min_expo_lstm":
                result["L_min_expo_lstm_km"]  = row["L_km"]
                result["T_min_expo_lstm_min"] = row["T_min"]
                result["E_min_expo_lstm"]     = row["E"]

    result["same_geometry_flag"] = _routes_identical(case_dir, tag)

    # Primer/ultimo vertice real para marcadores
    route_time_gj = case_dir / f"route_min_time_{VAR}_{tag}.geojson"
    if route_time_gj.exists():
        first_c, last_c = _geojson_first_last(route_time_gj)
        real_olat, real_olon = first_c[1], first_c[0]
        real_dlat, real_dlon = last_c[1],  last_c[0]
    else:
        real_olat, real_olon = olat, olon
        real_dlat, real_dlon = dlat, dlon

    # --- S5: figura
    out_png  = case_dir / f"figura_{cid}.png"
    out_html = case_dir / f"mapa_{cid}.html"
    ok, _ = _run([
        py, f"{sd}/p5_figura_tfg.py",
        "--route-time",  str(case_dir / f"route_min_time_{VAR}_{tag}.geojson"),
        "--route-idw",   str(case_dir / f"route_min_expo_idw_{VAR}_{tag}.geojson"),
        "--route-lstm",  str(case_dir / f"route_min_expo_lstm_{VAR}_{tag}.geojson"),
        "--metrics-csv", str(met_csv),
        "--origin-name", oname,
        "--dest-name",   dname,
        "--origin-coords", str(real_olat), str(real_olon),
        "--dest-coords",   str(real_dlat), str(real_dlon),
        "--outdir",      str(case_dir),
        "--dpi",         str(args.dpi),
        "--time-tag",    tag,
    ], "S5-figura-gan")
    if not ok:
        result["status"] = "error_s5"
    else:
        # p5 genera con time-tag como sufijo; renombrar a nombres limpios
        gen_png  = case_dir / f"figura_rutas_{VAR}_{tag}.png"
        gen_html = case_dir / f"mapa_rutas_{VAR}_{tag}.html"
        if gen_png.exists():
            if out_png.exists():
                out_png.unlink()
            gen_png.rename(out_png)
        if gen_html.exists():
            if out_html.exists():
                out_html.unlink()
            gen_html.rename(out_html)
        result["png_path"]  = str(out_png)
        result["html_path"] = str(out_html)

    return result


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gan-dir",   required=True,
                    help="Directorio raiz del batch sin GAN (resultados/batch_no_gan)")
    ap.add_argument("--outdir",       required=True,
                    help="Directorio raiz de salida del batch GAN")
    ap.add_argument("--python-exe",   required=True)
    ap.add_argument("--scripts-dir",  required=True)
    ap.add_argument("--model",        required=True,
                    help="Ruta al generador GAN (.keras)")
    ap.add_argument("--air-csv",      required=True)
    ap.add_argument("--stations-csv", required=True)
    ap.add_argument("--stress-factor", type=float, default=1.5)
    ap.add_argument("--noise-scale",   type=float, default=0.3)
    ap.add_argument("--seed",          type=int,   default=42)
    ap.add_argument("--r-idw-km",  type=float, default=10.0,
                    help="Radio IDW en km (default: 10.0 — definitivo TFG)")
    ap.add_argument("--k-idw",     type=int,   default=5,
                    help="Vecinos IDW (default: 5 — definitivo TFG)")
    ap.add_argument("--p-idw",     type=float, default=2.0,
                    help="Exponente IDW (default: 2.0 — definitivo TFG)")
    ap.add_argument("--buffer-km", type=float, default=3.0)
    ap.add_argument("--speed-kmh", type=float, default=4.8,
                    help="Velocidad peatonal km/h (default: 4.8)")
    ap.add_argument("--dpi",       type=int,   default=300)
    return ap.parse_args()


def main():
    args = parse_args()
    outdir_root = Path(args.outdir)
    outdir_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for case in CASES:
        row = run_case(case, args, outdir_root)
        summary_rows.append(row)
        print(f"  -> {case['case_id']}: {row['status']}  same_geom={row['same_geometry_flag']}")

    summary = pd.DataFrame(summary_rows)
    summary_path = outdir_root / "summary_gan_temporal.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\n[OK] summary_gan_temporal.csv -> {summary_path}")

    ok_count   = (summary["status"] == "ok").sum()
    same_count = int(summary["same_geometry_flag"].sum())
    print(f"[INFO] Casos completados: {ok_count}/{len(CASES)}")
    if same_count > 0:
        print(f"[WARN] {same_count} casos con rutas identicas (same_geometry_flag=True)")


if __name__ == "__main__":
    main()
