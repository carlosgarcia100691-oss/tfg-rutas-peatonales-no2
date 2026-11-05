#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
p3_snapshot_observed.py
=======================
Etapa C del pipeline: extracción del snapshot observado real en t0.

Lee el CSV de observaciones a 60min y extrae, para cada estación,
el valor real observado más cercano a t0 dentro de una ventana de tolerancia.

Este CSV representa el campo estático observado que se usará para
calcular la ruta de mínima exposición IDW (campo real, no predicho).

Salida: CSV con columnas fecha, estacion, lat, lon, <var>
        Una fila por estación.

Uso:
  python scripts/p3_snapshot_observed.py
    --air-csv      <aire_60min_raw.csv>
    --stations-csv <stations_all.csv>
    --var          NO2
    --time         "2023-05-15 08:00"
    --out          <snapshot_obs_NO2_20230515_0800.csv>
"""
from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser(
        description="p3_snapshot_observed: extrae snapshot observado real en t0"
    )
    ap.add_argument("--air-csv", required=True,
                    help="aire_60min_raw.csv (fecha UTC, estacion, <var>)")
    ap.add_argument("--stations-csv", required=True,
                    help="stations_all.csv (estacion, lat, lon)")
    ap.add_argument("--var", default="NO2",
                    help="Variable a extraer (default: NO2)")
    ap.add_argument("--time", required=True,
                    help='t0 local, ej: "2023-05-15 08:00"')
    ap.add_argument("--tz", default="Europe/Madrid")
    ap.add_argument("--tol-h", type=float, default=1.0,
                    help="Tolerancia en horas para buscar el valor más cercano a t0 (default: 1.0)")
    ap.add_argument("--out", required=True,
                    help="CSV de salida con el snapshot observado")
    return ap.parse_args()


def main():
    args = parse_args()

    # --- Carga
    air = pd.read_csv(args.air_csv)
    need = {"fecha", "estacion", args.var}
    if not need.issubset(air.columns):
        raise SystemExit(f"[FATAL] {args.air_csv} debe tener columnas: {need}")
    # Leer como naive UTC (S1 guarda sin tzinfo). Toda la lógica temporal es UTC naive.
    # format='mixed' tolera variaciones en el formato ISO (con o sin milisegundos, etc.)
    air["fecha"] = pd.to_datetime(air["fecha"], errors="coerce", format="mixed")
    air[args.var] = pd.to_numeric(air[args.var], errors="coerce")

    st = pd.read_csv(args.stations_csv)
    if not {"estacion", "lat", "lon"}.issubset(st.columns):
        raise SystemExit("[FATAL] stations_csv requiere columnas: estacion, lat, lon")

    # t0 en UTC naive (coherente con cómo S1 guarda las fechas)
    t0_local = pd.to_datetime(args.time)
    t0_utc = (
        t0_local
        .tz_localize(args.tz, nonexistent="shift_forward", ambiguous="NaT")
        .tz_convert("UTC")
        .tz_localize(None)   # → naive UTC, igual que air["fecha"]
    )
    tol = pd.Timedelta(hours=args.tol_h)

    print(f"[INFO] t0 UTC = {t0_utc}  tolerancia = ±{args.tol_h}h")

    # --- Ventana alrededor de t0
    window = air.loc[
        (air["fecha"] >= t0_utc - tol) & (air["fecha"] <= t0_utc + tol)
    ].copy()

    if window.empty:
        raise SystemExit(
            f"[FATAL] No hay observaciones en ventana ±{args.tol_h}h alrededor de {t0_utc}. "
            "Prueba a aumentar --tol-h."
        )

    # Para cada estación, tomar el registro más cercano a t0
    window["_dt"] = (window["fecha"] - t0_utc).abs()
    idx = window.groupby("estacion")["_dt"].idxmin()
    snap = window.loc[idx].copy()
    snap = snap.dropna(subset=[args.var])

    # Añadir coordenadas
    snap = snap.merge(st[["estacion", "lat", "lon"]], on="estacion", how="inner")
    snap = snap[["fecha", "estacion", "lat", "lon", args.var]].reset_index(drop=True)

    if snap.empty:
        raise SystemExit("[FATAL] Snapshot vacío tras merge con estaciones.")

    # Convertir fecha a string ISO para legibilidad
    snap["fecha"] = snap["fecha"].astype(str)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    snap.to_csv(out_path, index=False)

    print(f"[OK] snapshot_obs -> {out_path}  (estaciones={len(snap)})")
    print(f"[INFO] {args.var}: min={snap[args.var].min():.2f}  "
          f"median={snap[args.var].median():.2f}  max={snap[args.var].max():.2f}")


if __name__ == "__main__":
    main()
