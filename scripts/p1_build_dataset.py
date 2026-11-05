#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
p1_build_dataset.py
===================
Etapa A+B del pipeline: lectura y normalización de JSON de aire.

Lee los JSON de aire a 60min y a 10min (formato CITCOM.ai).
Soporta automáticamente:
  - ficheros .json sueltos en el directorio
  - ficheros .zip que contienen .json en su interior

Produce:
  - aire_60min_raw.csv  (fecha UTC, estacion, NO2, PM10, PM25, O3, SO2)
  - aire_10min_raw.csv  (fecha UTC, estacion, NO2)
  - stations_all.csv    (estacion, lat, lon)

Las coordenadas de estaciones se extraen de los propios JSON de aire.
No se usa meteorología en esta versión del pipeline.

Uso:
  python scripts/p1_build_dataset.py
    --aire-json-60min <dir>
    --aire-json-10min <dir>
    --outdir          <dir>
"""
from __future__ import annotations
import argparse
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Utilidades de parseo
# ---------------------------------------------------------------------------

def _parse_date(v) -> pd.Timestamp:
    """Convierte campo de fecha (string ISO o dict $date) a Timestamp UTC naive."""
    if isinstance(v, dict) and "$date" in v:
        s = v["$date"]
    else:
        s = v
    ts = pd.to_datetime(s, utc=True, errors="coerce")
    if ts is pd.NaT:
        return pd.NaT
    return ts.tz_convert("UTC").tz_localize(None)


def _to_float(x) -> float:
    if x is None:
        return math.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x.strip().replace(",", "."))
        except Exception:
            return math.nan
    return math.nan


def _pick_first(d: dict, *keys):
    for k in keys:
        if k in d:
            return d[k]
    return None


def _norm_est(s) -> str:
    """Normaliza identificador de estación a mayúsculas sin espacios."""
    if not pd.notna(s):
        return ""
    raw = str(s).strip().upper()
    for suf in ("_60M", "_10M", "_60m", "_10m"):
        if raw.endswith(suf.upper()):
            raw = raw[: -len(suf)]
    return raw


# ---------------------------------------------------------------------------
# Carga de registros: JSON suelto o dentro de ZIP
# ---------------------------------------------------------------------------

def _parse_json_text(txt: str) -> list:
    """Parsea texto JSON como array o NDJSON."""
    txt = txt.strip()
    if not txt:
        return []
    if txt[0] == "[":
        try:
            arr = json.loads(txt)
            return arr if isinstance(arr, list) else []
        except Exception:
            return []
    recs = []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except Exception:
            continue
    return recs


def _year_from_name(name: str) -> str | None:
    """Extrae el año (4 dígitos) del nombre de un fichero."""
    import re
    m = re.search(r"(20\d{2})", name)
    return m.group(1) if m else None


def load_all_records(directory: Path) -> list:
    """
    Carga todos los registros JSON de un directorio.
    Soporta .json sueltos y .zip con .json dentro.

    Deduplicación por año: si para un mismo año hay tanto un JSON suelto
    como un ZIP, se usa solo el JSON suelto (ya descomprimido) y se omite el ZIP.
    Si solo hay ZIP, se usa el ZIP.
    """
    files = sorted(directory.iterdir())

    # Años cubiertos por JSON sueltos
    years_in_json = set()
    for fp in files:
        if fp.suffix.lower() == ".json":
            y = _year_from_name(fp.name)
            if y:
                years_in_json.add(y)

    all_records = []

    for fp in files:
        if fp.suffix.lower() == ".json":
            try:
                txt = fp.read_text(encoding="utf-8", errors="ignore")
                recs = _parse_json_text(txt)
                all_records.extend(recs)
                print(f"  [JSON] {fp.name}: {len(recs)} registros")
            except Exception as e:
                print(f"  [WARN] No se pudo leer {fp.name}: {e}")

        elif fp.suffix.lower() == ".zip":
            # Omitir ZIP si su año ya está cubierto por un JSON suelto
            y = _year_from_name(fp.name)
            if y and y in years_in_json:
                print(f"  [SKIP] {fp.name} (año {y} ya cubierto por JSON suelto)")
                continue
            try:
                with zipfile.ZipFile(fp, "r") as zf:
                    json_names = [n for n in zf.namelist() if n.lower().endswith(".json")]
                    for jname in json_names:
                        with zf.open(jname) as jf:
                            txt = io.TextIOWrapper(jf, encoding="utf-8", errors="ignore").read()
                        recs = _parse_json_text(txt)
                        all_records.extend(recs)
                        print(f"  [ZIP/{fp.name}] {jname}: {len(recs)} registros")
            except Exception as e:
                print(f"  [WARN] No se pudo leer ZIP {fp.name}: {e}")

    return all_records


# ---------------------------------------------------------------------------
# Extracción de coordenadas de estaciones
# ---------------------------------------------------------------------------

def _extract_stations(records: list) -> dict[str, tuple[float, float]]:
    """Devuelve {estacion: (lat, lon)} a partir de una lista de registros."""
    seen: dict[str, tuple[float, float]] = {}
    for r in records:
        loc = r.get("location") or {}
        coords = loc.get("coordinates")
        if not (isinstance(coords, list) and len(coords) >= 2):
            continue
        try:
            lon, lat = float(coords[0]), float(coords[1])
        except (TypeError, ValueError):
            continue
        est = _norm_est(r.get("refPointOfInterest") or r.get("entityId") or "")
        if est:
            seen[est] = (lat, lon)
    return seen


# ---------------------------------------------------------------------------
# Construcción de aire_60min_raw.csv
# ---------------------------------------------------------------------------

def build_aire_60min(json_dir: Path, outdir: Path):
    print(f"\n[S1] Cargando aire 60min desde {json_dir} ...")
    records = load_all_records(json_dir)
    if not records:
        raise SystemExit(f"[FATAL] No se encontraron registros en {json_dir}")

    stations = _extract_stations(records)
    rows = []

    for r in records:
        fecha = _parse_date(_pick_first(r, "dateObserved", "dateObservedGMT0"))
        if pd.isna(fecha):
            continue
        est = _norm_est(r.get("refPointOfInterest") or r.get("entityId") or "")
        if not est:
            continue

        def _get(prefix, allow_full=False):
            v = _pick_first(r, f"{prefix}ValueOrigin", f"{prefix}Value")
            if v is None and allow_full:
                v = r.get(f"{prefix}ValueFull")
            return _to_float(v)

        rows.append((
            fecha, est,
            _get("NO2"),
            _get("PM10", allow_full=True),
            _get("PM25"),
            _get("O3"),
            _get("SO2"),
        ))

    if not rows:
        raise SystemExit("[FATAL] aire_60min: ningún registro con fecha válida.")

    df = pd.DataFrame(rows, columns=["fecha", "estacion", "NO2", "PM10", "PM25", "O3", "SO2"])
    df = df.dropna(subset=["fecha"]).sort_values("fecha")
    df["estacion"] = df["estacion"].map(_norm_est)
    df = df.drop_duplicates(subset=["fecha", "estacion"], keep="last")

    out = outdir / "aire_60min_raw.csv"
    df.to_csv(out, index=False)
    print(f"[OK] aire_60min_raw.csv → {out}  (n={len(df)})")
    return df, stations


# ---------------------------------------------------------------------------
# Construcción de aire_10min_raw.csv
# ---------------------------------------------------------------------------

def build_aire_10min(json_dir: Path, outdir: Path):
    print(f"\n[S1] Cargando aire 10min desde {json_dir} ...")
    records = load_all_records(json_dir)
    if not records:
        raise SystemExit(f"[FATAL] No se encontraron registros en {json_dir}")

    stations = _extract_stations(records)
    rows = []

    for r in records:
        fecha = _parse_date(_pick_first(r, "dateObserved", "dateObservedGMT0"))
        if pd.isna(fecha):
            continue
        est = _norm_est(r.get("refPointOfInterest") or r.get("entityId") or "")
        if not est:
            continue
        no2 = _to_float(_pick_first(r, "NO2ValueOrigin", "NO2Value"))
        rows.append((fecha, est, no2))

    if not rows:
        raise SystemExit("[FATAL] aire_10min: ningún registro con fecha válida.")

    df = pd.DataFrame(rows, columns=["fecha", "estacion", "NO2"])
    df = df.dropna(subset=["fecha"]).sort_values("fecha")
    df["estacion"] = df["estacion"].map(_norm_est)
    df = df.drop_duplicates(subset=["fecha", "estacion"], keep="last")

    out = outdir / "aire_10min_raw.csv"
    df.to_csv(out, index=False)
    print(f"[OK] aire_10min_raw.csv → {out}  (n={len(df)})")
    return df, stations


# ---------------------------------------------------------------------------
# Construcción de stations_all.csv
# ---------------------------------------------------------------------------

def build_stations(
    stations_60: dict[str, tuple[float, float]],
    stations_10: dict[str, tuple[float, float]],
    outdir: Path,
) -> pd.DataFrame:
    # Preferimos coordenadas de 60min (más estables); completamos con 10min
    merged = {**stations_10, **stations_60}
    if not merged:
        raise SystemExit("[FATAL] No se detectaron coordenadas de estaciones.")

    df = pd.DataFrame(
        [{"estacion": k, "lat": v[0], "lon": v[1]} for k, v in merged.items()]
    ).sort_values("estacion").reset_index(drop=True)

    out = outdir / "stations_all.csv"
    df.to_csv(out, index=False)
    print(f"[OK] stations_all.csv → {out}  (n={len(df)})")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="p1_build_dataset: JSON de aire → CSVs normalizados"
    )
    ap.add_argument("--aire-json-60min", required=True,
                    help="Directorio con JSON/ZIP de aire a 60min")
    ap.add_argument("--aire-json-10min", required=True,
                    help="Directorio con JSON/ZIP de aire a 10min")
    ap.add_argument("--outdir", required=True,
                    help="Directorio de salida para los CSVs")
    return ap.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_60, stations_60 = build_aire_60min(Path(args.aire_json_60min), outdir)
    df_10, stations_10 = build_aire_10min(Path(args.aire_json_10min), outdir)
    df_st = build_stations(stations_60, stations_10, outdir)

    print("\n" + "="*60)
    print("RESUMEN S1")
    print("="*60)
    print(f"aire_60min_raw : {len(df_60):>8,} registros  |  columnas: {list(df_60.columns)}")
    print(f"aire_10min_raw : {len(df_10):>8,} registros  |  columnas: {list(df_10.columns)}")
    print(f"stations_all   : {len(df_st):>8} estaciones |  columnas: {list(df_st.columns)}")
    print("\n--- aire_10min_raw (primeras 10 filas) ---")
    print(df_10.head(10).to_string(index=False))
    print("\n--- aire_60min_raw (primeras 10 filas) ---")
    print(df_60.head(10).to_string(index=False))
    print("\n--- stations_all (primeras 10 filas) ---")
    print(df_st.head(10).to_string(index=False))
    print("\n[OK] p1_build_dataset completado.")


if __name__ == "__main__":
    main()
