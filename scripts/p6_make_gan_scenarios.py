#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
p6_make_gan_scenarios.py
========================
Genera N escenarios de estres de NO2 usando el generador GAN ya entrenado.

El generador GAN (models/gan_no2_vectors_G.keras) produce vectores de 11
estaciones a partir de ruido latente de dimension 64. La salida del generador
esta en espacio normalizado; este script la reescala a unidades fisicas de NO2
usando las estadisticas del historico real (media y std por estacion).

Cada escenario se guarda como un CSV compatible con p4_compute_routes.py:
  estacion, lat, lon, NO2

Ademas se guarda un manifest con los parametros de cada escenario.

Uso:
  python scripts/p6_make_gan_scenarios.py
    --model      <ruta al generador .keras>
    --air-csv    <aire_60min_raw.csv>
    --stations-csv <stations_all.csv>
    --n-scenarios  10
    --latent-dim   64
    --stress-factor 1.5
    --seed         42
    --outdir       <dir de salida>
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# TF silencioso
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from tensorflow import keras


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",         required=True)
    ap.add_argument("--air-csv",       required=True)
    ap.add_argument("--stations-csv",  required=True)
    ap.add_argument("--n-scenarios",   type=int,   default=10)
    ap.add_argument("--latent-dim",    type=int,   default=64)
    ap.add_argument("--stress-factor", type=float, default=1.5,
                    help="Multiplica la std del ruido latente para generar escenarios de estres (>1 = mas extremo)")
    ap.add_argument("--seed",          type=int,   default=42)
    ap.add_argument("--outdir",        required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Cargar generador
    G = keras.models.load_model(args.model, compile=False)
    print(f"[OK] Generador cargado: input={G.input_shape} output={G.output_shape}")

    # --- Estadisticas reales por estacion (para reescalar)
    air = pd.read_csv(args.air_csv)
    air["NO2"] = pd.to_numeric(air["NO2"], errors="coerce")
    stats = air.groupby("estacion")["NO2"].agg(["mean", "std"]).reset_index()
    stats["std"] = stats["std"].fillna(1.0).clip(lower=1.0)

    # --- Coordenadas de estaciones
    st = pd.read_csv(args.stations_csv)
    # Orden canonico de estaciones (mismo que el generador fue entrenado)
    # El generador produce 11 valores -> asumimos orden alfabetico de estaciones
    stations_ordered = sorted(st["estacion"].unique())
    n_stations = len(stations_ordered)
    assert G.output_shape[1] == n_stations, (
        f"El generador produce {G.output_shape[1]} valores pero hay {n_stations} estaciones"
    )

    # Merge stats con orden canonico
    stats = stats.set_index("estacion").reindex(stations_ordered)
    means = stats["mean"].values
    stds  = stats["std"].values

    # Coordenadas en orden canonico
    st_idx = st.set_index("estacion").reindex(stations_ordered)
    lats = st_idx["lat"].values
    lons = st_idx["lon"].values

    # --- Generar escenarios
    rng = np.random.default_rng(args.seed)
    scenarios = []

    for i in range(args.n_scenarios):
        # Ruido latente con stress_factor > 1 desplaza hacia colas de la distribucion
        # Usamos semillas distintas para variedad
        z = rng.normal(loc=0.0, scale=args.stress_factor, size=(1, args.latent_dim)).astype("float32")
        raw = G.predict(z, verbose=0)[0]  # shape (11,)

        # Reescalar: interpretar salida del generador como z-score y convertir a NO2
        # raw esta en [-1,1] aprox (tanh) o en espacio normalizado
        # Usamos: NO2 = mean + raw * std * stress_factor
        no2 = means + raw * stds * args.stress_factor
        no2 = np.clip(no2, 0.0, 200.0)  # NO2 fisicamente no negativo, cap en 200

        # Guardar CSV del escenario
        df_sc = pd.DataFrame({
            "estacion": stations_ordered,
            "lat": lats,
            "lon": lons,
            "NO2": no2,
        })
        sc_path = outdir / f"gan_scenario_{i+1:02d}.csv"
        df_sc.to_csv(sc_path, index=False)

        scenarios.append({
            "scenario_id": f"scenario_{i+1:02d}",
            "path": str(sc_path),
            "NO2_mean": float(no2.mean()),
            "NO2_min":  float(no2.min()),
            "NO2_max":  float(no2.max()),
            "NO2_std":  float(no2.std()),
            "stress_factor": args.stress_factor,
            "seed_offset": i,
        })
        print(f"  [OK] scenario_{i+1:02d}: NO2 mean={no2.mean():.1f} "
              f"min={no2.min():.1f} max={no2.max():.1f}")

    # --- Manifest
    manifest_path = outdir / "manifest_gan_scenarios.csv"
    pd.DataFrame(scenarios).to_csv(manifest_path, index=False)
    print(f"[OK] Manifest -> {manifest_path}")
    print(f"[OK] {args.n_scenarios} escenarios GAN generados en {outdir}")


if __name__ == "__main__":
    main()
