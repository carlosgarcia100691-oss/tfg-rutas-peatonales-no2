#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
p6_gan_temporal.py
==================
S6 del pipeline TFG: generacion de escenario GAN temporal.

Toma las predicciones LSTM multi-horizonte (CSV con columnas
time, estacion, lat, lon, NO2) y produce una version estresada
temporalmente coherente usando el generador GAN ya entrenado.

Logica:
  Para cada paso temporal t_k en el CSV LSTM:
    1. Extrae el vector de NO2 de las 11 estaciones en t_k.
    2. Normaliza con estadisticas del historico real (z-score por estacion).
    3. Construye entrada al generador: [vector_normalizado | ruido_latente]
       donde vector_normalizado ocupa las primeras 11 dims del espacio latente
       y el resto se rellena con ruido gaussiano escalado.
    4. Genera vector GAN -> desnormaliza -> aplica factor de estres.
    5. Clip a [0, 200] para mantener fisicamente plausible.

La salida tiene el mismo formato que el CSV LSTM:
  time, estacion, lat, lon, NO2

Uso:
  python scripts/p6_gan_temporal.py
    --lstm-csv      <lstm_preds_NO2_YYYYMMDD_HHMM.csv>
    --model         <models/gan_no2_vectors_G.keras>
    --air-csv       <resultados/pipeline/aire_60min_raw.csv>
    --stations-csv  <resultados/pipeline/stations_all.csv>
    --stress-factor 1.5
    --noise-scale   0.3
    --seed          42
    --out           <gan_temporal_NO2_YYYYMMDD_HHMM.csv>
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from tensorflow import keras


def parse_args():
    ap = argparse.ArgumentParser(
        description="p6_gan_temporal: escenario GAN temporal sobre predicciones LSTM"
    )
    ap.add_argument("--lstm-csv",     required=True,
                    help="CSV de predicciones LSTM (time, estacion, lat, lon, NO2)")
    ap.add_argument("--model",        required=True,
                    help="Ruta al generador GAN (.keras)")
    ap.add_argument("--air-csv",      required=True,
                    help="aire_60min_raw.csv para estadisticas de normalizacion")
    ap.add_argument("--stations-csv", required=True,
                    help="stations_all.csv con coordenadas")
    ap.add_argument("--var",          default="NO2")
    ap.add_argument("--stress-factor", type=float, default=1.5,
                    help="Factor de amplificacion de la perturbacion GAN (default: 1.5)")
    ap.add_argument("--noise-scale",   type=float, default=0.3,
                    help="Escala del ruido latente adicional (default: 0.3)")
    ap.add_argument("--seed",          type=int,   default=42)
    ap.add_argument("--out",           required=True,
                    help="CSV de salida con escenario GAN temporal")
    return ap.parse_args()


def main():
    args = parse_args()
    var = args.var
    rng = np.random.default_rng(args.seed)

    # --- Cargar generador GAN
    G = keras.models.load_model(args.model, compile=False)
    latent_dim = G.input_shape[1]   # 64
    n_stations  = G.output_shape[1]  # 11
    print(f"[OK] Generador: latent_dim={latent_dim}  n_stations={n_stations}")

    # --- Estadisticas historicas por estacion (para z-score)
    air = pd.read_csv(args.air_csv)
    air[var] = pd.to_numeric(air[var], errors="coerce")
    stats = (
        air.groupby("estacion")[var]
        .agg(["mean", "std"])
        .reset_index()
    )
    stats["std"] = stats["std"].fillna(1.0).clip(lower=1.0)

    # --- Orden canonico de estaciones (alfabetico, igual que el generador)
    st = pd.read_csv(args.stations_csv)
    stations_ordered = sorted(st["estacion"].unique())
    assert len(stations_ordered) == n_stations, (
        f"El generador espera {n_stations} estaciones pero hay {len(stations_ordered)}"
    )

    stats_idx = stats.set_index("estacion").reindex(stations_ordered)
    means = stats_idx["mean"].values.astype(float)
    stds  = stats_idx["std"].values.astype(float)

    # --- Cargar predicciones LSTM
    lstm = pd.read_csv(args.lstm_csv)
    time_col = next((c for c in ("time", "fecha", "timestamp") if c in lstm.columns), None)
    if time_col is None:
        raise SystemExit("[FATAL] lstm_csv sin columna temporal (time/fecha).")
    lstm[time_col] = pd.to_datetime(lstm[time_col], format="mixed", errors="coerce")
    lstm[var] = pd.to_numeric(lstm[var], errors="coerce")

    steps = sorted(lstm[time_col].dropna().unique())
    print(f"[INFO] Pasos temporales LSTM: {len(steps)}")

    # --- Generar escenario GAN temporal paso a paso
    results = []
    for step_idx, ts in enumerate(steps):
        df_step = lstm.loc[lstm[time_col] == ts].copy()

        # Vector de NO2 en orden canonico
        df_step_idx = df_step.set_index("estacion").reindex(stations_ordered)
        # fillna con la media de cada estacion (usando dict para compatibilidad pandas)
        means_dict = {est: float(means[i]) for i, est in enumerate(stations_ordered)}
        no2_series = df_step_idx[var].copy()
        no2_series = no2_series.fillna(pd.Series(means_dict))
        no2_base = no2_series.values.astype(float)

        # Normalizar a z-score
        z_base = (no2_base - means) / stds  # shape (11,)

        # Construir entrada al generador:
        # Las primeras min(11, latent_dim) dims = estado LSTM normalizado
        # El resto = ruido gaussiano con noise_scale
        z_input = rng.normal(0.0, args.noise_scale, size=(latent_dim,)).astype("float32")
        n_cond = min(n_stations, latent_dim)
        z_input[:n_cond] = z_base[:n_cond].astype("float32")

        # Pasar por generador
        raw_out = G.predict(z_input.reshape(1, -1), verbose=0)[0]  # shape (11,)

        # Desnormalizar la salida del generador
        no2_gan_raw = means + raw_out * stds

        # Perturbacion final: mezcla del base LSTM con la salida GAN
        # no2_gan = base + stress_factor * (gan_raw - base)
        # = base * (1 - stress_factor) + gan_raw * stress_factor
        no2_gan = no2_base + args.stress_factor * (no2_gan_raw - no2_base)
        no2_gan = np.clip(no2_gan, 0.0, 200.0)

        # Coordenadas
        st_idx = st.set_index("estacion").reindex(stations_ordered)
        lats = st_idx["lat"].values
        lons = st_idx["lon"].values

        for i, est in enumerate(stations_ordered):
            results.append({
                time_col:    ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                "estacion":  est,
                "lat":       float(lats[i]),
                "lon":       float(lons[i]),
                var:         float(no2_gan[i]),
                f"{var}_lstm_base": float(no2_base[i]),
            })

        print(f"  [t{step_idx+1}] {ts}  NO2_base_mean={no2_base.mean():.1f}  "
              f"NO2_gan_mean={no2_gan.mean():.1f}  "
              f"delta_mean={no2_gan.mean()-no2_base.mean():+.1f}")

    out_df = pd.DataFrame(results).sort_values([time_col, "estacion"]).reset_index(drop=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"[OK] Escenario GAN temporal -> {out_path}  (filas={len(out_df)})")
    print(f"[INFO] Columnas: {list(out_df.columns)}")
    print(f"[INFO] NO2 GAN: mean={out_df[var].mean():.1f}  "
          f"min={out_df[var].min():.1f}  max={out_df[var].max():.1f}")


if __name__ == "__main__":
    main()
