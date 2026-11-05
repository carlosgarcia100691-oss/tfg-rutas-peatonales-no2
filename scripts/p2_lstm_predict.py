#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
p2_lstm_predict.py
==================
Etapa D del pipeline: predicción LSTM univariante por estación.

Para cada estación:
  1. Extrae la serie histórica de NO2 a 10min anterior a t0.
  2. Entrena una LSTM univariante (solo NO2) sobre esa serie.
  3. Predice de forma autoregresiva los valores en t0, t0+10min, ..., t0+(horizon-steps-1)*10min.

IDW NO se aplica aquí. Este script solo produce predicciones por estación.
El campo espacial se construye en p4_compute_routes.py.

Salida: CSV con columnas time(UTC), estacion, lat, lon, NO2
        Una fila por estación por paso temporal.

Uso:
  python scripts/p2_lstm_predict.py
    --air-csv      <aire_10min_raw.csv>
    --stations-csv <stations_all.csv>
    --var          NO2
    --time         "2023-05-15 08:00"
    --lookback     144
    --horizon-steps 6
    --out          <lstm_preds_NO2_20230515_0800.csv>
"""
from __future__ import annotations
import argparse
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Keras / TensorFlow
try:
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.callbacks import EarlyStopping
    KERAS_OK = True
except Exception:
    KERAS_OK = False


def _set_seeds(seed: int):
    """Fija semillas para reproducibilidad en numpy, random y TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)
    if KERAS_OK:
        tf.random.set_seed(seed)


# ---------------------------------------------------------------------------
# Modelo LSTM mínimo
# ---------------------------------------------------------------------------

def _make_lstm(lookback: int) -> "Sequential":
    model = Sequential([
        LSTM(32, input_shape=(lookback, 1)),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def _build_supervised(series: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(len(series) - lookback):
        X.append(series[i: i + lookback])
        y.append(series[i + lookback])
    X = np.array(X).reshape(-1, lookback, 1)
    y = np.array(y).reshape(-1, 1)
    return X, y


# ---------------------------------------------------------------------------
# Predicción por estación
# ---------------------------------------------------------------------------

def predict_station(
    series: np.ndarray,
    lookback: int,
    horizon: int,
    epochs: int,
    allow_fallback: bool,
) -> np.ndarray | None:
    """
    Entrena LSTM sobre `series` y predice `horizon` pasos hacia adelante.
    Devuelve array de shape (horizon,) o None si no hay datos suficientes
    y allow_fallback=False.
    """
    if len(series) < lookback + 8:
        if not allow_fallback:
            return None
        # persistencia: repite el último valor conocido
        last = series[-1] if len(series) > 0 else np.nan
        return np.full(horizon, last)

    if not KERAS_OK:
        last = series[-1]
        return np.full(horizon, last)

    scaler = StandardScaler()
    s = scaler.fit_transform(series.reshape(-1, 1)).ravel()

    X, y = _build_supervised(s, lookback)
    if len(X) < 8:
        last = series[-1]
        return np.full(horizon, last)

    model = _make_lstm(lookback)
    cb = EarlyStopping(patience=3, restore_best_weights=True, monitor="loss", verbose=0)
    model.fit(X, y, epochs=epochs, batch_size=32, verbose=0, callbacks=[cb])

    # predicción autoregresiva
    cur = s[-lookback:].reshape(1, lookback, 1)
    preds_scaled = []
    for _ in range(horizon):
        yhat = model.predict(cur, verbose=0)[0, 0]
        preds_scaled.append(yhat)
        cur = np.roll(cur, -1, axis=1)
        cur[0, -1, 0] = yhat

    preds = scaler.inverse_transform(
        np.array(preds_scaled).reshape(-1, 1)
    ).ravel()
    return preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="p2_lstm_predict: predicción LSTM univariante por estación a 10min"
    )
    ap.add_argument("--air-csv", required=True,
                    help="aire_10min_raw.csv (fecha UTC, estacion, NO2)")
    ap.add_argument("--stations-csv", required=True,
                    help="stations_all.csv (estacion, lat, lon)")
    ap.add_argument("--var", default="NO2",
                    help="Variable a predecir (default: NO2)")
    ap.add_argument("--time", required=True,
                    help='t0 local, ej: "2023-05-15 08:00"')
    ap.add_argument("--tz", default="Europe/Madrid")
    ap.add_argument("--lookback", type=int, default=144,
                    help="Ventana de inferencia LSTM en pasos de 10min (default: 144 = 24h)")
    ap.add_argument("--train-extra", type=int, default=432,
                    help="Pasos extra de histórico para entrenamiento, además del lookback (default: 432 = 72h)")
    ap.add_argument("--horizon-steps", type=int, default=6,
                    help="Pasos de 10min a predecir (default: 6 = 60min)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--allow-fallback", action="store_true",
                    help="Si falta histórico, usa persistencia para esa estación")
    ap.add_argument("--seed", type=int, default=42,
                    help="Semilla aleatoria para reproducibilidad (default: 42)")
    ap.add_argument("--out", required=True,
                    help="CSV de salida con predicciones")
    return ap.parse_args()


def main():
    args = parse_args()
    _set_seeds(args.seed)

    # --- Carga de datos
    air = pd.read_csv(args.air_csv)
    need = {"fecha", "estacion", args.var}
    if not need.issubset(air.columns):
        raise SystemExit(f"[FATAL] {args.air_csv} debe tener columnas: {need}")
    # Leer como naive UTC (S1 guarda sin tzinfo). Toda la lógica temporal es UTC naive.
    air["fecha"] = pd.to_datetime(air["fecha"], errors="coerce", utc=False)
    air[args.var] = pd.to_numeric(air[args.var], errors="coerce")

    st = pd.read_csv(args.stations_csv)
    if not {"estacion", "lat", "lon"}.issubset(st.columns):
        raise SystemExit("[FATAL] stations_csv requiere columnas: estacion, lat, lon")
    st = st.set_index("estacion")

    # t0 en UTC naive (coherente con cómo S1 guarda las fechas)
    t0_local = pd.to_datetime(args.time)
    t0_utc = (
        t0_local
        .tz_localize(args.tz, nonexistent="shift_forward", ambiguous="NaT")
        .tz_convert("UTC")
        .tz_localize(None)   # → naive UTC, igual que air["fecha"]
    )
    # alinear a múltiplo de 10min
    t0_utc = t0_utc.floor("10min")

    # ventana histórica: lookback (inferencia) + train_extra (entrenamiento)
    step = pd.Timedelta(minutes=10)
    total_hist = args.lookback + args.train_extra
    start_hist = t0_utc - total_hist * step

    # instantes de predicción (UTC)
    pred_times = [t0_utc + i * step for i in range(args.horizon_steps)]

    print(f"[INFO] t0 UTC = {t0_utc}")
    print(f"[INFO] Semilla: {args.seed}")
    print(f"[INFO] Horizonte: {args.horizon_steps} pasos × 10min")
    print(f"[INFO] Ventana histórica: lookback={args.lookback} + train_extra={args.train_extra} = {total_hist} pasos")
    print(f"[INFO] Desde {start_hist} hasta {t0_utc}")

    results = []
    stations = sorted(air["estacion"].dropna().unique())
    ok_cnt = persist_cnt = 0

    for est in stations:
        # coordenadas
        if est not in st.index:
            print(f"[WARN] Estación {est} sin coordenadas, omitida.")
            continue
        lat = float(st.loc[est, "lat"])
        lon = float(st.loc[est, "lon"])

        # serie histórica a 10min
        mask = (
            (air["estacion"] == est)
            & (air["fecha"] >= start_hist)
            & (air["fecha"] < t0_utc)
        )
        sub = (
            air.loc[mask, ["fecha", args.var]]
            .copy()
            .sort_values("fecha")
        )
        sub[args.var] = pd.to_numeric(sub[args.var], errors="coerce")

        # remuestrear a rejilla exacta de 10min e interpolar huecos pequeños
        sub = sub.set_index("fecha").resample("10min").mean()
        sub[args.var] = (
            sub[args.var]
            .interpolate(method="time", limit=3)
            .ffill(limit=3)
            .bfill(limit=3)
        )
        sub = sub.reset_index().dropna(subset=[args.var])

        series = sub[args.var].astype(float).values

        preds = predict_station(
            series,
            lookback=args.lookback,
            horizon=args.horizon_steps,
            epochs=args.epochs,
            allow_fallback=args.allow_fallback,
        )

        if preds is None:
            print(f"[WARN] {est}: histórico insuficiente y fallback desactivado, omitida.")
            continue

        # clip a valores no negativos (NO2 >= 0)
        preds = np.clip(preds, 0.0, None)

        if len(series) >= args.lookback + 8 and KERAS_OK:
            ok_cnt += 1
        else:
            persist_cnt += 1

    # Guardar instantes de predicción como ISO sin tzinfo (naive UTC)
        for ts, val in zip(pred_times, preds):
            results.append({
                "time": ts.strftime("%Y-%m-%dT%H:%M:%S"),  # naive UTC, sin +00:00
                "estacion": est,
                "lat": lat,
                "lon": lon,
                args.var: float(val),
            })

    if not results:
        raise SystemExit("[FATAL] No se generaron predicciones.")

    out_df = (
        pd.DataFrame(results)
        .sort_values(["time", "estacion"])
        .reset_index(drop=True)
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"[OK] Predicciones -> {out_path}  (filas={len(out_df)})")
    print(f"[INFO] Estaciones entrenadas con LSTM: {ok_cnt}  con persistencia: {persist_cnt}")


if __name__ == "__main__":
    main()
