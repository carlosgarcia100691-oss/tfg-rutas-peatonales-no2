# -*- coding: utf-8 -*-
import argparse, json, os, numpy as np, pandas as pd
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--long-csv', required=True)                     # aire_10min_2023_NO2.csv
    ap.add_argument('--station-order-json', required=True)           # el del LSTM
    ap.add_argument('--scaler-csv', required=True)                   # el del LSTM (mu, sd por estación)
    ap.add_argument('--out-npy', required=True)                      # X.npy (T x N) en z-score
    args = ap.parse_args()

    with open(args.station_order_json,'r',encoding='utf-8') as f:
        stations = json.load(f)
    scaler = pd.read_csv(args.scaler_csv).set_index('estacion').loc[stations]
    mu = scaler['mu'].to_numpy(dtype=np.float32)
    sd = scaler['sd'].replace(0,1.0).to_numpy(dtype=np.float32)

    df = pd.read_csv(args.long_csv)
    df['time'] = pd.to_datetime(df['time'], utc=True, errors='coerce')
    df = df.dropna(subset=['time','estacion','NO2'])
    df['time'] = df['time'].dt.floor('10min')
    df = df.sort_values(['time','estacion'])
    wide = df.pivot_table(index='time', columns='estacion', values='NO2', aggfunc='mean').sort_index()
    missing = [s for s in stations if s not in wide.columns]
    if missing:
        raise SystemExit(f"[ERROR] Faltan estaciones en este CSV: {missing}")

    W = wide[stations].to_numpy(dtype=np.float32)    # T x N
    Z = (W - mu) / sd                                # z-score
    Z = np.nan_to_num(Z, nan=0.0)
    os.makedirs(os.path.dirname(args.out_npy), exist_ok=True)
    np.save(args.out_npy, Z)
    print("[OK] Dataset GAN →", args.out_npy, " shape:", Z.shape)
if __name__ == '__main__':
    main()
