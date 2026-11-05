#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
p4_compute_routes.py
====================
Etapas E+F del pipeline: construcción del campo IDW y cálculo de rutas.

Calcula tres rutas sobre la red peatonal de València:

  1. Mínimo tiempo
     Dijkstra estándar. Peso = longitud / velocidad peatonal.
     Baseline temporal puro, sin campo de contaminación.

  2. Mínima exposición IDW estático
     Dijkstra estándar. IDW sobre snapshot observado real en t0.
     Campo fijo: no cambia durante el recorrido.
     Peso = IDW(midpoint) × longitud.

  3. Mínima exposición LSTM+IDW dinámica
     Dijkstra time-dependent. Para cada arista, el campo IDW se evalúa
     en el instante de entrada del peatón en esa arista (t_entrada).
     t_entrada = t0 + tiempo_acumulado_hasta_nodo_origen_de_la_arista.
     El paso temporal k = floor(t_entrada_seg / 600) selecciona el campo
     IDW precalculado para ese instante (600 seg = 10 min).
     Peso = IDW_k(midpoint) × longitud.

Salidas:
  route_min_time_<VAR>_<TAG>.geojson
  route_min_expo_idw_<VAR>_<TAG>.geojson
  route_min_expo_lstm_<VAR>_<TAG>.geojson
  routes_metrics_<VAR>_<TAG>.csv

Uso:
  python scripts/p4_compute_routes.py
    --snapshot-csv <snapshot_obs_NO2_20230515_0800.csv>
    --lstm-csv     <lstm_preds_NO2_20230515_0800.csv>
    --var          NO2
    --time         "2023-05-15 08:00"
    --origin       39.4856445 -0.3494007
    --dest         39.4736445 -0.3254007
    --r-idw-km     2.5
    --k-idw        3
    --p-idw        2.6
    --outdir       <dir>
"""
from __future__ import annotations
import argparse
import heapq
import json
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
from sklearn.neighbors import BallTree


# ---------------------------------------------------------------------------
# Compatibilidad OSMnx (varias versiones)
# ---------------------------------------------------------------------------

def _largest_component(G):
    try:
        from osmnx.utils_graph import largest_component
        return largest_component(G, strongly=False)
    except Exception:
        pass
    try:
        from osmnx.utils_graph import get_largest_component
        return get_largest_component(G, strongly=False)
    except Exception:
        pass
    H = G.to_undirected(as_view=True) if G.is_directed() else G
    nodes = max(nx.connected_components(H), key=len)
    return G.subgraph(nodes).copy()


def _nearest_node(G, lat, lon):
    try:
        return ox.distance.nearest_nodes(G, X=[lon], Y=[lat])[0]
    except Exception:
        return ox.get_nearest_node(G, (lat, lon))


# ---------------------------------------------------------------------------
# IDW con BallTree (haversine)
# ---------------------------------------------------------------------------

def build_idw_fn(stations_df: pd.DataFrame, var: str, k: int, p: float, r_km: float):
    """
    Devuelve una función idw(lat_deg, lon_deg) -> float.
    Construida sobre los valores de `var` en `stations_df`.
    """
    df = stations_df.dropna(subset=["lat", "lon", var]).copy()
    if df.empty:
        return lambda lat, lon: np.nan

    vals = df[var].astype(float).values
    lat_r = np.deg2rad(df["lat"].values)
    lon_r = np.deg2rad(df["lon"].values)
    tree = BallTree(np.c_[lat_r, lon_r], metric="haversine")
    r_rad = r_km / 6371.0
    k_eff = min(k, len(vals))

    def idw(lat_deg: float, lon_deg: float) -> float:
        q = np.deg2rad([[lat_deg, lon_deg]])
        dists, idxs = tree.query(q, k=k_eff)
        dists = dists[0]
        idxs = idxs[0]
        mask = dists <= r_rad
        if not mask.any():
            mask = np.ones(k_eff, dtype=bool)
        d = np.maximum(dists[mask], 1e-9)
        v = vals[idxs[mask]]
        w = 1.0 / (d ** p)
        return float(np.sum(w * v) / np.sum(w))

    return idw


# ---------------------------------------------------------------------------
# Grafo peatonal
# ---------------------------------------------------------------------------

def get_graph(origin, dest, buffer_km: float, network_type: str = "walk"):
    ox.settings.log_console = False
    mid_lat = (origin[0] + dest[0]) / 2.0
    mid_lon = (origin[1] + dest[1]) / 2.0
    dist = max(500.0, buffer_km * 1000.0)
    G = ox.graph_from_point(
        (mid_lat, mid_lon), dist=dist,
        network_type=network_type, simplify=True
    )
    G = _largest_component(G)
    return G


def _edge_midpoint(G, u, v, data):
    if "geometry" in data and data["geometry"] is not None:
        mid = data["geometry"].interpolate(0.5, normalized=True)
        return mid.y, mid.x
    return (
        (G.nodes[u]["y"] + G.nodes[v]["y"]) / 2.0,
        (G.nodes[u]["x"] + G.nodes[v]["x"]) / 2.0,
    )


# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------

def route_min_time(G, origin, dest, speed_mps: float):
    """Dijkstra estándar por tiempo de tránsito."""
    for u, v, k, d in G.edges(keys=True, data=True):
        d["_time"] = float(d.get("length", 1.0)) / speed_mps
    on = _nearest_node(G, origin[0], origin[1])
    dn = _nearest_node(G, dest[0], dest[1])
    return nx.shortest_path(G, on, dn, weight="_time")


def route_min_expo_static(G, origin, dest, idw_fn, speed_mps: float):
    """Dijkstra estándar con campo IDW estático."""
    for u, v, k, d in G.edges(keys=True, data=True):
        length = float(d.get("length", 1.0))
        lat_m, lon_m = _edge_midpoint(G, u, v, d)
        c = idw_fn(lat_m, lon_m)
        c = max(c, 0.0) if np.isfinite(c) else 1e6
        d["_expo_static"] = c * length
        d["_time"] = length / speed_mps
    on = _nearest_node(G, origin[0], origin[1])
    dn = _nearest_node(G, dest[0], dest[1])
    return nx.shortest_path(G, on, dn, weight="_expo_static")


def route_min_expo_dynamic(
    G, origin, dest,
    idw_fns: list,          # lista de funciones IDW, una por paso de 10min
    speed_mps: float,
    step_sec: float = 600.0,
):
    """
    Dijkstra time-dependent.

    La cola almacena (expo_acumulada, tiempo_acumulado_seg, nodo).
    Al expandir la arista (u -> v):
      k = min(floor(t_entrada_seg / step_sec), len(idw_fns)-1)
      c = idw_fns[k](midpoint)
      coste_arista = c * longitud
      tiempo_arista = longitud / speed_mps
    """
    on = _nearest_node(G, origin[0], origin[1])
    dn = _nearest_node(G, dest[0], dest[1])

    # dist_expo[node] = (expo_acumulada, tiempo_acumulado_seg)
    INF = float("inf")
    dist_expo = {n: INF for n in G.nodes()}
    dist_time = {n: INF for n in G.nodes()}
    dist_expo[on] = 0.0
    dist_time[on] = 0.0
    prev = {on: None}

    # heap: (expo_acum, tiempo_acum, nodo)
    heap = [(0.0, 0.0, on)]

    while heap:
        expo_u, t_u, u = heapq.heappop(heap)
        if expo_u > dist_expo[u]:
            continue
        if u == dn:
            break
        for v, edge_dict in G.adj[u].items():
            # multigrafo: tomar la arista de menor longitud
            d = min(edge_dict.values(), key=lambda x: x.get("length", 1.0))
            length = float(d.get("length", 1.0))
            lat_m, lon_m = _edge_midpoint(G, u, v, d)

            # seleccionar campo IDW según instante de entrada
            k_step = min(int(t_u / step_sec), len(idw_fns) - 1)
            c = idw_fns[k_step](lat_m, lon_m)
            c = max(c, 0.0) if np.isfinite(c) else 1e6

            expo_edge = c * length
            t_edge = length / speed_mps

            new_expo = expo_u + expo_edge
            new_t = t_u + t_edge

            if new_expo < dist_expo[v]:
                dist_expo[v] = new_expo
                dist_time[v] = new_t
                prev[v] = u
                heapq.heappush(heap, (new_expo, new_t, v))

    # reconstruir camino
    if dn not in prev:
        raise RuntimeError("No se encontró camino al destino (ruta dinámica).")
    path = []
    cur = dn
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


# ---------------------------------------------------------------------------
# Métricas de ruta
# ---------------------------------------------------------------------------

def path_metrics(G, path, idw_fn, speed_mps: float):
    """Devuelve (L_km, T_min, E) para una ruta con campo IDW dado."""
    L = T = E = 0.0
    for u, v in zip(path[:-1], path[1:]):
        d = min(G.get_edge_data(u, v).values(), key=lambda x: x.get("length", 1.0))
        length = float(d.get("length", 0.0))
        lat_m, lon_m = _edge_midpoint(G, u, v, d)
        c = idw_fn(lat_m, lon_m)
        c = max(c, 0.0) if np.isfinite(c) else 0.0
        L += length
        T += length / speed_mps
        E += c * length
    return L / 1000.0, T / 60.0, E


def path_metrics_dynamic(G, path, idw_fns: list, speed_mps: float, step_sec: float = 600.0):
    """
    Evalua la exposicion real de una ruta cuando el campo varia en el tiempo.

    Para cada arista, el campo IDW se selecciona segun el instante en que
    el peaton entra en esa arista (tiempo acumulado desde el inicio).
    Esto es la evaluacion correcta cuando la 'realidad' es una serie temporal
    (por ejemplo, el escenario GAN completo).

    idw_fns: lista de funciones IDW, una por paso temporal (igual que en route_min_expo_dynamic).
    """
    L = T = E = 0.0
    t_acum = 0.0  # tiempo acumulado en segundos desde el inicio del recorrido
    for u, v in zip(path[:-1], path[1:]):
        d = min(G.get_edge_data(u, v).values(), key=lambda x: x.get("length", 1.0))
        length = float(d.get("length", 0.0))
        lat_m, lon_m = _edge_midpoint(G, u, v, d)
        # Seleccionar campo IDW segun instante de entrada en la arista
        k_step = min(int(t_acum / step_sec), len(idw_fns) - 1)
        c = idw_fns[k_step](lat_m, lon_m)
        c = max(c, 0.0) if np.isfinite(c) else 0.0
        L += length
        t_edge = length / speed_mps
        T += t_edge
        E += c * length
        t_acum += t_edge
    return L / 1000.0, T / 60.0, E


# ---------------------------------------------------------------------------
# GeoJSON
# ---------------------------------------------------------------------------

def path_to_geojson(G, path, name: str) -> dict:
    coords = [(G.nodes[n]["x"], G.nodes[n]["y"]) for n in path]
    return {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {"name": name},
        }],
    }


def save_geojson(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="p4_compute_routes: calcula las tres rutas sobre red peatonal"
    )
    ap.add_argument("--snapshot-csv", required=True,
                    help="snapshot_obs_<VAR>_<TAG>.csv")
    ap.add_argument("--lstm-csv", required=True,
                    help="lstm_preds_<VAR>_<TAG>.csv (todos los pasos temporales)")
    ap.add_argument("--var", default="NO2")
    ap.add_argument("--time", required=True,
                    help='t0 local, ej: "2023-05-15 08:00"')
    ap.add_argument("--tz", default="Europe/Madrid")
    ap.add_argument("--origin", nargs=2, type=float, required=True,
                    metavar=("LAT", "LON"))
    ap.add_argument("--dest", nargs=2, type=float, required=True,
                    metavar=("LAT", "LON"))
    ap.add_argument("--buffer-km", type=float, default=3.0)
    ap.add_argument("--speed-kmh", type=float, default=4.8,
                    help="Velocidad peatonal en km/h (default: 4.8)")
    ap.add_argument("--r-idw-km", type=float, default=10.0,
                    help="Radio de búsqueda IDW en km (default: 10.0)")
    ap.add_argument("--k-idw", type=int, default=5,
                    help="Número de vecinos IDW (default: 5)")
    ap.add_argument("--p-idw", type=float, default=2.0,
                    help="Exponente de distancia IDW (default: 2.0)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--eval-dynamic", action="store_true",
                    help="Evaluar exposicion de todas las rutas con el campo temporal "
                         "real (serie lstm-csv), no con el snapshot estatico. "
                         "Usar cuando lstm-csv es el escenario GAN real.")
    return ap.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    var = args.var
    speed_mps = args.speed_kmh / 3.6
    origin = (args.origin[0], args.origin[1])
    dest = (args.dest[0], args.dest[1])

    # tag para nombres de fichero
    t0_local = pd.to_datetime(args.time)
    tag = t0_local.strftime("%Y%m%d_%H%M")

    # --- Carga snapshot observado
    snap = pd.read_csv(args.snapshot_csv)
    for col in ("lat", "lon", var):
        if col not in snap.columns:
            raise SystemExit(f"[FATAL] snapshot_csv sin columna '{col}'")
    snap[var] = pd.to_numeric(snap[var], errors="coerce")

    # --- Carga predicciones LSTM (todos los pasos)
    lstm = pd.read_csv(args.lstm_csv)
    time_col = next((c for c in ("time", "fecha", "timestamp") if c in lstm.columns), None)
    if time_col is None:
        raise SystemExit("[FATAL] lstm_csv sin columna temporal (time/fecha).")
    # Leer como naive UTC (S2 guarda sin tzinfo)
    lstm[time_col] = pd.to_datetime(lstm[time_col], errors="coerce", utc=False)
    lstm[var] = pd.to_numeric(lstm[var], errors="coerce")

    # Pasos temporales disponibles (UTC, ordenados)
    steps_utc = sorted(lstm[time_col].dropna().unique())
    n_steps = len(steps_utc)
    print(f"[INFO] Pasos LSTM disponibles: {n_steps}  ({steps_utc[0]} … {steps_utc[-1]})")

    # --- Construir función IDW estática (snapshot observado)
    idw_static = build_idw_fn(snap, var, args.k_idw, args.p_idw, args.r_idw_km)

    # --- Construir lista de funciones IDW dinámicas (una por paso LSTM)
    idw_dynamic: list = []
    for ts in steps_utc:
        df_step = lstm.loc[lstm[time_col] == ts].copy()
        idw_dynamic.append(
            build_idw_fn(df_step, var, args.k_idw, args.p_idw, args.r_idw_km)
        )

    # --- Grafo peatonal
    print("[INFO] Descargando red peatonal de OSMnx…")
    G = get_graph(origin, dest, buffer_km=args.buffer_km)
    print(f"[INFO] Grafo: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas")

    # --- Ruta 1: mínimo tiempo
    print("[INFO] Calculando ruta mínimo tiempo…")
    path_time = route_min_time(G, origin, dest, speed_mps)

    # --- Ruta 2: mínima exposición IDW estático
    print("[INFO] Calculando ruta mínima exposición IDW estático…")
    path_idw = route_min_expo_static(G, origin, dest, idw_static, speed_mps)

    # --- Ruta 3: mínima exposición LSTM+IDW dinámica
    print("[INFO] Calculando ruta mínima exposición LSTM+IDW dinámica…")
    path_lstm = route_min_expo_dynamic(
        G, origin, dest, idw_dynamic, speed_mps, step_sec=600.0
    )

    # --- Métricas
    # Si --eval-dynamic: evaluar todas las rutas con el campo temporal real
    # (la serie lstm-csv, que en el caso GAN es el escenario GAN completo).
    # Esto mide la exposicion real que acumula cada estrategia de routing
    # cuando el campo evoluciona en el tiempo.
    # Si no: evaluar con el campo estatico en t0 (comportamiento por defecto).
    if args.eval_dynamic:
        L_t, T_t, E_t = path_metrics_dynamic(G, path_time, idw_dynamic, speed_mps)
        L_i, T_i, E_i = path_metrics_dynamic(G, path_idw,  idw_dynamic, speed_mps)
        L_l, T_l, E_l = path_metrics_dynamic(G, path_lstm, idw_dynamic, speed_mps)
    else:
        L_t, T_t, E_t = path_metrics(G, path_time, idw_static, speed_mps)
        L_i, T_i, E_i = path_metrics(G, path_idw,  idw_static, speed_mps)
        L_l, T_l, E_l = path_metrics(G, path_lstm, idw_static, speed_mps)

    metrics = pd.DataFrame([
        {"mode": "min_time",      "L_km": L_t, "T_min": T_t, "E": E_t},
        {"mode": "min_expo_idw",  "L_km": L_i, "T_min": T_i, "E": E_i},
        {"mode": "min_expo_lstm", "L_km": L_l, "T_min": T_l, "E": E_l},
    ])

    # --- Guardar GeoJSON
    gj_time = path_to_geojson(G, path_time, "min_time")
    gj_idw  = path_to_geojson(G, path_idw,  "min_expo_idw")
    gj_lstm = path_to_geojson(G, path_lstm, "min_expo_lstm")

    p_time = outdir / f"route_min_time_{var}_{tag}.geojson"
    p_idw  = outdir / f"route_min_expo_idw_{var}_{tag}.geojson"
    p_lstm = outdir / f"route_min_expo_lstm_{var}_{tag}.geojson"
    p_met  = outdir / f"routes_metrics_{var}_{tag}.csv"

    save_geojson(gj_time, p_time)
    save_geojson(gj_idw,  p_idw)
    save_geojson(gj_lstm, p_lstm)
    metrics.to_csv(p_met, index=False)

    print(f"[OK] route_min_time      -> {p_time}")
    print(f"[OK] route_min_expo_idw  -> {p_idw}")
    print(f"[OK] route_min_expo_lstm -> {p_lstm}")
    print(f"[OK] routes_metrics      -> {p_met}")
    print()
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
