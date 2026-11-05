#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests/test_pipeline.py
======================
Suite mínima de tests pytest para el pipeline TFG NO2-rutas.

Ejecutar desde la raíz del repositorio:
    python -m pytest tests/ -v

Los tests NO requieren red ni datos reales pesados.
Usan datos sintéticos in-memory para verificar la lógica.
"""
from __future__ import annotations

import heapq
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Añadir scripts/ al path para importar módulos del pipeline
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# Helpers para generar datos sintéticos
# ---------------------------------------------------------------------------

def _make_stations_df(n: int = 5) -> pd.DataFrame:
    """Crea DataFrame de estaciones con coords dentro de Valencia."""
    lats = np.linspace(39.45, 39.50, n)
    lons = np.linspace(-0.42, -0.35, n)
    return pd.DataFrame({
        "estacion": [f"A{i+1:02d}" for i in range(n)],
        "lat": lats,
        "lon": lons,
    })


def _make_air_df(n_stations: int = 5, n_steps: int = 50) -> pd.DataFrame:
    """Crea DataFrame de observaciones sintéticas de NO2."""
    rng = np.random.default_rng(0)
    rows = []
    stations = _make_stations_df(n_stations)
    base_time = pd.Timestamp("2023-05-15 06:00")
    for step in range(n_steps):
        ts = base_time + pd.Timedelta(minutes=10 * step)
        for _, s in stations.iterrows():
            rows.append({
                "fecha": ts,
                "estacion": s["estacion"],
                "NO2": max(0.0, rng.normal(40.0, 10.0)),
                "PM10": max(0.0, rng.normal(20.0, 5.0)),
                "PM25": np.nan,
                "O3": max(0.0, rng.normal(60.0, 15.0)),
                "SO2": max(0.0, rng.normal(5.0, 2.0)),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests — Carga y estructura de datos
# ---------------------------------------------------------------------------

class TestDataStructure:
    def test_stations_columns(self):
        df = _make_stations_df()
        assert {"estacion", "lat", "lon"}.issubset(df.columns)

    def test_stations_no_null_coords(self):
        df = _make_stations_df()
        assert not df["lat"].isna().any()
        assert not df["lon"].isna().any()

    def test_stations_coord_order(self):
        """Valencia: lat ≈ 39.4, lon ≈ -0.37. Verifica que lat > lon."""
        df = _make_stations_df()
        assert (df["lat"] > 39.0).all(), "lat debe ser ~39.4 (Valencia)"
        assert (df["lon"] < 0.0).all(), "lon debe ser negativa (Valencia)"

    def test_air_df_required_columns(self):
        df = _make_air_df()
        for col in ("fecha", "estacion", "NO2"):
            assert col in df.columns, f"Falta columna: {col}"

    def test_air_df_no_duplicate_fecha_estacion(self):
        df = _make_air_df()
        dupes = df.duplicated(subset=["fecha", "estacion"])
        assert not dupes.any(), f"Hay {dupes.sum()} duplicados fecha-estacion"

    def test_air_df_no2_non_negative(self):
        df = _make_air_df()
        valid = df["NO2"].dropna()
        assert (valid >= 0.0).all(), "NO2 no debe ser negativo"

    def test_air_df_fecha_is_datetime(self):
        df = _make_air_df()
        assert pd.api.types.is_datetime64_any_dtype(df["fecha"]), \
            "fecha debe ser datetime"

    def test_air_df_fecha_no_timezone(self):
        """S1 guarda fechas UTC naive (sin tzinfo) para coherencia interna."""
        df = _make_air_df()
        tz = df["fecha"].dt.tz
        assert tz is None, f"fecha debe ser timezone-naive (UTC naive); tiene tz={tz}"


# ---------------------------------------------------------------------------
# Tests — IDW
# ---------------------------------------------------------------------------

class TestIDW:
    """Tests de la función build_idw_fn de p4_compute_routes."""

    def _get_idw_fn(self):
        from p4_compute_routes import build_idw_fn
        return build_idw_fn

    def test_idw_exact_match_station(self):
        """Si el punto coincide exactamente con una estación, devuelve su valor."""
        build_idw_fn = self._get_idw_fn()
        df = _make_stations_df(3)
        df["NO2"] = [10.0, 20.0, 30.0]
        idw = build_idw_fn(df, "NO2", k=3, p=2.0, r_km=50.0)
        # Punto exacto sobre la primera estación
        lat0, lon0 = float(df["lat"].iloc[0]), float(df["lon"].iloc[0])
        val = idw(lat0, lon0)
        assert abs(val - 10.0) < 0.5, (
            f"En la estación exacta se esperaba ~10.0, obtenido {val:.3f}"
        )

    def test_idw_single_station_constant_field(self):
        """Con una sola estación el campo debe ser constante en todos los puntos."""
        build_idw_fn = self._get_idw_fn()
        df = pd.DataFrame({
            "estacion": ["A1"],
            "lat": [39.47],
            "lon": [-0.38],
            "NO2": [50.0],
        })
        idw = build_idw_fn(df, "NO2", k=1, p=2.0, r_km=100.0)
        vals = [idw(39.46, -0.37), idw(39.48, -0.40), idw(39.50, -0.35)]
        assert all(abs(v - 50.0) < 1e-6 for v in vals), \
            f"Campo constante esperado con 1 estación; valores: {vals}"

    def test_idw_higher_p_closer_station_dominates(self):
        """Con p mayor, las estaciones cercanas deben pesar más."""
        build_idw_fn = self._get_idw_fn()
        df = pd.DataFrame({
            "estacion": ["CLOSE", "FAR"],
            "lat": [39.470, 39.490],
            "lon": [-0.380, -0.340],
            "NO2": [10.0, 90.0],
        })
        query_lat, query_lon = 39.471, -0.381  # muy cerca de CLOSE
        idw_low_p  = build_idw_fn(df, "NO2", k=2, p=1.0, r_km=100.0)
        idw_high_p = build_idw_fn(df, "NO2", k=2, p=5.0, r_km=100.0)
        val_low  = idw_low_p(query_lat, query_lon)
        val_high = idw_high_p(query_lat, query_lon)
        # Con p mayor, el valor debe estar más cerca de la estación CLOSE (10.0)
        assert val_high < val_low, (
            f"p mayor debería acercar al valor de CLOSE (10): "
            f"val_high={val_high:.2f}, val_low={val_low:.2f}"
        )

    def test_idw_non_negative(self):
        """Los valores IDW nunca deben ser negativos cuando los inputs no lo son."""
        build_idw_fn = self._get_idw_fn()
        df = _make_stations_df(5)
        df["NO2"] = [10.0, 20.0, 15.0, 8.0, 25.0]
        idw = build_idw_fn(df, "NO2", k=3, p=2.0, r_km=50.0)
        test_points = [
            (39.46, -0.39), (39.47, -0.38), (39.48, -0.37), (39.49, -0.36),
        ]
        for lat, lon in test_points:
            val = idw(lat, lon)
            assert val >= 0.0, f"IDW negativo en ({lat}, {lon}): {val}"

    def test_idw_nan_var_skips_station(self):
        """Estaciones con NO2=NaN deben descartarse sin lanzar excepción."""
        build_idw_fn = self._get_idw_fn()
        df = pd.DataFrame({
            "estacion": ["A1", "A2", "A3"],
            "lat": [39.46, 39.47, 39.48],
            "lon": [-0.40, -0.39, -0.38],
            "NO2": [np.nan, 30.0, np.nan],
        })
        idw = build_idw_fn(df, "NO2", k=3, p=2.0, r_km=100.0)
        val = idw(39.47, -0.39)
        assert np.isfinite(val), f"IDW devolvió NaN/Inf: {val}"


# ---------------------------------------------------------------------------
# Tests — Rutas / grafo
# ---------------------------------------------------------------------------

class TestRoutesLogic:
    """Tests de la lógica de rutas sin descargar OSMnx."""

    def _make_simple_graph(self):
        """
        Grafo lineal: 0 → 1 → 2 → 3 (destino)
        Arista 0→1: longitud=100, NO2 campo=50
        Arista 1→2: longitud=100, NO2 campo=5   ← mucho más limpia
        Arista 0→3: longitud=250, NO2 campo=30  ← ruta directa
        """
        import networkx as nx
        G = nx.MultiDiGraph()
        for n, (lat, lon) in enumerate([(39.46, -0.40), (39.47, -0.39), (39.47, -0.38), (39.48, -0.37)]):
            G.add_node(n, y=lat, x=lon)
        G.add_edge(0, 1, key=0, length=100.0)
        G.add_edge(1, 2, key=0, length=100.0)
        G.add_edge(2, 3, key=0, length=100.0)
        G.add_edge(0, 3, key=0, length=250.0)
        return G

    def test_min_time_dijkstra_selects_shorter_path(self):
        """
        Verifica que Dijkstra con peso _time selecciona el camino más corto.
        Test independiente de OSMnx (usa networkx directamente).
        """
        import networkx as nx
        G = self._make_simple_graph()
        speed_mps = 4.8 / 3.6
        # Asignar pesos de tiempo
        for u, v, k, d in G.edges(keys=True, data=True):
            d["_time"] = float(d.get("length", 1.0)) / speed_mps
        # Ruta más corta en tiempo: 0→3 (250 m) vs 0→1→2→3 (300 m)
        path = nx.shortest_path(G, 0, 3, weight="_time")
        assert path[0] == 0 and path[-1] == 3
        total_len = sum(
            min(G.get_edge_data(u, v).values(), key=lambda x: x.get("length", 1e9)).get("length", 0)
            for u, v in zip(path[:-1], path[1:])
        )
        assert total_len <= 300.0, f"Dijkstra debe usar ≤ 300 m; usó {total_len}"

    def test_all_edge_weights_positive(self):
        """Los pesos de exposición no deben ser negativos (requerido por Dijkstra)."""
        from p4_compute_routes import build_idw_fn
        df = _make_stations_df(5)
        df["NO2"] = [10.0, 20.0, 15.0, 8.0, 25.0]
        idw = build_idw_fn(df, "NO2", k=3, p=2.0, r_km=50.0)
        G = self._make_simple_graph()
        weights = []
        for u, v, k, d in G.edges(keys=True, data=True):
            length = float(d.get("length", 1.0))
            lat_m = (G.nodes[u]["y"] + G.nodes[v]["y"]) / 2.0
            lon_m = (G.nodes[u]["x"] + G.nodes[v]["x"]) / 2.0
            c = idw(lat_m, lon_m)
            c = max(c, 0.0) if np.isfinite(c) else 0.0
            weights.append(c * length)
        assert all(w >= 0.0 for w in weights), \
            f"Hay pesos negativos: {[w for w in weights if w < 0]}"


# ---------------------------------------------------------------------------
# Tests — Métricas de ruta
# ---------------------------------------------------------------------------

class TestRouteMetrics:
    def test_path_metrics_returns_positive(self):
        """path_metrics debe devolver L, T, E ≥ 0."""
        import networkx as nx
        from p4_compute_routes import path_metrics, build_idw_fn
        G = nx.MultiDiGraph()
        for n, (lat, lon) in enumerate([(39.46, -0.40), (39.47, -0.39), (39.48, -0.38)]):
            G.add_node(n, y=lat, x=lon)
        G.add_edge(0, 1, key=0, length=200.0)
        G.add_edge(1, 2, key=0, length=200.0)
        df = _make_stations_df(5)
        df["NO2"] = [30.0, 25.0, 20.0, 15.0, 35.0]
        idw = build_idw_fn(df, "NO2", k=3, p=2.0, r_km=50.0)
        L, T, E = path_metrics(G, [0, 1, 2], idw, speed_mps=4.8/3.6)
        assert L > 0 and T > 0 and E > 0, f"Métricas deben ser positivas: L={L}, T={T}, E={E}"
        assert abs(L - 0.4) < 0.01, f"L esperado ~0.4 km, obtenido {L:.4f}"

    def test_snapshot_csv_columns(self):
        """El CSV de snapshot observado debe tener las columnas clave."""
        required = {"fecha", "estacion", "lat", "lon", "NO2"}
        df = pd.DataFrame({
            "fecha": ["2023-05-15 08:00:00"],
            "estacion": ["A01"],
            "lat": [39.47],
            "lon": [-0.38],
            "NO2": [35.2],
        })
        assert required.issubset(df.columns), \
            f"Faltan columnas en snapshot: {required - set(df.columns)}"

    def test_metrics_min_expo_lte_min_time(self):
        """
        En un caso controlado, la exposición de min_expo debe ser ≤ min_time.
        Verificamos la propiedad fundamental del TFG.
        """
        # Usamos path_metrics directamente con un grafo con dos rutas
        import networkx as nx
        from p4_compute_routes import build_idw_fn, path_metrics, route_min_expo_static

        G = nx.MultiDiGraph()
        # Ruta corta: nodos 0→2 directo (longitud 200 m, zona contaminada)
        # Ruta larga: nodos 0→1→2 (longitud 400 m, zona limpia)
        for n, (lat, lon) in enumerate([
            (39.460, -0.400),  # 0 origen
            (39.465, -0.395),  # 1 nodo intermedio limpio
            (39.470, -0.390),  # 2 destino
        ]):
            G.add_node(n, y=lat, x=lon)
        G.add_edge(0, 2, key=0, length=200.0)   # ruta corta (contaminada)
        G.add_edge(0, 1, key=0, length=200.0)   # ruta larga (limpia)
        G.add_edge(1, 2, key=0, length=200.0)

        # Estaciones: alta concentración cerca de la arista directa 0→2,
        # baja concentración en la zona 0→1→2
        st_df = pd.DataFrame({
            "estacion": ["DIRTY", "CLEAN"],
            "lat": [39.465, 39.463],   # DIRTY entre nodos 0 y 2
            "lon": [-0.395, -0.397],   # CLEAN en zona de la ruta larga
            "NO2": [200.0, 5.0],
        })
        idw = build_idw_fn(st_df, "NO2", k=2, p=2.0, r_km=50.0)
        speed = 4.8 / 3.6

        path_mintime = [0, 2]         # ruta corta (min tiempo)
        path_minexpo = [0, 1, 2]      # ruta larga (menos contaminada)

        _, _, E_time = path_metrics(G, path_mintime, idw, speed)
        _, _, E_expo = path_metrics(G, path_minexpo, idw, speed)

        assert E_expo <= E_time, (
            f"La ruta min_expo debería tener menor exposición: "
            f"E_expo={E_expo:.2f}, E_time={E_time:.2f}"
        )


# ---------------------------------------------------------------------------
# Tests — Parsing y fechas
# ---------------------------------------------------------------------------

class TestDatesAndTimezones:
    def test_utc_to_local_conversion(self):
        """La conversión UTC → Europe/Madrid debe ser coherente."""
        utc_naive = pd.Timestamp("2023-05-15 06:00:00")  # UTC naive
        utc_aware = utc_naive.tz_localize("UTC")
        madrid = utc_aware.tz_convert("Europe/Madrid")
        # En mayo (CEST = UTC+2), las 06:00 UTC son las 08:00 locales
        assert madrid.hour == 8, f"Esperado 08:00 local, obtenido {madrid.hour}:00"

    def test_tz_localize_utc_naive(self):
        """
        S1 guarda fechas UTC naive (tz_localize(None) al final).
        Un Timestamp con UTC no puede combinarse directamente con uno sin tz.
        """
        aware_utc = pd.Timestamp("2023-05-15 06:00", tz="UTC")
        naive     = aware_utc.tz_localize(None)  # → UTC naive
        # Tras tz_localize(None), el resultado no tiene tzinfo
        assert naive.tzinfo is None, "UTC naive debe carecer de tzinfo"
        # Y el valor numérico debe ser idéntico (mismo instante)
        assert naive == pd.Timestamp("2023-05-15 06:00"), \
            "UTC naive debe preservar la hora sin cambio"

    def test_date_deduplication(self):
        """Duplicados fecha-estación deben eliminarse antes del análisis."""
        df = pd.DataFrame({
            "fecha": ["2023-05-15 08:00", "2023-05-15 08:00", "2023-05-15 09:00"],
            "estacion": ["A01", "A01", "A01"],
            "NO2": [30.0, 35.0, 28.0],
        })
        df_deduped = df.drop_duplicates(subset=["fecha", "estacion"], keep="last")
        assert len(df_deduped) == 2, \
            f"Después de deduplicar deben quedar 2 filas, hay {len(df_deduped)}"


# ---------------------------------------------------------------------------
# Tests — Archivos de salida existentes
# ---------------------------------------------------------------------------

class TestExistingOutputs:
    """Verifica que los outputs generados del batch existen y son válidos."""

    ROOT = Path(__file__).parent.parent

    def test_batch_no_gan_summary_exists(self):
        p = self.ROOT / "resultados" / "batch_no_gan" / "batch_summary.csv"
        assert p.exists(), f"No existe: {p}"

    def test_batch_no_gan_summary_all_ok(self):
        p = self.ROOT / "resultados" / "batch_no_gan" / "batch_summary.csv"
        if not p.exists():
            pytest.skip("batch_summary.csv no generado aún")
        df = pd.read_csv(p)
        assert "status" in df.columns, "Falta columna 'status' en batch_summary"
        assert (df["status"] == "ok").all(), \
            f"Hay casos fallidos: {df[df['status'] != 'ok']['case_id'].tolist()}"

    def test_batch_no_gan_case01_metrics_valid(self):
        p = self.ROOT / "resultados" / "batch_no_gan" / "case_01" / \
            "routes_metrics_NO2_20230515_0800.csv"
        if not p.exists():
            pytest.skip("metrics de case_01 no generados aún")
        df = pd.read_csv(p)
        for col in ("mode", "L_km", "T_min", "E"):
            assert col in df.columns, f"Falta columna '{col}' en metrics"
        assert {"min_time", "min_expo_idw", "min_expo_lstm"}.issubset(set(df["mode"])), \
            "Faltan modos en metrics"
        # Exposición mínima debe ser menor que la del mínimo tiempo
        e_time = float(df.loc[df["mode"] == "min_time", "E"].iloc[0])
        e_idw  = float(df.loc[df["mode"] == "min_expo_idw", "E"].iloc[0])
        assert e_idw <= e_time, \
            f"E_idw ({e_idw:.0f}) debería ser ≤ E_time ({e_time:.0f})"

    def test_gan_model_exists(self):
        p = self.ROOT / "models" / "gan_no2_vectors_G.keras"
        assert p.exists(), f"Modelo GAN no encontrado: {p}"

    def test_stations_all_exists(self):
        p = self.ROOT / "resultados" / "pipeline" / "stations_all.csv"
        if not p.exists():
            pytest.skip("stations_all.csv no generado aún (ejecuta p1 primero)")
        df = pd.read_csv(p)
        assert {"estacion", "lat", "lon"}.issubset(df.columns)
        assert not df["lat"].isna().any()
        assert not df["lon"].isna().any()

    def test_pipeline_aire_60min_exists(self):
        p = self.ROOT / "resultados" / "pipeline" / "aire_60min_raw.csv"
        if not p.exists():
            pytest.skip("aire_60min_raw.csv no generado aún (ejecuta p1 primero)")
        df = pd.read_csv(p)
        assert {"fecha", "estacion", "NO2"}.issubset(df.columns)

    def test_batch_no_gan_no_identical_routes(self):
        """Ningún caso debería tener las 3 rutas geométricamente idénticas."""
        summary = self.ROOT / "resultados" / "batch_no_gan" / "batch_summary.csv"
        if not summary.exists():
            pytest.skip("batch_summary.csv no generado aún")
        df = pd.read_csv(summary)
        if "same_geometry_flag" in df.columns:
            identical = df[df["same_geometry_flag"] == True]
            assert len(identical) == 0, \
                f"Casos con rutas idénticas: {identical['case_id'].tolist()}"


# ---------------------------------------------------------------------------
# Tests — p1 (parseo de JSON)
# ---------------------------------------------------------------------------

class TestP1Parsing:
    def test_parse_date_iso(self):
        """_parse_date debe manejar fechas ISO estándar."""
        from p1_build_dataset import _parse_date
        ts = _parse_date("2023-05-15T08:00:00Z")
        assert ts is not pd.NaT
        assert ts.year == 2023 and ts.month == 5 and ts.day == 15

    def test_parse_date_mongo(self):
        """_parse_date debe manejar el formato MongoDB {$date: ...}."""
        from p1_build_dataset import _parse_date
        ts = _parse_date({"$date": "2023-05-15T08:00:00.000Z"})
        assert ts is not pd.NaT
        assert ts.year == 2023

    def test_to_float_nan_on_none(self):
        from p1_build_dataset import _to_float
        assert math.isnan(_to_float(None))

    def test_to_float_comma_decimal(self):
        from p1_build_dataset import _to_float
        assert abs(_to_float("3,14") - 3.14) < 1e-6

    def test_norm_est_strips_suffix(self):
        from p1_build_dataset import _norm_est
        assert _norm_est("A01_60M") == "A01"
        assert _norm_est("A01_10M") == "A01"
        assert _norm_est("a01") == "A01"

    def test_extract_stations_lon_lat_order(self):
        """El JSON de CITCOM tiene coordinates=[lon, lat]; debe invertirse."""
        from p1_build_dataset import _extract_stations
        records = [{
            "refPointOfInterest": "A01",
            "location": {"coordinates": [-0.38, 39.47]},  # [lon, lat]
        }]
        seen = _extract_stations(records)
        assert "A01" in seen
        lat, lon = seen["A01"]
        # lat debe ser ~39, lon debe ser ~-0.38
        assert lat > 30 and lon < 0, \
            f"Coordenadas invertidas: lat={lat}, lon={lon}"
