# scripts/cargar_meteo_mejorado.py

import pandas as pd
import numpy as np
from pathlib import Path
import json

pd.set_option('future.no_silent_downcasting', True)

RUTA_BASE = Path(__file__).resolve().parent.parent / "datos" / "meteo"
LON_BOUNDS = (-0.6, 0.1)
LAT_BOUNDS = (39.3, 39.6)

def cargar_datos_meteo(ruta_json: Path) -> pd.DataFrame:
    with open(ruta_json, encoding='utf-8') as f:
        raw = pd.json_normalize(json.load(f))

    raw = raw.replace({'': np.nan, ' ': np.nan}).infer_objects(copy=False)

    # elegir columna de estación
    for col in ('entityId','refPointOfInterest','gis_id'):
        if col in raw.columns:
            id_col = col
            break
    else:
        id_col = None

    if id_col:
        raw['estacion'] = raw[id_col]
    else:
        raw['estacion'] = np.nan

    # fecha
    if 'dateObserved.$date' in raw.columns:
        raw['fecha'] = pd.to_datetime(raw['dateObserved.$date'], errors='coerce')
    else:
        raw['fecha'] = pd.to_datetime(raw['dateObserved'], errors='coerce')

    # coordenadas
    coords = None
    if 'location.coordinates' in raw.columns:
        mask = raw['location.coordinates'].apply(lambda x: isinstance(x,list) and len(x)==2)
        coords = pd.DataFrame(raw.loc[mask,'location.coordinates'].tolist(),
                              columns=['lon','lat'], index=raw.index[mask])

    # variables
    mapping = {
        'temperatureValue':'temperature',
        'relativeHumidityValue':'humidity',
        'windSpeedValue':'wind_speed',
        'windDirectionValue':'wind_direction',
        'atmosphericPressureValue':'pressure',
        'precipitationValue':'precipitation',
    }
    parts = {'fecha': raw['fecha'], 'estacion': raw['estacion']}
    if coords is not None:
        parts.update({'lon': coords['lon'], 'lat': coords['lat']})

    for orig, new in mapping.items():
        if orig in raw.columns:
            col = pd.to_numeric(raw[orig], errors='coerce')
            parts[new] = col
            parts[f'{new}_was_na'] = col.isna()

    df = pd.concat(parts, axis=1).drop_duplicates(subset=['estacion','fecha'])
    # filtrar lon/lat dentro de Valencia
    if 'lon' in df.columns and 'lat' in df.columns:
        df = df[df['lon'].between(*LON_BOUNDS) & df['lat'].between(*LAT_BOUNDS)]

    return df

def cargar_meteo_10min_mejorado() -> pd.DataFrame:
    carpeta = RUTA_BASE / '10min'
    dfs = []
    # antes usábamos sólo meteo_10min_*.json, ahora cargamos TODOS los .json
    for ruta in sorted(carpeta.glob('*.json')):
        print(f"[meteo 10min] cargando → {ruta.name}")
        dfs.append(cargar_datos_meteo(ruta))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def cargar_meteo_60min_mejorado() -> pd.DataFrame:
    carpeta = RUTA_BASE / '60min'
    dfs = []
    for ruta in sorted(carpeta.glob('meteo_60min_*.json')):
        print(f"[meteo 60min] cargando → {ruta.name}")
        dfs.append(cargar_datos_meteo(ruta))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
