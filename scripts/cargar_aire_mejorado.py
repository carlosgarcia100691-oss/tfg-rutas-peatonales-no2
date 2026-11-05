import pandas as pd
import numpy as np
from pathlib import Path
import json

RUTA_BASE = Path(__file__).resolve().parent.parent / "datos" / "aire"

RANGOS = {
    'NO2':  (0, 500),
    'PM10': (0, 500),
    'PM25': (0, 500),
    'O3':   (0, 500),
    'SO2':  (0, 500),
}


def cargar_datos_aire(ruta_json):
    ruta = Path(ruta_json)
    with open(ruta, encoding='utf-8') as f:
        data = json.load(f)
    df = pd.json_normalize(data)

    df.replace({'': np.nan, ' ': np.nan}, inplace=True)

    # fecha y estación
    fecha_col = 'dateObserved.$date' if 'dateObserved.$date' in df.columns else 'dateObserved'
    df['fecha'] = pd.to_datetime(df[fecha_col], errors='coerce')
    df['estacion'] = df['entityId'].combine_first(df.get('refPointOfInterest'))

    # conversión de valores origin → numéricos
    mapeo = [
        ('NO2ValueOrigin','NO2'),
        ('PM10ValueOrigin','PM10'),
        ('PM25ValueOrigin','PM25'),
        ('O3ValueOrigin','O3'),
        ('SO2ValueOrigin','SO2'),
    ]
    valid_cols = []
    for orig, target in mapeo:
        if orig in df.columns:
            df[target] = pd.to_numeric(df[orig], errors='coerce')
            valid_cols.append(target)

    # indicadores _was_na
    for col in valid_cols:
        df[f'{col}_was_na'] = df[col].isna()

    # coordenadas
    if 'location.coordinates' in df.columns:
        df = df.dropna(subset=['location.coordinates'])
        df = df[df['location.coordinates'].apply(lambda x: isinstance(x, list) and len(x)==2)]
        coords = pd.DataFrame(df['location.coordinates'].tolist(),
                              index=df.index, columns=['lon','lat'])
        df = pd.concat([df.drop(columns=['location.coordinates']), coords], axis=1)

    # duplicados
    df.drop_duplicates(subset=['estacion','fecha'], inplace=True)

    # validación de rangos sin eliminar nulos
    for col in valid_cols:
        mn, mx = RANGOS[col]
        mask = df[col].between(mn, mx) | df[col].isna()
        df = df[mask]

    # columnas finales
    cols_finales = ['fecha','estacion','lon','lat']
    for col in valid_cols:
        cols_finales.extend([col, f'{col}_was_na'])

    return df[cols_finales]


def cargar_por_ano_mejorado(nombre_base, carpeta, anios):
    dfs = {}
    for anio in anios:
        ruta = RUTA_BASE / carpeta / f"{nombre_base}_{anio}.json"
        print(f"[aire] cargando {anio} → {ruta.name}")
        dfs[anio] = cargar_datos_aire(ruta)
    return dfs


if __name__ == "__main__":
    for base, carpeta in [('aire_10min','10min'),('aire_60min','60min'),('aire_24h','24h')]:
        dfs = cargar_por_ano_mejorado(base, carpeta, [2022,2023,2024])
        for anio, df in dfs.items():
            print(f"{base} {anio}: {df.shape}")
