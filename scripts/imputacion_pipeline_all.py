# scripts/imputacion_pipeline_all.py

import pandas as pd
import numpy as np
from pathlib import Path

# Imputers experimentales
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.preprocessing import FunctionTransformer

# Carga de datos mejorada
from cargar_aire_mejorado import cargar_por_ano_mejorado
from cargar_meteo_mejorado import cargar_meteo_60min_mejorado

# ==================================================
# CONFIGURACIÓN
# ==================================================
ANIOS_AIRE    = [2022, 2023, 2024]
VARS_AIRE     = ['NO2', 'PM10', 'PM25', 'O3', 'SO2']
VARS_METEO    = ['temperature', 'humidity', 'wind_speed', 'wind_direction', 'pressure', 'precipitation']

# ==================================================
# CONSTRUCCIÓN DEL PIPELINE
# ==================================================
def build_imputation_pipeline(vars_list, spatial=False):
    """
    Crea un pipeline que:
      1) FFILL/BFILL para huecos cortos,
      2) interpolación lineal para huecos medios,
      3) IterativeImputer para el resto,
      opcionalmente añade KNN espacial para SO2 y precipitación.
    """
    steps = []
    # 1) Fast forward/backfill
    steps.append(('fbfill', FunctionTransformer(lambda df: df.ffill(limit=1).bfill(limit=1))))
    # 2) Interpolación lineal bidireccional
    steps.append(('interp', FunctionTransformer(lambda df: df.interpolate(method='linear', limit_direction='both'))))
    # 3) Imputación multivariante
    steps.append(('iter', IterativeImputer(max_iter=10, random_state=0)))
    from sklearn.pipeline import Pipeline
    return Pipeline(steps)

# ==================================================
# EJECUCIÓN PRINCIPAL
# ==================================================
if __name__ == '__main__':
    # --- AIRE 60min ---
    dfs_air = cargar_por_ano_mejorado('aire_60min','60min',ANIOS_AIRE)
    df_air  = pd.concat(dfs_air.values(), ignore_index=True)
    print("Antes de imputar (aire):", df_air[VARS_AIRE].isna().sum().to_dict())
    pipe_air = build_imputation_pipeline(VARS_AIRE, spatial=True)
    arr_air  = pipe_air.fit_transform(df_air[VARS_AIRE])
    df_air_imp = df_air.copy()
    df_air_imp[VARS_AIRE] = arr_air
    print("Después de imputar (aire):", df_air_imp[VARS_AIRE].isna().sum().to_dict())

    # --- METEO 60min ---
    df_meteo = cargar_meteo_60min_mejorado()
    vars_m   = [v for v in VARS_METEO if v in df_meteo.columns]
    print("Antes de imputar (meteo):", df_meteo[vars_m].isna().sum().to_dict())
    pipe_met = build_imputation_pipeline(vars_m)
    arr_met  = pipe_met.fit_transform(df_meteo[vars_m])
    df_met_imp = df_meteo.copy()
    df_met_imp[vars_m] = arr_met
    print("Después de imputar (meteo):", df_met_imp[vars_m].isna().sum().to_dict())

    # Guardado opcional…
    dir_out = Path(__file__).resolve().parent.parent / "resultados" / "imputacion"
    dir_out.mkdir(parents=True, exist_ok=True)
    df_air_imp.to_csv(dir_out/"aire_60min_imputado.csv", index=False)
    df_met_imp.to_csv(dir_out/"meteo_60min_imputado.csv", index=False)
    print(f"[OK] Resultados guardados en {dir_out}")
