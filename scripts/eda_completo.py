# scripts/eda_completo.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from cargar_aire_mejorado import cargar_por_ano_mejorado
from cargar_meteo_mejorado import cargar_meteo_10min_mejorado, cargar_meteo_60min_mejorado

# Directorio de salida para tablas y gráficos
DIR_OUT = Path(__file__).resolve().parent.parent / 'resultados' / 'graficos'
DIR_OUT.mkdir(parents=True, exist_ok=True)


def resumen_estaciones(df: pd.DataFrame, nombre: str):
    n_est = df['estacion'].nunique()
    pct_mis = df['estacion'].isna().mean() * 100
    print(f"{nombre}: {n_est} estaciones únicas, {pct_mis:.2f}% missing")
    return n_est, pct_mis


def eda_textual(df: pd.DataFrame, pref: str):
    stats   = df.describe().round(2)
    n_null  = df.isna().sum()
    pct_null = (n_null / len(df) * 100).round(2)
    # Guardar CSV
    stats.to_csv(DIR_OUT / f"{pref}_describe.csv")
    pd.concat([n_null, pct_null], axis=1, keys=['n_null','pct_null']) \
      .to_csv(DIR_OUT / f"{pref}_nulos.csv")
    print(f"[OK] Estadísticas guardadas para {pref}")


def graficos_num(df: pd.DataFrame, pref: str):
    num_cols = df.select_dtypes(include='number').columns
    # Histogramas
    for col in num_cols:
        plt.figure()
        sns.histplot(df[col].dropna(), kde=True, bins=30)
        plt.title(f"Histograma {pref} - {col}")
        plt.savefig(DIR_OUT / f"hist_{pref}_{col}.png")
        plt.close()
    # Matriz de correlación
    plt.figure(figsize=(8,6))
    sns.heatmap(df[num_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm')
    plt.title(f"Correlación {pref}")
    plt.savefig(DIR_OUT / f"corr_{pref}.png")
    plt.close()
    print(f"[OK] Gráficos numéricos para {pref}")


def tendencia(df: pd.DataFrame, pref: str, freq='M'):
    ts = df.set_index('fecha').select_dtypes(include='number').resample(freq).mean()
    for col in ts.columns:
        plt.figure()
        ts[col].plot(title=f"Tendencia {pref} - {col}")
        plt.savefig(DIR_OUT / f"trend_{pref}_{col}.png")
        plt.close()
    print(f"[OK] Tendencias para {pref}")


if __name__ == '__main__':
    años = [2022, 2023, 2024]

    # === EDA para AIRE ===
    for res in ['10min', '60min', '24h']:
        # Carga por año -> dict año→df
        dfs = cargar_por_ano_mejorado(f"aire_{res}", res, años)
        # Concatenar todas las series temporales
        df_air = pd.concat(dfs.values(), ignore_index=True)
        nombre = f"aire_{res}_all"

        resumen_estaciones(df_air, nombre)
        eda_textual(df_air, nombre)
        graficos_num(df_air, nombre)
        tendencia(df_air, nombre, freq='M')  # mensual

    # === EDA para METEO 10min ===
    df_m10 = cargar_meteo_10min_mejorado()
    nombre = 'meteo_10min_all'
    resumen_estaciones(df_m10, nombre)
    eda_textual(df_m10, nombre)
    graficos_num(df_m10, nombre)
    tendencia(df_m10, nombre, freq='M')

    # === EDA para METEO 60min ===
    df_m60 = cargar_meteo_60min_mejorado()
    nombre = 'meteo_60min_all'
    resumen_estaciones(df_m60, nombre)
    eda_textual(df_m60, nombre)
    graficos_num(df_m60, nombre)
    tendencia(df_m60, nombre, freq='M')
