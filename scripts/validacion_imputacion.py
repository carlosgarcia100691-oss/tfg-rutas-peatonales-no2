import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Cargar datos preprocesados sin imputación
from cargar_aire_mejorado import cargar_por_ano_mejorado
from cargar_meteo_mejorado import cargar_meteo_60min_mejorado

# Directorio de salida
OUT_DIR = Path(__file__).resolve().parent.parent / "resultados" / "imputacion"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------
# Configuración
# --------------------------
RANDOM_STATE = 42
MASK_FRACTION = 0.05  # 5% de los valores reales

# Métodos a evaluar
def imputar_lineal(series):
    return series.interpolate(limit_direction='both')

iter_imputer = IterativeImputer(random_state=RANDOM_STATE)
knn_imputer  = KNNImputer(n_neighbors=5)

METHODS = {
    'linear': imputar_lineal,
    'iterative': lambda s: pd.Series(iter_imputer.fit_transform(s.to_frame()).ravel(), index=s.index),
    'knn':      lambda s: pd.Series(knn_imputer.fit_transform(s.to_frame()).ravel(), index=s.index),
}

# --------------------------
# Función de validación
# --------------------------
def validar_variable(df, var):
    serie = df[var]
    disponible = serie.dropna()
    n_mask = int(len(disponible) * MASK_FRACTION)
    mask_idx = disponible.sample(n=n_mask, random_state=RANDOM_STATE).index

    y_true = serie.loc[mask_idx]
    s_mask = serie.copy()
    s_mask.loc[mask_idx] = np.nan

    resultados = []
    for name, func in METHODS.items():
        # Imputar
        s_imp = func(s_mask)
        # Evaluar
        y_pred = s_imp.loc[mask_idx]
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        resultados.append({'variable': var, 'metodo': name, 'MAE': mae, 'MSE': mse})

    return resultados

# --------------------------
# Main
# --------------------------
def main():
    # Ejemplo en meteorología 60min
    df_meteo = cargar_meteo_60min_mejorado()
    vars_meteo = ['temperature', 'humidity', 'wind_speed', 'wind_direction', 'pressure', 'precipitation']
    vars_meteo = [v for v in vars_meteo if v in df_meteo.columns]

    all_results = []
    for var in vars_meteo:
        print(f"Validando imputación para: {var}")
        all_results.extend(validar_variable(df_meteo, var))

    df_res = pd.DataFrame(all_results)
    df_res.to_csv(OUT_DIR / "resultados_validacion_meteo_60min.csv", index=False)
    with open(OUT_DIR / "masking_test_meteo_60min.tex", 'w') as f:
        f.write(df_res.to_latex(index=False, float_format="%.3f"))
    print("[OK] Validación completada y resultados guardados.")

if __name__ == '__main__':
    main()
