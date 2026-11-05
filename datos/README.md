# Datos

Los datos originales no se incluyen en el repositorio por su volumen (≈1,4 GB) y porque
proceden de la infraestructura del proyecto europeo **CITCOM.AI** en València.

## Contenido utilizado

Series temporales georreferenciadas de calidad del aire de las 11 estaciones de la red
de València (2022–2024), en formato JSON o ZIP con JSON en su interior:

| Variable | Descripción |
|---|---|
| NO₂ | Dióxido de nitrógeno (variable objetivo) |
| PM10, PM2.5 | Partículas en suspensión |
| O₃ | Ozono troposférico |
| SO₂ | Dióxido de azufre |

Se emplearon también series meteorológicas (temperatura, humedad, viento, presión y
precipitación) en el análisis exploratorio y la imputación.

## Organización esperada

```
datos/
├── aire/
│   ├── 10min/   # ficheros .json o .zip con datos cada 10 minutos
│   └── 60min/   # ficheros .json o .zip con datos horarios
└── meteo/
    ├── 10min/
    └── 60min/
```

`scripts/p1_build_dataset.py` lee todos los `.json` y `.zip` de cada carpeta y genera los
CSV normalizados en `resultados/pipeline/`.

Los resultados de los experimentos ya calculados se incluyen en `resultados/`, por lo que
pueden consultarse sin necesidad de los datos originales.
