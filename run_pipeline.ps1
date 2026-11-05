# run_pipeline.ps1
# =================
# Script PowerShell para ejecutar el pipeline TFG NO2-rutas de principio a fin.
# Ejecutar desde la raiz del repositorio:
#   .\run_pipeline.ps1
#
# Requisitos previos:
#   - Entorno virtual instalado en .venv\ con las dependencias de requirements.txt
#   - Datos de aire en datos\aire\60min\ y datos\aire\10min\
#   - Modelo GAN en models\gan_no2_vectors_G.keras

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Rutas base (relativas, sin hardcodear usuario)
# ---------------------------------------------------------------------------
$ROOT   = $PSScriptRoot
$PY     = Join-Path $ROOT ".venv\Scripts\python.exe"
$SD     = Join-Path $ROOT "scripts"
$PD     = Join-Path $ROOT "resultados\pipeline"
$BATCH  = Join-Path $ROOT "resultados\batch_no_gan"
$BATCH_GAN = Join-Path $ROOT "resultados\batch_gan"
$MODEL  = Join-Path $ROOT "models\gan_no2_vectors_G.keras"
$AIR60  = Join-Path $ROOT "datos\aire\60min"
$AIR10  = Join-Path $ROOT "datos\aire\10min"

# ---------------------------------------------------------------------------
# Verificar entorno
# ---------------------------------------------------------------------------
if (-not (Test-Path $PY)) {
    Write-Error "No se encuentra el interprete Python en $PY"
    Write-Host "Crea el entorno con: python -m venv .venv && .\.venv\Scripts\pip install -r requirements.txt"
    exit 1
}
Write-Host "[INFO] Python: $PY"

# ---------------------------------------------------------------------------
# Crear directorios de salida
# ---------------------------------------------------------------------------
New-Item -ItemType Directory -Force $PD      | Out-Null
New-Item -ItemType Directory -Force $BATCH   | Out-Null
New-Item -ItemType Directory -Force $BATCH_GAN | Out-Null
Write-Host "[INFO] Directorios de salida creados."

# ---------------------------------------------------------------------------
# S1 — Construir CSVs base de aire (aire_60min_raw, aire_10min_raw, stations_all)
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "=========================================="
Write-Host "S1 — p1_build_dataset.py"
Write-Host "=========================================="
& $PY "$SD\p1_build_dataset.py" `
    --aire-json-60min "$AIR60" `
    --aire-json-10min "$AIR10" `
    --outdir "$PD"
if ($LASTEXITCODE -ne 0) { Write-Error "S1 FALLO"; exit 1 }

# ---------------------------------------------------------------------------
# S2 — Predicciones LSTM para el caso de ejemplo (t0 = 2023-05-15 08:00)
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "=========================================="
Write-Host "S2 — p2_lstm_predict.py (caso ejemplo)"
Write-Host "=========================================="
& $PY "$SD\p2_lstm_predict.py" `
    --air-csv      "$PD\aire_10min_raw.csv" `
    --stations-csv "$PD\stations_all.csv" `
    --var NO2 `
    --time "2023-05-15 08:00" `
    --lookback 144 `
    --horizon-steps 6 `
    --epochs 15 `
    --seed 42 `
    --allow-fallback `
    --out "$PD\lstm_preds_NO2_20230515_0800.csv"
if ($LASTEXITCODE -ne 0) { Write-Error "S2 FALLO"; exit 1 }

# ---------------------------------------------------------------------------
# S3 — Snapshot observado en t0
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "=========================================="
Write-Host "S3 — p3_snapshot_observed.py"
Write-Host "=========================================="
& $PY "$SD\p3_snapshot_observed.py" `
    --air-csv      "$PD\aire_60min_raw.csv" `
    --stations-csv "$PD\stations_all.csv" `
    --var NO2 `
    --time "2023-05-15 08:00" `
    --out "$PD\snapshot_obs_NO2_20230515_0800.csv"
if ($LASTEXITCODE -ne 0) { Write-Error "S3 FALLO"; exit 1 }

# ---------------------------------------------------------------------------
# S4 — Rutas (IDW + Dijkstra): 3 rutas + métricas
# Parámetros definitivos del TFG: k=5, p=2.0, r=10 km, speed=4.8 km/h
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "=========================================="
Write-Host "S4 — p4_compute_routes.py"
Write-Host "=========================================="
& $PY "$SD\p4_compute_routes.py" `
    --snapshot-csv "$PD\snapshot_obs_NO2_20230515_0800.csv" `
    --lstm-csv     "$PD\lstm_preds_NO2_20230515_0800.csv" `
    --var NO2 `
    --time "2023-05-15 08:00" `
    --origin 39.4692 -0.4059 `
    --dest   39.4736 -0.3254 `
    --k-idw 5 --p-idw 2.0 --r-idw-km 10 `
    --speed-kmh 4.8 `
    --buffer-km 3.0 `
    --outdir "$PD"
if ($LASTEXITCODE -ne 0) { Write-Error "S4 FALLO"; exit 1 }

# ---------------------------------------------------------------------------
# S5 — Figura PNG + mapa HTML interactivo
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "=========================================="
Write-Host "S5 — p5_figura_tfg.py"
Write-Host "=========================================="
& $PY "$SD\p5_figura_tfg.py" `
    --route-time  "$PD\route_min_time_NO2_20230515_0800.geojson" `
    --route-idw   "$PD\route_min_expo_idw_NO2_20230515_0800.geojson" `
    --route-lstm  "$PD\route_min_expo_lstm_NO2_20230515_0800.geojson" `
    --metrics-csv "$PD\routes_metrics_NO2_20230515_0800.csv" `
    --origin-name "Olivereta" `
    --dest-name   "Cabanyal" `
    --origin-coords 39.4692 -0.4059 `
    --dest-coords   39.4736 -0.3254 `
    --outdir "$PD"
if ($LASTEXITCODE -ne 0) { Write-Error "S5 FALLO"; exit 1 }

Write-Host ""
Write-Host "=========================================="
Write-Host "PIPELINE BASICO COMPLETADO"
Write-Host "  Resultados en: $PD"
Write-Host "=========================================="

# ---------------------------------------------------------------------------
# S6a (OPCIONAL) — Batch completo 10 casos sin GAN
# Descomenta si quieres regenerar todos los casos
# ---------------------------------------------------------------------------
<#
Write-Host ""
Write-Host "=========================================="
Write-Host "S6a — p6_batch_no_gan.py (10 casos)"
Write-Host "=========================================="
& $PY "$SD\p6_batch_no_gan.py" `
    --pipeline-dir "$PD" `
    --outdir       "$BATCH" `
    --python-exe   "$PY" `
    --scripts-dir  "$SD" `
    --k-idw 5 --p-idw 2.0 --r-idw-km 10 `
    --speed-kmh 4.8 --buffer-km 3.0
if ($LASTEXITCODE -ne 0) { Write-Error "S6a FALLO"; exit 1 }
#>

# ---------------------------------------------------------------------------
# S7 (OPCIONAL) — Batch completo 10 casos con GAN temporal
# Requiere haber ejecutado S6a primero
# ---------------------------------------------------------------------------
<#
Write-Host ""
Write-Host "=========================================="
Write-Host "S7 — p7_batch_gan.py (10 casos GAN)"
Write-Host "=========================================="
& $PY "$SD\p7_batch_gan.py" `
    --no-gan-dir   "$BATCH" `
    --outdir       "$BATCH_GAN" `
    --python-exe   "$PY" `
    --scripts-dir  "$SD" `
    --model        "$MODEL" `
    --air-csv      "$PD\aire_60min_raw.csv" `
    --stations-csv "$PD\stations_all.csv" `
    --k-idw 5 --p-idw 2.0 --r-idw-km 10 `
    --speed-kmh 4.8 `
    --stress-factor 1.5 --noise-scale 0.3 --seed 42
if ($LASTEXITCODE -ne 0) { Write-Error "S7 FALLO"; exit 1 }
#>

# ---------------------------------------------------------------------------
# TESTS (OPCIONAL)
# ---------------------------------------------------------------------------
<#
Write-Host ""
Write-Host "=========================================="
Write-Host "TESTS — pytest"
Write-Host "=========================================="
& $PY -m pytest "$ROOT\tests\" -v
#>

Write-Host ""
Write-Host "[DONE] Pipeline ejecutado correctamente."
