#!/usr/bin/env bash
# Ejecutá esto para ver el commit 10 (métricas vasculares + score de riesgo) en vivo.
# Uso: bash ver_commit10.sh [nombre_imagen_sin_extension] [nombre_sesion]
# Ejemplo: bash ver_commit10.sh 26_training paciente03

cd "$(dirname "$0")"
IMG="${1:-21_training}"
SESSION="${2:-demo}"
PYTHONPATH="$(pwd)" python3 vascular/risk_score.py \
  --input "fundus_processed/${IMG}_processed.png" \
  --mask "masks/${IMG}_processed_mask.png" \
  --session "$SESSION" --show
