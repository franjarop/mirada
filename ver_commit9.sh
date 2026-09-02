#!/usr/bin/env bash
# Ejecutá esto para ver el commit 9 (segmentación de vasos) en vivo.
# Uso: bash ver_commit9.sh [nombre_imagen_sin_extension]
# Ejemplo: bash ver_commit9.sh 25_training

cd "$(dirname "$0")"
IMG="${1:-21_training}"
PYTHONPATH="$(pwd)" python3 vascular/segmentation.py \
  --input "fundus_processed/${IMG}_processed.png" \
  --show --save
