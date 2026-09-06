#!/bin/bash
# Atajo para anotar la mácula a mano.
# Uso: bash annotate_macula.sh [carpeta de imágenes]  (default: dataset DRIVE)
cd "$(dirname "$0")"
CARPETA="${1:-fundus_images/dataset_drive/DRIVE/training/images}"
PYTHONPATH=. python fundus/annotate_disc.py --target macula --batch "$CARPETA"
