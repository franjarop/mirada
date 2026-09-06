#!/bin/bash
# Atajo para anotar el disco óptico a mano.
# Uso: bash annotate.sh [carpeta de imágenes]  (default: dataset DRIVE)
cd "$(dirname "$0")"
CARPETA="${1:-fundus_images/dataset_drive/DRIVE/training/images}"
PYTHONPATH=. python fundus/annotate_disc.py --batch "$CARPETA"
