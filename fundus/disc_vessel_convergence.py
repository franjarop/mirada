"""
Tercera señal para el esquema híbrido de disco óptico: el punto donde convergen los vasos
sanguíneos principales. Ni la heurística de brillo (roi_extractor.detect_optic_disc) ni el
modelo SMDG (disc_model_smdg.py) usan esta señal directamente, y es la que falla en los 2 casos
conocidos donde la heurística engancha un reflejo especular (23_training) o el disco queda
sobreexpuesto (34_training) — en ambos casos hay reflejos/brillo confuso pero los vasos
grandes igual convergen en un solo punto.

Usa el segmentador de vasos ya entrenado (commit 9, vascular/segmentation.py + models/unet_drive.pth).
IMPORTANTE: requiere la imagen ORIGINAL (no CLAHE) — segment() ya aplica internamente el
dominio correcto (ver nota en vascular/segmentation.py: el modelo se entrenó sobre DRIVE crudo
y sobre-detecta vasos si se le da una imagen con CLAHE).

El disco se estima como el punto de mayor "densidad de calibre" — la suma de calibre de vasos
en una ventana local — en vez de densidad simple de píxeles: los troncos principales son pocos
pero gruesos justo donde convergen en el disco, mientras que la densidad de píxeles a secas
puede ser mayor en zonas con muchos capilares finos ramificados lejos del disco (ver
fundus/calibrate_disc_threshold.py — la densidad simple daba resultados inconsistentes).
"""

import cv2
import numpy as np

from vascular.segmentation import load_model, segment
from vascular.caliber import compute_skeleton, caliber_map

DEFAULT_MODEL_PATH = "models/unet_drive.pth"

_model_cache = {}


def load_vessel_model(path: str = DEFAULT_MODEL_PATH):
    if path not in _model_cache:
        _model_cache[path] = load_model(path)
    return _model_cache[path]


def detect_disc_by_convergence(frame: np.ndarray, model=None, threshold: float = 0.5,
                                kernel_size: int = 81):
    """
    frame: imagen ORIGINAL de fondo de ojo (no CLAHE).
    Retorna (x, y) del punto de mayor densidad de calibre vascular, o None si no hay vasos.
    """
    if model is None:
        model = load_vessel_model()
    m, mean, std = model
    vessel_mask, dt_ms = segment(m, mean, std, frame, threshold)
    if vessel_mask.sum() == 0:
        return None

    skeleton = compute_skeleton(vessel_mask)
    calibers = caliber_map(vessel_mask, skeleton)
    ys, xs = np.where(skeleton)

    caliber_img = np.zeros(vessel_mask.shape, np.float32)
    caliber_img[ys, xs] = calibers

    k = kernel_size + 1 - kernel_size % 2
    density = cv2.GaussianBlur(caliber_img, (k, k), 0)

    idx = int(np.argmax(density))
    y, x = np.unravel_index(idx, density.shape)
    return (int(x), int(y))
