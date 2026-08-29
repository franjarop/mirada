"""
Commit 7 — Indicador de nitidez para adquisición de fondo de ojo.
Usa varianza del Laplaciano: más alta = imagen más nítida.
"""

import cv2
import numpy as np


# Umbral empírico: por debajo = borroso, por encima = nítido
SHARPNESS_THRESHOLD = 80.0


def compute_sharpness(frame: np.ndarray) -> float:
    """Retorna varianza del Laplaciano del canal de brillo. Mayor = más nítido."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def is_sharp(frame: np.ndarray, threshold: float = SHARPNESS_THRESHOLD) -> bool:
    return compute_sharpness(frame) >= threshold


def sharpness_percent(frame: np.ndarray, max_val: float = 500.0) -> int:
    """Convierte varianza a porcentaje (0-100) para mostrar en UI."""
    v = compute_sharpness(frame)
    return int(min(v / max_val * 100, 100))
