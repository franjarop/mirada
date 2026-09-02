"""
Commit 9 — Superpone la máscara de vasos sobre la imagen de retina original.

DRIVE solo da ground truth binario de vasos (no distingue arteria/vena), así
que acá se resaltan todos los vasos detectados en un único color (rojo).
Clasificar arteria/vena por separado necesitaría otro dataset (ej. HRF, LES-AV).
"""

import cv2
import numpy as np

VESSEL_COLOR_BGR = (0, 0, 255)  # rojo


def overlay_vessels(image_bgr: np.ndarray, vessel_mask: np.ndarray,
                     color=VESSEL_COLOR_BGR, alpha: float = 0.65) -> np.ndarray:
    """Mezcla los vasos (mask booleana o 0/255) en color sobre la imagen original."""
    mask = vessel_mask.astype(bool)
    out = image_bgr.copy()
    overlay = np.full_like(image_bgr, color, dtype=np.uint8)
    blended = cv2.addWeighted(image_bgr, 1 - alpha, overlay, alpha, 0)
    out[mask] = blended[mask]
    return out


def side_by_side(image_bgr: np.ndarray, vessel_mask: np.ndarray, win_size=None) -> np.ndarray:
    """Concatena original (izquierda) + vasos superpuestos (derecha), cada uno redimensionado a win_size."""
    right = overlay_vessels(image_bgr, vessel_mask)
    left = image_bgr
    if win_size is not None:
        left = cv2.resize(left, win_size)
        right = cv2.resize(right, win_size)
    return np.hstack([left, right])


ARTERY_COLOR_BGR = (0, 0, 255)   # rojo
VEIN_COLOR_BGR = (255, 0, 0)     # azul


def overlay_av(image_bgr: np.ndarray, vessel_mask: np.ndarray, p_artery: np.ndarray,
                p_vein: np.ndarray, alpha: float = 0.65) -> np.ndarray:
    """
    Vasos coloreados por clasificación arteria (rojo) / vena (azul) — vascular/av_classifier.py
    (commit 10, entrenado sobre RITE). El clasificador tiene ~72.5% de exactitud: esperá algunos
    tramos de vaso con el color "equivocado", no es perfecto.
    """
    mask = vessel_mask.astype(bool)
    is_artery = (p_artery >= p_vein) & mask

    out = image_bgr.copy()
    artery_overlay = np.full_like(image_bgr, ARTERY_COLOR_BGR, dtype=np.uint8)
    vein_overlay = np.full_like(image_bgr, VEIN_COLOR_BGR, dtype=np.uint8)
    blended_artery = cv2.addWeighted(image_bgr, 1 - alpha, artery_overlay, alpha, 0)
    blended_vein = cv2.addWeighted(image_bgr, 1 - alpha, vein_overlay, alpha, 0)

    out[is_artery] = blended_artery[is_artery]
    out[mask & ~is_artery] = blended_vein[mask & ~is_artery]
    return out


def side_by_side_av(image_bgr: np.ndarray, vessel_mask: np.ndarray, p_artery: np.ndarray,
                     p_vein: np.ndarray, win_size=None) -> np.ndarray:
    """Concatena original (izquierda) + arteria/vena coloreadas (derecha)."""
    right = overlay_av(image_bgr, vessel_mask, p_artery, p_vein)
    left = image_bgr
    if win_size is not None:
        left = cv2.resize(left, win_size)
        right = cv2.resize(right, win_size)
    return np.hstack([left, right])
