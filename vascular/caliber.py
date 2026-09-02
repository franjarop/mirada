"""
Commit 10 — Calibre y densidad vascular a partir de la máscara de vasos (commit 9).

El doc original de este commit pedía AVR (razón arteria/vena), pero DRIVE (y el modelo del
commit 9) no distingue arteria de vena — solo da vaso sí/no. Sin ground truth de A/V no hay
forma honesta de calcular un AVR real, así que se reemplaza por métricas que no necesitan esa
distinción: calibre promedio, variabilidad de calibre y densidad vascular. Decisión tomada con
la usuaria (sesión 2026-09-01).
"""

import numpy as np
from skimage.morphology import skeletonize
import cv2


def compute_skeleton(vessel_mask: np.ndarray) -> np.ndarray:
    """Esqueleto de 1px de la máscara de vasos (skimage, sobre máscara booleana)."""
    return skeletonize(vessel_mask.astype(bool))


def caliber_map(vessel_mask: np.ndarray, skeleton: np.ndarray) -> np.ndarray:
    """
    Calibre (diámetro, en píxeles) estimado en cada punto del esqueleto: 2x la distancia al
    borde más cercano del vaso (transformada de distancia), asumiendo sección ~circular local.
    """
    dist = cv2.distanceTransform((vessel_mask.astype(np.uint8) * 255), cv2.DIST_L2, 5)
    return dist[skeleton] * 2.0


def caliber_stats(vessel_mask: np.ndarray, fov_mask: np.ndarray = None) -> dict:
    """
    Retorna: calibre promedio y su desvío (px), densidad vascular (% del FOV cubierto por
    vasos) y longitud total del esqueleto (proxy de extensión de la red vascular).
    """
    skeleton = compute_skeleton(vessel_mask)
    calibers = caliber_map(vessel_mask, skeleton)

    if fov_mask is not None:
        fov_px = int(np.count_nonzero(fov_mask))
    else:
        fov_px = vessel_mask.size

    density = float(np.count_nonzero(vessel_mask)) / max(fov_px, 1)

    return {
        "mean_caliber_px": float(calibers.mean()) if calibers.size else 0.0,
        "std_caliber_px": float(calibers.std()) if calibers.size else 0.0,
        "vascular_density": density,
        "skeleton_length_px": int(np.count_nonzero(skeleton)),
    }
