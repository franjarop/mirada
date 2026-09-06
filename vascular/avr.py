"""
Commit 10 (extensión) — AVR real (razón arteria/vena).

Combina la máscara de vasos (commit 9, vascular/segmentation.py) con el clasificador A/V
(vascular/av_classifier.py, entrenado sobre RITE) para calcular un AVR de verdad — a
diferencia del proxy de vascular/risk_score.py, este SÍ distingue arteria de vena.

Simplificación respecto al protocolo clínico real (Knudtson revised formula, zona B con
las 6 arterias y 6 venas más gruesas): acá se promedia el calibre de TODOS los píxeles
clasificados como arteria/vena dentro de una zona anular alrededor del disco óptico
(entre 2x y 4x el radio del disco), no solo los 6 vasos principales. Es una aproximación
razonable pero no reproduce el estándar clínico exacto — ver disclaimer en risk_score.py.
"""

import numpy as np

from vascular.caliber import compute_skeleton, caliber_map
from fundus.roi_extractor import estimate_disc_radius

ARTERY, VEIN = 0, 1


def measurement_zone_mask(shape, disc_pos, disc_radius: float,
                           inner_factor: float = 2.0, outer_factor: float = 4.0) -> np.ndarray:
    """Anillo alrededor del disco (aprox. zona B clínica) donde se mide el AVR."""
    h, w = shape[:2]
    yy, xx = np.ogrid[:h, :w]
    dist = np.hypot(yy - disc_pos[1], xx - disc_pos[0])
    return (dist >= disc_radius * inner_factor) & (dist <= disc_radius * outer_factor)


def compute_avr(image_bgr: np.ndarray, vessel_mask: np.ndarray, p_artery: np.ndarray,
                 p_vein: np.ndarray, disc_pos, fov_mask: np.ndarray = None) -> dict:
    """
    Retorna dict con AVR y detalle: calibre promedio de arterias/venas dentro de la zona
    de medición, y cuántos píxeles de esqueleto se clasificaron de cada tipo.
    """
    disc_radius = estimate_disc_radius(image_bgr, disc_pos, fov_mask=fov_mask)
    zone = measurement_zone_mask(image_bgr.shape, disc_pos, disc_radius)

    skeleton = compute_skeleton(vessel_mask)
    skeleton_in_zone = skeleton & zone
    calibers = caliber_map(vessel_mask, skeleton_in_zone)

    ys, xs = np.where(skeleton_in_zone)
    is_artery = p_artery[ys, xs] >= p_vein[ys, xs]

    artery_calibers = calibers[is_artery]
    vein_calibers = calibers[~is_artery]

    artery_mean = float(artery_calibers.mean()) if artery_calibers.size else 0.0
    vein_mean = float(vein_calibers.mean()) if vein_calibers.size else 0.0
    avr = artery_mean / vein_mean if vein_mean > 0 else None

    return {
        "avr": round(avr, 3) if avr is not None else None,
        "artery_mean_caliber_px": round(artery_mean, 2),
        "vein_mean_caliber_px": round(vein_mean, 2),
        "artery_px_count": int(artery_calibers.size),
        "vein_px_count": int(vein_calibers.size),
        "disc_radius_px": round(disc_radius, 1),
    }
