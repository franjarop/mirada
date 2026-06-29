"""
Commit 5 — Conversión de píxeles a milímetros usando el iris como referencia.
El iris humano tiene un diámetro promedio de 11.7mm — lo usamos como escala.
"""

IRIS_DIAMETER_MM = 11.7   # diámetro promedio del iris humano en mm


def pixels_to_mm(pixels: float, iris_radius_px: float) -> float:
    """
    Convierte una distancia en píxeles a milímetros.
    Usa el radio del iris detectado como factor de escala.
    """
    if iris_radius_px <= 0:
        return 0.0
    mm_per_px = (IRIS_DIAMETER_MM / 2.0) / iris_radius_px
    return round(pixels * mm_per_px, 2)


def pupil_to_iris_ratio(pupil_radius_px: float, iris_radius_px: float) -> float:
    """
    Calcula la razón pupila/iris (PLR — Pupil-to-Limbus Ratio).
    Valor normal: 0.2–0.5. Útil para detectar midriasis o miosis.
    """
    if iris_radius_px <= 0:
        return 0.0
    return round(pupil_radius_px / iris_radius_px, 3)
