"""
Commit 8 — Normalización de color para imágenes de fondo de ojo.
El canal verde tiene el mayor contraste vascular (la hemoglobina absorbe
más luz verde que roja, y el azul suele estar saturado/ruidoso).
"""

import cv2
import numpy as np


def normalize_green_channel(frame: np.ndarray, clip_limit: float = 3.0,
                             tile_size: int = 8) -> np.ndarray:
    """Aplica CLAHE al canal verde. Retorna imagen de 1 canal (grises)."""
    green = frame[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(green)


def gray_world_balance(frame: np.ndarray) -> np.ndarray:
    """Corrige el tinte amarillo/rojo típico de fondo de ojo igualando el promedio de cada canal."""
    result = frame.astype(np.float32)
    b, g, r = cv2.split(result)
    avg = (b.mean() + g.mean() + r.mean()) / 3
    b *= avg / (b.mean() + 1e-6)
    g *= avg / (g.mean() + 1e-6)
    r *= avg / (r.mean() + 1e-6)
    balanced = cv2.merge([b, g, r])
    return np.clip(balanced, 0, 255).astype(np.uint8)


def remove_reflections(frame: np.ndarray, thresh: int = 240) -> np.ndarray:
    """Detecta brillos especulares (muy claros en los 3 canales) y los rellena por inpainting."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    if cv2.countNonZero(mask) == 0:
        return frame
    return cv2.inpaint(frame, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)


def apply_clahe_contrast(frame: np.ndarray, clip_limit: float = 2.5,
                          tile_size: int = 8) -> np.ndarray:
    """Aumenta contraste global aplicando CLAHE al canal L (luminancia) en espacio LAB."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
