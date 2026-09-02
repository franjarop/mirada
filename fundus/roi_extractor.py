"""
Commit 8 — Detección automática de disco óptico y mácula.

Heurística (sin red neuronal, válida para commit 8):
  - Disco óptico: centroide del blob más brillante y compacto de la retina.
  - Mácula: punto más oscuro cerca del centro óptico, lejos del disco.

Ambas búsquedas se limitan al área real de la retina (compute_fov_mask) para
no confundir el fondo negro del recorte circular con la estructura buscada.
"""

import cv2
import numpy as np


def compute_fov_mask(frame: np.ndarray, thresh: int = 10):
    """
    Detecta el área visible de la retina (excluye el fondo negro del recorte circular).
    Retorna (mask, centro, radio). Si no se detecta un contorno claro, usa la imagen completa.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    h, w = gray.shape[:2]
    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.full((h, w), 255, np.uint8), (w // 2, h // 2), min(w, h) // 2

    largest = max(contours, key=cv2.contourArea)
    clean_mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(clean_mask, [largest], -1, 255, -1)
    (cx, cy), radius = cv2.minEnclosingCircle(largest)
    return clean_mask, (int(cx), int(cy)), int(radius)


def detect_optic_disc(frame: np.ndarray, fov_mask: np.ndarray = None, blur_ksize: int = 51):
    """Retorna (x, y) del disco óptico, o None si no hay suficiente contraste."""
    green = frame[:, :, 1] if frame.ndim == 3 else frame
    blurred = cv2.GaussianBlur(green, (blur_ksize, blur_ksize), 0)

    if fov_mask is None:
        fov_mask = np.full(green.shape[:2], 255, np.uint8)

    valid = blurred[fov_mask > 0]
    if valid.size == 0:
        return None

    mean_val = float(valid.mean())
    max_val = float(valid.max())
    if max_val - mean_val < 15:   # no hay región suficientemente brillante
        return None

    # Centroide del blob más brillante (top 2%) — más robusto que un solo píxel máximo,
    # que puede caer sobre un reflejo puntual en vez del disco real.
    thresh_val = np.percentile(valid, 98)
    bright = ((blurred >= thresh_val) & (fov_mask > 0)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        _, _, _, max_loc = cv2.minMaxLoc(cv2.bitwise_and(blurred, blurred, mask=fov_mask))
        return max_loc

    largest = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest)
    if M["m00"] == 0:
        _, _, _, max_loc = cv2.minMaxLoc(cv2.bitwise_and(blurred, blurred, mask=fov_mask))
        return max_loc

    return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))


def detect_macula(frame: np.ndarray, disc_pos=None, fov_mask: np.ndarray = None,
                   fov_center=None, fov_radius: int = None, blur_ksize: int = 51):
    """Retorna (x, y) de la mácula, o None si no se detecta."""
    green = frame[:, :, 1] if frame.ndim == 3 else frame
    h, w = green.shape[:2]
    blurred = cv2.GaussianBlur(green, (blur_ksize, blur_ksize), 0)

    if fov_mask is None:
        fov_mask = np.full((h, w), 255, np.uint8)
    if fov_center is None:
        fov_center = (w // 2, h // 2)
    if fov_radius is None:
        fov_radius = min(w, h) // 2

    # La mácula está sobre el eje visual (centro óptico de la retina, no de la imagen)
    search_mask = np.zeros((h, w), np.uint8)
    cv2.circle(search_mask, fov_center, int(fov_radius * 0.45), 255, -1)
    search_mask = cv2.bitwise_and(search_mask, fov_mask)

    # Excluir el entorno del disco óptico para no confundirlo con la mácula
    if disc_pos is not None:
        cv2.circle(search_mask, disc_pos, int(fov_radius * 0.35), 0, -1)

    ys, xs = np.where(search_mask == 255)
    if xs.size == 0:
        return None

    vals = blurred[ys, xs]
    idx = int(np.argmin(vals))
    min_val = float(vals[idx])
    mean_val = float(blurred[fov_mask > 0].mean())
    if mean_val - min_val < 8:   # no hay región suficientemente oscura
        return None

    return (int(xs[idx]), int(ys[idx]))


def draw_roi(frame: np.ndarray, disc_pos=None, macula_pos=None) -> np.ndarray:
    """Dibuja círculos: amarillo = disco óptico, azul = mácula."""
    out = frame.copy()
    if disc_pos is not None:
        cv2.circle(out, disc_pos, 25, (0, 255, 255), 2)
        cv2.putText(out, "disco", (disc_pos[0] - 20, disc_pos[1] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    if macula_pos is not None:
        cv2.circle(out, macula_pos, 20, (255, 128, 0), 2)
        cv2.putText(out, "macula", (macula_pos[0] - 25, macula_pos[1] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 1)
    return out
