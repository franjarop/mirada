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

    Usa Otsu en vez de un threshold fijo: en imágenes ya pasadas por CLAHE el fondo no queda
    puro negro (ruido residual ~4-26), y con threshold=10 fijo el contorno se "fugaba" y
    terminaba cubriendo casi toda la imagen (~98%) en vez de solo el círculo de la retina —
    eso hacía que detect_optic_disc/detect_macula buscaran fuera del área real (ej. eligiendo
    un reflejo cerca del borde como "disco"). El resultado también se fuerza a un círculo
    macizo acotado a los límites de la imagen, en vez de un polígono de contorno que puede
    fugarse si el ruido de fondo se conecta con la retina al dilatar/cerrar.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    h, w = gray.shape[:2]
    otsu_val, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if otsu_val < thresh:  # Otsu degenerado (imagen casi sin fondo oscuro) -> usar threshold fijo
        _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.full((h, w), 255, np.uint8), (w // 2, h // 2), min(w, h) // 2

    largest = max(contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(largest)
    radius = min(radius, min(w, h) / 2)

    clean_mask = np.zeros((h, w), np.uint8)
    cv2.circle(clean_mask, (int(cx), int(cy)), int(radius), 255, -1)
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

    # Centroide ponderado por intensidad (al cuadrado) dentro del blob más brillante, en vez de
    # centroide uniforme: el blob puede ser asimétrico (ej. una extensión de brillo a lo largo de
    # un vaso/reflejo saliendo del disco real), y un centroide uniforme se corre hacia esa cola en
    # vez de quedarse sobre el núcleo real del disco. Ponderar por intensidad² tira el resultado
    # hacia el punto más brillante real sin caer en el problema original de un solo píxel (ruidoso).
    largest = max(contours, key=cv2.contourArea)
    blob_mask = np.zeros_like(bright)
    cv2.drawContours(blob_mask, [largest], -1, 255, -1)
    ys, xs = np.where(blob_mask > 0)
    if xs.size == 0:
        _, _, _, max_loc = cv2.minMaxLoc(cv2.bitwise_and(blurred, blurred, mask=fov_mask))
        return max_loc

    weights = blurred[ys, xs].astype(np.float64)
    weights = (weights - weights.min() + 1) ** 2
    cx = float(np.sum(xs * weights) / np.sum(weights))
    cy = float(np.sum(ys * weights) / np.sum(weights))
    return (int(cx), int(cy))


def estimate_disc_radius(frame: np.ndarray, disc_pos, fov_mask: np.ndarray = None,
                          blur_ksize: int = 51, search_radius: float = 80.0) -> float:
    """
    Radio real del disco óptico (sqrt(área/pi) del blob de brillo que contiene disc_pos, o el
    más cercano dentro de search_radius). A diferencia de una búsqueda independiente del blob
    más brillante de toda la imagen (como hacía antes vascular/avr.py, sin usar disc_pos para
    nada), esto ata el radio al mismo disco que ya se decidió dibujar/medir — si disc_pos vino
    del esquema híbrido (fundus/disc_hybrid.py) por convergencia de vasos en vez de brillo, el
    radio igual corresponde a ESE disco y no a otro blob brillante (ej. un reflejo) que quedó
    más grande o más brillante en otra parte de la imagen.
    """
    if disc_pos is None:
        return 30.0  # fallback razonable para imágenes ~565x584

    green = frame[:, :, 1] if frame.ndim == 3 else frame
    blurred = cv2.GaussianBlur(green, (blur_ksize, blur_ksize), 0)
    h, w = green.shape[:2]
    if fov_mask is None:
        fov_mask = np.full((h, w), 255, np.uint8)

    valid = blurred[fov_mask > 0]
    if valid.size == 0:
        return 30.0

    thresh_val = np.percentile(valid, 90)  # más laxo que el 98 de detect_optic_disc: acá interesa
                                            # el contorno completo del disco, no solo su núcleo
    bright = ((blurred >= thresh_val) & (fov_mask > 0)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 30.0

    point = (float(disc_pos[0]), float(disc_pos[1]))
    best, best_dist = None, None
    for c in contours:
        signed = cv2.pointPolygonTest(c, point, True)
        dist = 0.0 if signed >= 0 else -signed  # 0 si disc_pos cae adentro del contorno
        if best is None or dist < best_dist:
            best, best_dist = c, dist

    if best is None or best_dist > search_radius:
        return 30.0

    area = cv2.contourArea(best)
    return max(float(np.sqrt(area / np.pi)), 10.0)


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


def draw_roi(frame: np.ndarray, disc_pos=None, macula_pos=None, needs_review: bool = False,
             disc_radius: float = 25) -> np.ndarray:
    """Dibuja círculos: amarillo = disco óptico (rojo si el esquema híbrido marcó revisión), azul = mácula.
    disc_radius: tamaño real del círculo del disco (ver roi_extractor.estimate_disc_radius) —
    default 25 solo para llamadas viejas que no lo pasan."""
    out = frame.copy()
    if disc_pos is not None:
        color = (0, 0, 255) if needs_review else (0, 255, 255)
        cv2.circle(out, disc_pos, int(round(disc_radius)), color, 2)
        label = "disco (revisar)" if needs_review else "disco"
        cv2.putText(out, label, (disc_pos[0] - 20, disc_pos[1] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    if macula_pos is not None:
        cv2.circle(out, macula_pos, 20, (255, 128, 0), 2)
        cv2.putText(out, "macula", (macula_pos[0] - 25, macula_pos[1] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 1)
    return out
