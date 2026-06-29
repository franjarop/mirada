"""
Commit 3 — Pirámide gaussiana para magnificación euleriana de color.
"""

import cv2
import numpy as np


def build_gaussian_pyramid(frame: np.ndarray, levels: int) -> list:
    """
    Construye una pirámide gaussiana de `levels` niveles.
    Retorna lista de imágenes, de mayor a menor resolución.
    """
    pyramid = [frame.astype(np.float32)]
    for _ in range(levels - 1):
        pyramid.append(cv2.pyrDown(pyramid[-1]))
    return pyramid


def upsample_to(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Escala una imagen al tamaño objetivo con interpolación lineal."""
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
