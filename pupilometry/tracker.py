"""
Commit 5 — Seguimiento temporal de la pupila y detección de cambios bruscos.
"""

from collections import deque
import numpy as np


class PupilTracker:
    """
    Suaviza las mediciones de diámetro pupilar y detecta cambios abruptos.
    Un cambio > threshold_mm en menos de 1 segundo se considera significativo.
    """

    def __init__(self, window: int = 15, threshold_mm: float = 0.8):
        self.window = window
        self.threshold_mm = threshold_mm
        self._buf: deque = deque(maxlen=window)

    def update(self, diameter_mm: float, confidence: float) -> dict:
        """
        Agrega una nueva medición y retorna el estado actual.
        Solo acepta mediciones con confianza >= 0.5.
        """
        if confidence >= 0.5 and diameter_mm > 0:
            self._buf.append(diameter_mm)

        if len(self._buf) == 0:
            return {"diameter_mm": 0.0, "smooth_mm": 0.0,
                    "abrupt_change": False, "trend": "sin datos"}

        smooth = float(np.mean(self._buf))
        abrupt = False

        if len(self._buf) >= 4:
            recent = np.mean(list(self._buf)[-3:])
            older  = np.mean(list(self._buf)[:3])
            abrupt = abs(recent - older) > self.threshold_mm

        trend = "estable"
        if len(self._buf) >= 6:
            first_half = np.mean(list(self._buf)[:len(self._buf)//2])
            second_half = np.mean(list(self._buf)[len(self._buf)//2:])
            diff = second_half - first_half
            if diff > 0.3:
                trend = "dilatando"
            elif diff < -0.3:
                trend = "contrayendo"

        return {
            "diameter_mm": diameter_mm,
            "smooth_mm":   round(smooth, 2),
            "abrupt_change": abrupt,
            "trend": trend,
        }

    def reset(self):
        self._buf.clear()
