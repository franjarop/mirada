"""
Commit 6 — Medición de latencia del reflejo pupilar (PLR timing).

Uso: presiona ESPACIO para marcar el instante del estímulo luminoso.
El timer detecta automáticamente cuándo la pupila comienza a contraerse
y calcula la latencia en milisegundos. Normal: 200–500 ms.
"""

import time
from collections import deque
import numpy as np


class ReflexTimer:
    """
    Mide la latencia entre un estímulo luminoso y el inicio de la contracción pupilar.

    Protocolo:
      1. trigger(current_mm) — llama al encender la luz.
      2. update(diameter_mm) — llama en cada frame con la medición actual.
      3. get_latency_ms()    — devuelve la latencia detectada (None si aún no hay).
    """

    def __init__(self, window: int = 6, threshold_mm: float = 0.25):
        self._window       = window
        self._threshold_mm = threshold_mm
        self._buf: deque   = deque(maxlen=window)
        self._trigger_t: float | None  = None
        self._baseline_mm: float       = 0.0
        self._latency_ms: float | None = None

    # ------------------------------------------------------------------
    def trigger(self, current_mm: float):
        """Marca el instante del estímulo y guarda el diámetro basal."""
        self._trigger_t   = time.perf_counter()
        self._baseline_mm = current_mm
        self._latency_ms  = None
        self._buf.clear()

    def update(self, diameter_mm: float) -> "float | None":
        """
        Registra una medición. Devuelve latencia en ms en el instante en que
        la detecta por primera vez; None en cualquier otro caso.
        """
        if diameter_mm > 0:
            self._buf.append(diameter_mm)

        if self._trigger_t is None or self._latency_ms is not None:
            return None

        if len(self._buf) < 3:
            return None

        recent = float(np.mean(list(self._buf)[-3:]))
        if self._baseline_mm > 0 and (self._baseline_mm - recent) >= self._threshold_mm:
            elapsed = (time.perf_counter() - self._trigger_t) * 1000.0
            self._latency_ms = round(elapsed, 1)
            return self._latency_ms

        return None

    # ------------------------------------------------------------------
    def get_latency_ms(self) -> "float | None":
        return self._latency_ms

    @property
    def triggered(self) -> bool:
        return self._trigger_t is not None

    def reset(self):
        self._trigger_t   = None
        self._baseline_mm = 0.0
        self._latency_ms  = None
        self._buf.clear()
