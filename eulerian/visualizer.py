"""
Commit 4 — Visualizador de señal rPPG en tiempo real usando OpenCV.
Dibuja la señal como una gráfica de onda continua sobre un canvas negro.
"""

import cv2
import numpy as np
from collections import deque


class SignalPlot:
    """Gráfica de señal en tiempo real dibujada con OpenCV."""

    def __init__(self, width: int = 600, height: int = 200,
                 max_samples: int = 150, color=(0, 255, 128)):
        self.w = width
        self.h = height
        self.color = color
        self.buffer: deque = deque(maxlen=max_samples)
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)

    def update(self, value: float):
        self.buffer.append(value)

    def render(self, bpm: float = 0.0, confidence: float = 0.0) -> np.ndarray:
        canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        # Cuadrícula sutil
        for y in range(0, self.h, self.h // 4):
            cv2.line(canvas, (0, y), (self.w, y), (30, 30, 30), 1)

        if len(self.buffer) < 2:
            self._draw_labels(canvas, bpm, confidence)
            return canvas

        data = np.array(self.buffer, dtype=np.float32)

        # Normalizar a altura del canvas
        d_min, d_max = data.min(), data.max()
        rng = d_max - d_min if d_max - d_min > 1e-6 else 1.0
        norm = (data - d_min) / rng  # 0–1
        margin = int(self.h * 0.1)
        ys = ((1.0 - norm) * (self.h - 2 * margin) + margin).astype(int)

        n = len(ys)
        xs = np.linspace(0, self.w - 1, n).astype(int)

        # Dibujar señal
        for i in range(1, n):
            cv2.line(canvas, (xs[i-1], ys[i-1]), (xs[i], ys[i]), self.color, 2)

        self._draw_labels(canvas, bpm, confidence)
        return canvas

    def _draw_labels(self, canvas, bpm, confidence):
        if bpm > 0:
            color_bpm = (0, 255, 0) if confidence >= 0.6 else (0, 180, 255)
            cv2.putText(canvas, f"BPM: {bpm:.0f}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_bpm, 2)
            cv2.putText(canvas, f"Confianza: {confidence*100:.0f}%",
                        (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        else:
            cv2.putText(canvas, "Estabilizando señal...", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)
