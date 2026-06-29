"""
Commit 3 — Filtro temporal de banda para magnificación euleriana.
Aísla frecuencias entre freq_lo y freq_hi Hz (rango cardíaco: 0.4–4 Hz).
"""

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi
from collections import deque


class ButterBandpass:
    """
    Filtro Butterworth bandpass en tiempo real usando estado IIR.
    Procesa frame a frame sin necesidad de buffer completo.
    """

    def __init__(self, freq_lo: float, freq_hi: float, fps: float, order: int = 3):
        nyq = fps / 2.0
        lo = freq_lo / nyq
        hi = min(freq_hi / nyq, 0.99)
        self.sos = butter(order, [lo, hi], btype="bandpass", output="sos")
        self._zi = None  # se inicializa con el primer frame

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Aplica el filtro a un frame (H x W x C float32).
        Retorna el frame filtrado con la misma forma.
        """
        h, w, c = frame.shape
        flat = frame.reshape(1, -1).astype(np.float64)  # (1, H*W*C)

        if self._zi is None:
            zi_base = sosfilt_zi(self.sos)              # (n_sos, 2)
            # Escalar por el primer frame para evitar transitorio de paso grande
            self._zi = zi_base[:, :, np.newaxis] * flat  # (n_sos, 2, N)

        filtered, self._zi = sosfilt(self.sos, flat, zi=self._zi, axis=0)
        return filtered.reshape(h, w, c).astype(np.float32)

    def reset(self):
        self._zi = None


class SlidingBandpass:
    """
    Alternativa más simple: buffer deslizante + FFT ideal.
    Introduce latencia de buffer_size/2 frames pero es más estable visualmente.
    Usar cuando el IIR produce artefactos.
    """

    def __init__(self, freq_lo: float, freq_hi: float, fps: float, buffer_size: int = 32):
        self.freq_lo = freq_lo
        self.freq_hi = freq_hi
        self.fps = fps
        self.buffer: deque = deque(maxlen=buffer_size)

    def apply(self, frame: np.ndarray) -> np.ndarray | None:
        """
        Agrega el frame al buffer y retorna el frame central filtrado.
        Retorna None hasta que el buffer esté lleno.
        """
        self.buffer.append(frame.astype(np.float32))
        if len(self.buffer) < self.buffer.maxlen:
            return None

        buf = np.stack(self.buffer, axis=0)          # (T, H, W, C)
        T = buf.shape[0]
        freqs = np.fft.rfftfreq(T, d=1.0 / self.fps)

        fft = np.fft.rfft(buf, axis=0)
        mask = (freqs >= self.freq_lo) & (freqs <= self.freq_hi)
        fft[~mask] = 0
        filtered = np.fft.irfft(fft, n=T, axis=0)   # (T, H, W, C)

        return filtered[T // 2].astype(np.float32)  # frame central
