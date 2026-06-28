"""
Commit 1 — Diagnósticos de cámara: FPS real, resolución y latencia.
Puede usarse como módulo o ejecutarse directamente:
  python camera/diagnostics.py --device 0 --duration 10
"""

import cv2
import argparse
import time
import sys
from collections import deque


class Diagnostics:
    """Mide FPS real y latencia de captura usando ventana deslizante de 1 segundo."""

    def __init__(self, window=60):
        self._times = deque(maxlen=window)    # timestamps de frames
        self._latencies = deque(maxlen=window) # latencia por frame (ms)
        self._last_print = 0.0

    def update(self, latency_ms: float):
        now = time.perf_counter()
        self._times.append(now)
        self._latencies.append(latency_ms)

    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0

    def avg_latency(self) -> float:
        return sum(self._latencies) / len(self._latencies) if self._latencies else 0.0

    def print_terminal(self, width: int, height: int, interval: float = 1.0):
        """Imprime estadísticas en terminal cada `interval` segundos."""
        now = time.perf_counter()
        if now - self._last_print < interval:
            return
        self._last_print = now
        fps = self.fps()
        lat = self.avg_latency()
        print(f"\r  FPS real: {fps:5.1f}  |  Resolución: {width}x{height}  |  Latencia: {lat:.1f}ms    ",
              end="", flush=True)


def run_standalone(device: int, duration: int):
    """Modo standalone: mide durante `duration` segundos y muestra resumen."""
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir /dev/video{device}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"[INFO] Midiendo durante {duration}s — cámara: {width}x{height} @ {target_fps:.0f}fps solicitado")

    diag = Diagnostics(window=500)
    t_end = time.perf_counter() + duration
    frames = 0

    while time.perf_counter() < t_end:
        t0 = time.perf_counter()
        ret, _ = cap.read()
        if not ret:
            continue
        latency_ms = (time.perf_counter() - t0) * 1000
        diag.update(latency_ms)
        frames += 1
        diag.print_terminal(width, height)

    cap.release()

    fps_final = diag.fps()
    lat_final = diag.avg_latency()

    print(f"\n\n{'─'*50}")
    print(f"  Frames capturados : {frames}")
    print(f"  FPS real          : {fps_final:.1f}")
    print(f"  Resolución        : {width}x{height}")
    print(f"  Latencia promedio : {lat_final:.1f}ms")

    if fps_final >= 60:
        print(f"  Estado            : OK — FPS suficiente para el proyecto")
    elif fps_final >= 30:
        print(f"  Estado            : WARN — FPS bajo, revisa cable USB y puerto")
    else:
        print(f"  Estado            : ERROR — FPS muy bajo, problema con la cámara")
    print(f"{'─'*50}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Diagnóstico de cámara fisheye")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--duration", type=int, default=10, help="Segundos de medición (default: 10)")
    args = p.parse_args()
    run_standalone(args.device, args.duration)
