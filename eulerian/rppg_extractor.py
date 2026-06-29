"""
Commit 4 — Extracción de señal rPPG desde la esclera ocular.
Detecta el ojo con MediaPipe, extrae el canal verde del ROI y estima BPM en tiempo real.

Uso:
  python eulerian/rppg_extractor.py --device 0 --calib camera/calibration_data.npz
"""

import cv2
import numpy as np
import argparse
import sys
import time
from collections import deque
from scipy.signal import butter, sosfilt, find_peaks

from eulerian.eye_roi import EyeROI
from eulerian.visualizer import SignalPlot


WIN_VIDEO  = "Mirada — rPPG Ocular  (Q para salir)"
WIN_SIGNAL = "Mirada — Señal rPPG"

# Rango cardíaco normal: 40–180 BPM = 0.67–3.0 Hz
FREQ_LO = 0.67
FREQ_HI = 3.0


def parse_args():
    p = argparse.ArgumentParser(description="Extracción de rPPG desde el ojo")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--calib",  type=str, default=None)
    p.add_argument("--eye",    type=str, default="left", choices=["left", "right"])
    return p.parse_args()


def load_undistort_maps(calib_path, frame_size):
    if calib_path is None:
        return None, None
    try:
        data = np.load(calib_path)
        K, D = data["K"], data["D"]
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, frame_size, alpha=0.0)
        m1, m2 = cv2.initUndistortRectifyMap(K, D, None, new_K, frame_size, cv2.CV_16SC2)
        print(f"[OK] Calibración cargada: {calib_path}")
        return m1, m2
    except FileNotFoundError:
        print(f"[WARN] No se encontró {calib_path} — sin corrección fisheye")
        return None, None


def design_bandpass(fps):
    nyq = fps / 2.0
    sos = butter(3, [FREQ_LO / nyq, FREQ_HI / nyq], btype="bandpass", output="sos")
    return sos


def estimate_bpm(signal: np.ndarray, fps: float, sos) -> tuple:
    """
    Filtra la señal y estima BPM por detección de picos.
    Retorna (bpm, confianza) o (0, 0) si no hay suficiente señal.
    """
    if len(signal) < int(fps * 4):     # necesita mínimo 4 segundos
        return 0.0, 0.0

    filtered = sosfilt(sos, signal - signal.mean())

    # Distancia mínima entre picos: 40 BPM → 60/40=1.5s
    min_dist = int(fps * 0.4)
    peaks, props = find_peaks(filtered, distance=min_dist, prominence=0.001)

    if len(peaks) < 2:
        return 0.0, 0.0

    intervals = np.diff(peaks) / fps          # segundos entre picos
    bpm = 60.0 / intervals.mean()

    if not (40 <= bpm <= 180):
        return 0.0, 0.0

    # Confianza: regularidad de los intervalos (coef. variación inverso)
    cv = intervals.std() / intervals.mean() if intervals.mean() > 0 else 1.0
    confidence = max(0.0, min(1.0, 1.0 - cv * 2))

    return bpm, confidence


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 120)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir /dev/video{args.device}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or fps > 240:
        fps = 30.0
    print(f"[OK] Cámara: {w}x{h} @ {fps:.0f}fps")

    map1, map2 = load_undistort_maps(args.calib, (w, h))
    roi_detector = EyeROI(eye=args.eye)
    sos = design_bandpass(fps)
    plotter = SignalPlot(width=600, height=200, max_samples=int(fps * 8))

    # Buffer de señal rPPG cruda (canal verde medio del ROI)
    signal_buf: deque = deque(maxlen=int(fps * 12))  # 12 segundos
    bpm, confidence = 0.0, 0.0
    last_bpm_update = 0.0

    cv2.namedWindow(WIN_VIDEO,  cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_SIGNAL, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_VIDEO,  600, 400)
    cv2.resizeWindow(WIN_SIGNAL, 600, 200)
    cv2.moveWindow(WIN_VIDEO,    0,   0)
    cv2.moveWindow(WIN_SIGNAL,   0, 420)

    print(f"[INFO] Ojo: {args.eye} | Banda: {FREQ_LO}–{FREQ_HI} Hz")
    print("[INFO] Mantén la cabeza quieta — la señal necesita ~10s para estabilizarse.")
    print("[INFO] Presiona Q para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        if map1 is not None:
            frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

        roi, bbox, frame_annot = roi_detector.extract(frame)

        if roi is not None and roi.size > 0:
            # Canal verde normalizado como señal rPPG
            green_mean = float(roi[:, :, 1].mean())
            signal_buf.append(green_mean)
            plotter.update(green_mean)

            # Actualizar BPM cada segundo
            now = time.perf_counter()
            if now - last_bpm_update >= 1.0 and len(signal_buf) > int(fps * 4):
                sig = np.array(signal_buf, dtype=np.float32)
                bpm, confidence = estimate_bpm(sig, fps, sos)
                last_bpm_update = now

                if bpm > 0:
                    print(f"\r  BPM estimado: {bpm:.0f}  |  Confianza: {confidence*100:.0f}%    ",
                          end="", flush=True)

        # Overlay BPM en el video
        if bpm > 0:
            color = (0, 255, 0) if confidence >= 0.6 else (0, 180, 255)
            cv2.putText(frame_annot, f"BPM: {bpm:.0f}", (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
            cv2.putText(frame_annot, f"BPM: {bpm:.0f}", (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        signal_img = plotter.render(bpm, confidence)

        cv2.imshow(WIN_VIDEO,  frame_annot)
        cv2.imshow(WIN_SIGNAL, signal_img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    roi_detector.close()
    cv2.destroyAllWindows()
    print("\n[OK] Finalizado.")


if __name__ == "__main__":
    main()
