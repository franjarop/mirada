"""
Commit 3 — Pipeline de magnificación euleriana de color.
Amplifica cambios de color imperceptibles (pulso cardíaco, flujo sanguíneo).

Uso con video:
  python eulerian/magnify.py --input test_video.mp4 --alpha 50

Uso con cámara en vivo:
  python eulerian/magnify.py --device 0 --alpha 50 --calib camera/calibration_data.npz
"""

import cv2
import numpy as np
import argparse
import sys
import time

from eulerian.pyramid import build_gaussian_pyramid, upsample_to
from eulerian.temporal_filter import ButterBandpass


WIN = "Mirada — Magnificación euleriana  (Q para salir)"


def parse_args():
    p = argparse.ArgumentParser(description="Magnificación euleriana de color en tiempo real")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--device", type=int, help="Índice de cámara USB (ej: 0)")
    src.add_argument("--input",  type=str, help="Ruta a video de archivo (.mp4, .avi…)")
    p.add_argument("--alpha",   type=float, default=50.0, help="Intensidad de amplificación (default: 50)")
    p.add_argument("--freq-lo", type=float, default=0.4,  help="Frecuencia mínima en Hz (default: 0.4 = 24 BPM)")
    p.add_argument("--freq-hi", type=float, default=4.0,  help="Frecuencia máxima en Hz (default: 4.0 = 240 BPM)")
    p.add_argument("--levels",  type=int,   default=3,    help="Niveles de pirámide (default: 3)")
    p.add_argument("--calib",   type=str,   default=None, help="Archivo de calibración .npz (opcional)")
    return p.parse_args()


def load_undistort_maps(calib_path, frame_size):
    """Carga mapas de corrección fisheye si se proporciona calibración."""
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
        print(f"[WARN] No se encontró {calib_path} — corriendo sin corrección fisheye")
        return None, None


def open_source(args):
    if args.device is not None:
        cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 120)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        cap = cv2.VideoCapture(args.input)

    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la fuente de video")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or fps > 240:
        fps = 30.0  # fallback seguro
    print(f"[OK] Fuente: {w}x{h} @ {fps:.0f}fps")
    return cap, fps, w, h


def magnify_frame(frame_f32, filtered, alpha, levels):
    """
    Aplica la magnificación:
      1. Amplifica la señal filtrada por alpha
      2. La escala al tamaño original
      3. La suma al frame original — la imagen siempre es reconocible
      4. Recorta a [0, 255]
    """
    h, w = frame_f32.shape[:2]
    amplified = upsample_to(filtered * alpha, h, w)
    # Limitar cuánto puede alterar la imagen original (evita saturación total)
    amplified = np.clip(amplified, -80, 80)
    result = frame_f32 + amplified
    return np.clip(result, 0, 255).astype(np.uint8)


def add_label(img, text, pos=(10, 28)):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    args = parse_args()
    cap, fps, w, h = open_source(args)
    map1, map2 = load_undistort_maps(args.calib, (w, h))

    bandpass = ButterBandpass(args.freq_lo, args.freq_hi, fps)

    print(f"[INFO] Alpha: {args.alpha}  |  Banda: {args.freq_lo}–{args.freq_hi} Hz  |  Niveles: {args.levels}")
    print("[INFO] Apunta la cámara a tu muñeca o mejilla — espera 3-5 segundos para ver el efecto.")
    print("[INFO] Presiona Q para salir, +/- para ajustar alpha en vivo.")

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 960, 360)

    alpha = args.alpha
    fps_counter, t0 = 0, time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if map1 is not None:
            frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

        frame_f32 = frame.astype(np.float32)

        # Bajar resolución con pirámide para procesar menos pixeles
        pyramid = build_gaussian_pyramid(frame_f32, args.levels)
        level   = pyramid[-1]   # nivel más pequeño

        # Filtrar directamente en BGR — evita pérdida de señal por conversión
        filtered = bandpass.apply(level)   # float32, valores típicos 0.1–3.0

        # Amplificar y escalar al tamaño original
        magnified = magnify_frame(frame_f32, filtered, alpha, args.levels)

        # Mostrar lado a lado
        left  = cv2.resize(frame,     (480, 360))
        right = cv2.resize(magnified, (480, 360))
        add_label(left,  "Original")
        add_label(right, f"Magnificado  alpha={alpha:.0f}")
        combined = np.hstack([left, right])
        cv2.imshow(WIN, combined)

        fps_counter += 1
        elapsed = time.perf_counter() - t0
        if elapsed >= 2.0:
            fps_real = fps_counter / elapsed
            print(f"\r  FPS: {fps_real:.1f}  |  alpha: {alpha:.0f}  |  banda: {args.freq_lo}–{args.freq_hi} Hz    ",
                  end="", flush=True)
            fps_counter, t0 = 0, time.perf_counter()

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("+") or key == ord("="):
            alpha = min(alpha + 10, 300)
        elif key == ord("-"):
            alpha = max(alpha - 10, 1)

    cap.release()
    cv2.destroyAllWindows()
    print("\n[OK] Finalizado.")


if __name__ == "__main__":
    main()
