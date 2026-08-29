"""
Commit 7 — Adquisición de fondo de ojo con lente 20D.
Controles: E/D exposición  B/N brillo  ESPACIO captura  Q salir

Uso:
  python fundus/acquisition.py --device 0 --eye left --output fundus_images/
"""

import cv2
import numpy as np
import argparse
import sys

from fundus.quality_check import compute_sharpness, sharpness_percent, SHARPNESS_THRESHOLD
from fundus.storage import save_frame


# Rango de exposición soportado por V4L2 en modo manual (valor log₂ de microsegundos)
EXPOSURE_MIN = -13
EXPOSURE_MAX = -1
EXPOSURE_STEP = 1

BRIGHTNESS_MIN = -64
BRIGHTNESS_MAX = 64
BRIGHTNESS_STEP = 8


def parse_args():
    p = argparse.ArgumentParser(description="Adquisición de fondo de ojo")
    p.add_argument("--device",  type=int,   default=0)
    p.add_argument("--eye",     type=str,   default="left", choices=["left", "right"])
    p.add_argument("--output",  type=str,   default="fundus_images/")
    p.add_argument("--calib",   type=str,   default=None)
    return p.parse_args()


def open_camera(device: int):
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir /dev/video{device}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # Exposición manual para control preciso
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[OK] Cámara: {w}x{h} @ {cap.get(cv2.CAP_PROP_FPS):.0f}fps")
    return cap, w, h


def load_undistort_maps(calib_path, frame_size):
    if calib_path is None:
        return None, None
    try:
        import numpy as np
        data = np.load(calib_path)
        K, D = data["K"], data["D"]
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, frame_size, alpha=0.0)
        m1, m2 = cv2.initUndistortRectifyMap(K, D, None, new_K, frame_size, cv2.CV_16SC2)
        print(f"[OK] Calibración cargada: {calib_path}")
        return m1, m2
    except FileNotFoundError:
        return None, None


def draw_ui(frame, sharpness_pct: int, exposure: int, brightness: int,
            eye: str, saved_count: int):
    h, w = frame.shape[:2]

    # Barra de nitidez arriba
    bar_w = int(w * sharpness_pct / 100)
    color = (0, 200, 0) if sharpness_pct >= int(SHARPNESS_THRESHOLD / 5) else (0, 0, 220)
    cv2.rectangle(frame, (0, 0), (w, 18), (30, 30, 30), -1)
    cv2.rectangle(frame, (0, 0), (bar_w, 18), color, -1)
    label = "NITIDA" if color == (0, 200, 0) else "BORROSA"
    cv2.putText(frame, f"Nitidez: {sharpness_pct}%  [{label}]",
                (8, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Panel inferior con parámetros
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 36), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    exp_ms = round(2 ** (-exposure) / 1000, 1)
    cv2.putText(frame, f"Ojo: {eye.upper()}  |  Exp: {exp_ms}ms (E/D)  |  Brillo: {brightness} (B/N)  |  Guardadas: {saved_count}  |  ESPACIO=captura  Q=salir",
                (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)


def main():
    args = parse_args()
    cap, w, h = open_camera(args.device)
    map1, map2 = load_undistort_maps(args.calib, (w, h))

    exposure   = -6   # valor inicial razonable (~1/64s)
    brightness = 0
    saved_count = 0

    # Buffer del último segundo para elegir el frame más nítido al presionar ESPACIO
    best_frame      = None
    best_sharpness  = -1.0

    cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)

    WIN = "Mirada — Fondo de ojo (Q salir)"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 800, 480)

    print(f"[INFO] Ojo: {args.eye} | Salida: {args.output}")
    print("[INFO] E/D: exposición | B/N: brillo | ESPACIO: captura | Q: salir")

    sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        if map1 is not None:
            frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

        frame = cv2.filter2D(frame, -1, sharpening_kernel)

        sharp = compute_sharpness(frame)
        sharp_pct = sharpness_percent(frame)

        if sharp > best_sharpness:
            best_sharpness = sharp
            best_frame = frame.copy()

        display = frame.copy()
        draw_ui(display, sharp_pct, exposure, brightness, args.eye, saved_count)

        cv2.imshow(WIN, cv2.resize(display, (800, 480)))

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("e"):   # más exposición (imagen más brillante)
            exposure = min(exposure + EXPOSURE_STEP, EXPOSURE_MAX)
            cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
            best_sharpness = -1.0

        elif key == ord("d"):   # menos exposición
            exposure = max(exposure - EXPOSURE_STEP, EXPOSURE_MIN)
            cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
            best_sharpness = -1.0

        elif key == ord("b"):   # más brillo
            brightness = min(brightness + BRIGHTNESS_STEP, BRIGHTNESS_MAX)
            cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)

        elif key == ord("n"):   # menos brillo
            brightness = max(brightness - BRIGHTNESS_STEP, BRIGHTNESS_MIN)
            cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)

        elif key == ord(" "):   # guardar mejor frame del buffer
            if best_frame is not None:
                path = save_frame(best_frame, args.output, args.eye,
                                  exposure, sharpness_percent(best_frame))
                saved_count += 1
                best_sharpness = -1.0
                print(f"\n[OK] Frame guardado: {path}")
            else:
                print("\n[WARN] Sin frame disponible todavía")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[OK] Sesión terminada. {saved_count} imágenes guardadas en {args.output}")


if __name__ == "__main__":
    main()
