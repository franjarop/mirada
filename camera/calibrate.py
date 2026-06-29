"""
Commit 2 — Calibración de cámara fisheye con tablero de ajedrez.
Uso: python camera/calibrate.py --output camera/calibration_data.npz
     python camera/calibrate.py --output camera/calibration_data.npz --cols 9 --rows 6

Necesitas: tablero de ajedrez impreso, mínimo 9x6 cuadros.
El script captura 20 poses automáticamente cuando detecta el tablero.
"""

import cv2
import numpy as np
import argparse
import sys
import time


WINDOW = "Mirada — Calibración fisheye (Q para salir)"


def parse_args():
    p = argparse.ArgumentParser(description="Calibración de cámara fisheye")
    p.add_argument("--output", default="camera/calibration_data.npz")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--cols", type=int, default=9, help="Columnas internas del tablero (default: 9)")
    p.add_argument("--rows", type=int, default=6, help="Filas internas del tablero (default: 6)")
    p.add_argument("--square-mm", type=float, default=25.0, help="Tamaño del cuadro en mm (default: 25)")
    p.add_argument("--poses", type=int, default=20, help="Poses necesarias (default: 20)")
    return p.parse_args()


def make_object_points(cols, rows, square_mm):
    """Coordenadas 3D del tablero en el plano Z=0."""
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm
    return objp


def calibrate_fisheye(obj_pts, img_pts, img_size, poses):
    """Calibra usando el modelo estándar de OpenCV (robusto para lentes fisheye USB)."""
    flags = (
        cv2.CALIB_RATIONAL_MODEL |   # 6 coeficientes radiales — mejor para fisheye
        cv2.CALIB_FIX_ASPECT_RATIO
    )
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
        obj_pts, img_pts, img_size, None, None, flags=flags
    )
    return rms, K, D


def draw_status(frame, n_captured, n_needed, detected):
    h, w = frame.shape[:2]
    # Fondo semitransparente arriba
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    color = (0, 255, 0) if detected else (0, 180, 255)
    estado = "Tablero detectado" if detected else "Buscando tablero..."
    cv2.putText(frame, estado, (12, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    barra = f"Poses: {n_captured}/{n_needed}"
    cv2.putText(frame, barra, (12, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

    # Barra de progreso
    pct = n_captured / n_needed
    cv2.rectangle(frame, (w - 160, 10), (w - 10, 30), (60, 60, 60), -1)
    cv2.rectangle(frame, (w - 160, 10), (w - 160 + int(150 * pct), 30), (0, 220, 0), -1)


def main():
    args = parse_args()
    COLS, ROWS = args.cols, args.rows
    objp = make_object_points(COLS, ROWS, args.square_mm)

    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir /dev/video{args.device}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 800, 480)

    obj_pts, img_pts = [], []
    n_needed = args.poses
    last_capture = 0.0
    MIN_INTERVAL = 0.5  # segundos entre capturas para evitar poses similares

    print(f"[INFO] Tablero: {COLS}x{ROWS} cuadros | Tamaño: {args.square_mm}mm | Meta: {n_needed} poses")
    print(f"[INFO] Mueve el tablero lentamente frente a la cámara a distintos ángulos.")
    print()

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    while len(img_pts) < n_needed:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (COLS, ROWS), None)

        now = time.perf_counter()
        if found and (now - last_capture) > MIN_INTERVAL:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_pts.append(objp)
            img_pts.append(corners2)
            last_capture = now
            n = len(img_pts)
            print(f"\r  Poses capturadas: {n}/{n_needed}", end="", flush=True)
            cv2.drawChessboardCorners(frame, (COLS, ROWS), corners2, found)

        draw_status(frame, len(img_pts), n_needed, found)
        display = cv2.resize(frame, (800, 480))
        cv2.imshow(WINDOW, display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n[INFO] Calibración cancelada.")
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n\n[INFO] Calibrando con {n_needed} poses — espera un momento...")

    try:
        rms, K, D = calibrate_fisheye(obj_pts, img_pts, (img_w, img_h), n_needed)
    except cv2.error as e:
        print(f"[ERROR] Calibración fallida: {e}")
        print("        Intenta con más variedad de ángulos y posiciones del tablero.")
        sys.exit(1)

    np.savez(args.output, K=K, D=D, img_size=np.array([img_w, img_h]))

    print(f"  Error de reproyección: {rms:.2f}px", end="  ")
    if rms < 1.0:
        print("← BUENO")
    elif rms < 2.0:
        print("← ACEPTABLE (intenta con más poses para mejorar)")
    else:
        print("← ALTO — repite la calibración con más variedad de ángulos")

    print(f"  Calibración guardada en: {args.output}")
    print()
    print("[OK] Ahora ejecuta: python camera/undistort.py --calib", args.output)


if __name__ == "__main__":
    main()
