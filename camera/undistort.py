"""
Commit 2 — Corrección de distorsión fisheye en tiempo real.
Uso: python camera/undistort.py --calib camera/calibration_data.npz

Muestra dos ventanas lado a lado:
  IZQUIERDA: imagen original con efecto fisheye
  DERECHA:   imagen corregida (líneas rectas)
"""

import cv2
import numpy as np
import argparse
import sys


WIN = "Mirada — Corrección fisheye: Original | Corregida  (Q para salir)"


def parse_args():
    p = argparse.ArgumentParser(description="Corrección de distorsión fisheye en tiempo real")
    p.add_argument("--calib", default="camera/calibration_data.npz")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--balance", type=float, default=0.0,
                   help="0.0 = recorta todo negro, 1.0 = conserva todo (default: 0.0)")
    return p.parse_args()


def load_calibration(path):
    try:
        data = np.load(path)
        K = data["K"]
        D = data["D"]
        img_size = tuple(data["img_size"])
        print(f"[OK] Calibración cargada: {path}  |  imagen: {img_size[0]}x{img_size[1]}")
        return K, D, img_size
    except FileNotFoundError:
        print(f"[ERROR] No se encontró el archivo de calibración: {path}")
        print("        Ejecuta primero: python camera/calibrate.py --output camera/calibration_data.npz")
        sys.exit(1)


def build_maps(K, D, img_size, balance):
    """Pre-calcula los mapas de remapeo para no recalcular en cada frame."""
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, img_size, alpha=balance)
    map1, map2 = cv2.initUndistortRectifyMap(
        K, D, None, new_K, img_size, cv2.CV_16SC2
    )
    return map1, map2


def draw_grid(img, step=60, color=(40, 40, 40)):
    """Dibuja una cuadrícula para ver mejor la corrección de distorsión."""
    h, w = img.shape[:2]
    for x in range(0, w, step):
        cv2.line(img, (x, 0), (x, h), color, 1)
    for y in range(0, h, step):
        cv2.line(img, (0, y), (w, y), color, 1)


def add_label(img, text):
    cv2.putText(img, text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    args = parse_args()
    K, D, img_size = load_calibration(args.calib)

    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir /dev/video{args.device}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Pre-calcular mapas de remapeo (más eficiente que undistort por frame)
    map1, map2 = build_maps(K, D, img_size, args.balance)
    print("[INFO] Mostrando corrección en tiempo real. Presiona Q para salir.")

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 960, 360)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        undistorted = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

        # Redimensionar a 480x360 para mostrar lado a lado
        left  = cv2.resize(frame, (480, 360))
        right = cv2.resize(undistorted, (480, 360))

        # Cuadrícula para visualizar la corrección
        draw_grid(left)
        draw_grid(right)

        add_label(left, "Original (fisheye)")
        add_label(right, "Corregida")

        combined = np.hstack([left, right])
        cv2.imshow(WIN, combined)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[OK] Saliendo.")


if __name__ == "__main__":
    main()
