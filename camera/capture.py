"""
Commit 1 — Captura básica de cámara fisheye.
Uso: python camera/capture.py --device 0 --fps 120
Salir: tecla Q con el foco en la ventana de video.
"""

import cv2
import argparse
import sys
import time
from camera.diagnostics import Diagnostics


def parse_args():
    p = argparse.ArgumentParser(description="Captura de cámara fisheye USB")
    p.add_argument("--device", type=int, default=0, help="Índice del dispositivo (default: 0)")
    p.add_argument("--fps", type=int, default=120, help="FPS objetivo (default: 120)")
    p.add_argument("--width", type=int, default=640, help="Ancho de captura (default: 640)")
    p.add_argument("--height", type=int, default=480, help="Alto de captura (default: 480)")
    return p.parse_args()


def open_camera(device, fps, width, height):
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir /dev/video{device}")
        print("        Verifica que la cámara esté conectada: ls /dev/video*")
        sys.exit(1)

    # MJPEG es necesario para alcanzar 120fps en USB — sin esto el formato RAW
    # satura el ancho de banda y cae a ~8fps
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    real_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[OK] Cámara abierta: {real_w}x{real_h} @ {real_fps:.0f}fps (solicitado: {width}x{height} @ {fps}fps)")
    return cap, real_w, real_h


def draw_overlay(frame, fps_real, latency_ms):
    label = f"FPS: {fps_real:.0f}"
    cv2.putText(frame, label, (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, label, (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)


def main():
    args = parse_args()
    cap, real_w, real_h = open_camera(args.device, args.fps, args.width, args.height)
    diag = Diagnostics()

    WIN = "Mirada — Captura fisheye (Q para salir)"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 800, 480)

    print("[INFO] Mostrando video. Presiona Q en la ventana para salir.")
    print()

    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        latency_ms = (time.perf_counter() - t0) * 1000

        if not ret:
            print("[WARN] Frame perdido, reintentando...")
            continue

        diag.update(latency_ms)
        fps_real = diag.fps()

        draw_overlay(frame, fps_real, latency_ms)

        display = cv2.resize(frame, (800, 480))
        cv2.imshow(WIN, display)

        diag.print_terminal(real_w, real_h)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n[OK] Captura finalizada.")


if __name__ == "__main__":
    main()
