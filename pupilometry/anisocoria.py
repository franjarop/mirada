"""
Commit 6 — Detección de anisocoria y registro CSV.
Compara ambas pupilas simultáneamente, detecta diferencias de tamaño
y guarda cada medición en un CSV con timestamp.

Uso:
  python pupilometry/anisocoria.py --device 0 --session paciente01
  python pupilometry/anisocoria.py --device 0 --session paciente01 --calib camera/calibration_data.npz

Controles:
  ESPACIO  →  marcar estímulo luminoso (inicia timer de reflejo)
  E        →  registrar evento manual en CSV
  Q        →  salir y guardar CSV
"""

import cv2
import numpy as np
import argparse
import sys
import time
import mediapipe as mp

from pupilometry.measure   import pixels_to_mm, pupil_to_iris_ratio
from pupilometry.tracker   import PupilTracker
from pupilometry.logger    import PupilLogger
from pupilometry.reflex_timer import ReflexTimer
from pupilometry.detector  import detect_pupil_in_roi, get_iris_circle

LEFT_IRIS          = [474, 475, 476, 477]
RIGHT_IRIS         = [469, 470, 471, 472]

ANISOCORIA_MM      = 1.0   # diferencia clínicamente significativa
LOG_INTERVAL_S     = 0.5   # guardar en CSV cada 0.5 s

WIN = "Mirada — Anisocoria  (Q para salir)"


# ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Detección de anisocoria y registro CSV")
    p.add_argument("--device",  type=int, default=0)
    p.add_argument("--session", type=str, default="sesion")
    p.add_argument("--calib",   type=str, default=None)
    p.add_argument("--output",  type=str, default="sessions")
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
        print(f"[WARN] No se encontró {calib_path}")
        return None, None


# ──────────────────────────────────────────────────────────────────────
def process_eye(frame, lms, iris_lm_ids, w, h):
    """
    Detecta iris y pupila para un ojo. Devuelve (iris_cx, iris_cy, iris_r,
    pupil_cx, pupil_cy, pupil_r, conf) en coordenadas del frame completo.
    """
    iris_cx, iris_cy, iris_r = get_iris_circle(lms, iris_lm_ids, w, h)
    if iris_r <= 0:
        return 0, 0, 0.0, 0, 0, 0.0, 0.0

    margin = int(iris_r * 1.1)
    x1 = max(0, iris_cx - margin)
    y1 = max(0, iris_cy - margin)
    x2 = min(w, iris_cx + margin)
    y2 = min(h, iris_cy + margin)

    roi_color = frame[y1:y2, x1:x2]
    if roi_color.size == 0:
        return iris_cx, iris_cy, iris_r, 0, 0, 0.0, 0.0

    roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)

    # Máscara: fuera del iris → 255 (blanco)
    mask = np.zeros_like(roi_gray)
    cv2.circle(mask, (iris_cx - x1, iris_cy - y1), int(iris_r), 255, -1)
    roi_gray = cv2.bitwise_or(roi_gray, cv2.bitwise_not(mask))

    # Zoom digital
    roi_h, roi_w = roi_gray.shape
    zoom = max(1.0, 200.0 / max(roi_w, roi_h))
    if zoom > 1.0:
        roi_gray_z = cv2.resize(roi_gray, None, fx=zoom, fy=zoom,
                                interpolation=cv2.INTER_CUBIC)
        iris_r_z = iris_r * zoom
    else:
        roi_gray_z = roi_gray
        iris_r_z   = iris_r

    lcx_z, lcy_z, lrad_z, conf = detect_pupil_in_roi(roi_gray_z, iris_r_z)

    if lrad_z <= 0 or lrad_z >= iris_r_z * 0.85:
        return iris_cx, iris_cy, iris_r, 0, 0, 0.0, 0.0

    pupil_cx = x1 + int(lcx_z / zoom)
    pupil_cy = y1 + int(lcy_z / zoom)
    pupil_r  = lrad_z / zoom

    return iris_cx, iris_cy, iris_r, pupil_cx, pupil_cy, pupil_r, conf


# ──────────────────────────────────────────────────────────────────────
def draw_eye_info(frame, label, iris_cx, iris_cy, iris_r,
                  pupil_cx, pupil_cy, pupil_r, diam_mm, conf, col_x, row_y):
    """Dibuja los círculos y la etiqueta de un ojo."""
    if iris_r > 0:
        cv2.circle(frame, (iris_cx, iris_cy), int(iris_r), (0, 140, 255), 1)
    if pupil_r > 0:
        cv2.circle(frame, (pupil_cx, pupil_cy), int(pupil_r), (255, 255, 0), 2)
        cv2.circle(frame, (pupil_cx, pupil_cy), 2, (255, 255, 0), -1)

    color = (0, 255, 0) if conf >= 0.6 else (0, 180, 255)
    txt = f"{label}: {diam_mm:.1f}mm  ({conf*100:.0f}%)" if diam_mm > 0 else f"{label}: --"
    cv2.putText(frame, txt, (col_x, row_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)


# ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir /dev/video{args.device}")
        sys.exit(1)

    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[OK] Cámara: {w}x{h} @ {fps:.0f}fps")

    map1, map2 = load_undistort_maps(args.calib, (w, h))

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    tracker_izq = PupilTracker(window=15)
    tracker_der = PupilTracker(window=15)
    reflex      = ReflexTimer()
    logger      = PupilLogger(args.session, output_dir=args.output)

    last_log  = 0.0
    rows_written = 0

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 800, 480)

    print(f"[INFO] Sesión: {args.session}")
    print("[INFO] ESPACIO → estímulo de luz | E → evento manual | Q → guardar y salir")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        if map1 is not None:
            frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        izq_mm = der_mm = 0.0
        plr_izq = plr_der = 0.0
        conf_izq = conf_der = 0.0
        anisocoria = False
        latency_ms: "float | None" = None

        if result.multi_face_landmarks:
            lms = result.multi_face_landmarks[0].landmark

            # ── Ojo izquierdo ──────────────────────────────────────────
            icx_i, icy_i, ir_i, pcx_i, pcy_i, pr_i, cf_i = process_eye(
                frame, lms, LEFT_IRIS, w, h)
            if pr_i > 0:
                izq_mm  = pixels_to_mm(pr_i * 2, ir_i)
                plr_izq = pupil_to_iris_ratio(pr_i, ir_i)
                st_i    = tracker_izq.update(izq_mm, cf_i)
                conf_izq = cf_i
                latency_ms = reflex.update(izq_mm)

            # ── Ojo derecho ────────────────────────────────────────────
            icx_d, icy_d, ir_d, pcx_d, pcy_d, pr_d, cf_d = process_eye(
                frame, lms, RIGHT_IRIS, w, h)
            if pr_d > 0:
                der_mm  = pixels_to_mm(pr_d * 2, ir_d)
                plr_der = pupil_to_iris_ratio(pr_d, ir_d)
                st_d    = tracker_der.update(der_mm, cf_d)
                conf_der = cf_d

            # ── Anisocoria ────────────────────────────────────────────
            if izq_mm > 0 and der_mm > 0:
                anisocoria = abs(izq_mm - der_mm) >= ANISOCORIA_MM

            # ── Dibujar ambos ojos ────────────────────────────────────
            draw_eye_info(frame, "IZQ", icx_i, icy_i, ir_i,
                          pcx_i, pcy_i, pr_i, izq_mm, conf_izq, 10, h - 60)
            draw_eye_info(frame, "DER", icx_d, icy_d, ir_d,
                          pcx_d, pcy_d, pr_d, der_mm, conf_der, w // 2, h - 60)

            # ── Log CSV ───────────────────────────────────────────────
            now = time.perf_counter()
            if (izq_mm > 0 or der_mm > 0) and (now - last_log >= LOG_INTERVAL_S):
                logger.write(izq_mm, der_mm, plr_izq, plr_der, anisocoria)
                last_log = now
                rows_written += 1
                if rows_written % 20 == 0:
                    logger.flush()
                    print(f"\r  Izq: {izq_mm:.1f}mm  |  Der: {der_mm:.1f}mm  "
                          f"|  Dif: {abs(izq_mm-der_mm):.1f}mm  "
                          f"|  {'⚠ ANISOCORIA' if anisocoria else 'Normal'}    ",
                          end="", flush=True)

        # ── Panel inferior ────────────────────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 90), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        if anisocoria:
            diff = abs(izq_mm - der_mm)
            cv2.putText(frame, f"ANISOCORIA: {diff:.1f}mm",
                        (w // 2 - 140, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        # Latencia del reflejo
        lat = reflex.get_latency_ms()
        if reflex.triggered:
            lat_txt = f"Reflejo: {lat:.0f}ms" if lat else "Reflejo: midiendo..."
            cv2.putText(frame, lat_txt, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(frame, f"Registros: {rows_written}", (w - 190, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        display = cv2.resize(frame, (800, 480))
        cv2.imshow(WIN, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            baseline = izq_mm if izq_mm > 0 else der_mm
            reflex.trigger(baseline)
            logger.queue_event("estimulo_luz")
            print("\n[INFO] Estímulo marcado — midiendo latencia...")
        elif key == ord("e"):
            logger.queue_event("evento_manual")
            print("\n[INFO] Evento manual registrado en CSV.")

    cap.release()
    face_mesh.close()
    logger.close()
    cv2.destroyAllWindows()
    print(f"\n[OK] Finalizado. {rows_written} registros guardados.")


if __name__ == "__main__":
    main()
