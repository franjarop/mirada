"""
Commit 5 — Detección de pupila y medición de diámetro en tiempo real.
Usa MediaPipe para localizar el ojo, luego umbralización adaptativa para
segmentar la pupila. El iris sirve como referencia de escala (11.7mm).

Uso:
  python pupilometry/detector.py --device 0 --calib camera/calibration_data.npz
"""

import cv2
import numpy as np
import argparse
import sys
import time
import mediapipe as mp

from pupilometry.measure import pixels_to_mm, pupil_to_iris_ratio
from pupilometry.tracker import PupilTracker

# Landmarks MediaPipe para iris (requiere refine_landmarks=True)
LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

LEFT_EYE_CONTOUR  = [362, 382, 381, 380, 374, 373, 390, 249,
                      263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155,
                     133, 173, 157, 158, 159, 160, 161, 246]

WIN = "Mirada — Pupilometría  (Q para salir)"


def parse_args():
    p = argparse.ArgumentParser(description="Detección de pupila y medición de diámetro")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--calib",  type=str, default=None)
    p.add_argument("--eye",    type=str, default="left", choices=["left", "right"])
    return p.parse_args()


def load_undistort_maps(calib_path, frame_size):
    if calib_path is None:
        return None, None
    try:
        data  = np.load(calib_path)
        K, D  = data["K"], data["D"]
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, frame_size, alpha=0.0)
        m1, m2 = cv2.initUndistortRectifyMap(K, D, None, new_K, frame_size, cv2.CV_16SC2)
        print(f"[OK] Calibración cargada: {calib_path}")
        return m1, m2
    except FileNotFoundError:
        print(f"[WARN] No se encontró {calib_path}")
        return None, None


def get_iris_circle(lms, iris_lm_ids, w, h):
    """Calcula centro y radio del iris a partir de los 4 landmarks del iris."""
    pts = np.array([(lms[i].x * w, lms[i].y * h) for i in iris_lm_ids])
    cx, cy = pts.mean(axis=0)
    # Radio = distancia media desde el centro a los 4 puntos
    radii = np.linalg.norm(pts - [cx, cy], axis=1)
    return int(cx), int(cy), float(radii.mean())


def detect_pupil_in_roi(roi_gray: np.ndarray, iris_r: float = 0) -> tuple:
    """
    Detecta la pupila partiendo del pixel más oscuro dentro del iris.
    La pupila siempre es la región más negra del ojo.
    Retorna (cx, cy, radius, confidence) en coordenadas del ROI.
    """
    if roi_gray.size == 0 or iris_r <= 0:
        return 0, 0, 0, 0.0

    blurred = cv2.GaussianBlur(roi_gray, (5, 5), 0)

    # Ignorar píxeles brillantes del borde de la máscara (=255)
    valid = blurred.copy()
    valid[valid > 250] = 255    # los que pusimos en blanco no cuentan

    # Pixel más oscuro → centro aproximado de la pupila
    min_val = int(valid.min())
    if min_val > 80:            # si el mínimo es muy brillante, no hay pupila clara
        return 0, 0, 0, 0.0

    # Umbral adaptativo: mínimo + margen para capturar toda la pupila
    threshold = min(min_val + 40, 80)
    binary = (valid <= threshold).astype(np.uint8) * 255

    # Limpiar ruido
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, 0, 0.0

    h, w = roi_gray.shape
    best_cnt, best_score = None, 0.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        # Radio debe estar en rango pupila: 10%–70% del iris
        if r < iris_r * 0.10 or r > iris_r * 0.70:
            continue
        # Centro no debe estar en el borde
        if cx < 3 or cy < 3 or cx > w - 3 or cy > h - 3:
            continue
        peri = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * area / (peri ** 2 + 1e-6)
        score = circularity * area
        if score > best_score:
            best_score = score
            best_cnt = cnt

    if best_cnt is None:
        return 0, 0, 0, 0.0

    (cx, cy), r = cv2.minEnclosingCircle(best_cnt)
    peri = cv2.arcLength(best_cnt, True)
    circularity = 4 * np.pi * cv2.contourArea(best_cnt) / (peri ** 2 + 1e-6)

    # Confianza = circularidad × oscuridad relativa
    inner_mean = float(blurred[max(0,int(cy)-3):int(cy)+3,
                                max(0,int(cx)-3):int(cx)+3].mean())
    darkness = float(np.clip(1.0 - inner_mean / 80.0, 0.0, 1.0))
    confidence = float(np.clip(circularity * 0.6 + darkness * 0.4, 0.0, 1.0))

    return int(cx), int(cy), float(r), confidence


def draw_overlay(frame, pupil_cx, pupil_cy, pupil_r, iris_cx, iris_cy, iris_r,
                 diam_mm, ratio, state):
    """Dibuja el círculo cian sobre la pupila y las métricas en pantalla."""
    # Círculo del iris (naranja sutil)
    if iris_r > 0:
        cv2.circle(frame, (iris_cx, iris_cy), int(iris_r), (0, 140, 255), 1)

    # Círculo de la pupila (cian)
    if pupil_r > 0:
        cv2.circle(frame, (pupil_cx, pupil_cy), int(pupil_r), (255, 255, 0), 2)
        cv2.circle(frame, (pupil_cx, pupil_cy), 2, (255, 255, 0), -1)

    # Panel de métricas
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 80), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    if diam_mm > 0:
        conf_color = (0, 255, 0) if state.get("confidence", 0) >= 0.6 else (0, 180, 255)
        cv2.putText(frame, f"Diametro: {diam_mm:.1f}mm", (10, h - 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, conf_color, 2)
        cv2.putText(frame, f"PLR: {ratio:.2f}  |  {state.get('trend','')}", (10, h - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"Confianza: {state.get('confidence',0)*100:.0f}%",
                    (w - 170, h - 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        if state.get("abrupt_change"):
            cv2.putText(frame, "CAMBIO BRUSCO", (w//2 - 90, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Pupila no detectada", (10, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # Intentar autofocus; si no lo soporta, no pasa nada
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    # Reducir exposición para evitar motion blur (valor negativo = manual en V4L2)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)   # modo manual
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)       # ~1/64s; ajustar si queda oscuro

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

    iris_lm  = LEFT_IRIS  if args.eye == "left" else RIGHT_IRIS
    eye_lm   = LEFT_EYE_CONTOUR if args.eye == "left" else RIGHT_EYE_CONTOUR
    tracker  = PupilTracker(window=15)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 800, 480)
    cv2.namedWindow("DEBUG — ROI iris", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("DEBUG — ROI iris", 300, 300)

    print(f"[INFO] Ojo: {args.eye} | Resolución: {w}x{h}")
    print("[INFO] Apunta la cámara al ojo. Presiona Q para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        if map1 is not None:
            frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

        # Sharpening: resalta bordes para compensar blur de la lente
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
        frame = cv2.filter2D(frame, -1, kernel)

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        diam_mm = iris_r = pupil_r = 0.0
        iris_cx = iris_cy = pupil_cx = pupil_cy = 0
        ratio   = 0.0
        state   = {"confidence": 0.0, "trend": "", "abrupt_change": False}

        if result.multi_face_landmarks:
            lms = result.multi_face_landmarks[0].landmark

            # Iris
            iris_cx, iris_cy, iris_r = get_iris_circle(lms, iris_lm, w, h)

            # Usar el círculo del iris como ROI — la pupila siempre está dentro
            if iris_r > 0:
                margin = int(iris_r * 1.1)
                x1 = max(0, iris_cx - margin)
                y1 = max(0, iris_cy - margin)
                x2 = min(w, iris_cx + margin)
                y2 = min(h, iris_cy + margin)

                roi_color = frame[y1:y2, x1:x2]
                if roi_color.size > 0:
                    roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)

                    # Fuera del iris → 255 (blanco) para que no se confunda con la pupila oscura
                    mask = np.zeros_like(roi_gray)
                    cx_local = iris_cx - x1
                    cy_local = iris_cy - y1
                    cv2.circle(mask, (cx_local, cy_local), int(iris_r), 255, -1)
                    outside = cv2.bitwise_not(mask)
                    roi_gray = cv2.bitwise_or(roi_gray, outside)

                    # Zoom digital: escalar el ROI a mínimo 200px de ancho
                    # para tener suficientes píxeles para detectar la pupila
                    roi_h, roi_w = roi_gray.shape
                    zoom = max(1.0, 200.0 / max(roi_w, roi_h))
                    if zoom > 1.0:
                        roi_gray_z  = cv2.resize(roi_gray,  None, fx=zoom, fy=zoom,
                                                  interpolation=cv2.INTER_CUBIC)
                        roi_color_z = cv2.resize(roi_color, None, fx=zoom, fy=zoom,
                                                  interpolation=cv2.INTER_CUBIC)
                        iris_r_z = iris_r * zoom
                    else:
                        roi_gray_z  = roi_gray
                        roi_color_z = roi_color
                        iris_r_z    = iris_r

                    cv2.imshow("DEBUG — ROI iris", roi_color_z)

                    lcx, lcy, lrad_z, conf = detect_pupil_in_roi(roi_gray_z, iris_r_z)
                    lrad = lrad_z / zoom   # convertir radio de vuelta a escala original

                    # La pupila debe ser más pequeña que el iris
                    if lrad_z > 0 and lrad_z < iris_r_z * 0.85:
                        pupil_cx = x1 + int(lcx / zoom)
                        pupil_cy = y1 + int(lcy / zoom)
                        pupil_r  = lrad
                        diam_mm  = pixels_to_mm(lrad * 2, iris_r)
                        ratio    = pupil_to_iris_ratio(lrad, iris_r)
                        state    = tracker.update(diam_mm, conf)
                        state["confidence"] = conf

                        print(f"\r  Diámetro: {diam_mm:.1f}mm  |  PLR: {ratio:.2f}  "
                              f"|  {state['trend']}  |  Confianza: {conf*100:.0f}%    ",
                              end="", flush=True)

        draw_overlay(frame, pupil_cx, pupil_cy, pupil_r,
                     iris_cx, iris_cy, int(iris_r),
                     diam_mm, ratio, state)

        display = cv2.resize(frame, (800, 480))
        cv2.imshow(WIN, display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    face_mesh.close()
    cv2.destroyAllWindows()
    print("\n[OK] Finalizado.")


if __name__ == "__main__":
    main()
