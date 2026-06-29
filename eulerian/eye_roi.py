"""
Commit 4 — Detección de ROI ocular con MediaPipe FaceMesh.
Detecta 468 puntos del rostro y extrae la zona de la esclera (parte blanca del ojo).
"""

import cv2
import numpy as np
import mediapipe as mp

# Landmarks del contorno del ojo izquierdo y derecho (MediaPipe FaceMesh)
LEFT_EYE  = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33,    7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# Iris landmarks (para excluirlo del ROI)
LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]


class EyeROI:
    def __init__(self, eye: str = "left", padding: float = 0.15):
        """
        eye: "left" o "right"
        padding: fracción de margen alrededor del ojo (default: 15%)
        """
        self.landmarks = LEFT_EYE  if eye == "left"  else RIGHT_EYE
        self.iris_lm   = LEFT_IRIS if eye == "left"  else RIGHT_IRIS
        self.padding   = padding

        self.mp_face   = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,      # necesario para iris landmarks
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def extract(self, frame: np.ndarray) -> tuple:
        """
        Procesa el frame y retorna (roi, bbox, landmarks_img) o (None, None, None).

        roi          — imagen recortada de la región ocular (BGR)
        bbox         — (x1, y1, x2, y2) en píxeles del frame original
        frame_annot  — frame original con anotaciones dibujadas
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)

        frame_annot = frame.copy()

        if not result.multi_face_landmarks:
            cv2.putText(frame_annot, "Rostro no detectado", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return None, None, frame_annot

        lms = result.multi_face_landmarks[0].landmark

        # Coordenadas del contorno del ojo
        pts = np.array([(int(lms[i].x * w), int(lms[i].y * h)) for i in self.landmarks])

        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)

        # Padding
        pad_x = int((x2 - x1) * self.padding)
        pad_y = int((y2 - y1) * self.padding)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        roi = frame[y1:y2, x1:x2].copy()

        # Dibujar contorno del ojo en verde
        cv2.polylines(frame_annot, [pts], isClosed=True, color=(0, 255, 0), thickness=1)
        cv2.rectangle(frame_annot, (x1, y1), (x2, y2), (0, 200, 0), 2)

        # Marcar iris si hay landmarks de iris disponibles
        if len(lms) > max(self.iris_lm):
            iris_pts = [(int(lms[i].x * w), int(lms[i].y * h)) for i in self.iris_lm]
            iris_center = np.mean(iris_pts, axis=0).astype(int)
            iris_r = int(np.linalg.norm(np.array(iris_pts[0]) - iris_center))
            cv2.circle(frame_annot, tuple(iris_center), iris_r, (0, 140, 255), 1)

        return roi, (x1, y1, x2, y2), frame_annot

    def close(self):
        self.face_mesh.close()
