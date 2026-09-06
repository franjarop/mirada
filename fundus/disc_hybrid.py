"""
Esquema híbrido de disco óptico — combina 3 señales independientes:
  1. heurística de brillo (roi_extractor.detect_optic_disc)
  2. modelo entrenado sobre SMDG (disc_model_smdg.py, dominio CLAHE)
  3. punto de convergencia de vasos (disc_vessel_convergence.py, dominio original —
     el disco es donde convergen los troncos principales, señal que no depende del brillo)

Motivo de usar 3 señales en vez de 2: con heurística+modelo solamente, un desacuerdo no dice
cuál de los dos está mal. La convergencia de vasos es una tercera señal independiente (no le
afectan los mismos reflejos/sobreexposición que a las otras dos) que permite desempatar.

Política (calibrada en fundus/calibrate_disc_threshold.py sobre las 20 imágenes de DRIVE):
  - Si la heurística coincide con el modelo O con la convergencia (dentro del umbral de ese
    par), se confía en la heurística — es la más precisa cuando no está fallando (caso
    31_training, ver memoria del proyecto).
  - Si la heurística no coincide con NINGUNA de las otras dos, pero modelo y convergencia sí
    coinciden entre sí, se usa el punto medio de esos dos como corrección — la heurística es
    la que está fallando (ej. 23_training: enganchó un reflejo de borde).
  - Si ningún par coincide, no hay consenso: se mantiene la heurística como mejor estimación
    disponible sin corregir.

En cualquier caso donde la heurística no coincidió con al menos otra señal, needs_review queda
en True — aunque se haya podido autocorregir, señala que al menos un método falló ahí y vale la
pena que alguien lo mire.
"""

import numpy as np
import torch

from fundus.disc_model_smdg import DiscNet, predict as model_predict
from fundus.disc_vessel_convergence import detect_disc_by_convergence

DEFAULT_MODEL_PATH = "models/disc_smdg_clahe.pth"

# Calibrados en fundus/calibrate_disc_threshold.py: distancia máxima observada entre heurística
# y cada señal sobre las 18 imágenes "normales" de DRIVE (todas menos 23/34_training), con
# margen. MC_THRESHOLD_PX es más laxo porque ese par no involucra a la heurística — es el
# par de respaldo para cuando la heurística ya se descartó.
HM_THRESHOLD_PX = 50   # heurística vs modelo SMDG
HC_THRESHOLD_PX = 65   # heurística vs convergencia de vasos
MC_THRESHOLD_PX = 95   # modelo SMDG vs convergencia de vasos

_disc_model_cache = {}


def load_disc_model(path: str = DEFAULT_MODEL_PATH):
    if path not in _disc_model_cache:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = DiscNet()
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        _disc_model_cache[path] = model
    return _disc_model_cache[path]


def dist_px(a, b):
    if a is None or b is None:
        return None
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def detect_optic_disc_hybrid(frame: np.ndarray, clahe_img: np.ndarray, heuristic_pos,
                              disc_model=None, vessel_model=None) -> dict:
    """
    frame: imagen ORIGINAL (para la señal de convergencia de vasos — ver nota de dominio en
    vascular/segmentation.py).
    clahe_img: imagen tras balance de blancos + reflejos + CLAHE (para el modelo SMDG).
    heuristic_pos: resultado de roi_extractor.detect_optic_disc sobre clahe_img.
    """
    if heuristic_pos is None:
        return {"pos": None, "model_pos": None, "conv_pos": None,
                "needs_review": True, "reason": "heurística no detectó nada"}

    if disc_model is None:
        disc_model = load_disc_model()
    model_pos = model_predict(disc_model, clahe_img)
    conv_pos = detect_disc_by_convergence(frame, model=vessel_model)

    d_hm = dist_px(heuristic_pos, model_pos)
    d_hc = dist_px(heuristic_pos, conv_pos)
    d_mc = dist_px(model_pos, conv_pos)

    if (d_hm is not None and d_hm <= HM_THRESHOLD_PX) or (d_hc is not None and d_hc <= HC_THRESHOLD_PX):
        return {"pos": heuristic_pos, "model_pos": model_pos, "conv_pos": conv_pos,
                "needs_review": False, "reason": "heurística coincide con al menos otra señal"}

    if d_mc is not None and d_mc <= MC_THRESHOLD_PX:
        # Se usa conv_pos solo (no el punto medio con el modelo): en los 2 casos de calibración
        # (23/34_training) la convergencia sola cae más precisa sobre el disco que el promedio —
        # el modelo SMDG es la señal más floja de las 3, promediarlo la corre del lugar correcto.
        # d_mc <= umbral solo sirve acá para confirmar que la convergencia no se fue a otro lado
        # (protección contra el fallo que se vio en 24_training).
        return {"pos": conv_pos, "model_pos": model_pos, "conv_pos": conv_pos,
                "needs_review": True, "reason": "heurística no coincide con nada; corregido con convergencia de vasos"}

    return {"pos": heuristic_pos, "model_pos": model_pos, "conv_pos": conv_pos,
            "needs_review": True, "reason": "sin consenso entre las 3 señales"}
