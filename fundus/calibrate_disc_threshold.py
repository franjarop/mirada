"""
Calibra los umbrales del esquema híbrido de disco (disc_hybrid.py) sobre las 20 imágenes de
DRIVE, usando los casos ya conocidos como referencia:
  - Heurística correcta (revisada visualmente): 21, 25, 30, 31, 35, 39, 40_training
  - Heurística falla: 23_training (reflejo de borde), 34_training (iluminación pareja)
  - 24_training: agregado en la sesión del esquema híbrido — la heurística acierta pero la
    convergencia de vasos falla (enganchó un cruce de 2 vasos lejos del disco), sirve para
    verificar que el modelo SMDG desempata a favor de la heurística en ese caso.

Para cada imagen calcula las 3 distancias par a par (heurística-modelo, heurística-convergencia,
modelo-convergencia) y aplica la política de disc_hybrid.detect_optic_disc_hybrid, reportando
si el resultado final coincide con lo esperado.

Uso:
  PYTHONPATH=/home/mirada/mirada python fundus/calibrate_disc_threshold.py
"""

import cv2
from pathlib import Path

from fundus.preprocess import preprocess_image
from fundus.disc_hybrid import load_disc_model, detect_optic_disc_hybrid, dist_px
from fundus.disc_vessel_convergence import load_vessel_model
from fundus.roi_extractor import compute_fov_mask, detect_optic_disc

DRIVE_DIR = Path("fundus_images/dataset_drive/DRIVE/training/images")
KNOWN_GOOD = {"21", "25", "30", "31", "35", "39", "40"}
KNOWN_BAD = {"23", "34"}


def main():
    disc_model = load_disc_model()
    vessel_model = load_vessel_model()
    files = sorted(DRIVE_DIR.glob("*_training.tif"))

    print(f"{'#':>4} {'d_hm':>6} {'d_hc':>6} {'d_mc':>6}  {'needs_review':>13}  razón")
    for f in files:
        num = f.stem.split("_")[0]
        frame = cv2.imread(str(f))
        result = preprocess_image(frame)
        fov_mask, _, _ = compute_fov_mask(result["clahe"])
        heur = detect_optic_disc(result["clahe"], fov_mask=fov_mask)  # heurística cruda, no la ya corregida por el híbrido
        hybrid = detect_optic_disc_hybrid(frame, result["clahe"], heur,
                                           disc_model=disc_model, vessel_model=vessel_model)
        d_hm = dist_px(heur, hybrid["model_pos"])
        d_hc = dist_px(heur, hybrid["conv_pos"])
        d_mc = dist_px(hybrid["model_pos"], hybrid["conv_pos"])
        tag = "BUENA" if num in KNOWN_GOOD else ("MALA" if num in KNOWN_BAD else "")
        print(f"{num:>4} {d_hm:>6.0f} {d_hc:>6.0f} {d_mc:>6.0f}  "
              f"{str(hybrid['needs_review']):>13}  {hybrid['reason']} {tag}")


if __name__ == "__main__":
    main()
