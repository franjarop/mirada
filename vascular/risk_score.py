"""
Commit 10 — Métricas vasculares y score de riesgo.

El doc original pedía AVR (razón arteria/vena). Al principio no se calculaba (DRIVE/commit 9
no distinguen arteria de vena), pero se sumó vascular/av_classifier.py (entrenado sobre RITE,
que sí tiene esa distinción) + vascular/avr.py para calcular un AVR real. Si el modelo
(models/av_classifier.pth) no está entrenado, o no se detecta el disco óptico, cae de nuevo
al proxy sin AVR (calibre, variabilidad, densidad, tortuosidad).

Los umbrales del score están calibrados contra la distribución observada en las 20 imágenes
de DRIVE/RITE (ver notas abajo), NO contra un protocolo clínico estandarizado (Knudtson).
Es un proxy relativo para uso del proyecto, no un diagnóstico.

Uso:
  python vascular/risk_score.py --input fundus_processed/21_training_processed.png \
      --mask masks/21_training_processed_mask.png --session paciente01
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import cv2
import torch

from vascular.caliber import caliber_stats
from vascular.tortuosity import tortuosity_index
from vascular.avr import compute_avr
from vascular.av_classifier import infer_av_probs
from vascular.unet_model import UNet
from vascular.overlay import side_by_side_av
from fundus.roi_extractor import compute_fov_mask, detect_optic_disc

# Umbrales calibrados contra las 20 imágenes de DRIVE/RITE (~percentil 75 de cada métrica,
# o percentil 25 para AVR ya que valores más bajos = más riesgo). No son puntos de corte
# clínicos estandarizados — son relativos a este dataset de referencia. La literatura general
# de caliber vascular retinal (ej. estudios poblacionales tipo ARIC) asocia un AVR reducido
# con mayor riesgo cardiovascular/hipertensivo, en la misma dirección que se usa acá.
REL_VARIABILITY_HIGH = 0.68   # variabilidad de calibre (std/mean) por encima de esto: vaso irregular
TORTUOSITY_HIGH = 1.10        # tortuosidad por encima de esto: vasos más curvados de lo típico
DENSITY_LOW = 0.25            # densidad vascular por debajo de esto: posible rarefacción vascular
AVR_LOW = 0.76                 # AVR por debajo de esto (~p25 sobre las 20 imágenes de referencia): arterias angostas

DEFAULT_OUTPUT = "reports/"
AV_MODEL_PATH = "models/av_classifier.pth"


def parse_args():
    p = argparse.ArgumentParser(description="Métricas vasculares y score de riesgo (commit 10)")
    p.add_argument("--input", type=str, required=True, help="Imagen de retina pre-procesada")
    p.add_argument("--mask", type=str, required=True, help="Máscara de vasos (commit 9)")
    p.add_argument("--session", type=str, required=True, help="Identificador de sesión/paciente")
    p.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Carpeta de reportes (default: reports/)")
    p.add_argument("--av-model", type=str, default=AV_MODEL_PATH, help="Checkpoint del clasificador A/V")
    p.add_argument("--show", action="store_true", help="Muestra original + arteria (rojo) / vena (azul)")
    return p.parse_args()


def get_av_probs(image_bgr, av_model_path: Path):
    """Retorna (p_artery, p_vein) del clasificador A/V, o (None, None) si no está entrenado."""
    if not av_model_path.exists():
        return None, None
    ckpt = torch.load(av_model_path, map_location="cpu", weights_only=False)
    model = UNet(out_ch=2, base=ckpt["base_channels"], depth=ckpt["depth"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return infer_av_probs(model, rgb, ckpt["mean"], ckpt["std"], device="cpu")


def compute_metrics(image_bgr, vessel_mask, av_model_path: Path, av_probs=None) -> dict:
    fov_mask, _, _ = compute_fov_mask(image_bgr)
    cs = caliber_stats(vessel_mask, fov_mask=fov_mask)
    ti = tortuosity_index(vessel_mask)
    rel_variability = cs["std_caliber_px"] / cs["mean_caliber_px"] if cs["mean_caliber_px"] else 0.0

    metrics = {
        "mean_caliber_px": round(cs["mean_caliber_px"], 2),
        "caliber_relative_variability": round(rel_variability, 3),
        "vascular_density": round(cs["vascular_density"], 3),
        "tortuosity_index": round(ti["tortuosity_index"], 3),
        "tortuosity_segments_used": ti["segments_used"],
        "avr": None,
        "avr_unavailable_reason": None,
    }

    disc_pos = detect_optic_disc(image_bgr, fov_mask=fov_mask)
    if av_probs is None:
        av_probs = get_av_probs(image_bgr, av_model_path)
    p_artery, p_vein = av_probs

    if disc_pos is None:
        metrics["avr_unavailable_reason"] = "no se detectó el disco óptico"
    elif p_artery is None:
        metrics["avr_unavailable_reason"] = f"falta entrenar el modelo ({av_model_path})"
    else:
        avr_result = compute_avr(image_bgr, vessel_mask, p_artery, p_vein, disc_pos, fov_mask=fov_mask)
        metrics["avr"] = avr_result["avr"]
        metrics["avr_detail"] = avr_result

    return metrics


def compute_risk(metrics: dict) -> tuple:
    """Retorna (nivel, lista de factores presentes)."""
    factors = []
    if metrics["caliber_relative_variability"] > REL_VARIABILITY_HIGH:
        factors.append("calibre irregular")
    if metrics["tortuosity_index"] > TORTUOSITY_HIGH:
        factors.append("tortuosidad elevada")
    if metrics["vascular_density"] < DENSITY_LOW:
        factors.append("posible rarefacción vascular")
    if metrics["avr"] is not None and metrics["avr"] < AVR_LOW:
        factors.append("AVR reducido")

    if len(factors) == 0:
        level = "BAJO"
    elif len(factors) == 1:
        level = "MODERADO"
    else:
        level = "ALTO"
    return level, factors


def main():
    args = parse_args()
    input_path, mask_path = Path(args.input), Path(args.mask)

    if not input_path.exists():
        print(f"[ERROR] No existe: {input_path}")
        sys.exit(1)
    if not mask_path.exists():
        print(f"[ERROR] No existe la máscara: {mask_path}")
        print("[INFO] Generala primero con: python vascular/segmentation.py --input ... --save")
        sys.exit(1)

    image = cv2.imread(str(input_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        print("[ERROR] No se pudo leer la imagen o la máscara")
        sys.exit(1)
    vessel_mask = mask > 127

    av_model_path = Path(args.av_model)
    av_probs = get_av_probs(image, av_model_path)
    metrics = compute_metrics(image, vessel_mask, av_model_path, av_probs=av_probs)
    risk_level, factors = compute_risk(metrics)

    print("─" * 45)
    print(f"REPORTE VASCULAR — {args.session}")
    print("─" * 45)
    if metrics["avr"] is not None:
        print(f"AVR (razón arteria-vena):   {metrics['avr']}")
    else:
        print(f"AVR (razón arteria-vena): no disponible ({metrics['avr_unavailable_reason']})")
    print(f"Índice de tortuosidad:      {metrics['tortuosity_index']}")
    print(f"Calibre promedio de vasos:  {metrics['mean_caliber_px']} px")
    print(f"Variabilidad de calibre:    {metrics['caliber_relative_variability']}")
    print(f"Densidad vascular:          {metrics['vascular_density']}")
    print("─" * 45)
    print(f"Score de riesgo (proxy, no clínico): {risk_level}")
    if factors:
        print(f"Factores detectados: {', '.join(factors)}")
    print("─" * 45)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{args.session}_{date.today().isoformat()}.json"
    report = {
        "session": args.session,
        "date": date.today().isoformat(),
        "input": str(input_path),
        "mask": str(mask_path),
        "metrics": metrics,
        "risk_level": risk_level,
        "risk_factors": factors,
        "disclaimer": "Score proxy calibrado contra el dataset DRIVE/RITE, no validado clínicamente. "
                       "El AVR (si está disponible) usa una zona de medición simplificada, no el "
                       "protocolo clínico estandarizado (Knudtson revised formula, 6 vasos principales). "
                       "El clasificador arteria/vena tiene ~72% de exactitud (vs. 50% al azar) — en "
                       "algunas imágenes el AVR puede salir fisiológicamente implausible (>1.0).",
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Reporte guardado en: {report_path}")

    if args.show:
        p_artery, p_vein = av_probs
        if p_artery is None:
            print("[WARN] --show pedido pero no hay clasificador A/V entrenado, no hay nada que colorear")
        else:
            combined = side_by_side_av(image, vessel_mask, p_artery, p_vein, win_size=(480, 360))
            win_name = "Mirada — arteria (rojo) / vena (azul)"
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win_name, 960, 360)
            cv2.imshow(win_name, combined)
            print("[INFO] Presiona cualquier tecla sobre la ventana para continuar...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
