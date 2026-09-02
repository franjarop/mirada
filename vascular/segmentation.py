"""
Commit 9 — Segmentación de vasos sobre una imagen de fondo de ojo pre-procesada.

Uso:
  python vascular/segmentation.py --input fundus_processed/21_training_processed.png --show

Nota: el checkpoint por defecto (models/unet_drive.pth) corre en PyTorch/CPU.
La conversión a TensorRT queda para el commit 12 — ver vascular/unet_model.py --convert.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from vascular.unet_model import UNet, infer_full_image
from vascular.overlay import side_by_side
from fundus.roi_extractor import compute_fov_mask

DEFAULT_MODEL = "models/unet_drive.pth"
DEFAULT_OUTPUT = "masks/"
WIN_SIZE = (480, 360)


def parse_args():
    p = argparse.ArgumentParser(description="Segmentación de vasos retinales con U-Net")
    p.add_argument("--input", type=str, required=True, help="Imagen de retina pre-procesada")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Checkpoint .pth (default: models/unet_drive.pth)")
    p.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Carpeta de salida para la máscara")
    p.add_argument("--threshold", type=float, default=0.5, help="Umbral de probabilidad para binarizar (default: 0.5)")
    p.add_argument("--show", action="store_true", help="Muestra original + vasos superpuestos")
    p.add_argument("--save", action="store_true", help="Guarda la máscara en --output (default: no guarda)")
    p.add_argument("--win-size", type=int, nargs=2, default=list(WIN_SIZE), metavar=("W", "H"))
    return p.parse_args()


def load_model(model_path: Path):
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model = UNet(base=ckpt["base_channels"], depth=ckpt["depth"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, np.array(ckpt["mean"]), np.array(ckpt["std"])


def segment(model, mean, std, image_bgr: np.ndarray, threshold: float):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    t0 = time.time()
    prob = infer_full_image(model, rgb, mean, std, device="cpu")
    dt_ms = (time.time() - t0) * 1000
    fov_mask, _, _ = compute_fov_mask(image_bgr)
    fov_mask = cv2.erode(fov_mask, np.ones((15, 15), np.uint8))  # descarta el borde del recorte circular (falsos positivos por contraste alto)
    mask = (prob > threshold) & (fov_mask > 0)
    return mask, dt_ms


def main():
    args = parse_args()
    input_path = Path(args.input)
    model_path = Path(args.model)

    if not input_path.exists():
        print(f"[ERROR] No existe: {input_path}")
        sys.exit(1)
    if not model_path.exists():
        print(f"[ERROR] No existe el modelo: {model_path}")
        print("[INFO] Entrenalo primero con: python vascular/unet_model.py --train")
        sys.exit(1)

    image = cv2.imread(str(input_path))
    if image is None:
        print(f"[ERROR] No se pudo leer la imagen: {input_path}")
        sys.exit(1)

    print("→ Cargando modelo PyTorch (CPU)... ", end="")
    model, mean, std = load_model(model_path)
    print("OK")

    mask, dt_ms = segment(model, mean, std, image, args.threshold)
    vessel_px = int(mask.sum())
    print(f"→ Inferencia: {dt_ms:.0f}ms | Vasos detectados: {vessel_px:,} píxeles")

    if args.save:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{input_path.stem}_mask.png"
        cv2.imwrite(str(out_path), (mask.astype(np.uint8) * 255))
        print(f"→ Máscara guardada en: {out_path}")
    else:
        print("→ No guardada (falta --save)")

    if args.show:
        combined = side_by_side(image, mask, win_size=tuple(args.win_size))
        win_name = "Mirada — vasos retinales (izq: original | der: segmentado)"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, args.win_size[0] * 2, args.win_size[1])
        cv2.imshow(win_name, combined)
        print("[INFO] Presiona cualquier tecla sobre la ventana para continuar...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
