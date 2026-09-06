"""
Extensión del experimento disco/mácula (ver memoria del proyecto) — reentrena SOLO la
localización de disco óptico, esta vez con ground truth real e independiente (máscaras de
disco dibujadas a mano) en vez de las etiquetas de la heurística.

El modelo entrenado sobre IDRiD (fundus/disc_macula_model.py) no transfería bien a nuestras
imágenes (120px de error en disco) por brecha de dominio: IDRiD son fotos clínicas completas
sin el recorte circular ni el CLAHE de nuestro pipeline. SMDG (dataset estandarizado que junta
G1020/ORIGA/PAPILA/REFUGE/DRISHTI-GS/CRFO) sí trae fotos ya recortadas en círculo a 512x512,
mucho más parecidas a fundus_processed/, y 3103 imágenes con máscara real de disco (vs. 20
heurísticas de DRIVE). No trae mácula/fóvea — eso sigue con la heurística de roi_extractor.py.

Entrenamiento (una sola vez, tarda más que IDRiD por tener ~7x más imágenes):
  python fundus/disc_model_smdg.py --train --output models/disc_smdg.pth

Primer intento (sin domain matching): el modelo colapsaba a predecir casi siempre la misma
posición (x entre 332-394px, y entre 291-298px de 565px de ancho, sobre 20 imágenes de
fundus_processed/) — señal de que no aprendía a leer la imagen, solo el prior de posición de
SMDG. Ahora cada imagen de entrenamiento pasa por el mismo balance de blancos + eliminación de
reflejos + CLAHE que usa fundus/preprocess.py antes de llegar a detect_optic_disc, para que el
modelo vea el mismo dominio de color/contraste que verá en producción.

No se integra a preprocess.py automáticamente — queda como paso siguiente una vez validado
contra las 20 imágenes de DRIVE donde la heurística ya es 20/20 confiable.
"""

import argparse
import csv
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from fundus.color_balance import gray_world_balance
from fundus.preprocess import remove_reflections, apply_clahe_contrast

DEFAULT_DATA_DIR = "fundus_images/dataset_smdg"
DEFAULT_OUTPUT = "models/disc_smdg.pth"
IMG_SIZE = 256
ORIG_SIZE = 512  # todas las imágenes SMDG están estandarizadas a 512x512


class DiscNet(nn.Module):
    """Misma arquitectura que DiscMaculaNet (fundus/disc_macula_model.py) pero 2 salidas (solo disco)."""

    def __init__(self):
        super().__init__()
        chs = [3, 16, 32, 64, 128, 256]
        blocks = []
        for i in range(5):
            blocks += [
                nn.Conv2d(chs[i], chs[i + 1], 3, padding=1), nn.BatchNorm2d(chs[i + 1]),
                nn.ReLU(inplace=True), nn.MaxPool2d(2),
            ]
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 2), nn.Sigmoid(),  # disc_x, disc_y normalizados
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


def _fix_path(data_dir: Path, rel_path: str) -> Path:
    """El CSV apunta a '/full-fundus/NAME.png' pero el zip real anida la carpeta dos veces."""
    parts = rel_path.strip("/").split("/")
    return data_dir / parts[0] / parts[0] / parts[1]


def list_smdg_samples(data_dir: Path):
    """Retorna lista de (image_path, mask_path) para filas con fundus + máscara de disco."""
    csv_path = data_dir / "metadata - standardized.csv"
    samples = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["fundus"] and row["fundus_od_seg"]:
                img_path = _fix_path(data_dir, row["fundus"])
                mask_path = _fix_path(data_dir, row["fundus_od_seg"])
                if img_path.exists() and mask_path.exists():
                    samples.append((img_path, mask_path))
    return samples


def mask_centroid(mask_path: Path):
    """Centroide (x, y) en píxeles de la máscara de disco, o None si está vacía."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    ys, xs = np.where(mask > 127)
    if len(xs) < 20:
        return None
    return float(xs.mean()), float(ys.mean())


def apply_domain_transform(image_bgr: np.ndarray) -> np.ndarray:
    """Mismo balance de blancos + eliminación de reflejos + CLAHE que fundus/preprocess.py
    aplica antes de detectar disco/mácula (ver preprocess_image -> clahe_img)."""
    balanced = gray_world_balance(image_bgr)
    clean = remove_reflections(balanced)
    return apply_clahe_contrast(clean)


def split_train_val(samples, n_val=300, seed=0):
    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)
    return shuffled[n_val:], shuffled[:n_val]


class DiscDataset(torch.utils.data.Dataset):
    """
    Precarga y redimensiona todas las imágenes una sola vez en memoria, como en
    disc_macula_model.py. Guarda como uint8 (no float32) para no repetir el problema de
    memoria de la descarga del dataset: 3103 imágenes de 256x256x3 en uint8 ~ 610MB
    (en float32 serían ~2.4GB, riesgoso en una Jetson de 7.4GB de RAM).
    """

    def __init__(self, samples, img_size=IMG_SIZE, augment=True):
        self.img_size = img_size
        self.augment = augment
        self.images, self.coords = [], []
        for img_path, mask_path in samples:
            centroid = mask_centroid(mask_path)
            if centroid is None:
                continue
            cx, cy = centroid
            img_bgr = cv2.imread(str(img_path))
            h, w = img_bgr.shape[:2]
            img_bgr = apply_domain_transform(img_bgr)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_rgb = cv2.resize(img_rgb, (img_size, img_size))
            self.images.append(img_rgb.astype(np.uint8))
            self.coords.append(np.array([cx / w, cy / h], dtype=np.float32))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        arr = self.images[idx].astype(np.float32) / 255.0
        coords = self.coords[idx].copy()

        if self.augment:
            if random.random() < 0.5:  # flip horizontal: espeja x
                arr = arr[:, ::-1].copy()
                coords[0] = 1 - coords[0]
            if random.random() < 0.5:  # flip vertical: espeja y
                arr = arr[::-1, :].copy()
                coords[1] = 1 - coords[1]

        img_t = torch.from_numpy(arr.transpose(2, 0, 1)).float()
        return img_t, torch.from_numpy(coords)


@torch.no_grad()
def predict(model, image_bgr: np.ndarray, img_size=IMG_SIZE, device="cpu"):
    """Retorna (disc_x, disc_y) en píxeles de la imagen original."""
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (img_size, img_size)).astype(np.float32) / 255.0
    x = torch.from_numpy(resized.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    model.eval()
    out = model(x)[0].cpu().numpy()
    return int(out[0] * w), int(out[1] * h)


@torch.no_grad()
def evaluate(model, val_dataset: "DiscDataset", device="cpu", orig_size=ORIG_SIZE):
    model.eval()
    errs = []
    for i in range(len(val_dataset)):
        img_t, true_coords = val_dataset[i]
        pred = model(img_t.unsqueeze(0).to(device))[0].cpu().numpy()
        true_coords = true_coords.numpy()
        errs.append(np.hypot((pred[0] - true_coords[0]) * orig_size, (pred[1] - true_coords[1]) * orig_size))
    return float(np.mean(errs))


def train(args):
    data_dir = Path(args.data)
    samples = list_smdg_samples(data_dir)
    if not samples:
        print(f"[ERROR] No se encontraron imágenes/máscaras en {data_dir}")
        return
    train_s, val_s = split_train_val(samples, n_val=args.val_images)
    print(f"[INFO] SMDG: {len(train_s)} imágenes de train, {len(val_s)} de validación (de {len(samples)} totales)")

    print("[INFO] Cargando imágenes de train en memoria...")
    dataset = DiscDataset(train_s, img_size=args.img_size, augment=True)
    print("[INFO] Cargando imágenes de validación en memoria...")
    val_dataset = DiscDataset(val_s, img_size=args.img_size, augment=False)
    print(f"[INFO] Cargadas {len(dataset)} train / {len(val_dataset)} val (tras descartar máscaras vacías)")

    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    device = "cpu"
    model = DiscNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_err = float("inf")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        for x, y in tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)

        err = evaluate(model, val_dataset, device)
        dt = time.time() - t0
        print(f"-> Epoch {epoch}/{args.epochs} - loss: {running_loss / len(dataset):.5f} | "
              f"error disco: {err:.1f}px | {dt:.1f}s")

        if err < best_err:
            best_err = err
            torch.save({
                "state_dict": model.state_dict(),
                "img_size": args.img_size,
                "disc_error_px": err,
                "orig_size": ORIG_SIZE,
            }, output_path)
            print(f"  -> mejor modelo hasta ahora (error={best_err:.1f}px), guardado en {output_path}")

    print(f"\n[OK] Entrenamiento terminado. Mejor error de validacion: {best_err:.1f}px (sobre imagenes 512x512)")
    print(f"[OK] Checkpoint final en: {output_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Localización de disco óptico entrenada (SMDG)")
    p.add_argument("--train", action="store_true")
    p.add_argument("--data", type=str, default=DEFAULT_DATA_DIR)
    p.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--img-size", type=int, default=IMG_SIZE)
    p.add_argument("--val-images", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    return p.parse_args()


def main():
    args = parse_args()
    if args.train:
        train(args)
    else:
        print("[ERROR] Especificá --train")


if __name__ == "__main__":
    main()
