"""
Commit 8 (extensión) — Localización de disco óptico y mácula con un modelo entrenado.

La heurística de fundus/roi_extractor.py (blob más brillante/más oscuro) sigue funcionando,
pero tiene techo — falla en imágenes con brillo asimétrico (ver notas del commit 8/9 en
memoria del proyecto). Este módulo entrena un regresor liviano sobre IDRiD (516 imágenes,
subset "C. Localization", con centro de disco y mácula anotados a mano) para predecir
directamente las coordenadas (x, y) de ambos, sin depender de brillo/contraste.

Entrenamiento (una sola vez, ~15-20 min en CPU):
  python fundus/disc_macula_model.py --train --output models/disc_macula.pth

fundus/preprocess.py usa este modelo si el checkpoint existe; si no, cae a la heurística.
"""

import argparse
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

DEFAULT_DATA_DIR = "fundus_images/dataset_idrid/C.%20Localization/C. Localization"
DEFAULT_OUTPUT = "models/disc_macula.pth"
IMG_SIZE = 256


class DiscMaculaNet(nn.Module):
    """CNN chica: 5 bloques conv+pool -> global average pool -> FC -> 4 coords normalizadas [0,1]."""

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
            nn.Linear(128, 4), nn.Sigmoid(),  # disc_x, disc_y, fovea_x, fovea_y (normalizados)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


def _load_coords(data_dir: Path, split: str) -> pd.DataFrame:
    subset = "a" if split == "training" else "b"
    label = "Training" if split == "training" else "Testing"
    disc_csv = data_dir / "2. Groundtruths" / "1. Optic Disc Center Location" / f"{subset}. IDRiD_OD_Center_{label} Set_Markups.csv"
    fovea_csv = data_dir / "2. Groundtruths" / "2. Fovea Center Location" / f"IDRiD_Fovea_Center_{label} Set_Markups.csv"

    disc = pd.read_csv(disc_csv).dropna(subset=["Image No"])[["Image No", "X- Coordinate", "Y - Coordinate"]]
    disc.columns = ["image", "disc_x", "disc_y"]
    fovea = pd.read_csv(fovea_csv).dropna(subset=["Image No"])[["Image No", "X- Coordinate", "Y - Coordinate"]]
    fovea.columns = ["image", "fovea_x", "fovea_y"]
    return disc.merge(fovea, on="image")


def list_idrid_samples(data_dir: Path, split: str):
    """Retorna lista de (image_path, disc_x, disc_y, fovea_x, fovea_y) en píxeles originales."""
    img_dir = data_dir / "1. Original Images" / ("a. Training Set" if split == "training" else "b. Testing Set")
    df = _load_coords(data_dir, split)
    samples = []
    for _, row in df.iterrows():
        img_path = img_dir / f"{row['image']}.jpg"
        if img_path.exists():
            samples.append((img_path, float(row.disc_x), float(row.disc_y),
                             float(row.fovea_x), float(row.fovea_y)))
    return samples


def split_train_val(samples, n_val=40, seed=0):
    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)
    return shuffled[n_val:], shuffled[:n_val]


class DiscMaculaDataset(torch.utils.data.Dataset):
    """
    Precarga y redimensiona todas las imágenes una sola vez en memoria (413 imágenes de
    256x256 ~ 80MB, entra de sobra) — las IDRiD originales son JPEGs de 4288x2848,
    decodificarlas de disco en cada __getitem__() hacía cada epoch insoportablemente lento.
    """

    def __init__(self, samples, img_size=IMG_SIZE, augment=True):
        self.img_size = img_size
        self.augment = augment
        self.images, self.coords = [], []
        for path, disc_x, disc_y, fovea_x, fovea_y in samples:
            img = Image.open(path).convert("RGB")
            w, h = img.size
            img = img.resize((img_size, img_size))
            self.images.append(np.array(img).astype(np.float32) / 255.0)
            self.coords.append(np.array([disc_x / w, disc_y / h, fovea_x / w, fovea_y / h], dtype=np.float32))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        arr = self.images[idx].copy()
        coords = self.coords[idx].copy()

        if self.augment:
            if random.random() < 0.5:  # flip horizontal: espeja x
                arr = arr[:, ::-1].copy()
                coords[0], coords[2] = 1 - coords[0], 1 - coords[2]
            if random.random() < 0.5:  # flip vertical: espeja y
                arr = arr[::-1, :].copy()
                coords[1], coords[3] = 1 - coords[1], 1 - coords[3]

        img_t = torch.from_numpy(arr.transpose(2, 0, 1)).float()
        return img_t, torch.from_numpy(coords)


@torch.no_grad()
def predict(model, image_bgr: np.ndarray, img_size=IMG_SIZE, device="cpu"):
    """Retorna ((disc_x, disc_y), (fovea_x, fovea_y)) en píxeles de la imagen original."""
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (img_size, img_size)).astype(np.float32) / 255.0
    x = torch.from_numpy(resized.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    model.eval()
    out = model(x)[0].cpu().numpy()
    disc = (int(out[0] * w), int(out[1] * h))
    fovea = (int(out[2] * w), int(out[3] * h))
    return disc, fovea


@torch.no_grad()
def evaluate(model, val_dataset: "DiscMaculaDataset", device="cpu"):
    """
    Error euclidiano promedio para disco y mácula, en píxeles reales (usa las dimensiones
    originales de IDRiD, 4288x2848 para todas las imágenes del dataset). Opera sobre las
    imágenes ya cacheadas en memoria por val_dataset — no vuelve a leer/decodificar de disco.
    """
    model.eval()
    orig_w, orig_h = 4288, 2848  # todas las imágenes de IDRiD tienen esta resolución
    disc_errs, fovea_errs = [], []
    for i in range(len(val_dataset)):
        img_t, true_coords = val_dataset[i]
        pred = model(img_t.unsqueeze(0).to(device))[0].cpu().numpy()
        true_coords = true_coords.numpy()

        disc_errs.append(np.hypot((pred[0] - true_coords[0]) * orig_w, (pred[1] - true_coords[1]) * orig_h))
        fovea_errs.append(np.hypot((pred[2] - true_coords[2]) * orig_w, (pred[3] - true_coords[3]) * orig_h))
    return float(np.mean(disc_errs)), float(np.mean(fovea_errs))


def train(args):
    data_dir = Path(args.data)
    samples = list_idrid_samples(data_dir, "training")
    if not samples:
        print(f"[ERROR] No se encontraron imágenes/coordenadas en {data_dir}")
        return
    train_s, val_s = split_train_val(samples, n_val=args.val_images)
    print(f"[INFO] IDRiD: {len(train_s)} imágenes de train, {len(val_s)} de validación")

    dataset = DiscMaculaDataset(train_s, img_size=args.img_size, augment=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dataset = DiscMaculaDataset(val_s, img_size=args.img_size, augment=False)

    device = "cpu"
    model = DiscMaculaNet().to(device)
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

        disc_err, fovea_err = evaluate(model, val_dataset, device)
        dt = time.time() - t0
        avg_err = (disc_err + fovea_err) / 2
        print(f"→ Epoch {epoch}/{args.epochs} — loss: {running_loss / len(dataset):.5f} | "
              f"error disco: {disc_err:.1f}px | error mácula: {fovea_err:.1f}px | {dt:.1f}s")

        if avg_err < best_err:
            best_err = avg_err
            torch.save({
                "state_dict": model.state_dict(),
                "img_size": args.img_size,
                "disc_error_px": disc_err,
                "fovea_error_px": fovea_err,
            }, output_path)
            print(f"  ↳ mejor modelo hasta ahora (error prom={best_err:.1f}px), guardado en {output_path}")

    print(f"\n[OK] Entrenamiento terminado. Mejor error promedio: {best_err:.1f}px")
    print(f"[OK] Checkpoint final en: {output_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Localización de disco/mácula entrenada (IDRiD)")
    p.add_argument("--train", action="store_true")
    p.add_argument("--data", type=str, default=DEFAULT_DATA_DIR)
    p.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--img-size", type=int, default=IMG_SIZE)
    p.add_argument("--val-images", type=int, default=40)
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
