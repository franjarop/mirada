"""
Commit 10 (extensión) — Clasificador arteria/vena, para AVR real.

Entrena sobre RITE (fundus_images/dataset_rite era el genérico sin A/V real; el que sirve es
fundus_images/dataset_drive_av/AV_groundTruth — mismas 20 imágenes de DRIVE/training, con
máscara de arteria=rojo, vena=azul, superposición=verde, incierto=blanco).

No segmenta vasos de nuevo (para eso está vascular/unet_model.py, commit 9) — clasifica,
pixel a pixel, si un vaso ya detectado es arteria o vena. Se combinan en vascular/avr.py.

Entrenamiento (una sola vez):
  python vascular/av_classifier.py --train --output models/av_classifier.pth
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from vascular.unet_model import UNet, pad_to_multiple, DEPTH, BASE_CHANNELS

DEFAULT_DATA_DIR = "fundus_images/dataset_drive_av/AV_groundTruth/training"
DEFAULT_OUTPUT = "models/av_classifier.pth"
PATCH_SIZE = 64

ARTERY, VEIN, IGNORE = 0, 1, -1


def _load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def av_label_map(av_rgb: np.ndarray) -> np.ndarray:
    """
    Convierte la máscara de color (rojo=arteria, azul=vena, verde=superposición,
    blanco=incierto, negro=fondo) a un mapa de clases: 0=arteria, 1=vena, -1=ignorar
    (fondo/superposición/incierto — no hay señal confiable para esos píxeles).
    """
    r, g, b = av_rgb[..., 0].astype(int), av_rgb[..., 1].astype(int), av_rgb[..., 2].astype(int)
    labels = np.full(av_rgb.shape[:2], IGNORE, dtype=np.int64)
    labels[(r > 150) & (g < 100) & (b < 100)] = ARTERY
    labels[(b > 150) & (r < 100) & (g < 100)] = VEIN
    return labels


def list_rite_triplets(data_dir: Path):
    img_dir, av_dir = data_dir / "images", data_dir / "av"
    triplets = []
    for av_path in sorted(av_dir.glob("*.png")):
        img_path = img_dir / f"{av_path.stem}.tif"
        if img_path.exists():
            triplets.append((img_path, av_path))
    return triplets


def split_train_val(triplets, n_val=4, seed=0):
    rng = random.Random(seed)
    shuffled = triplets[:]
    rng.shuffle(shuffled)
    return shuffled[n_val:], shuffled[:n_val]


def compute_norm_stats(triplets):
    pixels = []
    for img_path, _ in triplets:
        img = _load_rgb(img_path).astype(np.float32) / 255.0
        pixels.append(img.reshape(-1, 3))
    pixels = np.concatenate(pixels, axis=0)
    return pixels.mean(axis=0), pixels.std(axis=0) + 1e-6


class RiteAVPatchDataset(torch.utils.data.Dataset):
    def __init__(self, triplets, patch_size=PATCH_SIZE, patches_per_epoch=1500, mean=None, std=None):
        self.patch_size = patch_size
        self.patches_per_epoch = patches_per_epoch
        self.mean, self.std = mean, std

        self.images, self.labels, self.labeled_coords = [], [], []
        for img_path, av_path in triplets:
            img = _load_rgb(img_path).astype(np.float32) / 255.0
            labels = av_label_map(_load_rgb(av_path))
            self.images.append(img)
            self.labels.append(labels)
            ys, xs = np.where(labels != IGNORE)
            self.labeled_coords.append((ys, xs))

    def __len__(self):
        return self.patches_per_epoch

    def __getitem__(self, _):
        idx = random.randrange(len(self.images))
        img, labels = self.images[idx], self.labels[idx]
        h, w = img.shape[:2]
        ps = self.patch_size
        ys, xs = self.labeled_coords[idx]

        i = random.randrange(len(ys))
        cy, cx = int(ys[i]), int(xs[i])
        cy = min(max(cy, ps // 2), h - ps // 2 - 1)
        cx = min(max(cx, ps // 2), w - ps // 2 - 1)
        y0, x0 = cy - ps // 2, cx - ps // 2

        patch_img = img[y0:y0 + ps, x0:x0 + ps].copy()
        patch_lbl = labels[y0:y0 + ps, x0:x0 + ps].copy()

        if random.random() < 0.5:
            patch_img, patch_lbl = patch_img[:, ::-1].copy(), patch_lbl[:, ::-1].copy()
        if random.random() < 0.5:
            patch_img, patch_lbl = patch_img[::-1, :].copy(), patch_lbl[::-1, :].copy()
        k = random.randrange(4)
        if k:
            patch_img = np.rot90(patch_img, k).copy()
            patch_lbl = np.rot90(patch_lbl, k).copy()

        patch_img = (patch_img - self.mean) / self.std
        img_t = torch.from_numpy(patch_img.transpose(2, 0, 1)).float()
        lbl_t = torch.from_numpy(patch_lbl).long()
        return img_t, lbl_t


@torch.no_grad()
def infer_av_probs(model, rgb_uint8: np.ndarray, mean, std, device="cpu", multiple=2 ** DEPTH):
    """Retorna (prob_arteria, prob_vena), cada una HxW en [0,1]."""
    img = rgb_uint8.astype(np.float32) / 255.0
    padded, orig_h, orig_w = pad_to_multiple(img, multiple)
    norm = (padded - mean) / std
    x = torch.from_numpy(norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    model.eval()
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return probs[ARTERY, :orig_h, :orig_w], probs[VEIN, :orig_h, :orig_w]


def evaluate(model, val_triplets, mean, std, device="cpu"):
    correct = total = 0
    for img_path, av_path in val_triplets:
        img = _load_rgb(img_path)
        labels = av_label_map(_load_rgb(av_path))
        p_artery, p_vein = infer_av_probs(model, img, mean, std, device)
        pred = np.where(p_artery >= p_vein, ARTERY, VEIN)

        valid = labels != IGNORE
        correct += int(np.sum((pred == labels) & valid))
        total += int(np.sum(valid))
    return correct / max(total, 1)


def train(args):
    data_dir = Path(args.data)
    triplets = list_rite_triplets(data_dir)
    if not triplets:
        print(f"[ERROR] No se encontraron pares imagen/av en {data_dir}")
        return
    train_t, val_t = split_train_val(triplets, n_val=args.val_images)
    print(f"[INFO] RITE: {len(train_t)} imágenes de train, {len(val_t)} de validación")

    mean, std = compute_norm_stats(train_t)
    dataset = RiteAVPatchDataset(train_t, patch_size=args.patch_size,
                                  patches_per_epoch=args.patches_per_epoch, mean=mean, std=std)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    device = "cpu"
    model = UNet(out_ch=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = -1.0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        for x, y in tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y, ignore_index=IGNORE)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)

        acc = evaluate(model, val_t, mean, std, device)
        dt = time.time() - t0
        print(f"→ Epoch {epoch}/{args.epochs} — loss: {running_loss / len(dataset):.4f} | "
              f"val accuracy (A/V): {acc:.3f} | {dt:.1f}s")

        if acc > best_acc:
            best_acc = acc
            torch.save({
                "state_dict": model.state_dict(),
                "mean": mean.tolist(), "std": std.tolist(),
                "base_channels": BASE_CHANNELS, "depth": DEPTH,
                "val_accuracy": acc,
            }, output_path)
            print(f"  ↳ mejor modelo hasta ahora (acc={best_acc:.3f}), guardado en {output_path}")

    print(f"\n[OK] Entrenamiento terminado. Mejor accuracy A/V de validación: {best_acc:.3f}")
    print(f"[OK] Checkpoint final en: {output_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Clasificador arteria/vena (para AVR real)")
    p.add_argument("--train", action="store_true")
    p.add_argument("--data", type=str, default=DEFAULT_DATA_DIR)
    p.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    p.add_argument("--patches-per-epoch", type=int, default=1500)
    p.add_argument("--val-images", type=int, default=4)
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
