"""
Commit 9 — Modelo U-Net para segmentación de vasos retinales.

No existe un checkpoint preentrenado público para esta arquitectura exacta,
así que en vez de "--download" (como sugiere el doc original) se entrena
desde cero sobre el dataset DRIVE ya descargado (fundus_images/dataset_drive/).

Entrenamiento (una sola vez, ~15-30 min en CPU de la Jetson):
  python vascular/unet_model.py --train --output models/unet_drive.pth

Conversión a TensorRT: requiere torch2trt/TensorRT instalados, que esta Jetson
no tiene configurados todavía (queda para el commit 12 — optimización Jetson).
  python vascular/unet_model.py --convert --input models/unet_drive.pth --output models/unet_trt.engine
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

DEFAULT_DATA_DIR = "fundus_images/dataset_drive/DRIVE/training"
DEFAULT_OUTPUT = "models/unet_drive.pth"
PATCH_SIZE = 64
BASE_CHANNELS = 16
DEPTH = 4  # 4 downsamplings -> factor 16, usado también para el padding en inferencia full-image


# ─────────────────────────────────────────────────────────────────────────
# Arquitectura
# ─────────────────────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """U-Net compacta (base_channels=16, depth=4) — pensada para entrenar en CPU con pocas imágenes."""

    def __init__(self, in_ch=3, out_ch=1, base=BASE_CHANNELS, depth=DEPTH):
        super().__init__()
        chs = [base * (2 ** i) for i in range(depth + 1)]  # ej: [16,32,64,128,256]

        self.downs = nn.ModuleList()
        prev = in_ch
        for c in chs[:-1]:
            self.downs.append(DoubleConv(prev, c))
            prev = c
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(chs[-2], chs[-1])

        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        rev = list(reversed(chs[:-1]))
        prev = chs[-1]
        for c in rev:
            self.ups.append(nn.ConvTranspose2d(prev, c, 2, stride=2))
            self.up_convs.append(DoubleConv(c * 2, c))
            prev = c

        self.out_conv = nn.Conv2d(prev, out_ch, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for up, conv, skip in zip(self.ups, self.up_convs, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = conv(x)

        return self.out_conv(x)


# ─────────────────────────────────────────────────────────────────────────
# Dataset DRIVE (patch-based, con augmentation)
# ─────────────────────────────────────────────────────────────────────────

def _load_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"))


def _load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def list_drive_triplets(data_dir: Path):
    """Empareja cada imagen con su máscara FOV y su ground truth de vasos (1st_manual)."""
    img_dir = data_dir / "images"
    fov_dir = data_dir / "mask"
    gt_dir = data_dir / "1st_manual"

    triplets = []
    for img_path in sorted(img_dir.glob("*")):
        stem = img_path.stem.replace("_training", "").replace("_test", "")
        fov_path = next(fov_dir.glob(f"{stem}_*mask*"), None)
        gt_path = next(gt_dir.glob(f"{stem}_manual*"), None)
        if fov_path is None or gt_path is None:
            continue
        triplets.append((img_path, fov_path, gt_path))
    return triplets


def split_train_val(triplets, n_val=4, seed=0):
    rng = random.Random(seed)
    shuffled = triplets[:]
    rng.shuffle(shuffled)
    return shuffled[n_val:], shuffled[:n_val]


class DrivePatchDataset(torch.utils.data.Dataset):
    """Extrae patches aleatorios (con foco en vasos) de las imágenes de entrenamiento."""

    def __init__(self, triplets, patch_size=PATCH_SIZE, patches_per_epoch=2000,
                 mean=None, std=None, vessel_focus=0.7):
        self.patch_size = patch_size
        self.patches_per_epoch = patches_per_epoch
        self.vessel_focus = vessel_focus
        self.mean = mean if mean is not None else np.array([0.5, 0.5, 0.5])
        self.std = std if std is not None else np.array([0.5, 0.5, 0.5])

        self.images, self.fovs, self.gts, self.vessel_coords = [], [], [], []
        for img_path, fov_path, gt_path in triplets:
            img = _load_rgb(img_path).astype(np.float32) / 255.0
            fov = (_load_gray(fov_path) > 127)
            gt = (_load_gray(gt_path) > 127)
            self.images.append(img)
            self.fovs.append(fov)
            self.gts.append(gt)
            ys, xs = np.where(gt & fov)
            self.vessel_coords.append((ys, xs))

    def __len__(self):
        return self.patches_per_epoch

    def _sample_center(self, idx):
        img = self.images[idx]
        h, w = img.shape[:2]
        ps = self.patch_size
        ys, xs = self.vessel_coords[idx]
        if len(ys) > 0 and random.random() < self.vessel_focus:
            i = random.randrange(len(ys))
            cy, cx = int(ys[i]), int(xs[i])
        else:
            cy, cx = random.randrange(h), random.randrange(w)
        cy = min(max(cy, ps // 2), h - ps // 2 - 1)
        cx = min(max(cx, ps // 2), w - ps // 2 - 1)
        return cy, cx

    def __getitem__(self, _):
        idx = random.randrange(len(self.images))
        img, gt = self.images[idx], self.gts[idx]
        ps = self.patch_size
        cy, cx = self._sample_center(idx)

        y0, x0 = cy - ps // 2, cx - ps // 2
        patch_img = img[y0:y0 + ps, x0:x0 + ps].copy()
        patch_gt = gt[y0:y0 + ps, x0:x0 + ps].astype(np.float32).copy()

        if random.random() < 0.5:
            patch_img, patch_gt = patch_img[:, ::-1].copy(), patch_gt[:, ::-1].copy()
        if random.random() < 0.5:
            patch_img, patch_gt = patch_img[::-1, :].copy(), patch_gt[::-1, :].copy()
        k = random.randrange(4)
        if k:
            patch_img = np.rot90(patch_img, k).copy()
            patch_gt = np.rot90(patch_gt, k).copy()

        patch_img = (patch_img - self.mean) / self.std
        img_t = torch.from_numpy(patch_img.transpose(2, 0, 1)).float()
        gt_t = torch.from_numpy(patch_gt).float().unsqueeze(0)
        return img_t, gt_t


def compute_norm_stats(triplets):
    pixels = []
    for img_path, fov_path, _ in triplets:
        img = _load_rgb(img_path).astype(np.float32) / 255.0
        fov = _load_gray(fov_path) > 127
        pixels.append(img[fov])
    pixels = np.concatenate(pixels, axis=0)
    return pixels.mean(axis=0), pixels.std(axis=0) + 1e-6


# ─────────────────────────────────────────────────────────────────────────
# Inferencia full-image (compartida con segmentation.py)
# ─────────────────────────────────────────────────────────────────────────

def pad_to_multiple(img: np.ndarray, multiple: int):
    h, w = img.shape[:2]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)) if img.ndim == 3 else ((0, pad_h), (0, pad_w)),
                     mode="reflect")
    return padded, h, w


@torch.no_grad()
def infer_full_image(model, rgb_uint8: np.ndarray, mean, std, device="cpu", multiple=2 ** DEPTH):
    img = rgb_uint8.astype(np.float32) / 255.0
    padded, orig_h, orig_w = pad_to_multiple(img, multiple)
    norm = (padded - mean) / std
    x = torch.from_numpy(norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    model.eval()
    logits = model(x)
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    return prob[:orig_h, :orig_w]


def dice_loss(prob, target, eps=1e-6):
    prob_f, target_f = prob.reshape(prob.size(0), -1), target.reshape(target.size(0), -1)
    inter = (prob_f * target_f).sum(1)
    union = prob_f.sum(1) + target_f.sum(1)
    return 1 - ((2 * inter + eps) / (union + eps)).mean()


def evaluate(model, val_triplets, mean, std, device="cpu"):
    """Sensibilidad, especificidad, accuracy y Dice sobre las imágenes de validación completas."""
    tp = tn = fp = fn = 0
    for img_path, fov_path, gt_path in val_triplets:
        img = _load_rgb(img_path)
        fov = _load_gray(fov_path) > 127
        gt = _load_gray(gt_path) > 127
        prob = infer_full_image(model, img, mean, std, device)
        pred = (prob > 0.5) & fov
        gt = gt & fov
        tp += int(np.sum(pred & gt))
        tn += int(np.sum(~pred & ~gt & fov))
        fp += int(np.sum(pred & ~gt))
        fn += int(np.sum(~pred & gt))

    sens = tp / (tp + fn + 1e-6)
    spec = tn / (tn + fp + 1e-6)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-6)
    return {"sensitivity": sens, "specificity": spec, "accuracy": acc, "dice": dice}


# ─────────────────────────────────────────────────────────────────────────
# Entrenamiento
# ─────────────────────────────────────────────────────────────────────────

def train(args):
    data_dir = Path(args.data)
    triplets = list_drive_triplets(data_dir)
    if not triplets:
        print(f"[ERROR] No se encontraron tríos imagen/mask/1st_manual en {data_dir}")
        return
    train_triplets, val_triplets = split_train_val(triplets, n_val=args.val_images)
    print(f"[INFO] DRIVE: {len(train_triplets)} imágenes de train, {len(val_triplets)} de validación")

    mean, std = compute_norm_stats(train_triplets)
    print(f"[INFO] Normalización (RGB) — media: {mean.round(3)}, std: {std.round(3)}")

    dataset = DrivePatchDataset(train_triplets, patch_size=args.patch_size,
                                 patches_per_epoch=args.patches_per_epoch, mean=mean, std=std)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=0)

    device = "cpu"
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_dice = -1.0
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
            prob = torch.sigmoid(logits)
            loss = F.binary_cross_entropy_with_logits(logits, y) + dice_loss(prob, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)

        metrics = evaluate(model, val_triplets, mean, std, device)
        dt = time.time() - t0
        print(f"→ Epoch {epoch}/{args.epochs} — loss: {running_loss / len(dataset):.4f} | "
              f"val sens: {metrics['sensitivity']:.3f} | val spec: {metrics['specificity']:.3f} | "
              f"val dice: {metrics['dice']:.3f} | {dt:.1f}s")

        if metrics["dice"] > best_dice:
            best_dice = metrics["dice"]
            torch.save({
                "state_dict": model.state_dict(),
                "mean": mean.tolist(), "std": std.tolist(),
                "base_channels": BASE_CHANNELS, "depth": DEPTH,
                "metrics": metrics,
            }, output_path)
            print(f"  ↳ mejor modelo hasta ahora (dice={best_dice:.3f}), guardado en {output_path}")

    print(f"\n[OK] Entrenamiento terminado. Mejor dice de validación: {best_dice:.3f}")
    print(f"[OK] Checkpoint final en: {output_path}")


def convert_to_tensorrt(args):
    try:
        import tensorrt  # noqa: F401
        import torch2trt  # noqa: F401
    except ImportError:
        print("[WARN] TensorRT/torch2trt no están instalados en esta Jetson todavía.")
        print("[WARN] La conversión queda diferida al commit 12 (optimización Jetson).")
        print("[INFO] Por ahora, vascular/segmentation.py corre inferencia directa en PyTorch (CPU).")
        return

    from torch2trt import torch2trt

    ckpt = torch.load(args.input, map_location="cpu", weights_only=False)
    model = UNet(base=ckpt["base_channels"], depth=ckpt["depth"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval().cuda()

    dummy = torch.zeros((1, 3, args.patch_size, args.patch_size)).cuda()
    model_trt = torch2trt(model, [dummy])

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_trt.state_dict(), args.output)
    print(f"[OK] Modelo TensorRT guardado en: {args.output}")


def parse_args():
    p = argparse.ArgumentParser(description="Modelo U-Net para segmentación de vasos retinales")
    p.add_argument("--train", action="store_true", help="Entrena desde cero sobre el dataset DRIVE")
    p.add_argument("--convert", action="store_true", help="Convierte un checkpoint .pth a TensorRT")
    p.add_argument("--download", action="store_true",
                    help=argparse.SUPPRESS)  # ver mensaje explicativo en main()

    p.add_argument("--data", type=str, default=DEFAULT_DATA_DIR, help="Carpeta DRIVE/training")
    p.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Ruta de salida del checkpoint")
    p.add_argument("--input", type=str, help="Checkpoint .pth de entrada (para --convert)")

    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    p.add_argument("--patches-per-epoch", type=int, default=1500,
                    help="~15 epochs x 1500 patches tarda unos 20-25 min en CPU (Jetson Orin Nano)")
    p.add_argument("--val-images", type=int, default=4, help="Cuántas imágenes de DRIVE/training reservar para validación")
    p.add_argument("--lr", type=float, default=1e-3)
    return p.parse_args()


def main():
    args = parse_args()
    if args.download:
        print("[INFO] No existe un checkpoint preentrenado público para esta arquitectura.")
        print("[INFO] Usá 'python vascular/unet_model.py --train' para entrenar sobre DRIVE en su lugar.")
        return
    if args.train:
        train(args)
    elif args.convert:
        if not args.input:
            print("[ERROR] --convert requiere --input <checkpoint.pth>")
            return
        convert_to_tensorrt(args)
    else:
        print("[ERROR] Especificá --train o --convert")


if __name__ == "__main__":
    main()
