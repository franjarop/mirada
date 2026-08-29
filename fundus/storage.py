"""
Commit 7 — Guardado de imágenes de fondo de ojo con metadata.
"""

import cv2
import json
import time
from pathlib import Path


def save_frame(frame, output_dir: str, eye: str, exposure: int,
               sharpness_pct: int) -> str:
    """
    Guarda frame como PNG y metadata JSON.
    Retorna la ruta del PNG guardado.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    stem = f"fundus_{eye.upper()}_{ts}"
    img_path = out / f"{stem}.png"
    meta_path = out / f"{stem}.json"

    cv2.imwrite(str(img_path), frame)

    meta = {
        "timestamp": ts,
        "eye": eye,
        "exposure_ms": exposure,
        "sharpness_pct": sharpness_pct,
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    return str(img_path)
