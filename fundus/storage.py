"""
Commit 7 — Guardado de imágenes de fondo de ojo con metadata.
También carga imágenes de entrada, incluyendo DICOM (retinografías de equipos clínicos).
"""

import cv2
import json
import time
import numpy as np
from pathlib import Path

DICOM_EXTS = (".dcm", ".dicom")

# Campos DICOM que se copian al reporte. No se incluyen datos que identifiquen al paciente
# (nombre, ID, fecha de nacimiento).
DICOM_META_FIELDS = ("Modality", "Manufacturer", "ManufacturerModelName", "StudyDate",
                     "ImageLaterality", "Laterality", "PhotometricInterpretation",
                     "BitsStored", "Rows", "Columns", "PixelSpacing")


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


def is_dicom(path) -> bool:
    """Por extensión, o por la firma "DICM" en el byte 128 (archivos DICOM sin extensión)."""
    path = Path(path)
    if path.suffix.lower() in DICOM_EXTS:
        return True
    try:
        with open(path, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except OSError:
        return False


def _dicom_to_bgr(ds) -> np.ndarray:
    try:
        # pixel_array ya convierte YBR (JPEG) a RGB
        arr = ds.pixel_array
    except Exception as e:
        raise RuntimeError(f"No se pudo decodificar el DICOM ({ds.file_meta.TransferSyntaxUID.name}): {e}. "
                           "Si está comprimido, instalá: pip install pylibjpeg pylibjpeg-libjpeg "
                           "pylibjpeg-openjpeg") from e

    photometric = ds.PhotometricInterpretation
    samples = int(getattr(ds, "SamplesPerPixel", 1))

    # Multi-frame: se usa el primer frame
    if int(getattr(ds, "NumberOfFrames", 1)) > 1:
        arr = arr[0]

    if photometric == "PALETTE COLOR":
        from pydicom.pixels import apply_color_lut
        arr = apply_color_lut(arr, ds)
        samples = 3

    # 16 bits u otros rangos → uint8 (el pipeline trabaja en 8 bits)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        arr = ((arr - lo) / (hi - lo) * 255.0) if hi > lo else np.zeros_like(arr)
        arr = arr.astype(np.uint8)

    if samples == 1:
        if photometric == "MONOCHROME1":
            arr = 255 - arr
        print("[AVISO] DICOM en escala de grises: el pipeline está pensado para retinografía "
              "a color, los resultados pueden ser peores.")
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def load_image(path):
    """
    Carga una imagen de fondo de ojo como BGR uint8 (mismo formato que cv2.imread).
    Acepta PNG/JPG/TIF y DICOM. Retorna None si no se puede leer.
    """
    path = Path(path)
    if not is_dicom(path):
        return cv2.imread(str(path))

    import pydicom
    try:
        ds = pydicom.dcmread(str(path))
        return _dicom_to_bgr(ds)
    except Exception as e:
        print(f"[ERROR] {path}: {e}")
        return None


def dicom_metadata(path) -> dict:
    """Metadata no identificatoria del DICOM, o {} si no es DICOM."""
    if not is_dicom(path):
        return {}
    import pydicom
    ds = pydicom.dcmread(str(path), stop_before_pixels=True)
    meta = {}
    for field in DICOM_META_FIELDS:
        value = getattr(ds, field, None)
        if value not in (None, ""):
            if field == "PixelSpacing":
                meta[field] = [float(v) for v in value]
            elif field in ("BitsStored", "Rows", "Columns"):
                meta[field] = int(value)
            else:
                meta[field] = str(value)
    return meta
