"""
Commit 8 — Pre-procesamiento de imagen de fondo de ojo.
Mejora contraste (CLAHE), elimina reflejos y detecta disco óptico / mácula.

Uso — una sola imagen:
  python fundus/preprocess.py --input fundus_images/fundus_L_001.png --show

Uso — todas las imágenes de una carpeta (dataset o sesión completa):
  python fundus/preprocess.py --batch fundus_images/ --output fundus_processed/
"""

import cv2
import numpy as np
import argparse
import sys
from pathlib import Path

from fundus.color_balance import normalize_green_channel, gray_world_balance
from fundus.roi_extractor import compute_fov_mask, detect_optic_disc, detect_macula, draw_roi


EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def parse_args():
    p = argparse.ArgumentParser(description="Pre-procesamiento de imagen retinal")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=str, help="Ruta a una imagen individual")
    src.add_argument("--batch", type=str, help="Carpeta con varias imágenes a procesar")
    p.add_argument("--output", type=str, default="fundus_processed/",
                    help="Carpeta de salida (default: fundus_processed/)")
    p.add_argument("--show", action="store_true", help="Muestra las 3 ventanas en pantalla")
    p.add_argument("--save", action="store_true", help="Guarda la imagen procesada en --output (default: no guarda)")
    p.add_argument("--win-size", type=int, nargs=2, default=list(WIN_SIZE), metavar=("W", "H"),
                    help="Tamaño de cada ventana (default: 640 480). Usar '320 240' para la pantalla de 7\"")
    return p.parse_args()


def remove_reflections(frame: np.ndarray, thresh: int = 240) -> np.ndarray:
    """Detecta brillos especulares (muy claros en los 3 canales) y los rellena por inpainting."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    if cv2.countNonZero(mask) == 0:
        return frame
    return cv2.inpaint(frame, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)


def apply_clahe_contrast(frame: np.ndarray, clip_limit: float = 2.5,
                          tile_size: int = 8) -> np.ndarray:
    """Aumenta contraste global aplicando CLAHE al canal L (luminancia) en espacio LAB."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def preprocess_image(frame: np.ndarray) -> dict:
    """Corre el pipeline completo. Retorna dict con cada etapa y las posiciones detectadas."""
    balanced = gray_world_balance(frame)
    clean = remove_reflections(balanced)
    clahe_img = apply_clahe_contrast(clean)
    green_norm = normalize_green_channel(clean)

    fov_mask, fov_center, fov_radius = compute_fov_mask(clahe_img)
    disc_pos = detect_optic_disc(clahe_img, fov_mask=fov_mask)
    macula_pos = detect_macula(clahe_img, disc_pos=disc_pos, fov_mask=fov_mask,
                                fov_center=fov_center, fov_radius=fov_radius)
    annotated = draw_roi(clahe_img, disc_pos, macula_pos)

    return {
        "original": frame,
        "clahe": clahe_img,
        "green": green_norm,
        "annotated": annotated,
        "disc_pos": disc_pos,
        "macula_pos": macula_pos,
    }


WIN_SIZE = (640, 480)   # tamaño para ver en monitor de desarrollo (320x240 es el de la pantalla de 7" final)


def show_windows(result: dict, win_size=WIN_SIZE):
    original = cv2.resize(result["original"], win_size)
    clahe_img = cv2.resize(result["annotated"], win_size)
    green = cv2.resize(cv2.cvtColor(result["green"], cv2.COLOR_GRAY2BGR), win_size)

    for name, img in (("Mirada — ORIGINAL", original),
                       ("Mirada — CLAHE", clahe_img),
                       ("Mirada — CANAL G", green)):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, *win_size)
        cv2.imshow(name, img)

    print("[INFO] Presiona cualquier tecla sobre una ventana para continuar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_one(path: Path, output_dir: Path, show: bool, save: bool, win_size=WIN_SIZE) -> bool:
    frame = cv2.imread(str(path))
    if frame is None:
        print(f"[WARN] No se pudo leer: {path}")
        return False

    result = preprocess_image(frame)

    if result["disc_pos"] is not None:
        print(f"→ Disco óptico detectado en: {result['disc_pos']}")
    else:
        print("→ Disco óptico: no detectado")

    if result["macula_pos"] is not None:
        print(f"→ Mácula detectada en: {result['macula_pos']}")
    else:
        print("→ Mácula: no detectada")

    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{path.stem}_processed.png"
        cv2.imwrite(str(out_path), result["annotated"])
        print(f"→ Imagen guardada en: {out_path}")
    else:
        print("→ No guardada (falta --save)")

    if show:
        show_windows(result, win_size)

    return True


def main():
    args = parse_args()
    output_dir = Path(args.output)

    if args.input:
        path = Path(args.input)
        if not path.exists():
            print(f"[ERROR] No existe: {path}")
            sys.exit(1)
        process_one(path, output_dir, args.show, args.save, tuple(args.win_size))

    else:
        batch_dir = Path(args.batch)
        if not batch_dir.is_dir():
            print(f"[ERROR] No existe la carpeta: {batch_dir}")
            sys.exit(1)

        files = sorted(f for f in batch_dir.rglob("*") if f.suffix.lower() in EXTS)
        if not files:
            print(f"[WARN] No se encontraron imágenes en {batch_dir}")
            sys.exit(1)

        print(f"[INFO] Procesando {len(files)} imágenes de {batch_dir}...")
        ok = 0
        for f in files:
            print(f"\n--- {f.name} ---")
            if process_one(f, output_dir, args.show, args.save, tuple(args.win_size)):
                ok += 1

        print(f"\n[OK] {ok}/{len(files)} imágenes procesadas. Guardadas en {output_dir}")


if __name__ == "__main__":
    main()
