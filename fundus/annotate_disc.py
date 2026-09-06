"""
Anotación manual de disco óptico o mácula — para validar con verdad de terreno real qué tan
bien anda la heurística/esquema híbrido actual (ver fundus/calibrate_disc_threshold.py y la
sesión 2026-09-05 del esquema híbrido de disco en la memoria del proyecto, donde esto se usó
para confirmar que el híbrido baja el error de 48.7px a 27.3px sin necesidad de reentrenar
nada). Marcás con un clic la posición real sobre la imagen ya procesada (mismo dominio CLAHE
que ve el modelo/heurística), y las coordenadas quedan guardadas en un CSV.

Para el disco, además del centro se puede ajustar el radio (rueda del mouse o teclas +/-):
arranca en el valor que calcula automáticamente roi_extractor.estimate_disc_radius, así que solo
hace falta tocarlo cuando ese cálculo se equivoca.

Uso — una imagen:
  python fundus/annotate_disc.py --input fundus_images/dataset_drive/DRIVE/training/images/34_training.tif
  python fundus/annotate_disc.py --target macula --input ...

Uso — una carpeta completa (retoma donde quedó si ya hay anotaciones guardadas):
  python fundus/annotate_disc.py --batch fundus_images/dataset_drive/DRIVE/training/images
  python fundus/annotate_disc.py --target macula --batch fundus_images/dataset_drive/DRIVE/training/images

Controles en la ventana: clic izquierdo = marcar el punto, rueda del mouse o +/- = agrandar
o achicar el círculo (solo disco), 's' = guardar y pasar a la siguiente, 'n' = saltar esta
imagen sin guardar, 'q' = salir (lo ya guardado queda guardado).
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from fundus.preprocess import preprocess_image
from fundus.roi_extractor import compute_fov_mask, estimate_disc_radius

DEFAULT_OUTPUT = {
    "disc": "fundus_annotations/disc_manual.csv",
    "macula": "fundus_annotations/macula_manual.csv",
}
EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
RADIUS_STEP = 2
DEFAULT_MACULA_RADIUS = 20


def load_existing(csv_path: Path, cols: list) -> dict:
    """cols: nombres de columnas de valor (sin 'image'), ej. ['disc_x','disc_y','disc_radius']."""
    labels = {}
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if all(c in row and row[c] != "" for c in cols):
                    labels[row["image"]] = tuple(int(float(row[c])) for c in cols)
    return labels


def save_labels(csv_path: Path, labels: dict, cols: list):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image"] + cols)
        for img, values in sorted(labels.items()):
            writer.writerow([img, *values])


def annotate_one(name: str, target: str, clahe_img: np.ndarray, existing=None, default_radius=None):
    """
    Muestra la imagen, deja marcar el punto con un clic (y ajustar el radio si target=="disc").
    Retorna (x,y) o (x,y,radius), None (saltear), o 'quit'.
    """
    has_radius = target == "disc"
    if existing is not None:
        pos = (existing[0], existing[1])
        radius = existing[2] if has_radius and len(existing) > 2 else default_radius
    else:
        pos = None
        radius = default_radius

    state = {"pos": pos, "radius": radius}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["pos"] = (x, y)
        elif has_radius and event == cv2.EVENT_MOUSEWHEEL:
            delta = RADIUS_STEP if flags > 0 else -RADIUS_STEP
            state["radius"] = max(5, (state["radius"] or default_radius) + delta)

    win = f"Marca {target} -- {name}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 700, 600)
    cv2.setMouseCallback(win, on_mouse)

    result = "n"
    while True:
        display = clahe_img.copy()
        if state["pos"] is not None:
            r = int(round(state["radius"])) if has_radius else 15
            cv2.circle(display, state["pos"], r, (0, 255, 0), 2)
        hint = f"clic=marcar {target}"
        if has_radius:
            hint += " | rueda o +/- = radio"
        hint += " | s=guardar | n=saltar | q=salir"
        cv2.putText(display, hint, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.imshow(win, display)
        key = cv2.waitKey(20) & 0xFF
        if has_radius and key in (ord("+"), ord("=")):
            state["radius"] = (state["radius"] or default_radius) + RADIUS_STEP
        elif has_radius and key in (ord("-"), ord("_")):
            state["radius"] = max(5, (state["radius"] or default_radius) - RADIUS_STEP)
        elif key == ord("s"):
            result = "save"
            break
        elif key == ord("n"):
            result = "skip"
            break
        elif key == ord("q"):
            result = "quit"
            break
    cv2.destroyWindow(win)

    if result == "quit":
        return "quit"
    if result != "save" or state["pos"] is None:
        return None
    if has_radius:
        return (state["pos"][0], state["pos"][1], int(round(state["radius"])))
    return state["pos"]


def main():
    p = argparse.ArgumentParser(description="Anotación manual de disco/mácula para validación")
    p.add_argument("--target", choices=["disc", "macula"], default="disc")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=str, help="Una imagen ORIGINAL (se le aplica el pipeline CLAHE acá mismo)")
    src.add_argument("--batch", type=str, help="Carpeta con imágenes ORIGINALES")
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    cols = [f"{args.target}_x", f"{args.target}_y"] + (["disc_radius"] if args.target == "disc" else [])
    csv_path = Path(args.output) if args.output else Path(DEFAULT_OUTPUT[args.target])
    labels = load_existing(csv_path, cols)
    print(f"[INFO] {len(labels)} anotaciones ya guardadas en {csv_path}")

    if args.input:
        files = [Path(args.input)]
    else:
        batch_dir = Path(args.batch)
        files = sorted(f for f in batch_dir.rglob("*") if f.suffix.lower() in EXTS)

    for f in files:
        frame = cv2.imread(str(f))
        if frame is None:
            print(f"[WARN] No se pudo leer: {f}")
            continue
        result = preprocess_image(frame)
        key = str(f)

        default_radius = None
        if args.target == "disc":
            fov_mask, _, _ = compute_fov_mask(result["clahe"])
            default_radius = estimate_disc_radius(result["clahe"], result["disc_pos"], fov_mask=fov_mask)
        else:
            default_radius = DEFAULT_MACULA_RADIUS

        values = annotate_one(f.name, args.target, result["clahe"],
                               existing=labels.get(key), default_radius=default_radius)
        if values == "quit":
            print("[INFO] Salida pedida — se guardó lo marcado hasta acá.")
            break
        if values is not None:
            labels[key] = values
            save_labels(csv_path, labels, cols)
            print(f"[OK] {f.name}: {args.target} en {values}")
        else:
            print(f"[SKIP] {f.name}")

    print(f"\n[OK] Anotaciones guardadas en: {csv_path} ({len(labels)} imágenes)")


if __name__ == "__main__":
    main()
