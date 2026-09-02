"""
Commit 10 — Índice de tortuosidad de los vasos mayores.

Tortuosidad = longitud de arco / longitud de cuerda (línea recta entre extremos) de cada
segmento de vaso entre bifurcaciones. 1.0 = vaso perfectamente recto; valores más altos
indican más curvatura. No requiere distinguir arteria/vena (a diferencia del AVR original
del doc — ver vascular/caliber.py para el porqué del cambio).
"""

import numpy as np
from scipy.ndimage import label

from vascular.caliber import compute_skeleton

NEIGHBORS_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
STEP_LEN = {(dy, dx): float(np.hypot(dy, dx)) for dy, dx in NEIGHBORS_8}


def _neighbor_count_map(skeleton: np.ndarray) -> np.ndarray:
    """Cantidad de vecinos (8-conectividad) de cada píxel del esqueleto."""
    padded = np.pad(skeleton.astype(np.uint8), 1)
    count = np.zeros_like(skeleton, dtype=np.uint8)
    for dy, dx in NEIGHBORS_8:
        count += padded[1 + dy:1 + dy + skeleton.shape[0], 1 + dx:1 + dx + skeleton.shape[1]]
    return count


def _walk_segment(pixel_set: set, start):
    """Camina la cadena de píxeles desde un extremo, sumando longitud de arco."""
    path = [start]
    visited = {start}
    current = start
    arc_len = 0.0
    while True:
        nxt = None
        for dy, dx in NEIGHBORS_8:
            cand = (current[0] + dy, current[1] + dx)
            if cand in pixel_set and cand not in visited:
                nxt = cand
                step = STEP_LEN[(dy, dx)]
                break
        if nxt is None:
            break
        arc_len += step
        visited.add(nxt)
        path.append(nxt)
        current = nxt
    return path, arc_len


def tortuosity_index(vessel_mask: np.ndarray, min_chord_px: float = 15.0) -> dict:
    """
    Retorna el índice de tortuosidad promedio sobre los segmentos de vaso (entre
    bifurcaciones) suficientemente largos, y cuántos segmentos se usaron.
    """
    skeleton = compute_skeleton(vessel_mask)
    neighbor_count = _neighbor_count_map(skeleton)

    # Quitar puntos de bifurcación (>2 vecinos) para partir el esqueleto en segmentos simples
    branch_points = (neighbor_count > 2) & skeleton
    segments_mask = skeleton & ~branch_points

    labeled, n_segments = label(segments_mask, structure=np.ones((3, 3)))
    ratios = []

    for seg_id in range(1, n_segments + 1):
        ys, xs = np.where(labeled == seg_id)
        if len(ys) < 4:
            continue
        pixel_set = set(zip(ys.tolist(), xs.tolist()))

        degree = {p: sum(1 for dy, dx in NEIGHBORS_8 if (p[0] + dy, p[1] + dx) in pixel_set)
                  for p in pixel_set}
        endpoints = [p for p, d in degree.items() if d == 1]
        if len(endpoints) < 2:
            continue  # segmento cerrado en loop o degenerado, se descarta

        start = endpoints[0]
        path, arc_len = _walk_segment(pixel_set, start)
        end = path[-1]  # extremo real alcanzado por la caminata (puede no ser endpoints[-1] si hay >2 extremos)
        chord_len = float(np.hypot(start[0] - end[0], start[1] - end[1]))

        if chord_len < min_chord_px or arc_len <= 0:
            continue

        ratios.append(arc_len / chord_len)

    if not ratios:
        return {"tortuosity_index": 1.0, "segments_used": 0}

    return {"tortuosity_index": float(np.mean(ratios)), "segments_used": len(ratios)}
