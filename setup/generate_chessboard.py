"""
Genera un tablero de ajedrez para calibración de cámara.
Uso: python setup/generate_chessboard.py
Genera: setup/tablero_calibracion.png
"""

import numpy as np
import cv2

COLS_INTERNOS = 9
ROWS_INTERNOS = 6
CUADROS_COLS  = COLS_INTERNOS + 1   # 10
CUADROS_ROWS  = ROWS_INTERNOS + 1   # 7

# A4 a 300 DPI: 2480 x 3508 px
DPI    = 300
A4_W   = 2480
A4_H   = 3508
MARGEN = 100  # px de margen blanco en cada lado

area_w = A4_W - 2 * MARGEN
area_h = A4_H - 2 * MARGEN

tam_cuadro = min(area_w // CUADROS_COLS, area_h // CUADROS_ROWS)

tablero_w = tam_cuadro * CUADROS_COLS
tablero_h = tam_cuadro * CUADROS_ROWS

offset_x = (A4_W - tablero_w) // 2
offset_y = (A4_H - tablero_h) // 2

img = np.ones((A4_H, A4_W), dtype=np.uint8) * 255

for r in range(CUADROS_ROWS):
    for c in range(CUADROS_COLS):
        if (r + c) % 2 == 0:
            x1 = offset_x + c * tam_cuadro
            y1 = offset_y + r * tam_cuadro
            x2 = x1 + tam_cuadro
            y2 = y1 + tam_cuadro
            cv2.rectangle(img, (x1, y1), (x2, y2), 0, -1)

tam_mm = round(tam_cuadro / DPI * 25.4)
texto = f"Calibracion Mirada  |  {COLS_INTERNOS}x{ROWS_INTERNOS} esquinas internas  |  cuadro ~{tam_mm}mm  |  imprimir al 100%"
cv2.putText(img, texto, (MARGEN, A4_H - 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2, cv2.LINE_AA)

out = "setup/tablero_calibracion.png"
cv2.imwrite(out, img)
print(f"[OK] Tablero generado: {out}")
print(f"     Tamaño de cuadro: ~{tam_mm}mm  ({tam_cuadro}px a {DPI}DPI)")
print(f"     Esquinas internas: {COLS_INTERNOS}x{ROWS_INTERNOS}")
print()
print("  IMPORTANTE al imprimir:")
print("  - Escala: 100% (sin ajustar a la pagina)")
print("  - Papel: A4")
print("  - Sin margenes adicionales de la impresora")
print(f"\n  Usa este comando para calibrar:")
print(f"  python camera/calibrate.py --cols {COLS_INTERNOS} --rows {ROWS_INTERNOS} --square-mm {tam_mm}")
