"""
Commit 6 — Registro de mediciones pupilares en CSV con timestamp.
"""

import csv
import os
from datetime import datetime
import pathlib


class PupilLogger:
    """
    Abre (o crea) un CSV de sesión y escribe una fila por frame detectable.
    Columnas: timestamp, izq_mm, der_mm, diferencia_mm, plr_izq, plr_der,
              anisocoria (0/1), evento (texto libre).
    """

    HEADER = [
        "timestamp", "izq_mm", "der_mm", "diferencia_mm",
        "plr_izq", "plr_der", "anisocoria", "evento"
    ]

    def __init__(self, session_name: str, output_dir: str = "sessions"):
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime("%Y-%m-%d")
        filename = f"{session_name}_{date_str}.csv"
        self.filepath = os.path.join(output_dir, filename)
        self._fh = open(self.filepath, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(self.HEADER)
        self._event_next: str = ""
        print(f"[OK] Registro CSV: {self.filepath}")

    def queue_event(self, event: str):
        """Agrega una etiqueta al próximo registro (ej. 'luz_encendida')."""
        self._event_next = event

    def write(self, izq_mm: float, der_mm: float,
              plr_izq: float, plr_der: float, anisocoria: bool):
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        diff = round(abs(izq_mm - der_mm), 2)
        evento = self._event_next
        self._event_next = ""
        self._writer.writerow([
            ts,
            round(izq_mm, 2),
            round(der_mm, 2),
            diff,
            round(plr_izq, 3),
            round(plr_der, 3),
            int(anisocoria),
            evento,
        ])

    def flush(self):
        self._fh.flush()

    def close(self):
        self._fh.flush()
        self._fh.close()
        print(f"[OK] CSV guardado: {self.filepath}")
