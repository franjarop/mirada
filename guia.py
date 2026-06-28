#!/usr/bin/env python3
"""
Mirada — Guía interactiva de proyecto
Uso:
  python guia.py               → menú principal
  python guia.py --siguiente   → muestra el próximo commit pendiente
  python guia.py --completar 0 → marca el commit 0 como completado
  python guia.py --estado      → muestra el estado de todos los commits
  python guia.py --commit 3    → muestra las instrucciones del commit 3
"""

import sys
import os
import json
import re
import argparse
import textwrap

ESTADO_FILE = os.path.join(os.path.dirname(__file__), ".mirada_estado.json")
MD_FILE     = os.path.join(os.path.dirname(__file__), "mirada_proyecto.md")

COMMITS = [
    (0,  "Entorno",      "Verificación del entorno Jetson"),
    (1,  "Cámara",       "Captura básica de cámara fisheye"),
    (2,  "Cámara",       "Corrección de distorsión fisheye"),
    (3,  "Euler",        "Pipeline de magnificación euleriana"),
    (4,  "Euler",        "Detección de ROI ocular para rPPG"),
    (5,  "Pupilometría", "Detección de pupila y diámetro"),
    (6,  "Pupilometría", "Anisocoria y registro CSV"),
    (7,  "Fondo de ojo", "Adquisición con lente 20D/28D"),
    (8,  "Fondo de ojo", "Pre-procesamiento de imagen retinal"),
    (9,  "Vascular",     "Segmentación de vasos con U-Net"),
    (10, "Vascular",     "Métricas AVR, tortuosidad y reporte"),
    (11, "Integración",  "Dashboard unificado en tiempo real"),
    (12, "Optimización", "Optimización para Jetson Orin Nano"),
]

# ── colores ANSI ────────────────────────────────────────────────────────────
R  = "\033[0m"       # reset
B  = "\033[1m"       # bold
DIM= "\033[2m"       # dim
CY = "\033[96m"      # cyan
GR = "\033[92m"      # green
YE = "\033[93m"      # yellow
RE = "\033[91m"      # red
BL = "\033[94m"      # blue
MG = "\033[95m"      # magenta
BG_CY  = "\033[106m" # bg cyan
BG_GR  = "\033[42m"  # bg green
BG_BL  = "\033[44m"  # bg blue


def ancho():
    try:
        return os.get_terminal_size().columns
    except Exception:
        return 80


def linea(char="─", color=DIM):
    print(f"{color}{char * ancho()}{R}")


def titulo(texto, color=CY):
    w = ancho()
    pad = max(0, (w - len(texto) - 2)) // 2
    print()
    linea("═", color)
    print(f"{color}{B}{' ' * pad}  {texto}  {' ' * pad}{R}")
    linea("═", color)


def subtitulo(texto, color=BL):
    print(f"\n{color}{B}  {texto}{R}")
    linea("─", DIM)


def wrap(texto, indent=4):
    w = ancho() - indent - 2
    lines = texto.split("\n")
    out = []
    for line in lines:
        if len(line) <= w:
            out.append(" " * indent + line)
        else:
            wrapped = textwrap.fill(line, width=w)
            for wl in wrapped.split("\n"):
                out.append(" " * indent + wl)
    return "\n".join(out)


# ── estado JSON ─────────────────────────────────────────────────────────────
def cargar_estado():
    if os.path.exists(ESTADO_FILE):
        with open(ESTADO_FILE, "r") as f:
            return json.load(f)
    return {"completados": []}


def guardar_estado(estado):
    with open(ESTADO_FILE, "w") as f:
        json.dump(estado, f, indent=2)


# ── actualizar el .md ────────────────────────────────────────────────────────
def actualizar_md(completados):
    if not os.path.exists(MD_FILE):
        print(f"{RE}  No se encontró {MD_FILE}{R}")
        return
    with open(MD_FILE, "r", encoding="utf-8") as f:
        contenido = f.read()

    for num, modulo, desc in COMMITS:
        patron_pendiente  = rf"(\| {num:2d} \| [^|]+ \| [^|]+ \|) ⬜ Pendiente "
        patron_completado = rf"(\| {num:2d} \| [^|]+ \| [^|]+ \|) ✅ Completado"

        if num in completados:
            contenido = re.sub(patron_pendiente,  r"\1 ✅ Completado", contenido)
        else:
            contenido = re.sub(patron_completado, r"\1 ⬜ Pendiente ", contenido)

    with open(MD_FILE, "w", encoding="utf-8") as f:
        f.write(contenido)


# ── extraer instrucciones del .md ────────────────────────────────────────────
def extraer_commit_del_md(num):
    if not os.path.exists(MD_FILE):
        return None
    with open(MD_FILE, "r", encoding="utf-8") as f:
        contenido = f.read()

    patron = rf"(## COMMIT {num} —.*?)(?=\n## COMMIT |\Z)"
    m = re.search(patron, contenido, re.DOTALL)
    return m.group(1).strip() if m else None


def renderizar_md(texto):
    """Renderiza markdown básico en la terminal con colores ANSI."""
    lineas = texto.split("\n")
    en_codigo = False
    buf_codigo = []

    for linea_txt in lineas:
        # bloques de código
        if linea_txt.strip().startswith("```"):
            if not en_codigo:
                en_codigo = True
                lang = linea_txt.strip()[3:]
                buf_codigo = []
                print(f"\n{DIM}  ┌─ {lang or 'código'} {'─'*(ancho()-10)}{R}")
            else:
                en_codigo = False
                for cl in buf_codigo:
                    print(f"{DIM}  │{R}{BL}  {cl}{R}")
                print(f"{DIM}  └{'─'*(ancho()-4)}{R}\n")
            continue

        if en_codigo:
            buf_codigo.append(linea_txt)
            continue

        # encabezados
        if linea_txt.startswith("### "):
            print(f"\n{YE}{B}  {linea_txt[4:]}{R}")
            print(f"{DIM}  {'─' * (len(linea_txt) - 2)}{R}")
        elif linea_txt.startswith("## "):
            titulo(linea_txt[3:], MG)
        elif linea_txt.startswith("# "):
            titulo(linea_txt[2:], CY)
        # líneas horizontales md
        elif linea_txt.strip() in ("---", "===", "***"):
            linea("─", DIM)
        # listas
        elif linea_txt.strip().startswith("- ") or linea_txt.strip().startswith("* "):
            item = linea_txt.strip()[2:]
            item = re.sub(r"`([^`]+)`", f"{BL}\\1{R}", item)
            item = re.sub(r"\*\*([^*]+)\*\*", f"{B}\\1{R}", item)
            print(f"  {CY}•{R}  {item}")
        # líneas normales
        elif linea_txt.strip():
            out = linea_txt
            out = re.sub(r"`([^`]+)`", f"{BL}\\1{R}", out)
            out = re.sub(r"\*\*([^*]+)\*\*", f"{B}\\1{R}", out)
            out = re.sub(r"\*([^*]+)\*", f"{DIM}\\1{R}", out)
            print(f"  {out}")
        else:
            print()


# ── vistas ───────────────────────────────────────────────────────────────────
def mostrar_estado():
    estado = cargar_estado()
    completados = estado.get("completados", [])
    titulo("Mirada — Estado del proyecto", CY)

    total = len(COMMITS)
    hechos = len(completados)
    pct = int(hechos / total * 100)

    bar_w = ancho() - 20
    filled = int(bar_w * hechos / total)
    barra = f"{GR}{'█' * filled}{DIM}{'░' * (bar_w - filled)}{R}"
    print(f"\n  Progreso: {barra} {GR}{B}{pct}%{R}  ({hechos}/{total})\n")

    for num, modulo, desc in COMMITS:
        if num in completados:
            icono = f"{GR}✅{R}"
            color_num = GR
            color_desc = DIM
        else:
            icono = f"{YE}⬜{R}"
            color_num = YE
            color_desc = R

        num_str = f"{color_num}COMMIT {num:02d}{R}"
        mod_str = f"{DIM}[{modulo}]{R}"
        print(f"  {icono}  {num_str}  {mod_str}  {color_desc}{desc}{R}")

    print()
    proximo = next((n for n, _, _ in COMMITS if n not in completados), None)
    if proximo is not None:
        print(f"  {BL}→ Próximo a trabajar: {B}COMMIT {proximo}{R}")
        print(f"  {DIM}  Ejecuta: python guia.py --commit {proximo}{R}")
    else:
        print(f"  {GR}{B}  ¡PROYECTO COMPLETADO! Todos los commits están listos.{R}")
    print()


def mostrar_commit(num):
    texto = extraer_commit_del_md(num)
    if not texto:
        print(f"\n{RE}  No se encontró el COMMIT {num} en el archivo .md{R}\n")
        return

    estado = cargar_estado()
    completados = estado.get("completados", [])
    ya_hecho = num in completados

    if ya_hecho:
        print(f"\n  {GR}{B}✅ Este commit ya está marcado como completado.{R}")
        print(f"  {DIM}Puedes releer las instrucciones de todas formas.\n{R}")

    renderizar_md(texto)

    print()
    linea("─", DIM)
    if not ya_hecho:
        print(f"\n  {YE}Cuando termines este commit, ejecúta:{R}")
        print(f"  {BL}{B}  python guia.py --completar {num}{R}\n")
    else:
        proximo = next((n for n, _, _ in COMMITS if n not in completados), None)
        if proximo is not None:
            print(f"\n  {GR}Este commit está listo. El siguiente es COMMIT {proximo}:{R}")
            print(f"  {BL}{B}  python guia.py --commit {proximo}{R}\n")


def completar_commit(num):
    if not any(n == num for n, _, _ in COMMITS):
        print(f"\n{RE}  El commit {num} no existe. Los commits van del 0 al 12.{R}\n")
        return

    estado = cargar_estado()
    completados = estado.get("completados", [])

    if num in completados:
        print(f"\n{YE}  El COMMIT {num} ya estaba marcado como completado.{R}\n")
        return

    completados.append(num)
    completados.sort()
    estado["completados"] = completados
    guardar_estado(estado)
    actualizar_md(completados)

    _, modulo, desc = next((c for c in COMMITS if c[0] == num), (num, "", ""))
    print(f"\n{GR}{B}  ✅ COMMIT {num} marcado como completado{R}")
    print(f"  {DIM}[{modulo}] {desc}{R}")
    print(f"  {DIM}El archivo mirada_proyecto.md ha sido actualizado.{R}\n")

    proximo = next((n for n, _, _ in COMMITS if n not in completados), None)
    if proximo is not None:
        _, mod_p, desc_p = next(c for c in COMMITS if c[0] == proximo)
        print(f"  {BL}→ Próximo commit a trabajar:{R}")
        print(f"     {B}COMMIT {proximo}{R} [{mod_p}] — {desc_p}")
        print(f"  {DIM}  Ejecuta: python guia.py --commit {proximo}{R}\n")
    else:
        titulo("¡PROYECTO COMPLETADO!", GR)
        print(f"  {GR}{B}Todos los commits están listos. ¡Excelente trabajo!{R}\n")


def mostrar_siguiente():
    estado = cargar_estado()
    completados = estado.get("completados", [])
    proximo = next((n for n, _, _ in COMMITS if n not in completados), None)
    if proximo is None:
        print(f"\n{GR}{B}  ¡Todos los commits están completados!{R}\n")
    else:
        mostrar_commit(proximo)


def menu_interactivo():
    titulo("Mirada — Guía Interactiva", CY)
    print(f"  {DIM}Sistema de diagnóstico ocular con IA en Jetson Orin Nano{R}\n")

    estado = cargar_estado()
    completados = estado.get("completados", [])
    hechos = len(completados)
    total = len(COMMITS)
    pct = int(hechos / total * 100)
    print(f"  {GR}Progreso: {hechos}/{total} commits completados ({pct}%){R}\n")

    print(f"  {B}¿Qué deseas hacer?{R}\n")
    print(f"  {CY}[1]{R}  Ver el estado de todos los commits")
    print(f"  {CY}[2]{R}  Ver instrucciones del próximo commit pendiente")
    print(f"  {CY}[3]{R}  Ver instrucciones de un commit específico")
    print(f"  {CY}[4]{R}  Marcar un commit como completado")
    print(f"  {CY}[5]{R}  Salir\n")

    try:
        opcion = input(f"  {YE}Ingresa el número (1-5): {R}").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return

    if opcion == "1":
        mostrar_estado()

    elif opcion == "2":
        mostrar_siguiente()

    elif opcion == "3":
        print(f"\n  {DIM}Commits disponibles: 0 al 12{R}")
        try:
            num = int(input(f"  {YE}Número de commit: {R}").strip())
            mostrar_commit(num)
        except (ValueError, EOFError):
            print(f"\n{RE}  Número inválido.{R}\n")

    elif opcion == "4":
        mostrar_estado()
        try:
            num = int(input(f"  {YE}¿Qué commit deseas marcar como completado? {R}").strip())
            completar_commit(num)
        except (ValueError, EOFError):
            print(f"\n{RE}  Número inválido.{R}\n")

    elif opcion == "5":
        print(f"\n  {DIM}Hasta la próxima. ¡Éxito con el proyecto!{R}\n")

    else:
        print(f"\n{RE}  Opción no válida. Ejecuta 'python guia.py' de nuevo.{R}\n")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Mirada — Guía interactiva del proyecto",
        add_help=True
    )
    parser.add_argument("--estado",    action="store_true", help="Ver estado de todos los commits")
    parser.add_argument("--siguiente", action="store_true", help="Ver instrucciones del próximo commit")
    parser.add_argument("--commit",    type=int, metavar="N", help="Ver instrucciones del commit N")
    parser.add_argument("--completar", type=int, metavar="N", help="Marcar el commit N como completado")

    args = parser.parse_args()

    if args.estado:
        mostrar_estado()
    elif args.siguiente:
        mostrar_siguiente()
    elif args.commit is not None:
        mostrar_commit(args.commit)
    elif args.completar is not None:
        completar_commit(args.completar)
    else:
        menu_interactivo()


if __name__ == "__main__":
    main()
