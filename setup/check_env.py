"""
Commit 0 — Verificación del entorno Jetson para el proyecto Mirada.
Ejecutar con: python setup/check_env.py
"""

import sys
import importlib
import subprocess

# Códigos ANSI
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(label, version=""):
    ver = f"  {version}" if version else ""
    print(f"  {GREEN}✓{RESET} {label:<30}{GREEN}OK{RESET}{ver}")

def fail(label, fix=""):
    print(f"  {RED}✗{RESET} {label:<30}{RED}FALTA{RESET}", end="")
    if fix:
        print(f"  →  {YELLOW}{fix}{RESET}", end="")
    print()

def warn(label, msg=""):
    print(f"  {YELLOW}~{RESET} {label:<30}{YELLOW}{msg}{RESET}")


def check_python():
    v = sys.version_info
    label = f"Python {v.major}.{v.minor}.{v.micro}"
    if v.major == 3 and v.minor >= 10:
        ok("Python 3.10+", label)
        return True
    else:
        fail("Python 3.10+", f"versión actual: {label} — instala Python 3.10")
        return False


def check_opencv():
    try:
        import cv2
        ver = cv2.__version__
        # Verificar soporte CUDA
        cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
        if cuda_devices > 0:
            ok("OpenCV + CUDA", f"v{ver}  ({cuda_devices} GPU)")
        else:
            warn("OpenCV (sin CUDA)", f"v{ver} — compila con CUDA para mejor rendimiento")
        return True
    except ImportError:
        fail("OpenCV", "pip install opencv-python  (o compilar con CUDA)")
        return False
    except AttributeError:
        # OpenCV existe pero sin módulo cuda
        import cv2
        warn("OpenCV (sin CUDA)", f"v{cv2.__version__} — considera compilar con soporte CUDA")
        return True


def check_numpy():
    try:
        import numpy as np
        ok("NumPy", f"v{np.__version__}")
        return True
    except ImportError:
        fail("NumPy", "pip install numpy")
        return False


def check_scipy():
    try:
        import scipy
        ok("SciPy", f"v{scipy.__version__}")
        return True
    except ImportError:
        fail("SciPy", "pip install scipy")
        return False


def check_mediapipe():
    try:
        import mediapipe as mp
        ok("MediaPipe", f"v{mp.__version__}")
        return True
    except ImportError:
        fail("MediaPipe", "pip install mediapipe")
        return False


def check_pyqt6():
    try:
        from PyQt6.QtCore import QT_VERSION_STR
        ok("PyQt6", f"v{QT_VERSION_STR}")
        return True
    except ImportError:
        fail("PyQt6", "pip install PyQt6  (necesario para el dashboard, commit 11)")
        return False


def check_cuda_runtime():
    """Verifica CUDA en el sistema (independiente de OpenCV)."""
    import os
    cuda_path = "/usr/local/cuda"
    if os.path.isdir(cuda_path):
        # Intentar leer versión
        for vfile in ["version.txt", "version.json"]:
            vpath = os.path.join(cuda_path, vfile)
            if os.path.exists(vpath):
                try:
                    with open(vpath) as f:
                        content = f.read().strip()
                    ver = content[:40]
                    ok("CUDA runtime", ver)
                    return True
                except Exception:
                    pass
        # Versión desconocida pero existe
        import glob
        libs = glob.glob(f"{cuda_path}/lib64/libcudart.so.*.*")
        if libs:
            ver = libs[0].split("libcudart.so.")[-1]
            ok("CUDA runtime", f"v{ver}")
            return True
        ok("CUDA runtime", "(versión desconocida)")
        return True
    else:
        warn("CUDA runtime", "no encontrado en /usr/local/cuda")
        return False


def check_jetson():
    """Verifica que es una Jetson Orin Nano."""
    try:
        with open("/proc/device-tree/model", "r") as f:
            model = f.read().strip().rstrip("\x00")
        if "Jetson" in model:
            ok("Jetson hardware", model[:50])
            return True
        else:
            warn("Jetson hardware", f"dispositivo: {model[:50]}")
            return False
    except FileNotFoundError:
        warn("Jetson hardware", "no se pudo detectar /proc/device-tree/model")
        return False


def main():
    print()
    print(f"{BOLD}{'─'*55}{RESET}")
    print(f"{BOLD}  MIRADA — Verificación del entorno (Commit 0){RESET}")
    print(f"{BOLD}{'─'*55}{RESET}")
    print()

    # Verificaciones obligatorias para los primeros commits
    print(f"  {BOLD}Hardware y sistema:{RESET}")
    check_jetson()
    check_cuda_runtime()
    print()

    print(f"  {BOLD}Dependencias Python — núcleo:{RESET}")
    r_python  = check_python()
    r_numpy   = check_numpy()
    r_scipy   = check_scipy()
    r_opencv  = check_opencv()
    print()

    print(f"  {BOLD}Dependencias Python — módulos avanzados:{RESET}")
    r_mp    = check_mediapipe()
    r_pyqt  = check_pyqt6()
    print()

    # Resultado final
    print(f"{BOLD}{'─'*55}{RESET}")

    core_ok = all([r_python, r_numpy, r_scipy, r_opencv])

    if core_ok:
        print(f"  {GREEN}{BOLD}ENTORNO LISTO — Puedes continuar al Commit 1{RESET}")
        if not r_mp:
            print(f"  {YELLOW}(MediaPipe hace falta en el Commit 4 — instálalo antes){RESET}")
        if not r_pyqt:
            print(f"  {YELLOW}(PyQt6 hace falta en el Commit 11 — instálalo antes){RESET}")
    else:
        print(f"  {RED}{BOLD}ENTORNO INCOMPLETO — Instala los paquetes marcados con ✗{RESET}")
        print(f"  {YELLOW}Ejecuta:  bash install.sh{RESET}")

    print(f"{BOLD}{'─'*55}{RESET}")
    print()

    return 0 if core_ok else 1


if __name__ == "__main__":
    sys.exit(main())
