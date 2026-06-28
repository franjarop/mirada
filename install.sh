#!/usr/bin/env bash
# install.sh — Instalación automática de dependencias para el proyecto Mirada
# Plataforma: NVIDIA Jetson Orin Nano (JetPack 5.x / 6.x)
# Uso: bash install.sh

set -e

GREEN='\033[92m'
YELLOW='\033[93m'
RED='\033[91m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${BOLD}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[ OK ]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERR ]${NC}  $*"; }

echo
echo -e "${BOLD}──────────────────────────────────────────${NC}"
echo -e "${BOLD}  MIRADA — Instalación de dependencias    ${NC}"
echo -e "${BOLD}──────────────────────────────────────────${NC}"
echo

# ── 1. Verificar que es una Jetson ───────────────────────────────────────────
if [ -f /proc/device-tree/model ]; then
    MODEL=$(cat /proc/device-tree/model | tr -d '\0')
    info "Hardware: $MODEL"
else
    warn "No se pudo detectar el modelo del dispositivo"
fi

# ── 2. Actualizar sistema ────────────────────────────────────────────────────
info "Actualizando lista de paquetes del sistema..."
sudo apt-get update -qq

# ── 3. Dependencias del sistema ──────────────────────────────────────────────
info "Instalando dependencias del sistema..."
sudo apt-get install -y -qq \
    python3-pip \
    python3-dev \
    python3-venv \
    libhdf5-dev \
    libhdf5-serial-dev \
    libatlas-base-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    v4l-utils
ok "Dependencias del sistema instaladas"

# ── 4. Actualizar pip ────────────────────────────────────────────────────────
info "Actualizando pip..."
pip3 install --upgrade pip --quiet

# ── 5. NumPy y SciPy ─────────────────────────────────────────────────────────
info "Instalando NumPy y SciPy..."
pip3 install --upgrade numpy scipy --quiet
ok "NumPy y SciPy instalados"

# ── 6. OpenCV ────────────────────────────────────────────────────────────────
info "Verificando OpenCV..."
if python3 -c "import cv2" 2>/dev/null; then
    CV_VER=$(python3 -c "import cv2; print(cv2.__version__)")
    ok "OpenCV ya instalado: v$CV_VER"
    CUDA_DEVS=$(python3 -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())" 2>/dev/null || echo "0")
    if [ "$CUDA_DEVS" -gt 0 ] 2>/dev/null; then
        ok "OpenCV con soporte CUDA ($CUDA_DEVS GPU)"
    else
        warn "OpenCV sin CUDA — para mejor rendimiento, compila desde fuentes con CUDA"
        warn "Guía: https://github.com/mdegans/nano_build_opencv"
        info "Instalando opencv-python (sin CUDA) como alternativa..."
        pip3 install opencv-python --quiet
    fi
else
    info "Instalando opencv-python..."
    pip3 install opencv-python --quiet
    ok "OpenCV instalado (sin CUDA)"
    warn "Para soporte CUDA nativo en Jetson, compila OpenCV desde fuentes"
fi

# ── 7. MediaPipe ─────────────────────────────────────────────────────────────
info "Instalando MediaPipe (necesario para Commit 4)..."
pip3 install mediapipe --quiet
ok "MediaPipe instalado"

# ── 8. Resto de dependencias del proyecto ────────────────────────────────────
info "Instalando resto de dependencias (Pillow, scikit-image, matplotlib...)..."
pip3 install \
    Pillow \
    scikit-image \
    matplotlib \
    tqdm \
    PyYAML \
    reportlab \
    --quiet
ok "Dependencias auxiliares instaladas"

# ── 9. PyQt6 ─────────────────────────────────────────────────────────────────
info "Instalando PyQt6 (necesario para Commit 11 — dashboard)..."
pip3 install PyQt6 --quiet && ok "PyQt6 instalado" || warn "PyQt6 falló — intenta manualmente: pip3 install PyQt6"

# ── 10. Verificación final ───────────────────────────────────────────────────
echo
echo -e "${BOLD}──────────────────────────────────────────${NC}"
info "Ejecutando verificación del entorno..."
echo
python3 setup/check_env.py

echo
echo -e "${BOLD}──────────────────────────────────────────${NC}"
echo -e "${GREEN}${BOLD}  Instalación completa${NC}"
echo -e "${BOLD}──────────────────────────────────────────${NC}"
echo
