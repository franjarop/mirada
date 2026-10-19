# Mirada — Sistema de Diagnóstico Ocular con IA
**Plataforma:** NVIDIA Jetson Orin Nano  
**Lenguaje:** Python 3.10  
**Pantalla:** 7 pulgadas (1024x600) conectada directamente  
**Cámara:** USB con lente externo 20D/28D para fondo de ojo  

---

## ¿Qué hace este proyecto?

Mirada es un sistema de **procesamiento de imágenes de fondo de ojo** sobre una Jetson Orin Nano:

1. **Adquisición** — captura imágenes de la retina con un lente oftalmológico externo (20D/28D), o
   trabaja sobre imágenes ya tomadas con otro dispositivo
2. **Pre-procesamiento** — balance de color, eliminación de reflejos, CLAHE, detección de disco
   óptico (esquema híbrido) y mácula
3. **Segmentación vascular** — U-Net entrenada en DRIVE para segmentar vasos, clasificación
   arteria/vena
4. **Evaluación vascular** — AVR, tortuosidad, calibre y densidad → score de riesgo cardiovascular
   (proxy, no validado clínicamente)

> Los módulos de BPM por rPPG, pupilometría y calibración de cámara (antiguos commits 1–6) se
> sacaron del proyecto el 2026-10-18 y quedaron archivados en `~/mirada_legacy/` (y en el
> historial de git). La numeración de commits se mantiene para no perder la referencia.

---

## Estado de los commits

| # | Módulo | Descripción | Estado |
|---|--------|-------------|--------|
| 0 | Entorno | Verificación del entorno Jetson | ✅ Completado |
| 7 | Fondo de ojo | Adquisición con lente 20D/28D | ✅ Completado |
| 8 | Fondo de ojo | Pre-procesamiento de imagen retinal | ✅ Completado |
| 9 | Vascular | Segmentación de vasos con U-Net | ✅ Completado |
| 10 | Vascular | Métricas AVR, tortuosidad y reporte | ✅ Completado |
| 11 | Integración | Dashboard de fondo de ojo | ⏸ Pospuesto |
| 12 | Optimización | Optimización para Jetson Orin Nano | ⬜ Pendiente |

> Para marcar un commit como completado, ejecuta: `python guia.py --completar 0`

---

## COMMIT 0 — Verificación del entorno Jetson

### ¿Qué hace este commit?
Verifica que la Jetson tenga todo lo necesario instalado: Python, OpenCV, NumPy, SciPy, scikit-image y PyTorch. Si algo falta, lo indica y dice cómo instalarlo.

### Archivos que se crean
- `setup/check_env.py` — script de verificación
- `requirements.txt` — lista de todas las dependencias del proyecto
- `install.sh` — script de instalación automática

### Instrucciones paso a paso

**Paso 1 — Crear la carpeta del proyecto**
```bash
mkdir -p ~/mirada/setup
cd ~/mirada
```

**Paso 2 — Crear el archivo requirements.txt**
```bash
# El archivo requirements.txt se te proporcionará junto con este commit
# Copiarlo a ~/mirada/requirements.txt
```

**Paso 3 — Ejecutar el verificador**
```bash
python setup/check_env.py
```

### ¿Qué verás al ejecutar?
```
Terminal (solo texto, sin ventanas):

✓ Python 3.10.x         — OK
✓ OpenCV 4.x + CUDA     — OK
✓ NumPy 1.x             — OK
✓ SciPy 1.x             — OK
~ PyTorch (sin CUDA)    — la U-Net corre en CPU

Si todo está en verde: el entorno está listo.
Si hay una ✗ roja: sigue la instrucción que aparece en pantalla.
```

### Señal de éxito
Cuando veas `ENTORNO LISTO` en verde, este commit está completo.

---

## COMMIT 7 — Adquisición con lente 20D/28D

### ¿Qué hace este commit?
Configura la cámara para capturar imágenes de fondo de ojo usando el lente oftalmológico externo. Incluye control manual de exposición y un indicador de nitidez para saber cuándo guardar.

### Archivos que se crean
- `fundus/acquisition.py` — control de exposición, indicador de nitidez, captura
- `fundus/quality_check.py` — descarta frames borrosos automáticamente
- `fundus/storage.py` — guarda imágenes con metadata (ojo, exposición, timestamp)

### Lo que necesitas tener listo
```
✓ Lente 20D o 28D con adaptador para la cámara USB
✓ Fuente de luz LED infrarroja o coaxial para iluminar la retina
✓ Silla o soporte para que el paciente mantenga la cabeza quieta
```

### Instrucciones paso a paso

**Paso 1 — Ejecutar en modo adquisición**
```bash
python fundus/acquisition.py --device 0 --eye left --output fundus_images/
```

**Controles durante la captura**
```
Teclado mientras corre el programa:
  E  →  aumentar exposición (imagen más brillante)
  D  →  disminuir exposición (imagen más oscura)
  B  →  aumentar brillo
  N  →  disminuir brillo
  ESPACIO  →  guardar el mejor frame del último segundo
  Q  →  salir
```

### ¿Qué verás al ejecutar?
```
En la pantalla de 7 pulgadas:
→ Video en vivo del fondo de ojo
→ Barra en la parte superior:
   VERDE:  imagen nítida — buen momento para capturar
   ROJO:   imagen borrosa — ajusta el lente o la posición
→ En la esquina: "Nitidez: 87% | Exp: 45ms"

Al presionar ESPACIO:
→ "Frame guardado: fundus_images/fundus_L_20240115_143022.png"
```

### Señal de éxito
Se guardó al menos una imagen PNG con la barra de nitidez en verde. La imagen muestra estructuras de la retina (vasos sanguíneos, disco óptico).

---

## COMMIT 8 — Pre-procesamiento de imagen retinal

### ¿Qué hace este commit?
Mejora automáticamente las imágenes de retina capturadas: aumenta el contraste, normaliza colores y detecta el disco óptico y la mácula para usarlos como referencia en el análisis vascular.

### Archivos que se crean
- `fundus/preprocess.py` — CLAHE para mejorar contraste, elimina reflejos
- `fundus/color_balance.py` — normaliza el canal verde (mayor contraste vascular)
- `fundus/roi_extractor.py` — detecta disco óptico y mácula automáticamente

### Instrucciones paso a paso

**Paso 1 — Procesar una imagen**
```bash
python fundus/preprocess.py --input fundus_images/fundus_L_001.png --show
```

**Paso 2 — Procesar todas las imágenes de una sesión**
```bash
python fundus/preprocess.py --batch fundus_images/ --output fundus_processed/
```

### ¿Qué verás al ejecutar?
```
En la pantalla de 7 pulgadas:
→ TRES ventanas (cada una de 320x240):
   1. ORIGINAL: imagen tal como salió de la cámara
   2. CLAHE:    contraste mejorado, vasos más visibles
   3. CANAL G:  canal verde normalizado (máximo contraste vascular)
   
→ Sobre la imagen: círculos marcando:
   AMARILLO: disco óptico (si detectado)
   AZUL:     zona macular (si detectado)

En la terminal:
→ Disco óptico detectado en: (312, 248)
→ Mácula detectada en: (198, 241)
→ Imagen guardada en: fundus_processed/fundus_L_001_processed.png
```

### Señal de éxito
Las imágenes procesadas muestran los vasos con mayor claridad que las originales. Se guardaron en `fundus_processed/`.

---

## COMMIT 9 — Segmentación de vasos con U-Net

### ¿Qué hace este commit?
Usa una red neuronal U-Net (preentrenada en el dataset DRIVE) para identificar y colorear todos los vasos sanguíneos de la retina. El modelo se convierte a TensorRT para correr eficientemente en la Jetson.

### Archivos que se crean
- `vascular/unet_model.py` — carga U-Net, lo convierte a TensorRT
- `vascular/segmentation.py` — segmenta vasos sobre imagen pre-procesada
- `vascular/overlay.py` — superpone máscara de vasos sobre imagen original

### Instrucciones paso a paso

**Paso 1 — Descargar el modelo preentrenado (solo la primera vez)**
```bash
python vascular/unet_model.py --download --output models/
```
```
Esto descarga el modelo U-Net preentrenado en DRIVE (~45MB).
Puede tardar unos minutos según la velocidad de internet.
```

**Paso 2 — Convertir a TensorRT (solo la primera vez, tarda ~5 minutos)**
```bash
python vascular/unet_model.py --convert --input models/unet_drive.pth --output models/unet_trt.engine
```
```
Verás mensajes de progreso. Es normal que tarde entre 3 y 8 minutos.
Solo se hace UNA vez. El archivo .engine queda guardado para siempre.
```

**Paso 3 — Segmentar una imagen**
```bash
python vascular/segmentation.py --input fundus_processed/fundus_L_001_processed.png --model models/unet_trt.engine --show
```

### ¿Qué verás al ejecutar?
```
En la pantalla de 7 pulgadas:
→ DOS ventanas (cada una de 480x360):
   IZQUIERDA: imagen de retina original
   DERECHA:   imagen con vasos coloreados en ROJO (arterias) y AZUL (venas estimadas)

En la terminal:
→ Cargando modelo TensorRT... OK
→ Inferencia: 48ms | GPU Jetson: 62% | Vasos detectados: 2,841 píxeles
→ Máscara guardada en: masks/fundus_L_001_mask.png
```

### Señal de éxito
La imagen derecha muestra la red vascular de la retina claramente coloreada. El tiempo de inferencia es menor a 200ms.

---

## COMMIT 10 — Métricas vasculares y reporte

### ¿Qué hace este commit?
Calcula métricas clínicas a partir de la segmentación de vasos: AVR (razón arteria-vena), tortuosidad de los vasos y genera un score de riesgo cardiovascular básico. Guarda todo en un reporte JSON.

### Archivos que se crean
- `vascular/avr.py` — calcula Arteriovenous Ratio (AVR) por zona
- `vascular/tortuosity.py` — índice de tortuosidad de los vasos mayores
- `vascular/risk_score.py` — score de riesgo cardiovascular con las métricas

### Instrucciones paso a paso

**Paso 1 — Calcular métricas**
```bash
python vascular/risk_score.py \
  --input fundus_processed/fundus_L_001_processed.png \
  --mask masks/fundus_L_001_mask.png \
  --session paciente01
```

### ¿Qué verás al ejecutar?
```
En la terminal:
→ ─────────────────────────────────────────
→ REPORTE VASCULAR — paciente01
→ ─────────────────────────────────────────
→ AVR (razón arteria-vena): 0.72
→ Índice de tortuosidad:    1.08
→ Calibre promedio arterial: 3.2 px
→ ─────────────────────────────────────────
→ Score riesgo cardiovascular: BAJO
→ ─────────────────────────────────────────
→ Reporte guardado en: reports/paciente01_2024-01-15.json
```

### Señal de éxito
Se generó el archivo JSON en `reports/`. El score de riesgo aparece como BAJO, MODERADO o ALTO.

---

## COMMIT 11 — Dashboard de fondo de ojo (⏸ pospuesto)

### ¿Qué hace este commit?
Interfaz gráfica para una sesión de análisis de fondo de ojo: capturar en vivo con el lente 20D
**o cargar una imagen ya tomada** con otro dispositivo → analizar (disco/mácula, vasos, AVR,
score) → resumen → reporte PDF.

Pospuesto: primero se prioriza mejorar el pipeline de procesamiento (commits 8–10).

### Archivos que se crearán
- `dashboard/main_window.py` — GUI (PyQt5)
- `dashboard/session_manager.py` — datos de la sesión
- `dashboard/report_generator.py` — PDF con reportlab

### Dependencias (cuando se retome)
```bash
sudo apt-get install -y python3-pyqt5
pip install reportlab
```

---

## COMMIT 12 — Optimización para Jetson Orin Nano

### ¿Qué hace este commit?
Perfila el pipeline de fondo de ojo, identifica los módulos más lentos y los optimiza usando CUDA streams paralelos. El objetivo es que el análisis de una imagen sea lo más rápido posible en la Jetson.

### Archivos que se crean
- `optimization/profiler.py` — mide latencia de cada módulo individualmente
- `optimization/cuda_pipeline.py` — pipelines CUDA paralelos
- `optimization/memory_manager.py` — gestión eficiente de memoria GPU/CPU

### Instrucciones paso a paso

**Paso 1 — Perfil del sistema actual**
```bash
python optimization/profiler.py --device 0 --duration 30
```

**Paso 2 — Aplicar optimizaciones**
```bash
python optimization/cuda_pipeline.py --apply
```

**Paso 3 — Verificar mejora**
```bash
python optimization/profiler.py --device 0 --duration 30 --compare
```

### ¿Qué verás al ejecutar?
```
En la terminal:
→ ─────────────────────────────────────
→ PERFIL DE RENDIMIENTO — Jetson Orin Nano
→ ─────────────────────────────────────
→ Pre-procesamiento:     20ms
→ Detección de disco:    25ms
→ Segmentación vasos:    45ms
→ ─────────────────────────────────────
→ TOTAL: ~90ms por imagen
→ GPU: 68% | RAM: 3.2GB / 8GB
→ ─────────────────────────────────────
```

### Señal de éxito
El análisis de una imagen completa baja de tiempo de forma medible respecto al perfil inicial. La GPU no supera el 80% de uso.

---

## Estructura del proyecto

```
mirada/
├── guia.py                    ← Script de guía interactiva (leer desde terminal)
├── mirada_proyecto.md         ← Este archivo (documentación completa)
├── requirements.txt
├── install.sh
├── setup/
│   └── check_env.py
├── fundus/
│   ├── acquisition.py
│   ├── quality_check.py
│   ├── storage.py
│   ├── preprocess.py
│   ├── color_balance.py
│   ├── roi_extractor.py
│   ├── disc_hybrid.py
│   ├── disc_model_smdg.py
│   ├── disc_macula_model.py
│   ├── disc_vessel_convergence.py
│   ├── annotate_disc.py
│   └── calibrate_disc_threshold.py
├── vascular/
│   ├── unet_model.py
│   ├── segmentation.py
│   ├── overlay.py
│   ├── av_classifier.py
│   ├── avr.py
│   ├── caliber.py
│   ├── tortuosity.py
│   └── risk_score.py
├── dashboard/                 ← (commit 11, pospuesto)
├── optimization/              ← (commit 12)
├── models/
├── fundus_images/
├── fundus_processed/
├── fundus_annotations/
├── masks/
└── reports/
```

---

*Documento generado para el proyecto Mirada — Prototipo académico de análisis de fondo de ojo con IA*  
*Última actualización: ver historial de commits*
