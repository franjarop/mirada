# Mirada — Sistema de Diagnóstico Ocular con IA
**Plataforma:** NVIDIA Jetson Orin Nano  
**Lenguaje:** Python 3.10  
**Pantalla:** 7 pulgadas (1024x600) conectada directamente  
**Cámara:** Fisheye Wide Angle USB 120fps  
**Lente externo:** 20D/28D para fondo de ojo  

---

## ¿Qué hace este proyecto?

Este sistema convierte una Jetson Orin Nano en un dispositivo de análisis ocular con 4 capacidades:

1. **Magnificación euleriana** — amplifica los cambios de color en el ojo para medir el pulso cardiaco sin contacto
2. **Pupilometría** — mide el diámetro de la pupila, detecta asimetría entre ojos y mide reflejos
3. **Fondo de ojo** — captura imágenes de la retina usando un lente oftalmológico externo (20D/28D)
4. **Evaluación vascular** — analiza los vasos de la retina con inteligencia artificial para detectar riesgos cardiovasculares y posible desprendimiento de retina

---

## Estado de los commits

| # | Módulo | Descripción | Estado |
|---|--------|-------------|--------|
| 0 | Entorno | Verificación del entorno Jetson | ⬜ Pendiente |
| 1 | Cámara | Captura básica de cámara fisheye | ⬜ Pendiente |
| 2 | Cámara | Corrección de distorsión fisheye | ⬜ Pendiente |
| 3 | Euler | Pipeline de magnificación euleriana | ⬜ Pendiente |
| 4 | Euler | Detección de ROI ocular para rPPG | ⬜ Pendiente |
| 5 | Pupilometría | Detección de pupila y diámetro | ⬜ Pendiente |
| 6 | Pupilometría | Anisocoria y registro CSV | ⬜ Pendiente |
| 7 | Fondo de ojo | Adquisición con lente 20D/28D | ⬜ Pendiente |
| 8 | Fondo de ojo | Pre-procesamiento de imagen retinal | ⬜ Pendiente |
| 9 | Vascular | Segmentación de vasos con U-Net | ⬜ Pendiente |
| 10 | Vascular | Métricas AVR, tortuosidad y reporte | ⬜ Pendiente |
| 11 | Integración | Dashboard unificado en tiempo real | ⬜ Pendiente |
| 12 | Optimización | Optimización para Jetson Orin Nano | ⬜ Pendiente |

> Para marcar un commit como completado, ejecuta: `python guia.py --completar 0`

---

## COMMIT 0 — Verificación del entorno Jetson

### ¿Qué hace este commit?
Verifica que la Jetson tenga todo lo necesario instalado: Python, OpenCV con CUDA, NumPy, SciPy y MediaPipe. Si algo falta, lo indica y dice cómo instalarlo.

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
✗ MediaPipe             — FALTA → ejecuta: pip install mediapipe

Si todo está en verde: el entorno está listo.
Si hay una ✗ roja: sigue la instrucción que aparece en pantalla.
```

### Señal de éxito
Cuando veas `ENTORNO LISTO — Puedes continuar al Commit 1` en verde, este commit está completo.

---

## COMMIT 1 — Captura básica de cámara fisheye

### ¿Qué hace este commit?
Abre la cámara USB fisheye, configura 120fps y muestra el video en vivo en la pantalla de 7 pulgadas. También imprime en terminal los fps reales y la resolución detectada.

### Archivos que se crean
- `camera/capture.py` — abre la cámara y muestra el video
- `camera/diagnostics.py` — mide fps real, resolución y latencia

### Instrucciones paso a paso

**Paso 1 — Conectar la cámara**
```
Conecta la cámara USB fisheye a cualquier puerto USB de la Jetson.
Espera 5 segundos para que el sistema la detecte.
```

**Paso 2 — Verificar que la Jetson la detectó**
```bash
ls /dev/video*
# Debes ver algo como: /dev/video0  /dev/video1
# El número 0 es normalmente tu cámara principal
```

**Paso 3 — Ejecutar la captura**
```bash
cd ~/mirada
python camera/capture.py --device 0 --fps 120
```

### ¿Qué verás al ejecutar?
```
En la pantalla de 7 pulgadas:
→ Ventana con el video en vivo de la cámara (tamaño 800x480 para que quepa bien)
→ En la esquina superior izquierda: "FPS: 118" en verde

En la terminal:
→ FPS real: 118.3 | Resolución: 1280x720 | Latencia: 8ms

Para salir: presiona la tecla Q con el mouse sobre la ventana de video.
```

### Señal de éxito
La ventana abre, se ve video en vivo y los FPS están por encima de 60. Si los FPS son menores a 30, hay un problema con la cámara o el cable USB.

---

## COMMIT 2 — Corrección de distorsión fisheye

### ¿Qué hace este commit?
La cámara fisheye deforma la imagen con efecto de ojo de pez (líneas curvas). Este commit calibra la cámara usando un tablero de ajedrez y después corrige la distorsión en tiempo real.

### Archivos que se crean
- `camera/calibrate.py` — captura poses del tablero para calibrar
- `camera/calibration_data.npz` — datos de calibración guardados (se usa en todos los commits siguientes)
- `camera/undistort.py` — aplica la corrección en tiempo real

### Lo que necesitas
```
Un tablero de ajedrez impreso en papel (mínimo 9x6 cuadros).
Puedes imprimirlo desde: https://calib.io/pages/camera-calibration-pattern-generator
Tamaño recomendado: hoja A4 o carta.
```

### Instrucciones paso a paso

**Paso 1 — Calibrar la cámara**
```bash
python camera/calibrate.py --output camera/calibration_data.npz
```
```
Lo que debes hacer mientras corre:
→ Sostén el tablero frente a la cámara
→ Muévelo lentamente a diferentes ángulos y posiciones
→ La terminal dirá cuántas poses capturó: "Poses: 15/20"
→ Necesita mínimo 20 poses para una buena calibración
→ Cuando llegue a 20, calcula automáticamente y guarda el archivo
```

**Paso 2 — Verificar la corrección**
```bash
python camera/undistort.py --calib camera/calibration_data.npz
```

### ¿Qué verás al ejecutar?
```
En la pantalla de 7 pulgadas:
→ DOS ventanas lado a lado (cada una de 480x360):
   IZQUIERDA: imagen original con efecto fisheye (líneas curvas)
   DERECHA:   imagen corregida (líneas rectas, sin distorsión)

En la terminal:
→ Error de reproyección: 0.43px   ← menor a 1.0 es bueno
→ Calibración guardada en: camera/calibration_data.npz
```

### Señal de éxito
El error de reproyección es menor a 1.0px. Las líneas que antes eran curvas ahora aparecen rectas en la ventana derecha.

---

## COMMIT 3 — Pipeline de magnificación euleriana

### ¿Qué hace este commit?
Implementa el algoritmo que amplifica cambios de color imperceptibles al ojo humano. Usado para visualizar el pulso cardíaco en la piel o en la conjuntiva ocular.

### Archivos que se crean
- `eulerian/pyramid.py` — construye la pirámide laplaciana de la imagen
- `eulerian/temporal_filter.py` — filtro de banda entre 0.4 y 4 Hz (frecuencia cardíaca)
- `eulerian/magnify.py` — pipeline completo con alpha (intensidad) ajustable

### Instrucciones paso a paso

**Paso 1 — Prueba con video de archivo (más fácil)**
```bash
cd ~/mirada
python eulerian/magnify.py --input test_video.mp4 --alpha 50 --freq-lo 0.4 --freq-hi 4.0
```

**Paso 2 — Prueba con cámara en vivo**
```bash
python eulerian/magnify.py --device 0 --alpha 50 --calib camera/calibration_data.npz
```
```
Consejo: apunta la cámara a tu muñeca o mejilla bajo buena iluminación.
Los cambios de color son muy sutiles — el algoritmo los amplifica para hacerlos visibles.
```

### ¿Qué verás al ejecutar?
```
En la pantalla de 7 pulgadas:
→ DOS ventanas (cada una de 480x360):
   IZQUIERDA: video original (piel o ojo, sin cambios visibles)
   DERECHA:   video magnificado (pulsaciones de color al ritmo del corazón)

El efecto es sutil. Si no lo ves, intenta subir el alpha:
   python eulerian/magnify.py --device 0 --alpha 100
```

### Señal de éxito
En la ventana derecha se ven cambios de color rítmicos que no son visibles en la izquierda.

---

## COMMIT 4 — Detección de ROI ocular para rPPG

### ¿Qué hace este commit?
Usa MediaPipe (IA de Google) para detectar el rostro y localizar exactamente la zona conjuntival del ojo. Luego extrae la señal de pulso (rPPG) de esa zona y estima los BPM en tiempo real.

### Archivos que se crean
- `eulerian/eye_roi.py` — detecta 468 puntos del rostro, extrae zona conjuntival
- `eulerian/rppg_extractor.py` — extrae señal rPPG del canal verde de la ROI
- `eulerian/visualizer.py` — muestra la señal en gráfica en tiempo real

### Instrucciones paso a paso

**Paso 1 — Instalar MediaPipe (solo la primera vez)**
```bash
pip install mediapipe
```

**Paso 2 — Ejecutar**
```bash
python eulerian/rppg_extractor.py --device 0 --calib camera/calibration_data.npz
```

### ¿Qué verás al ejecutar?
```
En la pantalla de 7 pulgadas:
→ Ventana 1 (600x400, parte superior):
   Video con un rectángulo VERDE sobre la zona blanca del ojo (conjuntiva)
   
→ Ventana 2 (600x200, parte inferior):
   Gráfica en tiempo real con la señal rPPG (ondas que suben y bajan)

En la terminal (actualiza cada segundo):
→ BPM estimado: 72  |  Confianza: 87%

IMPORTANTE: mantén la cabeza quieta. La señal necesita ~10 segundos para estabilizarse.
```

### Señal de éxito
La gráfica muestra ondas regulares y el BPM estimado está entre 50 y 100 pulsaciones.

---

## COMMIT 5 — Detección de pupila y medición de diámetro

### ¿Qué hace este commit?
Detecta la pupila en el video de la cámara, dibuja un círculo sobre ella y mide su diámetro en milímetros en tiempo real.

### Archivos que se crean
- `pupilometry/detector.py` — segmenta la pupila con umbralización adaptativa
- `pupilometry/measure.py` — convierte píxeles a milímetros con factor de calibración
- `pupilometry/tracker.py` — seguimiento temporal, detecta cambios bruscos

### Instrucciones paso a paso

**Paso 1 — Ejecutar**
```bash
python pupilometry/detector.py --device 0 --calib camera/calibration_data.npz
```

**Paso 2 — Prueba de respuesta pupilar**
```
Con el script corriendo:
→ Dirige una pequeña linterna hacia el ojo (no directamente, sino lateral)
→ Verás que el número de diámetro DISMINUYE (pupila se contrae con la luz)
→ Al apagar la linterna, el número AUMENTA (pupila se dilata en la oscuridad)
Este reflejo se llama reflejo fotomotor y es importante para diagnóstico.
```

### ¿Qué verás al ejecutar?
```
En la pantalla de 7 pulgadas:
→ Ventana con la imagen del ojo en close-up
→ Círculo CIAN dibujado sobre el borde de la pupila
→ En la esquina: "Diámetro: 4.2mm | Confianza: 94%"

En la terminal:
→ Diámetro pupila: 4.2mm | Ojo: izquierdo | Confianza: 94%
```

### Señal de éxito
El círculo cian sigue la pupila cuando el ojo se mueve. El diámetro cambia al variar la iluminación.

---

## COMMIT 6 — Anisocoria y registro CSV

### ¿Qué hace este commit?
Compara ambas pupilas simultáneamente, detecta diferencias de tamaño (anisocoria) y guarda todas las mediciones en un archivo CSV con timestamp para análisis posterior.

### Archivos que se crean
- `pupilometry/anisocoria.py` — compara ambas pupilas, calcula diferencia
- `pupilometry/reflex_timer.py` — mide latencia de respuesta (ms)
- `pupilometry/logger.py` — guarda mediciones en CSV con timestamp

### Instrucciones paso a paso

**Paso 1 — Ejecutar con nombre de sesión**
```bash
python pupilometry/anisocoria.py --device 0 --session paciente01 --calib camera/calibration_data.npz
```

### ¿Qué verás al ejecutar?
```
En la pantalla de 7 pulgadas:
→ Ventana con AMBOS ojos visibles simultáneamente
→ Cada pupila tiene su propio círculo y medición
→ Si la diferencia entre ambas pupilas es mayor a 1mm:
   ALERTA ROJA: "ANISOCORIA DETECTADA: diferencia 1.3mm"

En la terminal:
→ Ojo izq: 4.1mm | Ojo der: 4.2mm | Diferencia: 0.1mm — Normal

Al salir con Q:
→ Se guarda automáticamente: sessions/paciente01_2024-01-15.csv
```

### Señal de éxito
Se generó el archivo CSV en la carpeta `sessions/`. Puedes abrirlo en cualquier hoja de cálculo.

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
python fundus/acquisition.py --device 0 --eye left --output fundus_images/ --calib camera/calibration_data.npz
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

## COMMIT 11 — Dashboard unificado

### ¿Qué hace este commit?
Integra todos los módulos anteriores en una sola interfaz gráfica que muestra todo simultáneamente: video en vivo, BPM, pupilometría, imagen retinal y métricas vasculares.

### Archivos que se crean
- `dashboard/main_window.py` — GUI principal con todos los paneles
- `dashboard/session_manager.py` — gestiona una sesión completa de examen
- `dashboard/report_generator.py` — genera PDF con todos los hallazgos

### Instrucciones paso a paso

**Paso 1 — Instalar dependencia de GUI**
```bash
pip install PyQt6
```

**Paso 2 — Ejecutar el dashboard**
```bash
python dashboard/main_window.py --device 0 --calib camera/calibration_data.npz
```

### ¿Qué verás al ejecutar?
```
En la pantalla de 7 pulgadas:
→ Interfaz completa dividida en 4 paneles:
   SUPERIOR IZQUIERDA: Video en vivo + BPM en tiempo real
   SUPERIOR DERECHA:   Pupilometría bilateral (ambos ojos)
   INFERIOR IZQUIERDA: Imagen de fondo de ojo con vasos segmentados
   INFERIOR DERECHA:   Score vascular y alertas

→ Botón "NUEVA SESIÓN" para empezar un examen
→ Botón "GENERAR REPORTE PDF" al finalizar
```

### Señal de éxito
El dashboard abre sin errores y todos los paneles muestran datos en tiempo real.

---

## COMMIT 12 — Optimización para Jetson Orin Nano

### ¿Qué hace este commit?
Perfila el sistema completo, identifica los módulos más lentos y los optimiza usando CUDA streams paralelos. El objetivo es que todo corra a ≥10 fps de forma fluida.

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
→ rPPG (magnificación):  12ms
→ Pupilometría:           8ms
→ Segmentación vasos:    45ms
→ Dashboard render:      15ms
→ ─────────────────────────────────────
→ TOTAL: ~80ms por frame → 12.5 fps ✓
→ GPU: 68% | RAM: 3.2GB / 8GB
→ ─────────────────────────────────────
```

### Señal de éxito
El sistema completo corre a ≥10 fps de forma estable. La GPU no supera el 80% de uso.

---

## Estructura final del proyecto

```
mirada/
├── guia.py                    ← Script de guía interactiva (leer desde terminal)
├── mirada_proyecto.md    ← Este archivo (documentación completa)
├── requirements.txt
├── install.sh
├── camera/
│   ├── capture.py
│   ├── calibrate.py
│   ├── calibration_data.npz
│   └── undistort.py
├── eulerian/
│   ├── pyramid.py
│   ├── temporal_filter.py
│   ├── magnify.py
│   ├── eye_roi.py
│   ├── rppg_extractor.py
│   └── visualizer.py
├── pupilometry/
│   ├── detector.py
│   ├── measure.py
│   ├── tracker.py
│   ├── anisocoria.py
│   ├── reflex_timer.py
│   └── logger.py
├── fundus/
│   ├── acquisition.py
│   ├── quality_check.py
│   ├── storage.py
│   ├── preprocess.py
│   ├── color_balance.py
│   └── roi_extractor.py
├── vascular/
│   ├── unet_model.py
│   ├── segmentation.py
│   ├── overlay.py
│   ├── avr.py
│   ├── tortuosity.py
│   └── risk_score.py
├── dashboard/
│   ├── main_window.py
│   ├── session_manager.py
│   └── report_generator.py
├── optimization/
│   ├── profiler.py
│   ├── cuda_pipeline.py
│   └── memory_manager.py
├── models/
│   ├── unet_drive.pth
│   └── unet_trt.engine
├── fundus_images/
├── fundus_processed/
├── masks/
├── sessions/
└── reports/
```

---

*Documento generado para el proyecto Mirada — Prototipo académico de diagnóstico ocular con IA*  
*Última actualización: ver historial de commits*
