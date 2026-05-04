## 2025-tfg-gerbravo
##Proyecto de planificación de drones y detección temprana de cianobacterias 

##Autor: Germán Bravo Sánchez 
##Organización: Universidad Complutense de Madrid (UCM)

## Descripción del Proyecto
Este repositorio contiene el código desarrollado para la automatización, extracción de telemetría y detección por Visión Artificial de *blooms* de cianobacterias en reservas hídricas utilizando drones comerciales. 

El sistema está diseñado bajo el paradigma de **Edge Computing** y propone una solución \textit{software} modular que no requiere de conexión a Internet, solo conexión GPS. Su objetivo final es alimentar un **Gemelo Digital** del cuerpo de agua analizado, optimizando las misiones de reconocimiento y procesando vídeos con telemetría cerrada (OSD).

## Estructura del Código (`src/`)
El \textit{pipeline} de ejecución es secuencial. Los *scripts* están numerados según su orden lógico de uso:

1. **`01_planificador_tsp.py`**: Algoritmo que modela la misión de inspección como un Problema del Viajante (TSP) usando la métrica de Haversine para optimizar el vuelo y el gasto de batería.
2. **`02_extractor_ocr.py`**: Motor basado en EasyOCR que lee el flujo de vídeo del dron y extrae las coordenadas geográficas (OSD) incrustadas en pantalla.
3. **`03_extractor_fotogramas.py`**: Algoritmo de muestreo que extrae *frames* clave del vídeo capturado para generar el conjunto de imágenes base.
4. **`04_generador_sintetico.py`**: Motor de generación procedimental que utiliza máscaras HSV y fusión matemática para insertar *blooms* artificiales realistas sobre el agua, creando el *dataset* de entrenamiento.
5. **`05_entrenamiento_yolo.py`**: *Script* de configuración y entrenamiento de la red neuronal convolucional (YOLOv8) sobre el *dataset* generado.
6. **`06_inferencia_y_fusion.py`**: Ejecuta la detección final y cruza los fotogramas positivos de YOLOv8 con la telemetría del OCR, exportando un archivo `CSV` listo para inyectar en el Gemelo Digital.

## Instalación y Uso

1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/iscar-ucm/2025-tfg-gerbravo.git](https://github.com/iscar-ucm/2025-tfg-gerbravo.git)
   cd 2025-tfg-gerbravo
