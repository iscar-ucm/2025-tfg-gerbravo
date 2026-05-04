import cv2
import os
import csv
from ultralytics import YOLO

#rutas
CARPETA_RGB = r'C:\Users\GERMAN\Desktop\universidad\5_quinto\Trabajo_de_Fin_de_Grado\assets\fotosSucias_guaj1'
RUTA_MODELO_YOLO = r'C:\Users\GERMAN\Desktop\universidad\5_quinto\Trabajo_de_Fin_de_Grado\resultados\fase2\entrenamiento_yolo5\weights\best.pt'

CSV_TELEMETRIA = r'C:\Users\GERMAN\Desktop\universidad\5_quinto\Trabajo_de_Fin_de_Grado\resultados\fase1\guajaraz1_ir_datos.csv'

CARPETA_RESULTADOS_VISUALES = r'C:\Users\GERMAN\Desktop\universidad\5_quinto\Trabajo_de_Fin_de_Grado\resultados\fase3\visualizaciones_finales'
CSV_SALIDA = r'C:\Users\GERMAN\Desktop\universidad\5_quinto\Trabajo_de_Fin_de_Grado\resultados\fase4\coordenadas_feedback.csv'

# creacion de directorios por si no existen
os.makedirs(CARPETA_RESULTADOS_VISUALES, exist_ok=True)
os.makedirs(os.path.dirname(CSV_SALIDA), exist_ok=True)

# se carga el csv de coordenadas obtenido en la fase 1
print("<--- Cargando datos de telemetría --->")
diccionario_coordenadas = {}
if os.path.exists(CSV_TELEMETRIA):
    with open(CSV_TELEMETRIA, mode='r', encoding='utf-8') as f:
        # Se asume delimitador ';'
        lector_telemetria = csv.DictReader(f, delimiter=';') 
        for fila in lector_telemetria:
            try:
                # se guarda la latitud y longitud usando el segundo del video como identificador
                segundo = int(fila['Segundo_Video'])
                diccionario_coordenadas[segundo] = (fila['Latitud'], fila['Longitud'])
            except:
                continue
else:
    print(f"No se ha encontrado el CSV")

#inicializacion de YOLO
print("<--- Iniciando YOLO --->")
model_yolo = YOLO(RUTA_MODELO_YOLO)

#se lee todo lo que haya en la carperta, pero solo nos quedamos con las imagenes
archivos_rgb = [f for f in os.listdir(CARPETA_RGB) if f.endswith(('.jpg', '.png'))]

# se abre el csv para guardar lo que se encuentre
with open(CSV_SALIDA, mode='w', newline='', encoding='utf-8') as archivo_csv:
    escritor_csv = csv.writer(archivo_csv, delimiter=';')
    # se guardan las coordenadas en la foto, las coordenadas geograficas y la cofianza
    escritor_csv.writerow(['Fotograma', 'Confianza_YOLO', 'Pos_X1', 'Pos_Y1', 'Pos_X2', 'Pos_Y2', 'Latitud', 'Longitud'])

    print(f"\n<--- Iniciando detección --->")

    for nombre_foto in archivos_rgb:
        
        # se saca el segundo de la foto y se bucan coordenadas
        try:
            # corta el nombre frame_60.jpg y se queda con el 60
            segundo_foto = int(nombre_foto.split('_')[1].split('.')[0])
            # busca las coordenadas para ese segundo. Si no las tiene, pone "No detectado"
            latitud, longitud = diccionario_coordenadas.get(segundo_foto, ("No detectado", "No detectado"))
        except:
            latitud, longitud = ("Error formato", "Error formato")


        #creacion de ruta para la foto resultado
        ruta_rgb = os.path.join(CARPETA_RGB, nombre_foto)
        #se lee el archivo y convierte en matriz de pixels
        img_rgb = cv2.imread(ruta_rgb)
        
        if img_rgb is None: 
            continue

        # YOLO analiza la imagen
        resultados = model_yolo(img_rgb, verbose=False)[0]
        hay_detecciones = False

        #si YOLO encuentra blooms, los recorremos para revisar las cajas que pone
        for box in resultados.boxes:

            #marcamos que ha habido deteccion
            hay_detecciones = True
            #se extraen las coordenadas exactas de la imagen y la confianza
            x1_rgb, y1_rgb, x2_rgb, y2_rgb = map(int, box.xyxy[0])
            confianza = float(box.conf[0])
            
            # se guardan los datos en csv (INCLUYENDO LATITUD Y LONGITUD)
            escritor_csv.writerow([nombre_foto, f"{confianza:.3f}", x1_rgb, y1_rgb, x2_rgb, y2_rgb, latitud, longitud])
            
            #se dibujamos la caja verde y la etiqueta de texto sobre la foto con la confianza
            color_caja = (0, 255, 0) # Verde en formato BGR
            etiqueta = f"BLOOM | Conf: {confianza:.2f}"
            
            cv2.rectangle(img_rgb, (x1_rgb, y1_rgb), (x2_rgb, y2_rgb), color_caja, 2)
            cv2.putText(img_rgb, etiqueta, (x1_rgb, max(20, y1_rgb - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_caja, 2)

        #solo guardamos la foto si se ha detectado algo
        if hay_detecciones:
            cv2.imwrite(os.path.join(CARPETA_RESULTADOS_VISUALES, nombre_foto), img_rgb)
            print(f"[{nombre_foto}] ¡Alga detectada en LAT:{latitud} LON:{longitud}!")

print(f"\n <--- Detección finalizada --->")