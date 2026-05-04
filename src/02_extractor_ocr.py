import cv2
import easyocr
import math
import os
import re
import torch
import pandas as pd
from tkinter import filedialog
from tkinter import Tk

#se le indica a pytorch que busque el mejor algoritmo convolucional posible de la GPU
torch.backends.cudnn.benchmark = True

CARPETA_RESULTADOS = r'C:\Users\GERMAN\Desktop\universidad\5_quinto\Trabajo_de_Fin_de_Grado\resultados\fase1'

#comprueba si la ruta existe y sino, la crea
if not os.path.exists(CARPETA_RESULTADOS):
    os.makedirs(CARPETA_RESULTADOS)

#creacion de ventana
root = Tk()
root.withdraw() # ocultamos la ventana principal de fondo de tkinter para que solo se vea el explorador de archivos
root.attributes('-topmost', True)

print("Selecciona el video IR en la ventana que se acaba de abrir")
# abrimos el explorador de archivos
ruta_video = filedialog.askopenfilename(
    title="Selecciona el vdeo IR para extraer datos",
    filetypes=[("Archivos de video", "*.mp4")]
)

# si se cierra la ventana sin elegir nada, el programa se para
if not ruta_video:
    print("No se ha seleccionado ningun video. Cancelando operación...")
    exit()

#se crea la ruta absoluta para el csv basandonos en la ruta del video elegido
#y lo guardamos en la carpeta de resultados de la fase 1
nombre_video_limpio = os.path.splitext(os.path.basename(ruta_video))[0]
ruta_csv = os.path.join(CARPETA_RESULTADOS, f"{nombre_video_limpio}_datos.csv")

print("<--- Iniciando modelo EasyOCR --->")

#configuracion de EasyOCR y video
# gpu=True para usar paralelamente la gpu
reader = easyocr.Reader(['en'], gpu=True) 

cap = cv2.VideoCapture(ruta_video)
fps = cap.get(cv2.CAP_PROP_FPS)
fps = math.ceil(fps) if fps > 0 else 30 

total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
duracion_total = total_frames / fps

# gestion sobre la duracion del video, si el video dura mas 
# de 2 minutos, se recortan el primero y el ultimo, para evitar procesar el despegue y el aterrizaje.
if duracion_total > 120:
    segundo_inicio = 60
    segundo_fin = int(duracion_total) - 60
else:
    segundo_inicio = 0
    segundo_fin = int(duracion_total)

datos_finales = []
#segundo en el que se empieza quitando el despegue, normalmente el 60
segundo_actual = segundo_inicio 
# variable para guardar la fecha
fecha = "" 

print(f"video cargado. FPS detectados: {fps}. Empezando extraccion del segundo {segundo_inicio} al {segundo_fin}...")

#bucle para procesar segundo a segundo
while cap.isOpened():
    #comprueab si ha terminado o no de procesar todos los segundos
    if segundo_actual > segundo_fin:
        break

    frame_id = segundo_actual * fps 
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    
    #se lee un frame exacto del video
    ret, frame = cap.read()
    
    if not ret:
        break
    
    #se miden el alto y el ancho del frame
    alto, ancho, _ = frame.shape
    
    # recortes, para que solo se trate lo que nos interesa del frame
    roi_arriba = frame[0:int(alto*0.20), 0:ancho]
    roi_abajo = frame[int(alto*0.80):alto, 0:ancho]

    #aqui se dejan las imagenes(las dos secciones de 20%)
    # en blanco y negro, para mejor tratamiento 
    #y tambien se duplica su tamaño usando INTER_CUBIC
    roi_arriba = cv2.resize(roi_arriba, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    roi_arriba = cv2.cvtColor(roi_arriba, cv2.COLOR_BGR2GRAY)
    roi_abajo = cv2.resize(roi_abajo, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    roi_abajo = cv2.cvtColor(roi_abajo, cv2.COLOR_BGR2GRAY)

    # se ejecuta la red neuronal de el easyocr
    texto_arriba = reader.readtext(roi_arriba, detail=0)
    texto_abajo = reader.readtext(roi_abajo, detail=0)
    
    str_temperaturas = " ".join(texto_arriba)
    str_gps = " ".join(texto_abajo)
    
    #filtros para evitar confuciones y errores de escritura
    str_temperaturas = str_temperaturas.replace('O', '0').replace('o', '0').replace('S', '5').replace('s', '5').replace('Z', '2').replace('z', '2')
    str_gps = str_gps.replace('O', '0').replace('o', '0').replace('S', '5').replace('s', '5').replace('Z', '2').replace('z', '2')
    str_gps = str_gps.replace('L0N', 'LON')

    # se guardan los datos usando expresiones regulares, donde el - es opcional
    # y el resto de elementos pueden tener longitud 
    # variable(se detectan longitud, latitud y altura)
    match_lat = re.search(r'LAT:(-?\d+\.\d+)', str_gps)
    latitud = match_lat.group(1) if match_lat else "No detectado"

    match_lon = re.search(r'LON:(-?\d+\.\d+)', str_gps)
    longitud = match_lon.group(1) if match_lon else "No detectado"

    match_alt = re.search(r'H:\s*([^\s]+)', str_gps)
    altura = match_alt.group(1) if match_alt else "No detectado"

    #sacamos la fecha
    if fecha == "":
        match_fecha = re.search(r'(\d{4}[/.-]\d{1,2}[/.-]\d{1,2})', str_gps)
        if match_fecha:
            # se cambian los posibles puntos por barras para que la fecha siempre sea YYYY/MM/DD
            fecha = match_fecha.group(1).replace('.', '/').replace('-', '/')

    # se busca un patron de 2 digitos, separador, 2 digitos, separador, 1 o 2 digitos.
    match_hora = re.search(r'(\d{2}[:.,;]\d{2}[:.,;]\d{1,2})', str_gps)
    if match_hora:
        hora = match_hora.group(1)
        # se reemplaza cualquier punto, coma o punto y coma que haya leido el OCR por dos puntos, para evitar fallos
        hora = re.sub(r'[:.,;]', ':', hora)
        #si el OCR se comio el ultimo digito (ej: 04:57:3 en vez de 30), se rellena con un 0 al final
        if len(hora) == 7:
            hora += "0"
    else:
        hora = "No detectado"

    # buscamos todos los numeros decimales para sacar las 4 temperaturas
    numeros_temp = re.findall(r'\d+\.\d+', str_temperaturas)
    temp_max = numeros_temp[0] if len(numeros_temp) > 0 else "No detectado"
    temp_min = numeros_temp[1] if len(numeros_temp) > 1 else "No detectado"
    temp_avg = numeros_temp[2] if len(numeros_temp) > 2 else "No detectado"
    temp_center = numeros_temp[3] if len(numeros_temp) > 3 else "No detectado"

    # se ordenan las columnas desplazandolas y se inserta la fecha solo en la primera fila
    datos_finales.append({
        'Fecha_Vuelo': fecha if len(datos_finales) == 0 else "",
        'Segundo_Video': segundo_actual,
        'Latitud': latitud,
        'Longitud': longitud,
        'Altura': altura,
        'Hora': hora,
        'Temp_Max': temp_max,
        'Temp_Min': temp_min,
        'Temp_Avg': temp_avg,
        'Temp_Center': temp_center,
        ' ': '',
        '  ': '',
        '   ': '',
        '    ': '',
        '     ': '',
        'Texto_Bruto_OCR_Abajo': str_gps,
        'Texto_Bruto_OCR_Arriba': str_temperaturas
    })
    
    print(f"<--- Procesado segundo {segundo_actual} --->")
    segundo_actual += 1

cap.release()

#exportar a CSV
df = pd.DataFrame(datos_finales)
df.to_csv(ruta_csv, index=False, encoding='utf-8', sep=';')

print(f"Los datos se han guardado en: {ruta_csv}")