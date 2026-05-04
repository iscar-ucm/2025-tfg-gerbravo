import cv2
import os
import math
from tkinter import filedialog
from tkinter import Tk

CARPETA_RESULTADOS = r'C:\Users\GERMAN\Desktop\universidad\5_quinto\Trabajo_de_Fin_de_Grado\resultados\fase2'

#creacion de ventana
root = Tk()
root.withdraw() # ocultamos la ventana principal de fondo de tkinter
root.attributes('-topmost', True)

print("Selecciona el video RGB en la ventana que se acaba de abrir")
# abrimos el explorador de archivos
ruta_video = filedialog.askopenfilename(
    title="Selecciona el video RGB para extraer fotogramas",
    filetypes=[("Archivos de video", "*.mp4 *.MP4")]
)

# si se cierra la ventana sin elegir nada, el programa se para
if not ruta_video:
    print("No se ha seleccionado ningun video. Cancelando operación...")
    exit()

#se crea una carpeta para guardar los frames
nombre_video_limpio = os.path.splitext(os.path.basename(ruta_video))[0]
carpeta_destino = os.path.join(CARPETA_RESULTADOS, f"{nombre_video_limpio}_fotogramas")

#se crea si no existe
if not os.path.exists(carpeta_destino):
    os.makedirs(carpeta_destino)

print(f"<--- Video seleccionado: {ruta_video} --->")

#configuracion del video igual que en el anterior script
cap = cv2.VideoCapture(ruta_video)
fps = cap.get(cv2.CAP_PROP_FPS)
fps = math.ceil(fps) if fps > 0 else 30 

total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
duracion_total = total_frames / fps

#si el video dura mas de 2 minutos, se recortan el primero y el ultimo
if duracion_total > 120:
    segundo_inicio = 60
    segundo_fin = int(duracion_total) - 60
else:
    segundo_inicio = 0
    segundo_fin = int(duracion_total)

segundo_actual = segundo_inicio 
#se define cada cuantos segundos se extrae un frame
salto_segundos = 5 

print(f"Video cargado. FPS: {fps}. Empezando extraccion del segundo {segundo_inicio} al {segundo_fin}...")

#bucle para procesar el video y extraer fotos
while cap.isOpened():
    #control de finalizacion
    if segundo_actual > segundo_fin:
        break

    #salta al frame y con read se decodifica a matriz de pixeles
    frame_id = segundo_actual * fps
    #esta parte hace un salto sin tener que pasar por todos los frames que no nos interesan
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    
    ret, frame = cap.read()
    
    if not ret:
        break

    # se guarda frame como una imagen .jpg en una ruta absoluta
    nombre_foto = os.path.join(carpeta_destino, f"frame_{segundo_actual}.jpg")
    cv2.imwrite(nombre_foto, frame)
    
    print(f"Foto extraida y guardada: frame_{segundo_actual}.jpg")
    
    # se avanzan x segundos de golpe
    segundo_actual += salto_segundos

cap.release()

print(f"<--- Proceso terminado --->")