import cv2
import numpy as np
import os
import random
import math
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

# rutas
CARPETA_LIMPIAS_RGB = filedialog.askdirectory(title="Selecciona la carpeta con los fotogramas limpios")
CARPETA_SALIDA_RGB = filedialog.askdirectory(title="Selecciona la carpeta de salida para el dataset")

TEXTURAS = filedialog.askopenfilenames(title="Selecciona las fotos de texturas (puedes seleccionar varias a la vez)", filetypes=[("Imágenes", "*.jpg *.png *.jpeg"), ("Todos los archivos", "*.*")])

# se crean las carpetas image y labels para que el dataset este estructurado al acabar la ejecucion
os.makedirs(os.path.join(CARPETA_SALIDA_RGB, 'images'), exist_ok=True)
os.makedirs(os.path.join(CARPETA_SALIDA_RGB, 'labels'), exist_ok=True)

texturas_cargadas = []
# cargar texturas para aligerar la ejecucion
for r_txt in TEXTURAS:
    img_t = cv2.imread(r_txt)
    if img_t is not None: texturas_cargadas.append(img_t)

lista_fotos = [f for f in os.listdir(CARPETA_LIMPIAS_RGB) if f.endswith(('.jpg', '.png'))]
print(f"<--- Ejecutando --->")

con_bloom = 0
sin_bloom = 0

# iteramos sobre las fotos RGB
for nombre_foto in lista_fotos:
    nombre_base = os.path.splitext(nombre_foto)[0]
    
    # cargar RGB
    ruta_rgb = os.path.join(CARPETA_LIMPIAS_RGB, nombre_foto)
    img_fondo_rgb = cv2.imread(ruta_rgb)
    if img_fondo_rgb is None: continue

    h_rgb, w_rgb = img_fondo_rgb.shape[:2]

    # copia para no alterar la imagen base
    img_final_rgb = img_fondo_rgb.copy()
    
    # etiquetas
    ruta_lbl_rgb = os.path.join(CARPETA_SALIDA_RGB, 'labels', nombre_base + '.txt')
    
    # lista de etiquetas
    all_bboxes_rgb = []

    # la idea es poner algas solo en el 80% de imagenes
    if random.random() < 0.80:
        con_bloom += 1
        
        # separacion tierra agua
        hsv_fondo = cv2.cvtColor(img_fondo_rgb, cv2.COLOR_BGR2HSV)
        # de este color se va a entender el agua
        lower_agua = np.array([29, 101, 96])
        upper_agua = np.array([111, 229, 171])
        
        mask_agua_segmentada = cv2.inRange(hsv_fondo, lower_agua, upper_agua)
        kernel_clean = np.ones((15, 15), np.uint8)
        
        mask_sin_puntos_blancos = cv2.morphologyEx(mask_agua_segmentada, cv2.MORPH_OPEN, kernel_clean)
        mask_agua_valid_area = cv2.morphologyEx(mask_sin_puntos_blancos, cv2.MORPH_CLOSE, kernel_clean)

        num_manchas = random.randint(8, 18) 
        mascara_maestra_rgb = np.zeros((h_rgb, w_rgb), dtype=np.uint8)

        # preparacion textura RGB, se pasan las imagenes a hls y se calcula el tono medio de las mismas
        hls_fondo_rgb = cv2.cvtColor(img_fondo_rgb, cv2.COLOR_BGR2HLS)
        img_textura_elegida_rgb = random.choice(texturas_cargadas)
        img_textura_rgb = cv2.resize(img_textura_elegida_rgb, (w_rgb, h_rgb), interpolation=cv2.INTER_AREA)
        hls_textura_rgb = cv2.cvtColor(img_textura_rgb, cv2.COLOR_BGR2HLS)

        # se hace un redimensionamiento de la textura
        bg_h_avg = np.mean(hls_fondo_rgb[:, :, 0])
        text_h_avg = np.mean(hls_textura_rgb[:, :, 0])

        # se saca un tono medio de luminosisdad y color y esto se le resta al fondo original en un 80% y 40%
        hls_textura_rgb[:, :, 0] = np.mod(hls_textura_rgb[:, :, 0].astype(float) + (bg_h_avg - text_h_avg) * 0.8, 180).astype(np.uint8)
        bg_l_avg = np.mean(hls_fondo_rgb[:, :, 1])
        text_l_avg = np.mean(hls_textura_rgb[:, :, 1])
        hls_textura_rgb[:, :, 1] = np.clip(hls_textura_rgb[:, :, 1].astype(float) + (bg_l_avg - text_l_avg) * 0.4, 0, 255).astype(np.uint8)
        img_textura_rgb_corregida = cv2.cvtColor(hls_textura_rgb, cv2.COLOR_HLS2BGR)
        # el objetivo es que la textura que se esta usando se integre en el fondo

        # se ponen num_manchas, se intenta hacer en coordenadas aleatorias fuera de tierra
        for _ in range(num_manchas):
            reintentos = 0
            centro_valido = False
            while reintentos < 50 and not centro_valido:
                random_x = random.randint(int(w_rgb*0.05), int(w_rgb*0.95))
                random_y = random.randint(int(h_rgb*0.05), int(h_rgb*0.95))
                # comprobacion de mascara de tierra
                if mask_agua_valid_area[random_y, random_x] == 255:
                    centro_valido = True
                else:
                    reintentos += 1

            # si el centro propuesto no es valido, por ser tierra, se continua con otro centro
            if not centro_valido: continue

            # generacion en RGB
            # mascara para poder dibujar
            mascara_temp = np.zeros((h_rgb, w_rgb), dtype=np.uint8)
            # tamaño que tendra
            radio_base = random.randint(int(w_rgb*0.015), int(w_rgb*0.04)) 
            if radio_base < 2: radio_base = 2

            num_vertices = random.randint(8, 15) 
            puntos_rgb = []

            # utiliza trigonometria para calcular una forma relativamente organica para el bloom
            for i in range(num_vertices):
                angulo = (i / num_vertices) * 2 * math.pi
                radio_deformado = radio_base * random.uniform(0.5, 1.5)
                px = int(random_x + radio_deformado * math.cos(angulo))
                py = int(random_y + radio_deformado * math.sin(angulo))
                puntos_rgb.append([px, py])
                
            puntos_np = np.array(puntos_rgb, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mascara_temp, [puntos_np], 255)
            
            # si el bloom entra en colision con la tierra, se eliminan parte de la misma para que no haya colision
            # el resto se mantiene
            mascara_temp = cv2.bitwise_and(mascara_temp, mask_agua_valid_area)
            x, y, w, h = cv2.boundingRect(mascara_temp)
            
            if w > 5 and h > 5:
                # se guardan las etiquetas solo para RGB y las cajas a tener en cuenta
                yolo_x = (x + w / 2) / w_rgb
                yolo_y = (y + h / 2) / h_rgb
                yolo_w = w / w_rgb
                yolo_h = h / h_rgb
                all_bboxes_rgb.append(f"0 {yolo_x:.6f} {yolo_y:.6f} {yolo_w:.6f} {yolo_h:.6f}")
                mascara_maestra_rgb = cv2.bitwise_or(mascara_maestra_rgb, mascara_temp)

        # incorpora la mancha con desenfoque
        if len(all_bboxes_rgb) > 0:
            k_size_rgb = int(w_rgb * 0.05) | 1 
            mascara_maestra_rgb = cv2.GaussianBlur(mascara_maestra_rgb, (k_size_rgb, k_size_rgb), 0)
            mascara_maestra_rgb = cv2.bitwise_and(mascara_maestra_rgb, mask_agua_valid_area)

            mascara_float_rgb = (mascara_maestra_rgb.astype(float) / 255.0) * random.uniform(0.50, 0.70)
            mascara_float_rgb = np.stack([mascara_float_rgb]*3, axis=-1) 

            img_final_rgb = ((img_textura_rgb_corregida.astype(float) * mascara_float_rgb) + (img_fondo_rgb.astype(float) * (1.0 - mascara_float_rgb))).astype(np.uint8)

        cv2.imwrite(os.path.join(CARPETA_SALIDA_RGB, 'images', nombre_foto), img_final_rgb)

        # Se guardan las etiquetas YOLO
        with open(ruta_lbl_rgb, 'w') as f:
            for bbox in all_bboxes_rgb: f.write(bbox + '\n')

    else:
        # fotos limpias sin blooms se guardan
        sin_bloom += 1
        cv2.imwrite(os.path.join(CARPETA_SALIDA_RGB, 'images', nombre_foto), img_fondo_rgb)
        open(ruta_lbl_rgb, 'w').close()

print(f"Fotos infectadas: {con_bloom} | Fotos limpias: {sin_bloom}")
