import cv2
import numpy as np
import os
import random
import math
import csv
from tkinter import filedialog, Tk, messagebox

root = Tk()
root.withdraw()
root.attributes('-topmost', True)

def desplazar_coordenada(coord_str):
    try:
        coord = float(coord_str.replace(',', '.'))
        return round(coord + random.uniform(-0.0004, 0.0004), 7)
    except:
        return coord_str

print("<--- INICIANDO SIMULADOR FRANKENSTEIN (VISUAL + LÓGICO) --->")

# Selección de archivos
TEXTURAS = filedialog.askopenfilenames(
    title="0. Selecciona las FOTOS DE TEXTURAS (fake1, fake2...)",
    filetypes=[("Imágenes", "*.jpg *.png *.jpeg")]
)
if not TEXTURAS:
    print("Operación cancelada: No se seleccionaron texturas.")
    exit()

CSV_ENTRADA = filedialog.askopenfilename(title="1. Selecciona el CSV de predicciones (5 puntos)")
CARPETA_FOTOS = filedialog.askdirectory(title="2. Selecciona carpeta con 5 fotos LIMPIAS")
CARPETA_SALIDA = filedialog.askdirectory(title="3. Selecciona carpeta para los RESULTADOS")

if not (CSV_ENTRADA and CARPETA_FOTOS and CARPETA_SALIDA):
    print("Operación cancelada.")
    exit()

# se cargan las texturas
texturas_cargadas = [cv2.imread(r) for r in TEXTURAS if cv2.imread(r) is not None]
lista_fotos = sorted([f for f in os.listdir(CARPETA_FOTOS) if f.lower().endswith(('.jpg', '.png'))])[:5]

# se lee el CSV
with open(CSV_ENTRADA, mode='r', encoding='utf-8-sig') as f:
    linea = f.readline()
    delim = ';' if ';' in linea else ','
    f.seek(0)
    lector = csv.DictReader(f, delimiter=delim)
    lector.fieldnames = [n.strip() for n in lector.fieldnames]
    filas_csv = list(lector)


# 3 variables predictivos (índices 0, 1, 2) -> 1 exacto, 1 desplazado, y 1 fallo del predictor (0 blooms)
reglas_variables = [
    {"tipo": "Variable (Predictor)", "blooms": 1, "shift": False},
    {"tipo": "Variable (Predictor)", "blooms": 1, "shift": True},
    {"tipo": "Variable (Predictor)", "blooms": 0, "shift": False}
]
random.shuffle(reglas_variables)

# 2 fijos de zodiac (índices 3, 4) -> B >= 0, al azar (0, 1 o 2 blooms)
reglas_fijos = [
    {"tipo": "Fijo (Zodiac)", "blooms": random.choice([0, 1, 2]), "shift": False},
    {"tipo": "Fijo (Zodiac)", "blooms": random.choice([0, 1, 2]), "shift": False}
]

# se unen ambas listas para que los índices 0,1,2 sean variables y 3,4 sean fijos
reglas = reglas_variables + reglas_fijos


resultados_csv = []

for i in range(5):
    fila = filas_csv[i]
    foto_nombre = lista_fotos[i]
    regla = reglas[i]
    
    img = cv2.imread(os.path.join(CARPETA_FOTOS, foto_nombre))
    h, w = img.shape[:2]
    img_final = img.copy()
    
    if regla["blooms"] > 0:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([29, 101, 96]), np.array([111, 229, 171]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        
        hls_f = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        bg_h, bg_l = np.mean(hls_f[:,:,0]), np.mean(hls_f[:,:,1])
        
        mascara_maestra = np.zeros((h, w), dtype=np.uint8)
        
        for _ in range(regla["blooms"]):
            #textura y color
            txt = cv2.resize(random.choice(texturas_cargadas), (w, h))
            hls_t = cv2.cvtColor(txt, cv2.COLOR_BGR2HLS)
            hls_t[:,:,0] = np.mod(hls_t[:,:,0].astype(float) + (bg_h - np.mean(hls_t[:,:,0])) * 0.8, 180)
            hls_t[:,:,1] = np.clip(hls_t[:,:,1].astype(float) + (bg_l - np.mean(hls_t[:,:,1])) * 0.4, 0, 255)
            txt_corr = cv2.cvtColor(hls_t.astype(np.uint8), cv2.COLOR_HLS2BGR)
            
            rx, ry = random.randint(int(w*0.1), int(w*0.9)), random.randint(int(h*0.1), int(h*0.9))
            if mask[ry, rx] == 255:
                pts = []
                rad = random.randint(int(w*0.02), int(w*0.05))
                for n in range(random.randint(8, 12)):
                    a = (n/10)*2*math.pi
                    r = rad * random.uniform(0.6, 1.4)
                    pts.append([int(rx + r*math.cos(a)), int(ry + r*math.sin(a))])
                
                m_tmp = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(m_tmp, [np.array(pts, np.int32)], 255)
                mascara_maestra = cv2.bitwise_or(mascara_maestra, cv2.bitwise_and(m_tmp, mask))
        
        if np.any(mascara_maestra):
            blur = cv2.GaussianBlur(mascara_maestra, (int(w*0.05)|1, int(w*0.05)|1), 0)
            alpha = (cv2.bitwise_and(blur, mask).astype(float)/255.0) * random.uniform(0.5, 0.7)
            alpha = np.stack([alpha]*3, axis=-1)
            img_final = (txt_corr.astype(float)*alpha + img.astype(float)*(1-alpha)).astype(np.uint8)

    # coordenadas y medida
    lat = desplazar_coordenada(fila['lat']) if regla["shift"] else fila['lat']
    lon = desplazar_coordenada(fila['lon']) if regla["shift"] else fila['lon']
    medida = round(random.uniform(0.4, 0.95), 2) if regla["blooms"] > 0 else 0.0
    
    # se guardan los resultados
    cv2.imwrite(os.path.join(CARPETA_SALIDA, f"sim_{foto_nombre}"), img_final)
    resultados_csv.append({
        'Fotograma': f"sim_{foto_nombre}",
        'Tipo': regla["tipo"],
        'Latitud': lat,
        'Longitud': lon,
        'Medida_IA': medida,
        'Observacion': "Encontrado (Desplazado)" if regla["shift"] else ("Encontrado (Exacto)" if medida > 0 else "Agua limpia")
    })
    print(f"Procesada: {foto_nombre} -> {regla['tipo']}")

# se guarda el CSV
with open(os.path.join(CARPETA_SALIDA, "feedback_mision.csv"), 'w', newline='', encoding='utf-8') as f:
    escritor = csv.DictWriter(f, fieldnames=['Fotograma', 'Tipo', 'Latitud', 'Longitud', 'Medida_IA', 'Observacion'], delimiter=';')
    escritor.writeheader()
    escritor.writerows(resultados_csv)

messagebox.showinfo("Fin", "Misión simulada. Revisa la carpeta de resultados.")
