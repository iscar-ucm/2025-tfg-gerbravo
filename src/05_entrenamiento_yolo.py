from ultralytics import YOLO
import torch
import tkinter as tk
from tkinter import filedialog

torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()

    RUTA_YAML = filedialog.askopenfilename(title="Selecciona el archivo data.yaml de tu dataset", filetypes=[("Archivos YAML", "*.yaml"), ("Todos los archivos", "*.*")])
    CARPETA_PROYECTO = filedialog.askdirectory(title="Selecciona la carpeta donde se guardarán los resultados del proyecto (ej. resultados/fase2)")

    model = YOLO('yolov8s.pt') 

    print("<--- Iniciando el entrenamiento --->")
    #entrenamiento
    resultados = model.train(
        data=RUTA_YAML, 
        epochs=300,       
        batch=8,          
        imgsz=640, 
        device=0,
        workers=2,
        flipud=0.5,       
        fliplr=0.5,       
        hsv_s=0.5,        
        hsv_v=0.5,
        
        project=CARPETA_PROYECTO,
        name='entrenamiento_yolo' #nombre de la carpeta que se va a crear dentro de fase2
    )
    print("Fin del entrenamiento")
