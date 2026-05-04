from ultralytics import YOLO
import torch

torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    model = YOLO('yolov8s.pt') 

    print("<--- Iniciando el entrenamiento --->")
    #entrenamiento
    resultados = model.train(
        data=r'C:\Users\GERMAN\Desktop\universidad\5_quinto\Trabajo_de_Fin_de_Grado\entrenamiento\dataset_tfg\data.yaml', 
        epochs=300,       
        batch=8,          
        imgsz=640, 
        device=0,
        workers=2,
        flipud=0.5,       
        fliplr=0.5,       
        hsv_s=0.5,        
        hsv_v=0.5,
        
        project=r'C:\Users\GERMAN\Desktop\universidad\5_quinto\Trabajo_de_Fin_de_Grado\resultados\fase2',
        name='entrenamiento_yolo' #nombre de la carpeta que se va a crear dentro de fase2
    )
    print("Fin del entrenamiento")