import math
import itertools
import time
import sys
import os
import tkinter as tk
from tkinter import filedialog

class TspDron:
    def __init__(self, points):
        self.points = points
        self.N = len(points)
        self.R = 6371000.0 
        
        # NUEVO: Matriz de distancias NxN inicializada a 0
        self.matriz_distancias = [[0.0 for _ in range(self.N)] for _ in range(self.N)]
        self._precalcular_matriz() # Rellenamos la tabla nada más arrancar
        
    def _haversine(self, p1, p2):
        """Calcula la fórmula matemáticamente pesada una sola vez por par de puntos"""
        lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
        lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        return 2 * self.R * math.asin(math.sqrt(a))

    def _precalcular_matriz(self):
        """Rellena la tabla de distancias para no tener que calcularlas nunca más"""
        for i in range(self.N):
            for j in range(i + 1, self.N):
                dist = self._haversine(self.points[i], self.points[j])
                self.matriz_distancias[i][j] = dist
                self.matriz_distancias[j][i] = dist # La distancia A->B es igual a B->A

    def _distancia_ruta(self, ruta):
        """Suma las distancias de una ruta completa consultando la matriz (Acceso O(1))"""
        dist_total = 0.0
        for i in range(len(ruta) - 1):
            dist_total += self.matriz_distancias[ruta[i]][ruta[i+1]]
        dist_total += self.matriz_distancias[ruta[-1]][ruta[0]] 
        return dist_total

    def _solBruta(self):
        minDist = float('inf')
        caminoMejor = None
        for path in itertools.permutations(range(self.N)):
            distAct = self._distancia_ruta(path)
            if distAct < minDist:
                minDist = distAct
                caminoMejor = path
        return caminoMejor, minDist
    
    def _solHeuristica(self):
        no_visitados = set(range(1, self.N))
        path = [0]
        nodo_actual = 0

        # Vecino más cercano buscando en la matriz
        while no_visitados:
            siguiente_nodo = min(no_visitados, key=lambda nodo: self.matriz_distancias[nodo_actual][nodo])
            path.append(siguiente_nodo)
            no_visitados.remove(siguiente_nodo)
            nodo_actual = siguiente_nodo
        
        # 2-opt optimizado
        mejor_camino = path
        mejorado = True

        while mejorado:
            mejorado = False
            for i in range(1, len(mejor_camino) - 1):
                for j in range(i + 1, len(mejor_camino)):
                    nuevo_camino = mejor_camino[:i] + mejor_camino[i:j][::-1] + mejor_camino[j:]
                    if self._distancia_ruta(nuevo_camino) < self._distancia_ruta(mejor_camino):
                        mejor_camino = nuevo_camino
                        mejorado = True

        distTotal = self._distancia_ruta(mejor_camino)
        return tuple(mejor_camino), distTotal
        
    def resolver(self):
        if self.N <= 1:
            return "Solucion directa", (tuple(range(self.N)), 0.0)
        elif self.N <= 8:
            return "Fuerza bruta (Óptima)", self._solBruta()
        else:
            return "Heurística (2-opt)", self._solHeuristica()
        
def leer_coordenadas_archivo(ruta_archivo):
    coords = []
    try:
        with open(ruta_archivo, 'r', encoding='utf-8') as f:
            for linea in f:
                linea = linea.strip()
                #si hay algun comentario o linea sin nada nos lo saltamos
                if not linea or linea.startswith('#'): 
                    continue
                #se fragmenta la linea usando la , como separador
                partes = linea.split(',')
                if len(partes) >= 2:
                    #se convierten ambos fragmentos en float
                    coords.append((float(partes[0].strip()), float(partes[1].strip())))
    except Exception as e:
        print(f"Error al leer el archivo: {e}")
        sys.exit(1)
    return coords

if __name__ == "__main__":
    # Ocultar la ventana principal de tkinter para que solo salga el cuadro de diálogo
    root = tk.Tk()
    root.withdraw()

    print("Por favor, selecciona el archivo de coordenadas (.txt) en la ventana emergente...")
    
    # Abrir ventana de diálogo para seleccionar el archivo
    ruta_archivo = filedialog.askopenfilename(
        title="Selecciona el archivo de coordenadas (.txt)",
        filetypes=[("Archivos de texto", "*.txt"), ("Todos los archivos", "*.*")]
    )

    # Comprobar si el usuario canceló la selección
    if not ruta_archivo:
        print("\nERROR: No se seleccionó ningún archivo. Operación cancelada.\n")
        sys.exit(1)

    # Leer las coordenadas del archivo seleccionado
    coord_usuario = leer_coordenadas_archivo(ruta_archivo)

    if not coord_usuario:
        print("\nERROR: El archivo seleccionado está vacío o tiene un formato incorrecto.\n")
        sys.exit(1)

    print(f"\n<--- Ruta para {len(coord_usuario)} puntos --->")
    #se hace el calculo
    tsp = TspDron(coord_usuario)
    
    #lineas encargadas de cronometrar lo que tarda en ejecutarse el algoritmo y de ponerlo en marcha
    inicio = time.perf_counter()
    metodo, (ruta_indices, dist) = tsp.resolver()
    fin = time.perf_counter()

    # se escribe por consola la ruta calculada
    ruta_visual = " -> ".join(str(nodo) for nodo in ruta_indices) + f" -> {ruta_indices[0]}"

    #informacion de la ejecucion
    print(f"Método: {metodo}")
    print(f"Orden: {ruta_visual}")
    print(f"Tiempo: {(fin - inicio)*1000:.3f} ms")
    print(f"Distancia recorrida: {dist:.1f} metros")
    
    CARPETA_RESULTADOS = r'C:\Users\GERMAN\Desktop\universidad\5_quinto\Trabajo_de_Fin_de_Grado\resultados\fase0'
    if not os.path.exists(CARPETA_RESULTADOS):
        os.makedirs(CARPETA_RESULTADOS)
    
    # Al eliminar argparse, fijamos el nombre de salida por defecto
    nombre_salida = 'ruta_optimizada.csv'
    ruta_salida = os.path.join(CARPETA_RESULTADOS, nombre_salida)

    #guardado del resultado en un archivo CSV
    try:
        with open(ruta_salida, 'w', encoding='utf-8') as f:
            f.write(f"# Metodo: {metodo} | Distancia: {dist:.1f}m\n")
            f.write("latitud,longitud,orden\n")
            
            #guardamos las coordenadas ya en el orden en el que el dron tiene que hacer el recorrido
            for orden_secuencia, indice_original in enumerate(ruta_indices):
                lat, lon = coord_usuario[indice_original]
                f.write(f"{lat},{lon},{orden_secuencia}\n")
                
            #se añade la vuelta al primer punto
            lat_ini, lon_ini = coord_usuario[ruta_indices[0]]
            f.write(f"{lat_ini},{lon_ini},RETORNO\n")
            
        print(f"\nLa ruta se ha guardado en: {ruta_salida}")
    except Exception as e:
        print(f"\n[ERROR] No se pudo guardar el archivo de salida: {e}")