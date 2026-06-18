"""
============================================================
Script 03 - Pruebas e Inferencia del Modelo
Proyecto: Detección de Tornillos con YOLOv8
============================================================

Ejecutar con:
    python scripts/03_pruebas.py

    # Modo imagen única:
    python scripts/03_pruebas.py --imagen ruta/imagen.jpg

    # Modo carpeta de imágenes:
    python scripts/03_pruebas.py --carpeta dataset/images/test/

    # Modo cámara en tiempo real:
    python scripts/03_pruebas.py --camara
"""

# 'argparse' permite leer argumentos desde la línea de comandos,
# como --imagen, --carpeta, --camara, --test.
import argparse

# 'json' permite guardar los resultados de la inferencia en formato
# estructurado, fácil de leer por otros programas (ej. un dashboard).
import json

# 'time' se usa para medir cuánto tarda cada inferencia (en milisegundos).
import time

# 'Path' representa rutas de archivos/carpetas como objetos.
from pathlib import Path

# Clase principal de Ultralytics para cargar modelos YOLO y hacer
# predicciones (inferencia) o evaluaciones.
from ultralytics import YOLO


# ============================================================
# CONFIGURACIÓN
# ============================================================

# Ruta donde se guardó el mejor modelo entrenado (generado en el script 02).
MODELO_PATH = "models/mejor_modelo.pt"

# Umbral mínimo de confianza para aceptar una detección.
# Si el modelo "cree" con menos de 50% de certeza que algo es un tornillo,
# esa detección se descarta. Subir este valor reduce falsos positivos,
# pero puede aumentar falsos negativos (tornillos no detectados).
CONFIANZA_MIN = 0.50

# Umbral de IoU (Intersection over Union) para el algoritmo NMS
# (Non-Maximum Suppression). Cuando el modelo genera varias cajas
# superpuestas para el mismo objeto, NMS elimina las redundantes.
# Un IoU más alto permite que coexistan cajas más superpuestas.
IOU_THRESHOLD = 0.45

# Si es True, las imágenes con las detecciones dibujadas se guardan en disco.
GUARDAR_IMGS = True

# Carpeta donde se guardarán las imágenes anotadas y los reportes.
OUTPUT_DIR = "results/inferencias"

# Diccionario que traduce el ID numérico de clase a su nombre legible.
# Debe coincidir EXACTAMENTE con el orden definido en dataset/data.yaml.
CLASES = {
    0: "tornillo_presente",
    1: "tornillo_ausente",
}

# Colores (en formato BGR, que es el que usa OpenCV) para dibujar
# cada clase de forma distinta y fácil de identificar visualmente.
COLORES = {
    0: (0, 200, 0),    # verde  -> tornillo presente (correcto) ✅
    1: (0, 0, 220),    # rojo   -> tornillo ausente (defecto)  ❌
}


# ============================================================
# FUNCIONES
# ============================================================

def cargar_modelo():
    """Carga el modelo entrenado desde disco."""
    # Convertimos la ruta (string) en un objeto Path para poder
    # verificar fácilmente si el archivo existe.
    ruta = Path(MODELO_PATH)

    # Si el archivo de pesos no existe, no tiene sentido continuar:
    # lanzamos un error explicando qué hacer.
    if not ruta.exists():
        raise FileNotFoundError(
            f"❌ Modelo no encontrado en {ruta}\n"
            "   Ejecuta primero: python scripts/02_entrenamiento.py"
        )

    # Cargamos el modelo YOLO desde el archivo .pt (pesos entrenados).
    # str(ruta) convierte el objeto Path de vuelta a texto, ya que
    # YOLO() espera un string como argumento.
    model = YOLO(str(ruta))

    print(f"✅ Modelo cargado: {ruta}")

    # Retornamos el objeto modelo, ya listo para hacer predicciones.
    return model


def inferencia_imagen(model, ruta_imagen):
    """Ejecuta la detección sobre una imagen y retorna los resultados."""
    # Guardamos el momento exacto antes de iniciar la predicción,
    # para poder calcular cuánto tiempo tomó después.
    inicio = time.time()

    # model.predict() ejecuta la inferencia (forward pass) sobre la imagen
    # indicada y retorna una lista de objetos "Results" (uno por imagen).
    resultados = model.predict(
        source   = str(ruta_imagen),  # ruta de la imagen a analizar
        conf     = CONFIANZA_MIN,     # umbral mínimo de confianza
        iou      = IOU_THRESHOLD,     # umbral IoU para NMS
        save     = GUARDAR_IMGS,      # si se guarda la imagen anotada
        project  = OUTPUT_DIR,        # carpeta raíz de salida
        name     = "detecciones",     # subcarpeta específica
        exist_ok = True,               # no fallar si la carpeta ya existe
        verbose  = False,              # no imprimir logs internos de Ultralytics
    )

    # Calculamos el tiempo transcurrido y lo convertimos a milisegundos
    # (multiplicamos por 1000, ya que time.time() devuelve segundos).
    tiempo_ms = (time.time() - inicio) * 1000

    # Retornamos tanto la lista de resultados como el tiempo que tomó.
    return resultados, tiempo_ms


def analizar_resultado(resultado, nombre_archivo="imagen"):
    """Analiza los resultados de detección y retorna un resumen."""
    # "boxes" contiene todas las cajas (bounding boxes) detectadas
    # en esta imagen, junto con su clase y nivel de confianza.
    boxes = resultado.boxes

    # Si no hay cajas detectadas (boxes es None o tiene longitud 0):
    if boxes is None or len(boxes) == 0:
        print(f"   [{nombre_archivo}]  Sin detecciones.")
        # Retornamos un diccionario indicando que no se detectó nada.
        return {"archivo": nombre_archivo, "detecciones": []}

    # Lista donde guardaremos el detalle de cada detección individual.
    detecciones = []

    # enumerate(boxes) nos da el índice (i) y la caja (box) en cada vuelta.
    for i, box in enumerate(boxes):
        # box.cls[0] es un tensor con el ID de clase predicho; int() lo
        # convierte a un número entero normal de Python.
        clase_id = int(box.cls[0])

        # box.conf[0] es el nivel de confianza (0.0 a 1.0) de esta detección;
        # float() lo convierte a un número decimal normal.
        confianza = float(box.conf[0])

        # Buscamos el nombre legible de la clase en el diccionario CLASES.
        # Si por alguna razón el id no está registrado, usamos un nombre genérico.
        nombre_cls = CLASES.get(clase_id, f"clase_{clase_id}")

        # box.xyxy[0] contiene las coordenadas de la caja en formato
        # [x1, y1, x2, y2] (esquina superior izquierda y esquina inferior
        # derecha). .tolist() convierte el tensor a una lista normal de Python.
        coords = box.xyxy[0].tolist()

        # Agregamos un diccionario con toda la información de esta detección
        # a la lista "detecciones".
        detecciones.append({
            "id":        i + 1,                              # número secuencial (1, 2, 3...)
            "clase_id":  clase_id,                            # 0 o 1
            "clase":     nombre_cls,                          # nombre legible
            "confianza": round(confianza, 4),                 # redondeado a 4 decimales
            "bbox":      [round(c, 1) for c in coords],       # coordenadas redondeadas a 1 decimal
        })

    # Contamos cuántas detecciones son de tornillo PRESENTE (clase 0).
    presentes = sum(1 for d in detecciones if d["clase_id"] == 0)

    # Contamos cuántas detecciones son de tornillo AUSENTE (clase 1).
    ausentes = sum(1 for d in detecciones if d["clase_id"] == 1)

    # Si NO hay ningún tornillo ausente, la pieza se considera aprobada (OK).
    # Si hay al menos uno ausente, se marca como fallo, indicando cuántos faltan.
    estado = "✅ OK" if ausentes == 0 else f"❌ FALLO — {ausentes} tornillo(s) faltante(s)"

    # Mostramos en consola un resumen de esta imagen.
    print(f"   [{nombre_archivo}]")
    print(f"     Tornillos detectados: {presentes} presentes  |  {ausentes} ausentes")
    print(f"     Estado de la pieza:  {estado}")

    # Retornamos un diccionario con toda la información relevante de
    # esta imagen, que luego se usará para generar el reporte JSON final.
    return {
        "archivo":     nombre_archivo,
        "presentes":   presentes,
        "ausentes":    ausentes,
        "estado":      "ok" if ausentes == 0 else "fallo",
        "detecciones": detecciones,
    }


def evaluar_en_test(model):
    """Evalúa el modelo sobre el conjunto de prueba y calcula métricas."""
    print("\n📊 Evaluando sobre dataset de prueba...")

    # model.val() ejecuta una evaluación formal del modelo sobre un split
    # específico del dataset (en este caso "test"), calculando métricas
    # como precisión, recall y mAP de forma automática.
    metricas = model.val(
        data    = "dataset/data.yaml",  # configuración del dataset
        split   = "test",                # usamos el conjunto de prueba (no train/val)
        conf    = CONFIANZA_MIN,         # mismo umbral de confianza que en inferencia
        iou     = IOU_THRESHOLD,         # mismo umbral de IoU
        verbose = True,                  # mostrar el detalle del proceso en consola
    )

    # Retornamos el objeto de métricas para uso posterior si se necesita.
    return metricas


def guardar_reporte(resumen_inferencias):
    """Guarda un reporte JSON con todos los resultados."""
    # Creamos la carpeta de salida si no existe (incluyendo carpetas padre).
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Definimos la ruta completa del archivo de reporte.
    ruta_reporte = Path(OUTPUT_DIR) / "reporte_inferencia.json"

    # Contamos cuántas imágenes se analizaron en total.
    total = len(resumen_inferencias)

    # Contamos cuántas de esas imágenes tuvieron estado "ok"
    # (es decir, todos los tornillos estaban presentes).
    ok = sum(1 for r in resumen_inferencias if r.get("estado") == "ok")

    # Las que no fueron "ok" se consideran fallos (al menos un tornillo ausente).
    fallos = total - ok

    # Calculamos el porcentaje de piezas aprobadas. Evitamos división por
    # cero comprobando que "total" sea mayor que 0.
    tasa_ok = (ok / total * 100) if total > 0 else 0

    # Construimos el diccionario final que se convertirá a JSON.
    reporte = {
        "resumen": {
            "total_imagenes":   total,
            "piezas_ok":        ok,
            "piezas_con_fallo": fallos,
            "tasa_aprobacion":  round(tasa_ok, 2),
        },
        "detalle": resumen_inferencias,  # lista completa con el detalle de cada imagen
    }

    # Abrimos el archivo en modo escritura, con codificación UTF-8
    # (necesaria para que los emojis y acentos se guarden correctamente).
    with open(ruta_reporte, "w", encoding="utf-8") as f:
        # json.dump() escribe el diccionario "reporte" en el archivo "f".
        # indent=2 hace que el JSON sea legible (con sangría).
        # ensure_ascii=False permite guardar caracteres especiales (á, é, ✅...)
        # sin convertirlos en códigos de escape.
        json.dump(reporte, f, indent=2, ensure_ascii=False)

    print(f"\n📄 Reporte guardado en: {ruta_reporte}")
    print(f"   Tasa de aprobación: {tasa_ok:.1f}%  ({ok}/{total} piezas OK)")


def modo_camara(model):
    """Ejecuta detección en tiempo real usando la cámara web."""
    # Intentamos importar OpenCV solo dentro de esta función, ya que es
    # la única que lo necesita (así el resto del script funciona aunque
    # OpenCV no esté instalado).
    try:
        import cv2
    except ImportError:
        # Si la librería no está instalada, avisamos y salimos de la función.
        print("❌ OpenCV no instalado. Ejecuta: pip install opencv-python")
        return

    print("\n📷 Iniciando cámara... (presiona 'q' para salir)")

    # cv2.VideoCapture(0) abre la cámara con índice 0 (normalmente la
    # cámara principal/integrada del equipo).
    cap = cv2.VideoCapture(0)

    # Verificamos si la cámara se pudo abrir correctamente.
    if not cap.isOpened():
        print("❌ No se pudo acceder a la cámara.")
        return

    # Bucle infinito: se ejecuta continuamente hasta que el usuario
    # presione la tecla 'q' (ver más abajo).
    while True:
        # cap.read() captura un solo fotograma (frame) de la cámara.
        # Retorna "ret" (True si se leyó correctamente) y "frame"
        # (la imagen capturada, como un array de NumPy).
        ret, frame = cap.read()

        # Si no se pudo leer el frame (ej. cámara desconectada), salimos del bucle.
        if not ret:
            break

        # Ejecutamos la predicción del modelo directamente sobre el frame
        # capturado (sin necesidad de guardarlo primero en disco).
        resultados = model.predict(frame, conf=CONFIANZA_MIN, verbose=False)

        # resultados[0].plot() genera una nueva imagen con las cajas de
        # detección, etiquetas y confianza ya dibujadas encima del frame original.
        frame_anotado = resultados[0].plot()

        # Extraemos las cajas detectadas en este frame.
        boxes = resultados[0].boxes

        # Contamos cuántas detecciones son de tornillo ausente (clase 1).
        # Si "boxes" es None o está vacío, el resultado es 0.
        ausentes = sum(1 for b in boxes if int(b.cls[0]) == 1) if boxes else 0

        # Definimos el texto y color a mostrar según si hay fallos o no.
        texto = "PIEZA OK" if ausentes == 0 else f"FALLO: {ausentes} tornillo(s) faltante(s)"
        color = (0, 200, 0) if ausentes == 0 else (0, 0, 220)

        # cv2.putText() dibuja texto sobre la imagen:
        # - frame_anotado: la imagen donde se dibuja
        # - texto: el contenido a mostrar
        # - (10, 30): posición (x, y) en píxeles desde la esquina superior izq.
        # - cv2.FONT_HERSHEY_SIMPLEX: tipo de fuente
        # - 1.0: tamaño de la fuente (escala)
        # - color: color del texto en formato BGR
        # - 2: grosor de línea del texto
        cv2.putText(frame_anotado, texto, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        # cv2.imshow() abre/actualiza una ventana mostrando la imagen anotada.
        cv2.imshow("Deteccion de Tornillos - YOLOv8", frame_anotado)

        # cv2.waitKey(1) espera 1 milisegundo por una tecla presionada.
        # & 0xFF asegura compatibilidad entre sistemas operativos.
        # Si la tecla presionada es 'q', salimos del bucle (rompemos el while).
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Liberamos la cámara para que otros programas puedan usarla.
    cap.release()

    # Cerramos todas las ventanas de OpenCV que se hayan abierto.
    cv2.destroyAllWindows()

    print("✅ Cámara cerrada.")


# ============================================================
# EJECUCIÓN PRINCIPAL
# ============================================================

def main():
    """Función principal: interpreta los argumentos y ejecuta el modo correspondiente."""
    # Creamos el "parser" de argumentos, con una descripción que se muestra
    # si el usuario ejecuta el script con --help.
    parser = argparse.ArgumentParser(description="Inferencia - Detección de Tornillos YOLOv8")

    # Definimos cada argumento opcional que el script puede recibir:

    # --imagen: ruta a una sola imagen para analizar.
    parser.add_argument("--imagen", type=str, help="Ruta a una imagen individual")

    # --carpeta: ruta a una carpeta con varias imágenes para analizar todas.
    parser.add_argument("--carpeta", type=str, help="Ruta a una carpeta de imágenes")

    # --camara: si se incluye esta bandera (sin valor), activa el modo cámara.
    # action="store_true" significa que su valor será True si está presente,
    # y False si no se incluye en el comando.
    parser.add_argument("--camara", action="store_true", help="Usar cámara en tiempo real")

    # --test: si se incluye, evalúa el modelo formalmente sobre el split "test".
    parser.add_argument("--test", action="store_true", help="Evaluar sobre dataset de prueba")

    # parser.parse_args() lee los argumentos reales que el usuario escribió
    # en la terminal y los guarda en el objeto "args".
    args = parser.parse_args()

    print("=" * 55)
    print("  PRUEBAS E INFERENCIA - Detección de Tornillos")
    print("=" * 55)

    # Cargamos el modelo entrenado (necesario para cualquier modo).
    model = cargar_modelo()

    # Lista donde acumularemos el resumen de cada imagen analizada
    # (solo se usa en los modos --imagen y --carpeta).
    resumen = []

    # Evaluamos qué modo fue solicitado por el usuario, en orden de prioridad.

    if args.camara:
        # Si se pidió modo cámara, ejecutamos esa función y terminamos
        # (no genera "resumen" porque es un flujo continuo, no por imagen).
        modo_camara(model)

    elif args.test:
        # Si se pidió evaluación formal sobre el dataset de test.
        evaluar_en_test(model)

    elif args.imagen:
        # Si se especificó una imagen individual:
        ruta = Path(args.imagen)

        # Verificamos que el archivo realmente exista antes de procesarlo.
        if not ruta.exists():
            print(f"❌ No se encontró la imagen: {ruta}")
            return

        print(f"\n🔍 Analizando imagen: {ruta.name}")

        # Ejecutamos la inferencia sobre esa única imagen.
        resultados, ms = inferencia_imagen(model, ruta)

        # Analizamos el primer (y único) resultado retornado.
        r = analizar_resultado(resultados[0], ruta.name)

        # Mostramos cuánto tardó la inferencia, en milisegundos.
        print(f"     Tiempo de inferencia: {ms:.1f} ms")

        # Agregamos el resultado a la lista de resumen.
        resumen.append(r)

    elif args.carpeta:
        # Si se especificó una carpeta completa de imágenes:
        carpeta = Path(args.carpeta)

        # Mismo conjunto de extensiones válidas que en el script de preprocesamiento.
        extensiones = {".jpg", ".jpeg", ".png", ".bmp"}

        # Filtramos solo los archivos cuya extensión sea una imagen válida.
        imagenes = [f for f in carpeta.iterdir() if f.suffix.lower() in extensiones]

        # Si no se encontró ninguna imagen, avisamos y terminamos.
        if not imagenes:
            print(f"❌ No se encontraron imágenes en: {carpeta}")
            return

        print(f"\n🔍 Analizando {len(imagenes)} imágenes en: {carpeta}")

        # sorted() ordena las imágenes alfabéticamente, para que el
        # procesamiento sea predecible (no en orden aleatorio del sistema).
        for img in sorted(imagenes):
            # Ejecutamos inferencia sobre cada imagen, una por una.
            resultados, ms = inferencia_imagen(model, img)

            # Analizamos el resultado de esta imagen específica.
            r = analizar_resultado(resultados[0], img.name)

            # Agregamos el tiempo de inferencia al diccionario de resultado.
            r["tiempo_ms"] = round(ms, 1)

            # Lo añadimos a la lista general de resumen.
            resumen.append(r)

    else:
        # Si el usuario no especificó ningún modo (ej. solo ejecutó el
        # script sin argumentos), usamos la evaluación de test como
        # comportamiento por defecto.
        print("\n🔍 Modo por defecto: evaluando dataset de test...")
        evaluar_en_test(model)
        # 'return' termina la función aquí, ya que evaluar_en_test()
        # no produce un "resumen" por imagen que necesitemos guardar.
        return

    # Si acumulamos algún resumen (modos --imagen o --carpeta), lo guardamos
    # como un reporte JSON en disco.
    if resumen:
        guardar_reporte(resumen)

    print("\n✅ Pruebas completadas.")
    print(f"   Imágenes anotadas guardadas en: {OUTPUT_DIR}/detecciones/")


# Punto de entrada del script: solo se ejecuta "main()" si este archivo
# se corre directamente (no si se importa desde otro módulo).
if __name__ == "__main__":
    main()