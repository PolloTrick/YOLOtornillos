"""
============================================================
Script 02 - Entrenamiento del Modelo YOLOv8
Proyecto: Detección de Tornillos en Industria
============================================================

Ejecutar con:
    python scripts/02_entrenamiento.py

Requisitos:
    pip install ultralytics
"""

# 'os' permite interactuar con el sistema operativo. Se deja importado
# por si se necesita leer variables de entorno (ej. configurar GPU).
import os

# 'json' permite leer y escribir datos en formato JSON (texto estructurado).
# Lo usamos para guardar las métricas finales del entrenamiento.
import json

# 'time' permite medir cuánto tiempo tarda el entrenamiento.
import time

# 'Path' (de pathlib) representa rutas de archivos/carpetas como objetos,
# facilitando operaciones como crear carpetas o verificar si existen.
from pathlib import Path

# Importamos la clase YOLO de la librería "ultralytics", que es la que
# implementa toda la arquitectura, entrenamiento e inferencia de YOLOv8.
from ultralytics import YOLO


# ============================================================
# CONFIGURACIÓN DEL ENTRENAMIENTO
# ============================================================
# Usamos un diccionario (CONFIG) para agrupar TODOS los parámetros del
# entrenamiento en un solo lugar, facilitando ajustarlos sin buscar
# en medio del código.

CONFIG = {
    # Modelo base (preentrenado en COCO, un dataset genérico de 80 clases).
    # Partir de un modelo preentrenado (transfer learning) acelera mucho
    # el aprendizaje, ya que el modelo ya "sabe" detectar formas, bordes, etc.
    # Opciones: yolov8n.pt | yolov8s.pt | yolov8m.pt | yolov8l.pt | yolov8x.pt
    # n=nano (más rápido, menos preciso), x=extra-large (más lento, más preciso)
    "model_base": "yolov8n.pt",

    # Ruta al archivo YAML que describe el dataset: dónde están las imágenes
    # de train/val y cuáles son las clases (tornillo_presente/ausente).
    "data_yaml": "dataset/data.yaml",

    # --- HIPERPARÁMETROS DE ENTRENAMIENTO ---

    # Número de épocas: cuántas veces el modelo verá el dataset COMPLETO.
    # Más épocas = más aprendizaje, pero también más riesgo de sobreajuste
    # (que el modelo "memorice" en vez de "generalizar").
    "epochs": 50,

    # Tamaño al que se redimensionan las imágenes antes de entrenar (en píxeles).
    # 640x640 es el estándar de YOLOv8; valores más altos = más detalle pero
    # más lento.
    "imgsz": 640,

    # Cuántas imágenes se procesan juntas en cada paso de entrenamiento.
    # Un batch más grande acelera el entrenamiento si hay suficiente memoria
    # (RAM/VRAM), pero puede causar errores de "out of memory" si es muy alto.
    "batch": 16,

    # "Paciencia" para el early stopping (detención temprana):
    # si el modelo no mejora en 10 épocas consecutivas, el entrenamiento
    # se detiene automáticamente para no perder tiempo ni sobreajustar.
    "patience": 10,

    # Tasa de aprendizaje (learning rate) INICIAL. Controla qué tan grandes
    # son los ajustes que el modelo hace en cada paso. Muy alta = inestable,
    # muy baja = aprendizaje lento.
    "lr0": 0.01,

    # Tasa de aprendizaje FINAL (al terminar todas las épocas). YOLOv8 reduce
    # gradualmente el lr0 hasta llegar a este valor (decaimiento programado).
    "lrf": 0.001,

    # Momentum del optimizador SGD: ayuda a que el descenso del gradiente
    # mantenga "inercia" en la dirección correcta y no oscile tanto.
    "momentum": 0.937,

    # Weight decay (decaimiento de pesos): técnica de regularización que
    # penaliza pesos muy grandes en la red, ayudando a evitar el sobreajuste.
    "weight_decay": 0.0005,

    # --- AUMENTACIÓN DE DATOS (DATA AUGMENTATION) ---
    # Estas técnicas generan variaciones artificiales de las imágenes
    # originales durante el entrenamiento, para que el modelo aprenda
    # a reconocer tornillos en distintas condiciones (luz, ángulo, etc.)
    # sin necesidad de tener miles de fotos reales adicionales.

    # Variación aleatoria del tono de color (hue), en una escala de 0 a 1.
    "hsv_h": 0.015,

    # Variación aleatoria de la saturación del color.
    "hsv_s": 0.7,

    # Variación aleatoria del brillo/valor (value) de la imagen.
    "hsv_v": 0.4,

    # Probabilidad (0.5 = 50%) de voltear la imagen horizontalmente (espejo).
    "fliplr": 0.5,

    # Probabilidad de usar la técnica "mosaic", que combina 4 imágenes
    # distintas en una sola, ayudando al modelo a detectar objetos en
    # distintos contextos y tamaños.
    "mosaic": 1.0,

    # Rango de rotación aleatoria de la imagen, en grados (+/- 10°).
    "degrees": 10.0,

    # --- DIRECTORIO DE SALIDA ---

    # Carpeta raíz donde YOLOv8 guardará los resultados de cada ejecución.
    "project": "runs/detect",

    # Subcarpeta específica de esta ejecución (para identificar el experimento).
    "name": "tornillos_v1",

    # Si es True, guarda los checkpoints (pesos) del modelo durante el
    # entrenamiento, no solo el resultado final.
    "save": True,
}


# ============================================================
# FUNCIONES
# ============================================================

def verificar_dataset():
    """Verifica que el dataset esté preparado antes de entrenar."""
    # Convertimos la ruta del YAML (string) en un objeto Path.
    yaml_path = Path(CONFIG["data_yaml"])

    # Si el archivo de configuración del dataset no existe, no podemos
    # continuar: lanzamos una excepción con un mensaje claro para el usuario.
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"❌ No se encontró {yaml_path}\n"
            "   Ejecuta primero: python scripts/01_preprocesamiento.py"
        )

    # Verificamos que existan imágenes tanto en "train" como en "val".
    # Sin alguno de los dos conjuntos, YOLOv8 no puede entrenar correctamente.
    for split in ["train", "val"]:
        img_dir = Path(f"dataset/images/{split}")
        # Comprobamos que la carpeta exista Y que tenga al menos un archivo.
        if not img_dir.exists() or not any(img_dir.iterdir()):
            raise FileNotFoundError(
                f"❌ La carpeta {img_dir} está vacía.\n"
                "   Ejecuta primero: python scripts/01_preprocesamiento.py"
            )

    # Contamos cuántas imágenes hay en cada conjunto, solo para informar
    # al usuario (no afecta el entrenamiento).
    n_train = len(list(Path("dataset/images/train").glob("*")))
    n_val   = len(list(Path("dataset/images/val").glob("*")))
    print(f"✅ Dataset listo:  train={n_train} imágenes  |  val={n_val} imágenes")


def entrenar():
    """Carga el modelo base y ejecuta el entrenamiento."""
    # Informamos qué modelo base se va a cargar (ej. yolov8n.pt).
    print(f"\n🔄 Cargando modelo base: {CONFIG['model_base']}")

    # YOLO(...) descarga (si es necesario) y carga el modelo preentrenado.
    # La primera vez que se ejecuta, Ultralytics descarga el archivo .pt
    # automáticamente desde sus servidores.
    model = YOLO(CONFIG["model_base"])

    # Informamos cuántas épocas se van a entrenar.
    print(f"\n🚀 Iniciando entrenamiento por {CONFIG['epochs']} épocas...")

    # Guardamos el tiempo actual (en segundos desde 1970) para medir
    # cuánto dura el entrenamiento al finalizar.
    inicio = time.time()

    # model.train() es el método central de Ultralytics: ejecuta TODO el
    # ciclo de entrenamiento (forward pass, cálculo de pérdida, backward
    # pass, actualización de pesos) durante el número de épocas indicado.
    # Cada parámetro de CONFIG se pasa explícitamente como argumento.
    results = model.train(
        data         = CONFIG["data_yaml"],     # ruta al YAML del dataset
        epochs       = CONFIG["epochs"],        # número total de épocas
        imgsz        = CONFIG["imgsz"],         # tamaño de imagen de entrada
        batch        = CONFIG["batch"],         # imágenes por lote
        patience     = CONFIG["patience"],      # épocas de paciencia (early stopping)
        lr0          = CONFIG["lr0"],           # learning rate inicial
        lrf          = CONFIG["lrf"],           # learning rate final
        momentum     = CONFIG["momentum"],      # momentum del optimizador
        weight_decay = CONFIG["weight_decay"],  # regularización L2
        hsv_h        = CONFIG["hsv_h"],         # aumentación: tono
        hsv_s        = CONFIG["hsv_s"],         # aumentación: saturación
        hsv_v        = CONFIG["hsv_v"],         # aumentación: brillo
        fliplr       = CONFIG["fliplr"],        # aumentación: volteo horizontal
        mosaic       = CONFIG["mosaic"],        # aumentación: mosaico
        degrees      = CONFIG["degrees"],       # aumentación: rotación
        project      = CONFIG["project"],       # carpeta raíz de resultados
        name         = CONFIG["name"],          # nombre de esta ejecución
        save         = CONFIG["save"],          # guardar checkpoints
        verbose      = True,                    # mostrar progreso detallado en consola
    )

    # Calculamos cuánto tiempo (en segundos) tardó todo el entrenamiento.
    duracion = time.time() - inicio

    # Mostramos la duración convertida a minutos, con 1 decimal.
    print(f"\n⏱️  Tiempo total de entrenamiento: {duracion/60:.1f} minutos")

    # Retornamos tanto el modelo entrenado como el objeto de resultados
    # (que contiene las métricas finales).
    return model, results


def guardar_mejor_modelo():
    """Copia el mejor modelo a la carpeta /models para fácil acceso."""
    # Creamos la carpeta "models" si no existe (exist_ok=True evita error).
    Path("models").mkdir(exist_ok=True)

    # Ultralytics guarda automáticamente el mejor modelo (según la métrica
    # de validación) en esta ruta dentro de la carpeta de resultados.
    ruta_best = Path(f"{CONFIG['project']}/{CONFIG['name']}/weights/best.pt")

    # Verificamos que el archivo realmente se haya generado.
    if ruta_best.exists():
        # Importamos shutil aquí (import local) solo para esta función,
        # ya que es la única que lo necesita.
        import shutil
        # Definimos dónde queremos la copia final, con un nombre fijo
        # y fácil de referenciar desde otros scripts (03_pruebas.py).
        destino = Path("models/mejor_modelo.pt")
        # copy2 copia el archivo conservando metadatos.
        shutil.copy2(ruta_best, destino)
        print(f"✅ Mejor modelo guardado en: {destino}")
    else:
        # Si por alguna razón no se generó el archivo, avisamos sin
        # detener el script (no es un error fatal).
        print(f"⚠️  No se encontró el mejor modelo en {ruta_best}")


def mostrar_metricas(results):
    """Extrae y muestra las métricas principales del entrenamiento."""
    # Usamos try/except porque la estructura interna de "results" puede
    # variar entre versiones de Ultralytics; si algo falla, no queremos
    # que el script se detenga por completo.
    try:
        # results.results_dict es un diccionario con todas las métricas
        # finales calculadas sobre el conjunto de validación.
        metricas = results.results_dict

        print("\n📊 MÉTRICAS FINALES DEL MODELO:")
        print("─" * 45)

        # Diccionario que mapea un nombre legible -> la clave técnica
        # correspondiente dentro de "metricas". Usamos .get(clave, "N/A")
        # para evitar errores si alguna métrica no existe.
        metricas_mostrar = {
            "Precisión (P)":         metricas.get("metrics/precision(B)", "N/A"),
            "Recall (R)":            metricas.get("metrics/recall(B)",    "N/A"),
            "mAP@0.5":               metricas.get("metrics/mAP50(B)",     "N/A"),
            "mAP@0.5:0.95":          metricas.get("metrics/mAP50-95(B)",  "N/A"),
        }

        # Recorremos cada métrica para imprimirla con formato.
        for nombre, valor in metricas_mostrar.items():
            # Si el valor es un número decimal, lo mostramos con 4 decimales.
            if isinstance(valor, float):
                print(f"   {nombre:<22} {valor:.4f}")
            else:
                # Si es "N/A" (no disponible), lo mostramos tal cual.
                print(f"   {nombre:<22} {valor}")

        # Guardamos las métricas en un archivo JSON, para poder consultarlas
        # después sin tener que re-entrenar el modelo.
        Path("results").mkdir(exist_ok=True)
        with open("results/metricas_entrenamiento.json", "w") as f:
            # Solo guardamos los valores que sean números flotantes
            # (filtramos cualquier otro tipo de dato que no sea serializable
            # de forma sencilla a JSON).
            json.dump({k: float(v) for k, v in metricas.items() if isinstance(v, float)}, f, indent=2)
        print("\n   Métricas guardadas en: results/metricas_entrenamiento.json")

    except Exception as e:
        # Si algo sale mal al leer las métricas, mostramos el error
        # pero permitimos que el script continúe.
        print(f"⚠️  No se pudieron mostrar las métricas: {e}")


def interpretar_resultados():
    """Explica brevemente qué significan las métricas obtenidas."""
    # Bloque de texto fijo (no depende de variables) que sirve como
    # guía educativa para interpretar las métricas mostradas arriba.
    print("""
📖 GUÍA DE INTERPRETACIÓN:
─────────────────────────────────────────────────
  Precisión (P): De todos los tornillos detectados,
                 ¿qué porcentaje eran correctos?
                 -> 0.90 = 90% de detecciones válidas

  Recall (R):    De todos los tornillos reales,
                 ¿qué porcentaje detectó el modelo?
                 -> 0.88 = detectó el 88% de tornillos

  mAP@0.5:       Métrica estándar de detección.
                 -> >0.80 es bueno para producción
                 -> >0.90 es excelente

  mAP@0.5:0.95:  Métrica más estricta (multiescala).
                 -> >0.50 es aceptable
─────────────────────────────────────────────────
""")


# ============================================================
# EJECUCIÓN PRINCIPAL
# ============================================================
# Este bloque solo se ejecuta si corremos el archivo directamente
# (no si lo importamos desde otro script).
if __name__ == "__main__":
    # Encabezado visual.
    print("=" * 55)
    print("  ENTRENAMIENTO YOLOv8 - Detección de Tornillos")
    print("=" * 55)

    # 1. Verificar que el dataset esté listo (carpetas y archivos existen).
    verificar_dataset()

    # 2. Entrenar el modelo: carga el modelo base y ejecuta model.train().
    #    Retorna el modelo ya entrenado y el objeto de resultados/métricas.
    model, results = entrenar()

    # 3. Copiar el mejor modelo (best.pt) a la carpeta "models" con un
    #    nombre fijo, para que 03_pruebas.py lo encuentre fácilmente.
    guardar_mejor_modelo()

    # 4. Mostrar las métricas finales (precisión, recall, mAP) en consola
    #    y guardarlas en un archivo JSON.
    mostrar_metricas(results)

    # 5. Mostrar una guía de interpretación de esas métricas, para que
    #    el usuario entienda qué significan los números obtenidos.
    interpretar_resultados()

    # Mensaje final de cierre, indicando el siguiente script a ejecutar.
    print("=" * 55)
    print("  ✅ Entrenamiento completado.")
    print("  Siguiente paso: python scripts/03_pruebas.py")
    print("=" * 55)