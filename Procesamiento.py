"""
============================================================
Script 01 - Preprocesamiento del Dataset
Proyecto: Detección de Tornillos con YOLOv8
============================================================

Instrucciones:
1. Coloca tus imágenes en la carpeta dataset/images/raw/
2. Coloca tus anotaciones en formato YOLO (.txt) en dataset/labels/raw/
3. Ejecuta este script: python scripts/01_preprocesamiento.py

Fuentes recomendadas para obtener el dataset:
- Roboflow Universe: https://universe.roboflow.com/search?q=screws
- Kaggle: https://www.kaggle.com/search?q=screws+detection
"""

# 'os' permite interactuar con el sistema operativo (rutas, archivos, carpetas).
# En este script no se usa de forma directa, pero se deja importado por si se
# necesita en futuras extensiones (ej. variables de entorno).
import os

# 'shutil' (shell utilities) ofrece funciones para copiar y mover archivos.
# La usamos para copiar imágenes/etiquetas desde "raw" hacia train/val/test.
import shutil

# 'random' permite generar números y selecciones aleatorias.
# Lo usamos para mezclar el orden de las imágenes antes de dividirlas.
import random

# 'Path' (de pathlib) representa rutas de archivos/carpetas de forma orientada
# a objetos. Es más moderno y seguro que concatenar strings con "+".
from pathlib import Path


# ============================================================
# CONFIGURACIÓN
# ============================================================
# Esta sección agrupa todos los valores que el usuario podría querer cambiar,
# para no tener que buscarlos dispersos en el código.

# Carpeta donde el usuario coloca las imágenes ORIGINALES, sin organizar.
DATASET_RAW_IMAGES = "dataset/images/raw"

# Carpeta donde el usuario coloca las etiquetas ORIGINALES (.txt formato YOLO).
DATASET_RAW_LABELS = "dataset/labels/raw"

# Porcentaje de imágenes que irán al conjunto de ENTRENAMIENTO (70%).
# Es el conjunto más grande: el modelo "aprende" de estas imágenes.
SPLIT_TRAIN = 0.70

# Porcentaje de imágenes para VALIDACIÓN (20%).
# Se usa durante el entrenamiento para medir qué tan bien generaliza el modelo
# sin haber visto estas imágenes directamente en el aprendizaje.
SPLIT_VAL = 0.20

# Porcentaje de imágenes para PRUEBA/TEST (10%).
# Se usa solo al final, para evaluar el modelo de forma totalmente imparcial.
SPLIT_TEST = 0.10

# Semilla aleatoria fija. Al usar la misma semilla, random.shuffle() siempre
# mezclará las imágenes en el MISMO orden cada vez que se ejecute el script.
# Esto hace que el experimento sea reproducible (resultados consistentes).
SEED = 42


# ============================================================
# FUNCIONES
# ============================================================

def verificar_estructura():
    """Verifica que las carpetas necesarias existan."""
    # Lista de todas las carpetas que el proyecto necesita para funcionar.
    # Si alguna no existe, se creará automáticamente.
    carpetas = [
        "dataset/images/raw",
        "dataset/labels/raw",
        "dataset/images/train", "dataset/images/val", "dataset/images/test",
        "dataset/labels/train", "dataset/labels/val",  "dataset/labels/test",
    ]
    # Recorremos cada ruta de la lista, una por una.
    for carpeta in carpetas:
        # Path(carpeta) convierte el string en un objeto de ruta.
        # .mkdir() crea la carpeta física en el disco.
        # parents=True -> crea también las carpetas "padre" si no existen
        #                 (ej. si "dataset" no existe, la crea primero).
        # exist_ok=True -> si la carpeta YA existe, no lanza error, simplemente
        #                 continúa (evita que el script se detenga).
        Path(carpeta).mkdir(parents=True, exist_ok=True)
    # Mensaje informativo para el usuario, confirmando que todo está listo.
    print("✅ Estructura de carpetas verificada.")


def obtener_pares_imagen_etiqueta():
    """Obtiene los pares (imagen, etiqueta) del dataset raw."""
    # Conjunto (set) de extensiones de imagen válidas. Usamos un "set" {}
    # en vez de una lista [] porque la búsqueda "está en este conjunto"
    # es más rápida con sets.
    extensiones_img = {".jpg", ".jpeg", ".png", ".bmp"}

    # Construimos una lista con todos los archivos de la carpeta raw
    # cuya extensión esté dentro de "extensiones_img".
    imagenes = [
        f for f in Path(DATASET_RAW_IMAGES).iterdir()   # recorre cada archivo de la carpeta
        if f.suffix.lower() in extensiones_img          # filtra solo si la extensión coincide
    ]
    # f.suffix devuelve la extensión del archivo, ej: ".JPG"
    # .lower() la convierte a minúsculas, ej: ".jpg", para comparar sin
    # importar si el usuario nombró el archivo en mayúsculas o minúsculas.

    # Lista vacía donde guardaremos los pares válidos (imagen + su etiqueta).
    pares = []

    # Lista vacía donde guardaremos los nombres de imágenes que NO tienen
    # una etiqueta correspondiente (para avisar al usuario).
    sin_etiqueta = []

    # Recorremos cada imagen encontrada.
    for img in imagenes:
        # Construimos la ruta esperada de la etiqueta:
        # mismo nombre que la imagen, pero con extensión .txt,
        # ubicada en la carpeta de etiquetas raw.
        # img.stem = nombre del archivo SIN extensión (ej. "foto1" de "foto1.jpg")
        etiqueta = Path(DATASET_RAW_LABELS) / (img.stem + ".txt")

        # Comprobamos si ese archivo de etiqueta realmente existe en disco.
        if etiqueta.exists():
            # Si existe, agregamos la tupla (imagen, etiqueta) a la lista de pares.
            pares.append((img, etiqueta))
        else:
            # Si no existe, guardamos solo el NOMBRE de la imagen (img.name)
            # para reportarlo después.
            sin_etiqueta.append(img.name)

    # Si encontramos imágenes sin etiqueta, mostramos una advertencia.
    if sin_etiqueta:
        print(f"⚠️  {len(sin_etiqueta)} imágenes sin etiqueta encontradas:")
        # Mostramos solo las primeras 5 para no llenar la pantalla de texto.
        for nombre in sin_etiqueta[:5]:
            print(f"   - {nombre}")
        # Si hay más de 5, indicamos cuántas faltan por mostrar.
        if len(sin_etiqueta) > 5:
            print(f"   ... y {len(sin_etiqueta) - 5} más.")

    # Mostramos cuántos pares válidos (imagen+etiqueta) se encontraron en total.
    print(f"📦 Total de pares válidos (imagen + etiqueta): {len(pares)}")

    # Retornamos la lista de pares para que la función principal la use.
    return pares


def dividir_dataset(pares):
    """Divide el dataset en train, val y test de forma aleatoria."""
    # Fijamos la semilla aleatoria ANTES de mezclar, para que el orden
    # aleatorio sea siempre el mismo en cada ejecución (reproducibilidad).
    random.seed(SEED)

    # random.shuffle() reordena la lista "pares" de forma aleatoria,
    # IN-PLACE (modifica la lista original directamente, no crea una nueva).
    random.shuffle(pares)

    # Contamos cuántos elementos hay en total.
    n_total = len(pares)

    # Calculamos cuántas imágenes corresponden a "train" (ej. 70% del total).
    # int() redondea hacia abajo (trunca decimales) para obtener un número entero.
    n_train = int(n_total * SPLIT_TRAIN)

    # Calculamos cuántas imágenes corresponden a "val" (ej. 20% del total).
    n_val = int(n_total * SPLIT_VAL)

    # El resto (lo que no es train ni val) será automáticamente "test".
    # No necesitamos calcular n_test porque usamos slicing de listas más abajo.

    # Diccionario que mapea el nombre del split a su porción correspondiente
    # de la lista "pares", usando slicing (rebanado de listas):
    splits = {
        # Desde el inicio (0) hasta n_train (sin incluirlo).
        "train": pares[:n_train],
        # Desde n_train hasta n_train + n_val.
        "val":   pares[n_train:n_train + n_val],
        # Desde donde terminó "val" hasta el final de la lista.
        "test":  pares[n_train + n_val:],
    }

    # Mostramos al usuario cuántas imágenes quedaron en cada conjunto.
    for split, items in splits.items():
        # :5s formatea el texto a un ancho fijo de 5 caracteres (alineación).
        print(f"   {split:5s}: {len(items)} imágenes")

    # Retornamos el diccionario con las 3 listas (train/val/test).
    return splits


def copiar_archivos(splits):
    """Copia los archivos a sus carpetas correspondientes."""
    # Recorremos cada split (train, val, test) y su lista de pares.
    for split, pares in splits.items():
        # Definimos la carpeta destino de imágenes para este split.
        dir_img = Path(f"dataset/images/{split}")
        # Definimos la carpeta destino de etiquetas para este split.
        dir_lbl = Path(f"dataset/labels/{split}")

        # Recorremos cada par (imagen, etiqueta) de este split.
        for img, lbl in pares:
            # shutil.copy2() copia el archivo MANTENIENDO sus metadatos
            # (fecha de creación, permisos, etc.), a diferencia de copy()
            # que solo copia el contenido.
            # dir_img / img.name construye la ruta destino:
            # "dataset/images/train" / "foto1.jpg" -> "dataset/images/train/foto1.jpg"
            shutil.copy2(img, dir_img / img.name)
            shutil.copy2(lbl, dir_lbl / lbl.name)

    # Confirmamos al usuario que la copia terminó.
    print("✅ Archivos copiados a train/val/test.")


def analizar_clases():
    """Analiza la distribución de clases en el dataset de entrenamiento."""
    # Diccionario contador: clave = id de clase, valor = cuántas veces aparece.
    # Inicializamos en 0 las dos clases que sabemos que existen:
    # 0 = tornillo_presente, 1 = tornillo_ausente.
    conteo = {0: 0, 1: 0}

    # Ruta a la carpeta de etiquetas de entrenamiento (ya organizadas).
    dir_labels = Path("dataset/labels/train")

    # Si la carpeta no existe, o existe pero está vacía, no hay nada que analizar.
    # any(dir_labels.iterdir()) devuelve True si hay AL MENOS un archivo dentro.
    if not dir_labels.exists() or not any(dir_labels.iterdir()):
        print("⚠️  No hay etiquetas de entrenamiento para analizar.")
        # 'return' sin valor sale de la función inmediatamente.
        return

    # Recorremos cada archivo .txt dentro de la carpeta de etiquetas train.
    # glob("*.txt") busca todos los archivos que terminen en ".txt".
    for archivo in dir_labels.glob("*.txt"):
        # Abrimos el archivo en modo lectura (por defecto).
        # El "with" garantiza que el archivo se cierre automáticamente al final.
        with open(archivo) as f:
            # Cada línea del archivo representa UNA detección/objeto anotado.
            for linea in f:
                # .strip() elimina espacios y saltos de línea al inicio/final.
                linea = linea.strip()
                # Si la línea no está vacía (evita errores con líneas en blanco):
                if linea:
                    # El formato YOLO es: "clase cx cy ancho alto"
                    # .split() separa por espacios -> ["0", "0.51", "0.43", ...]
                    # [0] toma el primer elemento (el id de clase, como texto)
                    # int() lo convierte a número entero.
                    clase = int(linea.split()[0])
                    # Incrementamos el contador de esa clase en 1.
                    # .get(clase, 0) devuelve 0 si la clase no existía antes
                    # (por seguridad, en caso de que aparezca una clase nueva).
                    conteo[clase] = conteo.get(clase, 0) + 1

    # Sumamos todos los valores del diccionario para saber el total de objetos.
    total = sum(conteo.values())

    # Si no se encontró ningún objeto anotado, avisamos y salimos.
    if total == 0:
        print("⚠️  No se encontraron anotaciones.")
        return

    # Encabezado del reporte de distribución de clases.
    print("\n📊 Distribución de clases (train):")

    # Lista de nombres legibles, en el mismo orden que sus ids (0 y 1).
    nombres = ["tornillo_presente", "tornillo_ausente"]

    # enumerate() nos da el índice (0,1,...) junto con el valor de la lista.
    for idx, nombre in enumerate(nombres):
        # Obtenemos cuántas veces apareció esta clase (0 si no apareció nunca).
        cantidad = conteo.get(idx, 0)
        # Calculamos el porcentaje que representa sobre el total.
        porcentaje = (cantidad / total * 100) if total > 0 else 0
        # Creamos una "barra visual" usando el carácter █, con una longitud
        # proporcional al porcentaje (dividido entre 2 para que no sea enorme).
        barra = "█" * int(porcentaje / 2)
        # Imprimimos la fila formateada: índice, nombre, cantidad, porcentaje y barra.
        # <20 alinea el texto a la izquierda en 20 caracteres.
        # >5 alinea el número a la derecha en 5 caracteres.
        # 5.1f muestra 1 decimal con ancho mínimo de 5 caracteres.
        print(f"   [{idx}] {nombre:<20} {cantidad:>5} ({porcentaje:5.1f}%)  {barra}")


def generar_reporte(pares):
    """Genera un reporte de texto del preprocesamiento."""
    # f-string multilínea (triple comillas) que construye el contenido
    # del archivo de reporte, insertando variables con {}.
    reporte = f"""
REPORTE DE PREPROCESAMIENTO
============================
Dataset: Detección de Tornillos
Total de imágenes procesadas: {len(pares)}
División:
  - Train: {SPLIT_TRAIN*100:.0f}%
  - Val:   {SPLIT_VAL*100:.0f}%
  - Test:  {SPLIT_TEST*100:.0f}%
Semilla aleatoria: {SEED}
"""
    # Abrimos (o creamos) el archivo en modo escritura ("w" = write).
    # Si el archivo ya existe, su contenido anterior se SOBRESCRIBE.
    with open("results/reporte_preprocesamiento.txt", "w") as f:
        # Escribimos el string completo del reporte dentro del archivo.
        f.write(reporte)
    # Confirmamos al usuario dónde quedó guardado el archivo.
    print("📄 Reporte guardado en results/reporte_preprocesamiento.txt")


# ============================================================
# EJECUCIÓN PRINCIPAL
# ============================================================
# Este bloque solo se ejecuta cuando el archivo se corre DIRECTAMENTE
# (ej. "python scripts/01_preprocesamiento.py"), no cuando se importa
# como módulo desde otro script.
if __name__ == "__main__":
    # Encabezado visual del script.
    print("=" * 55)
    print("  PREPROCESAMIENTO - Detección de Tornillos YOLOv8")
    print("=" * 55)

    # Paso 1: nos asegura que todas las carpetas necesarias existan.
    verificar_estructura()

    # Paso 2: buscamos qué imágenes tienen su etiqueta correspondiente.
    pares = obtener_pares_imagen_etiqueta()

    # Si no se encontró ningún par válido, no tiene sentido continuar.
    if len(pares) == 0:
        print("\n❌ No se encontraron imágenes con etiquetas.")
        print("   Agrega tus imágenes en:  dataset/images/raw/")
        print("   Agrega tus etiquetas en: dataset/labels/raw/")
        print("\n   Tip: Descarga un dataset desde https://universe.roboflow.com")
        # exit(1) termina el programa con código de error 1
        # (por convención, 0 = éxito, distinto de 0 = algún tipo de error).
        exit(1)

    # Paso 3: dividimos los pares en train/val/test según los porcentajes.
    print("\n📂 Dividiendo dataset...")
    splits = dividir_dataset(pares)

    # Paso 4: copiamos físicamente los archivos a sus carpetas finales.
    print("\n📁 Copiando archivos...")
    copiar_archivos(splits)

    # Paso 5: mostramos cuántos objetos de cada clase hay en el set de train.
    analizar_clases()

    # Nos asegura que la carpeta "results" exista antes de escribir en ella.
    Path("results").mkdir(exist_ok=True)

    # Paso 6: generamos un archivo de texto resumen del proceso.
    generar_reporte(pares)

    # Mensaje final de éxito y siguiente paso a seguir.
    print("\n✅ Preprocesamiento completado exitosamente.")
    print("   Siguiente paso: ejecuta  python scripts/02_entrenamiento.py")