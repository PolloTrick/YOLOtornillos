# 🔩 Detección de Tornillos con YOLOv8

**Control de calidad automatizado en líneas de ensamblaje industrial**

Angel Francisco Jaramillo Pizeno 23310317
---

## 📋 Descripción

Este proyecto implementa un modelo de visión computacional basado en **YOLOv8** para detectar la **presencia o ausencia de tornillos** en piezas industriales. El objetivo es automatizar el control de calidad en líneas de manufactura, reduciendo errores humanos y aumentando la velocidad de inspección.

El modelo distingue entre dos estados:

| Clase | Descripción | Indicador |
|-------|-------------|-----------|
| `tornillo_presente` | Tornillo correctamente colocado | ✅ Pieza aprobada |
| `tornillo_ausente`  | Posición vacía — tornillo faltante | ❌ Pieza rechazada |

---

## 📁 Estructura del Repositorio

```
yolo-tornillos-industria/
│
├── dataset/
│   ├── images/
│   │   ├── train/         # Imágenes de entrenamiento (70%)
│   │   ├── val/           # Imágenes de validación (20%)
│   │   └── test/          # Imágenes de prueba (10%)
│   ├── labels/
│   │   ├── train/         # Etiquetas YOLO (.txt) de entrenamiento
│   │   ├── val/           # Etiquetas YOLO (.txt) de validación
│   │   └── test/          # Etiquetas YOLO (.txt) de prueba
│   └── data.yaml          # Configuración del dataset
│
├── scripts/
│   ├── 01_preprocesamiento.py   # División y organización del dataset
│   ├── 02_entrenamiento.py      # Entrenamiento del modelo YOLOv8
│   └── 03_pruebas.py            # Inferencia y evaluación
│
├── models/
│   └── mejor_modelo.pt          # Pesos del mejor modelo entrenado
│
├── results/
│   ├── metricas_entrenamiento.json
│   └── inferencias/
│
├── requirements.txt
└── README.md
```

---

## 🚀 Instalación y Uso

### 1. Clonar el repositorio

```bash
git clone https://github.com/TU_USUARIO/yolo-tornillos-industria.git
cd yolo-tornillos-industria
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Preparar el dataset

Descarga un dataset de tornillos (ver sección [Dataset](#-dataset)) y coloca las imágenes en `dataset/images/raw/` y las etiquetas en `dataset/labels/raw/`, luego ejecuta:

```bash
python scripts/01_preprocesamiento.py
```

### 4. Entrenar el modelo

```bash
python scripts/02_entrenamiento.py
```

### 5. Probar el modelo

```bash
# Evaluar sobre imágenes de prueba
python scripts/03_pruebas.py --test

# Analizar una imagen
python scripts/03_pruebas.py --imagen ruta/mi_pieza.jpg

# Analizar una carpeta de imágenes
python scripts/03_pruebas.py --carpeta dataset/images/test/

# Detección en tiempo real con cámara
python scripts/03_pruebas.py --camara
```

---

## 📦 Dataset

El dataset utilizado proviene de:

- **Fuente:** [Roboflow Universe – Screws Detection](https://universe.roboflow.com/search?q=screws)
- **Formato de etiquetas:** YOLO (.txt) — cada línea contiene `clase cx cy w h` (valores normalizados)
- **Distribución:** 70% train / 20% val / 10% test
- **Resolución de entrada:** 640 × 640 px

### Formato de etiqueta YOLO

```
# clase   cx      cy      ancho   alto
    0     0.512   0.437   0.098   0.112    ← tornillo_presente
    1     0.251   0.680   0.091   0.105    ← tornillo_ausente
```

---

## 🏋️ Entrenamiento

| Parámetro | Valor |
|-----------|-------|
| Arquitectura | YOLOv8 Nano (yolov8n) |
| Épocas | 50 |
| Tamaño de imagen | 640 × 640 px |
| Batch size | 16 |
| Optimizador | SGD |
| Learning rate | 0.01 → 0.001 |
| Early stopping | 10 épocas sin mejora |
| Augmentación | Flip, rotación ±10°, HSV |

### Métricas obtenidas

> *(Se actualizan después del entrenamiento)*

| Métrica | Valor |
|---------|-------|
| Precisión (P) | — |
| Recall (R) | — |
| mAP@0.5 | — |
| mAP@0.5:0.95 | — |

---

## 💡 Caso de Estudio: Aplicación en la Vida Real

### 🏭 Contexto y Problema

En la industria manufacturera —especialmente en los sectores **automotriz, electrónico y aeroespacial**— el ensamblaje de componentes requiere la colocación precisa de tornillos en posiciones específicas. Un tornillo faltante puede provocar:

- Fallas estructurales en el producto final
- Accidentes para el usuario final
- Costosos retiros del mercado (*product recalls*)
- Pérdidas económicas millonarias y daños a la reputación

Actualmente, esta inspección se realiza de manera **manual** por operarios al final de la línea, lo cual es lento, costoso y propenso a errores por fatiga o distracción.

---

### 🤖 Solución Propuesta

Implementar un **sistema de visión artificial** integrado directamente en la línea de ensamblaje, utilizando el modelo YOLOv8 entrenado para:

1. **Capturar** imágenes de cada pieza mediante una cámara industrial montada sobre la cinta transportadora
2. **Analizar** la pieza en tiempo real (<50 ms por imagen)
3. **Clasificar** cada posición de tornillo como presente o ausente
4. **Actuar** automáticamente: aprobar o rechazar la pieza antes de continuar al siguiente proceso

```
[Cinta transportadora]
        ↓
  [Cámara industrial]
        ↓
  [Modelo YOLOv8]         ← Este proyecto
        ↓
  ┌─────────────────┐
  │  tornillo OK?   │
  └─────────────────┘
     ↙         ↘
  ✅ OK      ❌ FALLO
  Continúa   Rechazar + Alerta
```

---

### 🎯 Casos de Uso Específicos

**1. Industria Automotriz**
Verificación de tornillos en carrocerías, motores y frenos antes del siguiente paso en la línea de ensamblaje. Una pieza con un tornillo faltante se desvía automáticamente sin detener la producción.

**2. Electrónica de Consumo**
Inspección de placas de circuito y carcasas de dispositivos (laptops, smartphones) para confirmar que todos los puntos de fijación están correctamente atornillados.

**3. Sector Aeroespacial**
Control de calidad en paneles y componentes estructurales donde cada tornillo es crítico para la seguridad del vuelo. El sistema genera registros auditables de cada inspección.

---

### 👥 Beneficiarios

| Actor | Beneficio |
|-------|-----------|
| **Empresa manufacturera** | Reduce defectos, aumenta velocidad de inspección, baja costos de calidad |
| **Operarios** | Elimina tareas repetitivas y fatigosas; se reasignan a roles de supervisión |
| **Cliente final** | Recibe un producto con mayor calidad y seguridad garantizada |
| **Área de calidad** | Obtiene reportes automáticos y trazabilidad completa por lote |

---

### 📈 Impacto Esperado

- **Velocidad:** inspección de 1 pieza cada ~50 ms (20 piezas/segundo)
- **Precisión:** reducción del error humano de ~3% a <0.5% en detección
- **Disponibilidad:** operación 24/7 sin fatiga
- **Trazabilidad:** registro fotográfico y JSON de cada pieza inspeccionada
- **ROI estimado:** reducción del 60–80% en costos de retrabajo por defectos no detectados

---

### 🔌 Arquitectura del Sistema

```
┌────────────────────────────────────────────────┐
│             LÍNEA DE PRODUCCIÓN                │
│                                                │
│  [Cámara IP / GigE]                            │
│         │                                      │
│  [PC Industrial / Edge Device (Jetson Nano)]   │
│         │                                      │
│  [Modelo YOLOv8 – Python]  ← Este repositorio  │
│         │                                      │
│  [PLC / SCADA]  ←  señal OK/NOOK vía OPC-UA   │
│         │                                      │
│  [Actuador: desviador de piezas]               │
└────────────────────────────────────────────────┘
         │
   [Dashboard web] → reportes por turno, por operario, por lote
```

---

## 📊 Resultados Visuales

> *(Agrega aquí imágenes de detecciones una vez entrenado el modelo)*

```
Ejemplo de salida esperada:
┌─────────────────────────────┐
│  🟩 tornillo_presente 0.97  │
│  🟥 tornillo_ausente  0.94  │
│  🟩 tornillo_presente 0.91  │
│                             │
│  Estado: ❌ FALLO           │
│  1 tornillo faltante        │
└─────────────────────────────┘
```

---

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor abre un *issue* primero para discutir los cambios que deseas realizar.

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.

---

*Proyecto desarrollado como parte del curso de Visión Computacional — Detección de Objetos con YOLO*