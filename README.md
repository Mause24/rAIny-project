# 🌧️ rAIny – Predicción de Lluvia con Inteligencia Artificial

**rAIny** es un proyecto de Machine Learning que busca predecir si habrá precipitación (lluvia) en un lugar específico, utilizando variables climáticas históricas como temperatura, humedad y velocidad del viento.

Este modelo está enfocado en datos recopilados en **Ponedera, Atlántico (Colombia)** durante los años **2020 a 2021**.

---

## 📌 Objetivo

Entrenar un modelo de clasificación binaria que diga si **lloverá o no**, con base en datos meteorológicos.

---

## 🧠 ¿Cómo funciona?

1. Se cargan datos históricos en formato CSV (`rAIny_training_dataset.csv`).
2. Se preprocesan los datos: limpieza, selección de características, y normalización.
3. Se construye un modelo de clasificación usando una red neuronal simple con TensorFlow.
4. Se entrena el modelo para predecir si **lloverá o no**.
5. Se evalúa el rendimiento del modelo con métricas como precisión, matriz de confusión y curva ROC.

---

## 🔍 Dataset

El conjunto de datos se encuentra en el archivo `rAIny_training_dataset.csv`, y contiene variables como:

- `temperatura_media`
- `humedad_relativa`
- `velocidad_viento`
- `presion`
- `precipitacion` (valor binario: 1 si llueve, 0 si no)

Todos los datos corresponden a registros tomados en Ponedera (Atlántico, Colombia) entre 2020 y 2021.

---

## 📁 Estructura del Proyecto

````plaintext
rAIny/
├── venv310/                   # Entorno virtual (NO incluir en producción)
├──data/
├────── rAIny_training_dataset.csv # Dataset con variables climáticas
├── main.py                    # Código principal de entrenamiento y evaluación
├── requirements.txt           # Lista de dependencias
└── README.md                  # Documentación del proyecto

---

## 🛠️ Requisitos

- Python 3.10
- pip (gestor de paquetes)
- Entorno virtual (`venv`)

---

## 🛠️ Tecnologías utilizadas

- **Python 3.10**
- **TensorFlow** – Red neuronal para clasificación binaria
- **pandas & numpy** – Manipulación de datos
- **scikit-learn** – Métricas de evaluación
- **matplotlib** – Visualización de resultados

---

## 🚀 Cómo usar el proyecto

1. Clona o descarga este repositorio.
2. Crea y activa un entorno virtual:
   ```bash
   python -m venv venv310
   # En Windows:
   ./venv310/Scripts/activate
   # En macOS/Linux:
   source venv310/bin/activate
3. Instala los paquetes necesarios:
   ```bash
   pip install -r requirements.txt
4. Ejecuta el programa:
   ```bash
   python main.py

## ✅ Resultados Esperados

El script `main.py` genera una evaluación del modelo incluyendo:

- Accuracy (Precisión)
- Matriz de confusión
- Gráfica de pérdida y precisión durante el entrenamiento
- Curva ROC

---

## 👥 Colaboradores

- **Omar Arenas**
- **Carlos Duran**

````
