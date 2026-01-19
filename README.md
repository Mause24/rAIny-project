# 🌧️ rAIny – Predicción de Lluvia con Inteligencia Artificial

**rAIny** es un proyecto de Machine Learning que predice la **probabilidad de lluvia** en un lugar específico utilizando variables climáticas históricas como temperatura, humedad, radiación solar y velocidad del viento.

Este modelo se enfoca en datos del **Sur del departamento Atlántico (Colombia)** y actualmente usa registros desde el año **2015 hasta la actualidad**, integrados desde múltiples archivos.

---

## 📌 Objetivo

Entrenar una red neuronal capaz de predecir la **probabilidad de precipitación**, expresada como un porcentaje entre 0% y 100%

---

## 🧠 ¿Cómo funciona?

1. Se cargan múltiples datasets CSV desde la carpeta `data/`, con separador `;`.
2. Se preprocesan los datos: limpieza, eliminación de valores nulos, selección de características y normalización.
3. Se define una red neuronal profunda con capas densas, regularización L2 y Dropout.
4. Se entrena el modelo para predecir la probabilidad de lluvia.
5. Se evalúa el rendimiento del modelo con métricas como precisión, recall, f1-score, y matriz de confusión.
6. Se guarda el modelo y el escalador para uso futuro.

---

## 🔍 Dataset

Los datos meteorológicos incluyen las siguientes variables:

- `T2M`: Temperatura a 2 metros (°C)
- `WS2M`: Velocidad del viento a 2 metros (m/s)
- `QV2M`: Humedad específica a 2 metros (g/kg)
- `RH2M`: Humedad relativa (%)
- `ALLSKY_SFC_SW_DWN`: Irradiancia solar (MJ/m²/día)
- `PS`: Presión superficial (kPa)
- `PRECTOTCORR`: Acumulación de precipitación corregida (usada para generar la etiqueta `GRP`)

---

## 📁 Estructura del Proyecto

```plaintext
rAIny/
├── venv*/                         # Entorno virtual (excluido en .gitignore)
├── data/                          # Datasets CSV por año (2015 a hoy)
│   ├── rAIny_training_dataset_2015.csv
│   ├── ...
├── models/                        # Modelos .h5 y escaladores .joblib guardados
│   ├── rAIny_model_YYYY-MM-DD_HH-MM.h5
│   ├── scaler_YYYY-MM-DD_HH-MM.joblib
├── main.py                        # Código principal de entrenamiento
├── predict.py                     # Script para predicción con nuevos datos
├── requirements.txt               # Dependencias del proyecto
└── README.md                      # Documentación del proyecto
```

---

## 🛠️ Requisitos

- Python 3.10
- pip (gestor de paquetes)
- joblib
- TensorFlow 2.x

---

## ⚙️ Tecnologías utilizadas

- **Python 3.10**
- **TensorFlow** – Red neuronal con Keras
- **pandas & numpy** – Procesamiento de datos
- **scikit-learn** – Escalado y métricas
- **matplotlib** – Visualización de entrenamiento
- **joblib** – Serialización del escalador

---

## 🚀 Cómo usar el proyecto

1. Clona o descarga este repositorio.
2. Crea y activa un entorno virtual:
   ```bash
   python -m venv venv
   ./venv/Scripts/activate  # Windows
   source venv/bin/activate  # macOS/Linux
   ```
3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
4. Ejecuta el entrenamiento:
   ```bash
   python train.py
   ```
5. Ejecuta el modelo ya entrenado con datos de prueba:
   ```bash
   python main.py
   ```

---

## ✅ Resultados Esperados

- Entrenamiento con regularización y prevención de sobreajuste (`Dropout`, `L2`, `EarlyStopping`)
- Predicción de probabilidad de lluvia para nuevas muestras
- Visualización de pérdida y precisión
- Matriz de confusión y métricas (accuracy, precision, recall, f1-score)
- Guardado automático del modelo y escalador con fecha y hora

---

## 👥 Colaboradores

- **Omar Arenas**
- **Carlos Duran**
