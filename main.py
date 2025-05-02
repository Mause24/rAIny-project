import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import numpy as np
import random
import os

# Semilla global para reproducibilidad
seed = 73
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

# ===============================
# 1. Leer el dataset y transformarlo en Dataframe con pandas
# ===============================

DATA_PATH = "data/rAIny_training_dataset.csv"
df = pd.read_csv(DATA_PATH)


# ===============================
# 2. Definir la variable de clasde IGR mediante un parametro binario (Llovio o no)
# ===============================

df["IGR"] = (df["PRECTOTCORR"] > 0.1).astype(int)

# ===============================
# 3. Definir las variables independientes que va a tomar el modelo para realizar la prediccion
# ===============================

features = [
    "ALLSKY_SFC_PAR_TOT",  # Radiación solar
    "T2M_MAX",  # Temp. máxima
    "T2M_MIN",  # Temp. mínima
    "T2M",  # Temp. media
    "RH2M",  # Humedad relativa
    "WS2M",  # Vel. viento media
    "WS2M_MAX",  # Vel. viento máx.
    "WS2M_MIN",  # Vel. viento mín.
    "PS",  # Presión superficial
]

X = df[features]
y = df["IGR"]

# ===============================
# 4. Normalizar y dividir los datos
# ===============================

# Escalador estándar: convierte los datos a media 0 y desviación estándar 1
scaler = StandardScaler()

# Aplicar escalado a X
X_scaled = scaler.fit_transform(X)

# Dividir en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,  # Datos normalizados
    y,  # Etiquetas
    test_size=0.2,
    random_state=73,  # TBBT Seed
    stratify=y,  # Mantiene la proporcion de clases
)

# ===============================
# 5. Construcción del modelo
# ===============================

# Definir el modelo secuencial con más capacidad
model = models.Sequential()

# Capa de entrada + capa 1 oculta
model.add(layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)))

# Capa 2 oculta
model.add(layers.Dense(32, activation="relu"))

# Capa de salida: un solo valor entre 0 y 1
model.add(layers.Dense(1, activation="sigmoid"))

# Compilar el modelo con Adam y binary_crossentropy
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# ===============================
# 6. Entrenamiento del modelo
# ===============================

# Parada temprana más tolerante
early_stop = EarlyStopping(
    monitor="val_loss",  # Observa la pérdida del conjunto de validación
    patience=30,  # Permite más épocas sin mejora
    restore_best_weights=True,  # Recupera los mejores pesos
)

# (Opcional) Ajuste de pesos de clase si es necesario
class_weight = {0: 1.4, 1: 1.0}  # Ligeramente más peso a la clase No Lluvia

# Entrenar el modelo
history = model.fit(
    X_train,
    y_train,
    epochs=150,  # Más épocas para refinar el aprendizaje
    batch_size=16,  # Tamaño de mini-lote
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    class_weight=class_weight,  # Activado
    verbose=1,
)

# ===============================
# 7. Evaluación con métricas
# ===============================

# 1. Obtener predicciones en el conjunto de prueba
y_pred_prob = model.predict(X_test)

# 2. Convertir probabilidades a clases (0 o 1) usando umbral de 0.5
# 2. Ajustar umbral para mejorar desempeño
umbral = 0.45
y_pred = (y_pred_prob > umbral).astype(int).flatten()

# 3. Mostrar matriz de confusión
print("\nMatriz de Confusión:")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["No Lluvia", "Lluvia"]
)
disp.plot(cmap="Blues")

# 4. Reporte completo con precision, recall y F1-score
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=["No Lluvia", "Lluvia"]))

# ===============================
# 7. Gráficas de entrenamiento
# ===============================

# Pérdida (loss)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Entrenamiento")
plt.plot(history.history["val_loss"], label="Validación")
plt.title("Pérdida (Loss)")
plt.xlabel("Épocas")
plt.ylabel("Pérdida")
plt.legend()

# Precisión (accuracy)
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Entrenamiento")
plt.plot(history.history["val_accuracy"], label="Validación")
plt.title("Precisión (Accuracy)")
plt.xlabel("Épocas")
plt.ylabel("Precisión")
plt.legend()

plt.tight_layout()
plt.show()
