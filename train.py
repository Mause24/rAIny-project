import pandas as pd
import tensorflow as tf
import numpy as np
import joblib
import random
import os
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import EarlyStopping

from datetime import datetime

# Semilla global para reproducibilidad
# seed = 73
# os.environ["PYTHONHASHSEED"] = str(seed)
# np.random.seed(seed)
# random.seed(seed)
# tf.random.set_seed(seed)

# ===============================
# 1. Leer el dataset y transformarlo en Dataframe con pandas
# ===============================

# Ruta a todos los CSVs
dataset_paths = glob.glob("data/rAIny_training_dataset_*.csv")

# Leer todos los datasets en una lista
dfs = [pd.read_csv(path, delimiter=";") for path in dataset_paths]

# Concatenar todos en un solo DataFrame
df = pd.concat(dfs, ignore_index=True)

# Reemplazar valores faltantes (-999) por NaN
df.replace(-999, np.nan, inplace=True)

# Opcional: eliminar filas con valores faltantes
df.dropna(inplace=True)

# ===============================
# 2. Definir la variable de clasde GRP mediante un parametro binario (Llovio o no)
# ===============================

df["GRP"] = (df["PRECTOTCORR"] > 0.1).astype(int)

print(df["GRP"].value_counts())
# ===============================
# 3. Definir las variables independientes que va a tomar el modelo para realizar la prediccion
# ===============================

features = [
    "T2M",  # Temperatura a 2 m
    "WS2M",  # Vel. viento a 2 m
    "QV2M",  # Humedad específica
    "RH2M",  # Humedad relativa
    "ALLSKY_SFC_SW_DWN",  # Irradiancia solar
    "PS",  # Presión superficial
]

X = df[features]
y = df["GRP"]

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
model.add(layers.Dense(256, activation="relu", input_shape=(X_train.shape[1],)))

# Capa 1 oculta
model.add(
    layers.Dense(
        128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)
    )
)

# Capa 2 oculta
model.add(layers.Dense(64, activation="relu"))

# Dropout para regular el tema del overfitting
model.add(layers.Dropout(0.3))

# Capa 3 oculta
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
    patience=80,  # Permite más épocas sin mejora
    restore_best_weights=True,  # Recupera los mejores pesos
)

# (Opcional) Ajuste de pesos de clase si es necesario
class_weight = {0: 1.9, 1: 1.0}

# Entrenar el modelo
history = model.fit(
    X_train,
    y_train,
    epochs=350,  # Más épocas para refinar el aprendizaje
    batch_size=128,  # Tamaño de mini-lote
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    class_weight=class_weight,  # Activado
    verbose=1,
)

# ===============================
# 7. Evaluación con métricas
# ===============================


def expected_calibration_error(y_true, y_prob, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    bin_counts = np.histogram(y_prob, bins=n_bins)[0]
    bin_weights = bin_counts / np.sum(bin_counts)
    ece = np.sum(bin_weights * np.abs(prob_true - prob_pred))
    return ece


# 1. Obtener predicciones en el conjunto de prueba
y_pred_prob = model.predict(X_test)

# Evaluación binaria (opcional, para métricas tradicionales)
umbral = 0.45
y_pred = (y_pred_prob > umbral).astype(int).flatten()

# Matriz de confusión
print("\nConfussion Matrix:")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Rain", "Rain"])
disp.plot(cmap="Blues")

# Reporte de métricas
print(f"\nClassfication report (threshold = {umbral}):")
print(classification_report(y_test, y_pred, target_names=["No rain", "Rain"]))

# Calcular ROC-AUC
roc_auc = roc_auc_score(y_test, y_pred_prob)
print(f"\nROC-AUC: {roc_auc:.4f}")

# Calcular puntos ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Graficar la curva ROC
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "k--", label="Random model")
plt.xlabel("False Positive Rate(FPR)")
plt.ylabel("Tasa de verdaderos positivos (TPR)")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Calcular la curva y el área
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
pr_auc = average_precision_score(y_test, y_pred_prob)

print(f"PR-AUC: {pr_auc:.4f}")

# Graficar
plt.figure(figsize=(6, 6))
plt.plot(recall, precision, label=f"PR curve (AUC = {pr_auc:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()

brier = brier_score_loss(y_test, y_pred_prob)
print(f"Brier Score: {brier:.4f}")

# Calcular el modelo base (predice probabilidad promedio)
p_ref = np.mean(y_test)
brier_base = brier_score_loss(y_test, np.full_like(y_test, p_ref))
bss = 1 - (brier / brier_base)
print(f"Brier Skill Score (BSS): {bss:.4f}")


# Calcular la curva de calibración
prob_true, prob_pred = calibration_curve(y_test, y_pred_prob, n_bins=10)

# Graficar
plt.figure(figsize=(6, 6))
plt.plot(prob_pred, prob_true, marker="o", label="Model")
plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
plt.xlabel("Average predicted probability")
plt.ylabel("Observed Frequency")
plt.title("Reliability Diagram")
plt.legend()
plt.show()

ece = expected_calibration_error(y_test, y_pred_prob)
print(f"Expected Calibration Error (ECE): {ece:.4f}")


sensitivity = recall_score(y_test, y_pred, pos_label=1)
print(f"Sensitivity (Lluvia): {sensitivity:.4f}")

# ===============================
# 8. Gráficas de entrenamiento
# ===============================

# Pérdida (loss)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Training")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("Loss")
plt.xlabel("Epochs")
plt.ylabel("Losts")
plt.legend()

# Precisión (accuracy)
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Precision")
plt.legend()

plt.tight_layout()
plt.show()

# ===============================
# 9. Mostrar primeras predicciones probabilísticasñan
#
# ===============================

print("\nExamples of rain probability of the test group:")
for i, prob in enumerate(y_pred_prob[:10]):
    print(f"Sample {i+1}: {prob[0]*100:.2f}% rain probability")


# ===============================
# 10. Guardar el modelo completo en formato .h5
# ===============================

# Crear carpeta 'models' si no existe
os.makedirs("models", exist_ok=True)

# Obtener fecha y hora actual
fecha_hora = datetime.now().strftime("%Y-%m-%d_%H-%M")
nombre_archivo = f"rAIny_model_{fecha_hora}.h5"

# Guardar el modelo
# model.save(f"models/{nombre_archivo}")
# print(f"Saved model as: models/{nombre_archivo}")

# ===============================
# 11. Guardar el escalador StandardScaler
# ===============================

# scaler_filename = f"models/scaler_{fecha_hora}.joblib"
# joblib.dump(scaler, scaler_filename)
# print(f"✅ Scaler saved as: {scaler_filename}")
