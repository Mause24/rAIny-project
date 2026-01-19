from tensorflow.keras.models import load_model
import joblib
import numpy as np

# Cargar modelo y escalador
modelo = load_model("models/rAIny_model_2025-11-10_13-58.h5")
scaler = joblib.load("models/scaler_2025-11-10_13-58.joblib")

# Datos nuevos (sin escalar)
nuevos_datos = np.array([[27.3, 2.1, 14.2, 80.0, 18.5, 101.2]])

# Escalarlos
nuevos_datos_scaled = scaler.transform(nuevos_datos)

# Predecir
prob = modelo.predict(nuevos_datos_scaled)[0][0]
print(f"Probabilidad de lluvia: {prob*100:.2f}%")
