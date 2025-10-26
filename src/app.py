# app.py
from utils import load_data, split_data, train_model, evaluate_model, plot_trees, save_model
import pandas as pd # Aunque utils lo importa, es buena práctica importarlo aquí si se usa directamente

# --- 1: Carga de datos ---
file_path = '../data/raw/diabetes.csv'
total_data = load_data(file_path)

# Mostrar las primeras filas como en el notebook
print("\nPrimeras 5 filas del dataset:")
print(total_data.head())

# --- 2: División de datos ---
# Se usa la función de utils ahora
X_train, X_test, y_train, y_test = split_data(total_data, 'Outcome')

# Mostrar las primeras filas de X_train como en el notebook
print("\nPrimeras 5 filas de X_train:")
print(X_train.head())
print("-" * 30)

# --- 3: Inicialización y entrenamiento del modelo ---
# Se usa la función de utils
model = train_model(X_train, y_train, random_state=42)
print("-" * 30)

# --- 4: Predicción y Evaluación ---
# Se usa la función de utils
accuracy, y_pred = evaluate_model(model, X_test, y_test)
print(f"\nPredicciones (primeras 20): {y_pred[:20]}") # Mostrar algunas predicciones como referencia
print(f"Resultado de Accuracy: {accuracy:.4f}")
print("-" * 30)

# --- 5: Visualización de Árboles ---
# Obtener nombres de características
feature_names = list(X_train.columns)
# Visualizar los primeros 4 árboles como en el notebook
plot_trees(model, feature_names, num_trees=4)
print("-" * 30)

# --- 6: Guardar Modelo
model_filepath = '../models/random_forest_diabetes_model.pkl'
save_model(model, model_filepath)
print("\n--- Proceso completado ---")