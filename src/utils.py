# utils.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt
import pickle

# Funciones para cargar datos y preparar datos

def load_data(path_or_url):
    """
    Carga datos desde una ruta local o URL.

    :param path_or_url: Ruta local o URL del archivo .csv
    :return: un dataframe de pandas
    """
    print(f"Cargando datos desde: {path_or_url}")
    df = pd.read_csv(path_or_url)
    print(f"Datos cargados correctamente")
    return df

def split_data(df, var_obj):
    """
    Divide el DataFrame en conjuntos de entrenamiento y prueba.

    :param df: DataFrame listo para el modelo
    :param var_obj: Nombre de la variable objetivo (string)
    :return: X_train, X_test, y_train, y_test
    """
    print(f"Dividiendo los datos en entrenamiento y prueba")
    # Selecciono la data para X quitando la variable objetivo
    X = df.drop(var_obj, axis=1)

    # Selecciono solo la variable objetivo para y
    y = df[var_obj]

    # Utilizo train_test_split para dividir la data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Datos divididos correctamente")
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, random_state=42):
    """
    Inicializa y entrena un modelo RandomForestClassifier.

    :param X_train: DataFrame de características de entrenamiento
    :param y_train: Serie de la variable objetivo de entrenamiento
    :param random_state: Semilla para reproducibilidad
    :return: Modelo RandomForestClassifier entrenado
    """
    print(f"Entrenando modelo RandomForestClassifier")
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    print("Modelo entrenado correctamente")
    return model

def evaluate_model(model, X_test, y_test):
    """
    Realiza predicciones y evalúa el modelo usando accuracy.

    :param model: Modelo entrenado
    :param X_test: DataFrame de características de prueba
    :param y_test: Serie de la variable objetivo de prueba
    :return: Accuracy del modelo
    """
    print(f"Realizando predicciones en el conjunto de prueba")
    y_pred = model.predict(X_test)
    print(f"Evaluando modelo")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy del modelo: {accuracy:.4f}")
    return accuracy, y_pred # Devuelve también y_pred por si se necesita

def plot_trees(model, feature_names, num_trees=4):
    """
    Visualiza los primeros árboles de decisión del RandomForest.

    :param model: Modelo RandomForest entrenado
    :param feature_names: Lista con los nombres de las características
    :param num_trees: Número de árboles a visualizar (default: 4)
    :return: None (muestra la gráfica)
    """
    print(f"Visualizando los primeros {num_trees} árboles del modelo")
    # Ajusta el tamaño de la figura según el número de árboles
    cols = 2
    rows = (num_trees + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10 * cols, 8 * rows))
    axes = axes.flatten() # Asegura que axes sea siempre iterable

    # Mostramos los primeros num_trees árboles
    for i in range(num_trees):
        if i < len(model.estimators_):
            ax = axes[i]
            tree.plot_tree(model.estimators_[i],
                           ax=ax,
                           feature_names=feature_names,
                           filled=True,
                           class_names=['No Diabetes', 'Diabetes'], # Asumiendo 0 y 1
                           rounded=True,
                           proportion=False,
                           precision=2,
                           fontsize=8) # Ajusta el tamaño de fuente si es necesario
            ax.set_title(f'Árbol Estimador {i+1}')
        else:
            # Oculta ejes si hay menos estimadores que subplots
             axes[i].axis('off')

    # Oculta ejes sobrantes si num_trees no es par
    for i in range(num_trees, len(axes)):
         axes[i].axis('off')

    plt.tight_layout() # Ajusta el layout para evitar solapamientos
    plt.show()
    print("Visualización de árboles completada")

def save_model(model, filepath):
    """
    Guarda el modelo entrenado en un archivo pickle.

    :param model: modelo entrenado
    :param filepath: ruta completa del archivo .pkl donde se guardará el modelo
    :return: None
    """
    print(f"Guardando el modelo en: {filepath}")
    with open(filepath, 'wb') as file:
        pickle.dump(model, file)
    print("Modelo guardado correctamente")

