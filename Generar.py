import pandas as pd
import numpy as np
import sklearn
import mlflow
import mlflow.sklearn
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


def cargar_datos(ruta):
    data = pd.read_csv(ruta, sep=";")
    print("Información del dataset:")
    print(data.info())
    print("\nValores nulos :")
    print(data.isnull().sum())
    print("\Datos descriptivos:")
    print(data.describe())

    # Renombrar columnas 
    data = data.rename(columns={"G1": "Nota1", "G2": "Nota2", "G3": "NotaFinal"})

    # Seleccionar los parametros
    data = data[["studytime", "failures", "absences", "NotaFinal"]]

    target = "NotaFinal"

    
    X = np.array(data.drop([target], axis=1))
    y = np.array(data[target])

    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y


# Entrenar y evaluar el modelo
def entrenar_y_evaluar_modelo(X, y, test_sizes, guardar_modelos):
    mejor_r2 = -np.inf
    mejor_test_size = None

    for test_size in test_sizes:
        
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=test_size)

        with mlflow.start_run(run_name=f"Regresión_Lineal_test_size_{test_size}"):

            
            linear = linear_model.LinearRegression()
            linear.fit(x_train, y_train)

            
            acc = linear.score(x_test, y_test)
            predictions = linear.predict(x_test)
            mae = mean_absolute_error(y_test, predictions)
            rmse = mean_squared_error(y_test, predictions) ** 0.5

            # Registrar parámetros 
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("normalization", "StandardScaler")

            # Registrar métricas 
            mlflow.log_metric("R2_score", acc)
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("RMSE", rmse)

            # Registrar coeficientes
            for i, coef in enumerate(linear.coef_):
                mlflow.log_param(f"coef_{i}", coef)

            
            if guardar_modelos:
                mlflow.sklearn.log_model(linear, f"modelo_regresion_test_size_{test_size}", input_example=x_test[:1])

            # Mostrar los resultados
            print(f"\nResultados para test_size = {test_size}")
            print(f"R2 Score: {acc}")
            print(f"MAE: {mae}")
            print(f"RMSE: {rmse}")
            print('Coeficientes: \n', linear.coef_)
            print('Término independiente: \n', linear.intercept_)

            # Guardar el mejor modelo 
            if acc > mejor_r2:
                mejor_r2 = acc
                mejor_test_size = test_size

    print(f"\nEl mejor modelo fue con test_size = {mejor_test_size} con un R2 Score de {mejor_r2:.4f}")
