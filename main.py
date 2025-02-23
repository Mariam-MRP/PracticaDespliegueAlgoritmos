import argparse
from Generar import cargar_datos, entrenar_y_evaluar_modelo

def main():
    parser = argparse.ArgumentParser(description="Entrenar y evaluar un modelo de regresión lineal variando el test size.")
    parser.add_argument('ruta', type=str, help="Ruta al archivo CSV con los datos.")
    parser.add_argument('--test_sizes', nargs='+', type=float, default=[0.1, 0.15, 0.2], help="Lista de tamaños de prueba. Ejemplo: --test_sizes 0.1 0.15 0.2")
    parser.add_argument('--guardar_modelos', action='store_true')

    
    args = parser.parse_args()

    
    X, y = cargar_datos(args.ruta)

    
    entrenar_y_evaluar_modelo(X, y, args.test_sizes, args.guardar_modelos)

if __name__ == "__main__":
    main()
