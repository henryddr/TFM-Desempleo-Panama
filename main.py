"""
Pipeline principal del proyecto TFM: Aplicacion de Machine Learning para la
Prediccion de Desempleo Regional en Panama.

Ejecuta todas las etapas del proyecto de forma secuencial:
  1. Procesamiento de datos crudos del INEC
  2. Feature engineering
  3. Entrenamiento y evaluacion de modelos
  4. Generacion de mapas y visualizaciones

Uso:
    python main.py              # Ejecutar todo el pipeline
    python main.py --desde 2    # Ejecutar desde el paso 2 en adelante
    python main.py --solo 3     # Ejecutar solo el paso 3
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.config import PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR


PASOS = {
    1: ("Procesamiento de datos INEC", "src.data.procesar_datos_inec"),
    2: ("Feature engineering", "src.features.crear_features"),
    3: ("Entrenamiento de modelos", "src.models.entrenar_modelo"),
    4: ("Generacion de mapas", "src.visualization.mapas"),
}


def verificar_prerequisitos(paso):
    """Verifica que los archivos necesarios para cada paso existan."""
    if paso == 2:
        f = PROCESSED_DATA_DIR / "desempleo_por_provincia.csv"
        if not f.exists():
            print(f"  ERROR: Falta {f.name}. Ejecute el paso 1 primero.")
            return False
    elif paso == 3:
        f = PROCESSED_DATA_DIR / "features_desempleo.csv"
        if not f.exists():
            print(f"  ERROR: Falta {f.name}. Ejecute el paso 2 primero.")
            return False
    elif paso == 4:
        f = REPORTS_DIR / "predicciones_modelo.csv"
        if not f.exists():
            print(f"  AVISO: {f.name} no encontrado. Los mapas de prediccion "
                  f"no se generaran. Ejecute el paso 3 primero.")
    return True


def ejecutar_paso(numero, nombre, modulo_path):
    """Ejecuta un paso del pipeline."""
    print()
    print("=" * 70)
    print(f"  PASO {numero}: {nombre.upper()}")
    print("=" * 70)

    if not verificar_prerequisitos(numero):
        return False

    inicio = time.time()
    try:
        modulo = __import__(modulo_path, fromlist=['main'])
        modulo.main()
        duracion = time.time() - inicio
        print(f"\n  Paso {numero} completado en {duracion:.1f}s")
        return True
    except Exception as e:
        duracion = time.time() - inicio
        print(f"\n  ERROR en paso {numero} ({duracion:.1f}s): {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline TFM: Prediccion de Desempleo Regional en Panama"
    )
    parser.add_argument(
        "--desde", type=int, choices=[1, 2, 3, 4], default=1,
        help="Paso desde el cual iniciar (default: 1)"
    )
    parser.add_argument(
        "--solo", type=int, choices=[1, 2, 3, 4], default=None,
        help="Ejecutar solo este paso"
    )
    args = parser.parse_args()

    print()
    print("#" * 70)
    print("#  PIPELINE TFM: PREDICCION DE DESEMPLEO REGIONAL - PANAMA")
    print("#" * 70)

    inicio_total = time.time()

    if args.solo:
        pasos_a_ejecutar = [args.solo]
    else:
        pasos_a_ejecutar = [p for p in PASOS if p >= args.desde]

    resultados = {}
    for num in pasos_a_ejecutar:
        nombre, modulo = PASOS[num]
        ok = ejecutar_paso(num, nombre, modulo)
        resultados[num] = ok
        if not ok and num < max(pasos_a_ejecutar):
            print(f"\n  Pipeline detenido en paso {num}. "
                  f"Corrija el error y ejecute: python main.py --desde {num}")
            break

    # Resumen final
    duracion_total = time.time() - inicio_total
    print()
    print("#" * 70)
    print("#  RESUMEN DEL PIPELINE")
    print("#" * 70)
    for num, ok in resultados.items():
        estado = "OK" if ok else "ERROR"
        print(f"  Paso {num}: {PASOS[num][0]:<35} [{estado}]")
    print(f"\n  Duracion total: {duracion_total:.1f}s")

    exitos = sum(resultados.values())
    total = len(resultados)
    if exitos == total:
        print(f"\n  Pipeline completado: {exitos}/{total} pasos exitosos")
    else:
        print(f"\n  Pipeline con errores: {exitos}/{total} pasos exitosos")
        sys.exit(1)


if __name__ == "__main__":
    main()
