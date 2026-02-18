"""
Script para descargar todos los cuadros estadísticos del INEC
Encuesta de Mercado Laboral - Octubre 2024

Total: 43 archivos Excel con datos detallados de:
- Población ocupada y desocupada
- Tasas de desempleo por provincia
- Sector de actividad económica
- Nivel educativo
- Y mucho más

Autor: Henry De Gracia
Fecha: Enero 2025
"""

import sys
import os

# Configurar codificación UTF-8 para Windows
if sys.platform == 'win32':
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    if sys.stderr.encoding != 'utf-8':
        sys.stderr.reconfigure(encoding='utf-8')

import requests
from pathlib import Path
import time

# Configuración
ROOT_DIR = Path(__file__).parent.parent.parent
RAW_DATA_DIR = ROOT_DIR / "data" / "raw" / "inec_octubre_2024"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://www.inec.gob.pa/archivos/"

# Lista completa de archivos disponibles
CUADROS = {
    'Cuadro 1': 'P0579518620250124083403Cuadro 1.xlsx',
    'Cuadro 1A': 'P0705547520250124083639Cuadro 1A.xlsx',
    'Cuadro 2': 'P0289562520250124083712Cuadro 2.xlsx',
    'Cuadro 3': 'P0705547520250124084230Cuadro 3.xlsx',
    'Cuadro 4': 'P0705547520250124084848Cuadro 4.xlsx',
    'Cuadro 5': 'P0705547520250124084921Cuadro 5.xlsx',
    'Cuadro 6': 'P0705547520250124085041Cuadro 6.xlsx',
    'Cuadro 7': 'P053342420250124085121Cuadro 7.xlsx',
    'Cuadro 8': 'P0579518620250124085153Cuadro 8.xlsx',
    'Cuadro 9': 'P0289562520250124085232cuadro 9.xlsx',
    'Cuadro 10': 'P0579518620250124085300Cuadro 10.xlsx',
    'Cuadro 11': 'P0705547520250124085408Cuadro 11.xlsx',
    'Cuadro 12': 'P030194820250124085443Cuadro 12.xlsx',
    'Cuadro 13': 'P0705547520250124085534Cuadro 13.xlsx',
    'Cuadro 14': 'P0705547520250124085621Cuadro 14.xlsx',
    'Cuadro 15': 'P053342420250124085717Cuadro 15.xlsx',
    'Cuadro 16': 'P0705547520250124085813Cuadro 16.xlsx',
    'Cuadro 17': 'P0705547520250124090325Cuadro 17.xlsx',
    'Cuadro 18': 'P0705547520250124094522Cuadro 18.xlsx',
    'Cuadro 19': 'P053342420250124094618Cuadro 19.xlsx',
    'Cuadro 20': 'P0579518620250124094654Cuadro 20.xlsx',
    'Cuadro 21': 'P0705547520250124094739Cuadro 21.xlsx',
    'Cuadro 22': 'P0705547520250124094821Cuadro 22.xlsx',
    'Cuadro 23': 'P0705547520250124094901Cuadro 23.xlsx',
    'Cuadro 24': 'P0705547520250124094944Cuadro 24.xlsx',
    'Cuadro 25': 'P0705547520250124095018Cuadro 25.xlsx',
    'Cuadro 26': 'P053342420250124095048Cuadro 26.xlsx',
    'Cuadro 27': 'P0705547520250124095137Cuadro 27.xlsx',
    'Cuadro 28': 'P053342420250124095223Cuadro 28.xlsx',
    'Cuadro 29': 'P053342420250124095331Cuadro 29.xlsx',
    'Cuadro 30': 'P0579518620250124095403Cuadro 30.xlsx',
    'Cuadro 31': 'P0289562520250124095437Cuadro 31.xlsx',
    'Cuadro 32': 'P053342420250124095509Cuadro 32.xlsx',
    'Cuadro 33': 'P0579518620250124095557Cuadro 33.xlsx',
    'Cuadro 34': 'P053342420250124095635Cuadro 34.xlsx',
    'Cuadro 35': 'P0289562520250124095704Cuadro 35.xlsx',
    'Cuadro 36': 'P0579518620250124095802Cuadro 36.xlsx',
    'Cuadro 37': 'P030194820250124095835Cuadro 37.xlsx',
    'Cuadro 38': 'P0289562520250124095916Cuadro 38.xlsx',
    'Cuadro 39': 'P053342420250124100001Cuadro 39.xlsx',
    'Grafica 1': 'P0705547520250124100944Grafica 1.xlsx',
    'Grafica 2': 'P053342420250124101024Grafica 2.xlsx',
    'Grafica 3': 'P0705547520250124101100Grafica 3.xlsx',
    'Grafica 4': 'P053342420250124101131Grafica 4.xlsx',
}


def descargar_archivo(nombre, archivo):
    """
    Descarga un archivo Excel del INEC

    Args:
        nombre (str): Nombre descriptivo del archivo
        archivo (str): Nombre del archivo en el servidor

    Returns:
        bool: True si la descarga fue exitosa
    """
    try:
        url = BASE_URL + archivo
        ruta_destino = RAW_DATA_DIR / archivo

        # Verificar si ya existe
        if ruta_destino.exists():
            print(f"⏭️  {nombre}: Ya existe, saltando...")
            return True

        print(f"📥 Descargando: {nombre}")

        # Descargar con timeout
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()

        # Guardar archivo
        with open(ruta_destino, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        tamaño_kb = ruta_destino.stat().st_size / 1024
        print(f"   ✅ Descargado: {tamaño_kb:.1f} KB")

        # Pausa pequeña para no saturar el servidor
        time.sleep(0.5)

        return True

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"   ⚠️  No encontrado (404)")
        else:
            print(f"   ❌ Error HTTP {e.response.status_code}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Error de conexión: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Error inesperado: {e}")
        return False


def main():
    """
    Función principal para descargar todos los cuadros
    """
    print("\n" + "="*70)
    print("📊 DESCARGANDO CUADROS ESTADÍSTICOS DEL INEC")
    print("="*70)
    print(f"Fuente: Encuesta de Mercado Laboral - Octubre 2024")
    print(f"Total de archivos: {len(CUADROS)}")
    print(f"Carpeta destino: {RAW_DATA_DIR}")
    print("="*70)
    print()

    exitosos = 0
    fallidos = 0
    saltados = 0

    for i, (nombre, archivo) in enumerate(CUADROS.items(), 1):
        print(f"[{i}/{len(CUADROS)}] ", end="")

        # Verificar si ya existe
        ruta = RAW_DATA_DIR / archivo
        if ruta.exists():
            print(f"⏭️  {nombre}: Ya existe")
            saltados += 1
            continue

        resultado = descargar_archivo(nombre, archivo)
        if resultado:
            exitosos += 1
        else:
            fallidos += 1

    # Resumen final
    print("\n" + "="*70)
    print("📊 RESUMEN DE DESCARGA")
    print("="*70)
    print(f"✅ Exitosos: {exitosos}")
    print(f"⏭️  Saltados (ya existían): {saltados}")
    print(f"❌ Fallidos: {fallidos}")
    print(f"📁 Total de archivos: {len(CUADROS)}")

    if fallidos > 0:
        print(f"\n⚠️  Hubo {fallidos} archivos que no se pudieron descargar")
        print("   Esto puede deberse a:")
        print("   - Archivos temporalmente no disponibles")
        print("   - Problemas de conexión")
        print("   - URLs actualizadas en el servidor")
        print("\n   Puedes volver a ejecutar este script más tarde")

    print("\n" + "="*70)
    print("✅ PROCESO COMPLETADO")
    print("="*70)
    print(f"\n📂 Archivos guardados en: {RAW_DATA_DIR}")
    print("\n💡 PRÓXIMO PASO:")
    print("   Explora los archivos descargados para identificar cuáles")
    print("   contienen los datos más relevantes para tu modelo.")
    print("\n   Archivos clave probablemente incluyen:")
    print("   - Desempleo por provincia")
    print("   - Población económicamente activa")
    print("   - Nivel educativo de la población")
    print("   - Sector de actividad económica")


if __name__ == "__main__":
    main()
