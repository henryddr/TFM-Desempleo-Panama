"""
Script para descargar series históricas de la Encuesta de Mercado Laboral
INEC - Panamá (2015-2024)

Este script descarga los cuadros estadísticos de las encuestas de:
- Marzo 2015
- Agosto 2015
- Marzo 2016
- Agosto 2016
- Marzo 2017
- Agosto 2017
- Agosto 2018
- Agosto 2019
- Septiembre 2020 (telefónica por COVID)
- Octubre 2021
- Agosto 2023
- Octubre 2024

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
from bs4 import BeautifulSoup
import time

# Configuración
ROOT_DIR = Path(__file__).parent.parent.parent
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://www.inec.gob.pa"

# IDs de publicaciones disponibles
PUBLICACIONES = {
    '2012_Agosto': {
        'id': 485,
        'carpeta': 'inec_agosto_2012'
    },
    '2013_Agosto': {
        'id': 557,
        'carpeta': 'inec_agosto_2013'
    },
    '2014_Agosto': {
        'id': 636,
        'carpeta': 'inec_agosto_2014'
    },
    '2015_Marzo': {
        'id': 682,
        'carpeta': 'inec_marzo_2015'
    },
    '2015_Agosto': {
        'id': 717,
        'carpeta': 'inec_agosto_2015'
    },
    '2016_Marzo': {
        'id': 751,
        'carpeta': 'inec_marzo_2016'
    },
    '2016_Agosto': {
        'id': 784,
        'carpeta': 'inec_agosto_2016'
    },
    '2017_Marzo': {
        'id': 818,
        'carpeta': 'inec_marzo_2017'
    },
    '2017_Agosto': {
        'id': 856,
        'carpeta': 'inec_agosto_2017'
    },
    '2018_Agosto': {
        'id': 904,
        'carpeta': 'inec_agosto_2018'
    },
    '2019_Agosto': {
        'id': 971,
        'carpeta': 'inec_agosto_2019'
    },
    '2020_Septiembre': {
        'id': 1037,
        'carpeta': 'inec_septiembre_2020'
    },
    '2021_Octubre': {
        'id': 1106,
        'carpeta': 'inec_octubre_2021'
    },
    '2023_Agosto': {
        'id': 1209,
        'carpeta': 'inec_agosto_2023'
    },
    # 2024 ya lo descargamos anteriormente
}


def extraer_enlaces_excel(id_publicacion):
    """
    Extrae los enlaces de archivos Excel de una página de publicación del INEC

    Args:
        id_publicacion (int): ID de la publicación

    Returns:
        list: Lista de tuplas (nombre, url) de archivos Excel
    """
    try:
        url = f"{BASE_URL}/publicaciones/Default3.aspx?ID_PUBLICACION={id_publicacion}&ID_CATEGORIA=5&ID_SUBCATEGORIA=38"

        print(f"   🔍 Explorando publicación ID {id_publicacion}...")

        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Parsear HTML
        soup = BeautifulSoup(response.content, 'html.parser')

        # Buscar enlaces a archivos Excel
        archivos = []
        for link in soup.find_all('a', href=True):
            href = link.get('href', '')

            # Verificar si es un archivo Excel
            if '.xls' in href.lower():
                nombre = link.text.strip() or link.get('title', '')

                # Construir URL completa
                if href.startswith('../'):
                    url_completa = BASE_URL + '/' + href.replace('../', '')
                elif href.startswith('/'):
                    url_completa = BASE_URL + href
                elif not href.startswith('http'):
                    url_completa = BASE_URL + '/' + href
                else:
                    url_completa = href

                # Limpiar nombre
                if not nombre:
                    # Extraer nombre del archivo de la URL
                    nombre = href.split('/')[-1]

                archivos.append((nombre, url_completa))

        print(f"   ✅ Encontrados {len(archivos)} archivos Excel")
        return archivos

    except Exception as e:
        print(f"   ❌ Error al extraer enlaces: {e}")
        return []


def descargar_archivo(nombre, url, carpeta_destino):
    """
    Descarga un archivo Excel

    Args:
        nombre (str): Nombre del archivo
        url (str): URL del archivo
        carpeta_destino (Path): Carpeta donde guardar

    Returns:
        bool: True si la descarga fue exitosa
    """
    try:
        # Limpiar nombre de archivo
        nombre_archivo = url.split('/')[-1]
        ruta_destino = carpeta_destino / nombre_archivo

        # Verificar si ya existe
        if ruta_destino.exists():
            return True  # Ya existe

        # Descargar
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()

        # Guardar
        with open(ruta_destino, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        tamaño_kb = ruta_destino.stat().st_size / 1024
        print(f"      ✅ {nombre_archivo}: {tamaño_kb:.1f} KB")

        time.sleep(0.3)  # Pausa pequeña
        return True

    except Exception as e:
        print(f"      ❌ Error: {e}")
        return False


def descargar_publicacion(nombre_periodo, info):
    """
    Descarga todos los archivos de una publicación

    Args:
        nombre_periodo (str): Nombre del período (ej: '2018_Agosto')
        info (dict): Información de la publicación

    Returns:
        tuple: (exitosos, fallidos)
    """
    print(f"\n{'='*70}")
    print(f"📅 {nombre_periodo.replace('_', ' ')}")
    print(f"{'='*70}")

    carpeta = RAW_DATA_DIR / info['carpeta']
    carpeta.mkdir(parents=True, exist_ok=True)

    # Extraer enlaces
    archivos = extraer_enlaces_excel(info['id'])

    if not archivos:
        print("   ⚠️  No se encontraron archivos Excel en esta publicación")
        return 0, 0

    # Descargar archivos
    print(f"   📥 Descargando {len(archivos)} archivos...")

    exitosos = 0
    fallidos = 0

    for nombre, url in archivos:
        if descargar_archivo(nombre, url, carpeta):
            exitosos += 1
        else:
            fallidos += 1

    print(f"\n   ✅ Exitosos: {exitosos}")
    if fallidos > 0:
        print(f"   ❌ Fallidos: {fallidos}")

    return exitosos, fallidos


def main():
    """
    Función principal
    """
    print("\n" + "="*70)
    print("📊 DESCARGANDO SERIES HISTÓRICAS DEL INEC (2015-2024)")
    print("="*70)
    print(f"Total de períodos: {len(PUBLICACIONES)}")
    print(f"Carpeta destino: {RAW_DATA_DIR}")
    print("="*70)

    total_exitosos = 0
    total_fallidos = 0

    for nombre_periodo, info in PUBLICACIONES.items():
        exitosos, fallidos = descargar_publicacion(nombre_periodo, info)
        total_exitosos += exitosos
        total_fallidos += fallidos

    # Resumen final
    print("\n" + "="*70)
    print("📊 RESUMEN GENERAL")
    print("="*70)
    print(f"✅ Total archivos descargados: {total_exitosos}")
    if total_fallidos > 0:
        print(f"❌ Total archivos fallidos: {total_fallidos}")
    print(f"📁 Períodos procesados: {len(PUBLICACIONES)}")

    print("\n" + "="*70)
    print("✅ PROCESO COMPLETADO")
    print("="*70)
    print(f"\n📂 Revisa las carpetas en: {RAW_DATA_DIR}")
    print("\n💡 PRÓXIMOS PASOS:")
    print("   1. Explorar los archivos descargados")
    print("   2. Identificar cuadros con datos por provincia")
    print("   3. Crear script de consolidación de series temporales")
    print("   4. Procesar y limpiar los datos")

    print("\n🎯 DATOS NECESARIOS PARA EL MODELO:")
    print("   - Tasa de desempleo por provincia (trimestral)")
    print("   - Población económicamente activa")
    print("   - Nivel educativo por región")
    print("   - Sector de actividad económica")


if __name__ == "__main__":
    main()
