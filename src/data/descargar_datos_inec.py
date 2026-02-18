"""
Script para descargar datos del INEC (Instituto Nacional de Estadística y Censo de Panamá)

Fuentes:
- Portal de Datos Abiertos: datosabiertos.gob.pa
- INEC Database Hub: inec.gob.pa/dbshub/
- Publicaciones oficiales

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
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import time

# Configuración de rutas
ROOT_DIR = Path(__file__).parent.parent.parent
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# URLs de fuentes de datos
URLS = {
    # Portal de datos abiertos
    'datos_abiertos_inec': 'https://www.datosabiertos.gob.pa/organization/instituto-nacional-de-estadistica-y-censo',

    # Reportes específicos
    'eml_octubre_2024': 'https://www.inec.gob.pa/archivos/P0705547520250124083204Comentarios EML octubre 2024.pdf',

    # Base de datos hub
    'dbshub': 'https://www.inec.gob.pa/dbshub/',
}

def descargar_archivo(url, nombre_archivo, carpeta_destino=RAW_DATA_DIR):
    """
    Descarga un archivo desde una URL

    Args:
        url (str): URL del archivo a descargar
        nombre_archivo (str): Nombre con el que guardar el archivo
        carpeta_destino (Path): Carpeta donde guardar el archivo

    Returns:
        bool: True si la descarga fue exitosa
    """
    try:
        print(f"📥 Descargando: {nombre_archivo}")
        print(f"   Desde: {url}")

        # Realizar la petición
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()

        # Guardar el archivo
        ruta_completa = carpeta_destino / nombre_archivo

        with open(ruta_completa, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"✅ Descargado exitosamente en: {ruta_completa}")
        return True

    except requests.exceptions.RequestException as e:
        print(f"❌ Error al descargar {nombre_archivo}: {e}")
        return False
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return False


def descargar_reporte_octubre_2024():
    """
    Descarga el reporte más reciente de la Encuesta de Mercado Laboral (Octubre 2024)
    """
    print("\n" + "="*70)
    print("📊 DESCARGANDO REPORTE EML OCTUBRE 2024")
    print("="*70)

    url = URLS['eml_octubre_2024']
    nombre = "EML_Octubre_2024.pdf"

    return descargar_archivo(url, nombre)


def crear_registro_descarga():
    """
    Crea un archivo JSON con el registro de descargas realizadas
    """
    registro = {
        'fecha_descarga': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'archivos_descargados': [],
        'fuentes': URLS,
        'notas': [
            'Datos bajo licencia CC BY 4.0',
            'Atribución requerida: INEC - Panamá',
            'Para datos completos, visitar: https://www.inec.gob.pa/dbshub/'
        ]
    }

    # Listar archivos descargados
    for archivo in RAW_DATA_DIR.glob('*'):
        if archivo.is_file() and archivo.name != '.gitkeep':
            registro['archivos_descargados'].append({
                'nombre': archivo.name,
                'tamaño_mb': round(archivo.stat().st_size / (1024*1024), 2),
                'fecha_modificacion': datetime.fromtimestamp(
                    archivo.stat().st_mtime
                ).strftime('%Y-%m-%d %H:%M:%S')
            })

    # Guardar registro
    ruta_registro = RAW_DATA_DIR / 'registro_descargas.json'
    with open(ruta_registro, 'w', encoding='utf-8') as f:
        json.dump(registro, f, indent=2, ensure_ascii=False)

    print(f"\n📝 Registro de descargas guardado en: {ruta_registro}")


def listar_datasets_disponibles():
    """
    Muestra información sobre los datasets disponibles del INEC
    """
    print("\n" + "="*70)
    print("📋 DATASETS DISPONIBLES DEL INEC")
    print("="*70)

    datasets = [
        {
            'nombre': 'Encuesta de Mercado Laboral (EML)',
            'descripcion': 'Tasa de desempleo trimestral por provincia',
            'periodo': '2018-2024',
            'frecuencia': 'Trimestral',
            'formato': 'PDF (Reportes), Excel/CSV (datos procesados)',
            'url_manual': 'https://www.inec.gob.pa/publicaciones/Default2.aspx?ID_CATEGORIA=5&ID_SUBCATEGORIA=38'
        },
        {
            'nombre': 'Censo de Población y Vivienda 2023',
            'descripcion': 'Datos demográficos por corregimiento',
            'periodo': '2023',
            'frecuencia': 'Decenal',
            'formato': 'Excel, CSV',
            'url_manual': 'https://www.inec.gob.pa/redpan/index_censospma.html'
        },
        {
            'nombre': 'Panamá en Cifras',
            'descripcion': '170+ variables estadísticas',
            'periodo': 'Histórico',
            'frecuencia': 'Variable',
            'formato': 'Plataforma web interactiva',
            'url_manual': 'https://www.inec.gob.pa/publicaciones/Default2.aspx?ID_CATEGORIA=17&ID_SUBCATEGORIA=45'
        },
        {
            'nombre': 'Datos Abiertos INEC',
            'descripcion': 'Múltiples datasets en formato abierto',
            'periodo': 'Variable',
            'frecuencia': 'Continua',
            'formato': 'CSV, XLS, JSON',
            'url_manual': 'https://www.datosabiertos.gob.pa/organization/instituto-nacional-de-estadistica-y-censo'
        }
    ]

    for i, ds in enumerate(datasets, 1):
        print(f"\n{i}. {ds['nombre']}")
        print(f"   Descripción: {ds['descripcion']}")
        print(f"   Período: {ds['periodo']}")
        print(f"   Frecuencia: {ds['frecuencia']}")
        print(f"   Formato: {ds['formato']}")
        print(f"   📎 URL: {ds['url_manual']}")


def mostrar_instrucciones_descarga_manual():
    """
    Muestra instrucciones para descargar datos manualmente desde el portal INEC
    """
    print("\n" + "="*70)
    print("📖 INSTRUCCIONES PARA DESCARGA MANUAL")
    print("="*70)

    instrucciones = """
    Debido a que el INEC utiliza páginas dinámicas y formularios web,
    algunos datos deben descargarse manualmente. Sigue estos pasos:

    📍 PASO 1: Acceder al Portal de Datos Abiertos
    ────────────────────────────────────────────
    URL: https://www.datosabiertos.gob.pa/organization/instituto-nacional-de-estadistica-y-censo

    1. Filtrar por formato: XLS o CSV
    2. Buscar datasets relacionados con:
       - Mercado laboral
       - Empleo/Desempleo
       - Encuesta de hogares

    3. Descargar y guardar en: data/raw/

    📍 PASO 2: Descargar Datos del INEC Database Hub
    ────────────────────────────────────────────────
    URL: https://www.inec.gob.pa/dbshub/

    1. Navegar por categoría: "Trabajo"
    2. Seleccionar: "Encuesta de Mercado Laboral"
    3. Elegir años: 2018-2024
    4. Exportar en formato Excel o CSV
    5. Guardar en: data/raw/

    📍 PASO 3: Publicaciones Oficiales
    ───────────────────────────────────
    URL: https://www.inec.gob.pa/publicaciones/Default2.aspx?ID_CATEGORIA=5&ID_SUBCATEGORIA=38

    1. Buscar reportes trimestrales 2018-2024
    2. Descargar PDFs y Excel asociados
    3. Guardar en: data/raw/

    📍 PASO 4: Datos Geoespaciales (Shapefiles)
    ────────────────────────────────────────────
    Contactar a: rcervantes@contraloria.gob.pa
    Solicitar: Shapefiles de corregimientos de Panamá
    O buscar en: Portal de Datos Abiertos del Gobierno

    ⚠️  IMPORTANTE:
    ───────────────
    - Registra las fuentes en docs/fuentes_datos.md
    - Mantén la licencia CC BY 4.0
    - Atribuye siempre al INEC
    """

    print(instrucciones)


def main():
    """
    Función principal para ejecutar el proceso de descarga
    """
    print("\n" + "="*70)
    print("🇵🇦 DESCARGADOR DE DATOS DEL INEC - PANAMÁ")
    print("="*70)
    print(f"📁 Carpeta de destino: {RAW_DATA_DIR}")
    print(f"📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Mostrar datasets disponibles
    listar_datasets_disponibles()

    # Intentar descarga automática del reporte más reciente
    print("\n" + "="*70)
    print("⚡ INTENTANDO DESCARGAS AUTOMÁTICAS")
    print("="*70)

    exito = descargar_reporte_octubre_2024()

    if exito:
        print("\n✅ Descarga automática completada")
        crear_registro_descarga()

    # Mostrar instrucciones para descarga manual
    mostrar_instrucciones_descarga_manual()

    print("\n" + "="*70)
    print("✅ PROCESO COMPLETADO")
    print("="*70)
    print(f"\n📂 Revisa los archivos en: {RAW_DATA_DIR}")
    print("📖 Lee las instrucciones arriba para obtener más datos")
    print("\n💡 TIP: Algunos datos requieren descarga manual desde el portal web")
    print("   Visita: https://www.inec.gob.pa/dbshub/")


if __name__ == "__main__":
    main()
