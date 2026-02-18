"""
Script para descargar datos del Banco Mundial para Panamá

API del Banco Mundial: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392

Indicadores relevantes para el modelo de desempleo:
- PIB per cápita
- Tasa de crecimiento PIB
- Inflación
- Gasto público en educación
- Población urbana
- etc.

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

# Configuración
ROOT_DIR = Path(__file__).parent.parent.parent
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Código de país para Panamá
PAIS = "PA"

# API del Banco Mundial
BASE_URL = "https://api.worldbank.org/v2"

# Indicadores relevantes para el modelo
INDICADORES = {
    'NY.GDP.PCAP.CD': 'PIB per cápita (US$ corrientes)',
    'NY.GDP.MKTP.KD.ZG': 'Crecimiento del PIB (% anual)',
    'FP.CPI.TOTL.ZG': 'Inflación, precios al consumidor (% anual)',
    'SE.XPD.TOTL.GD.ZS': 'Gasto público en educación (% del PIB)',
    'SP.URB.TOTL.IN.ZS': 'Población urbana (% del total)',
    'SP.POP.TOTL': 'Población total',
    'SL.UEM.TOTL.ZS': 'Desempleo total (% de la fuerza laboral)',
    'SL.UEM.1524.ZS': 'Desempleo juvenil (% de la fuerza laboral de 15-24 años)',
    'SL.TLF.TOTL.IN': 'Fuerza laboral total',
    'SI.POV.GINI': 'Índice de Gini',
    'NY.GDP.MKTP.CD': 'PIB (US$ corrientes)',
    'NE.EXP.GNFS.ZS': 'Exportaciones de bienes y servicios (% del PIB)',
    'NE.IMP.GNFS.ZS': 'Importaciones de bienes y servicios (% del PIB)',
}

def descargar_indicador(codigo_indicador, fecha_inicio=2018, fecha_fin=2024):
    """
    Descarga un indicador específico del Banco Mundial

    Args:
        codigo_indicador (str): Código del indicador
        fecha_inicio (int): Año de inicio
        fecha_fin (int): Año final

    Returns:
        pd.DataFrame: DataFrame con los datos del indicador
    """
    try:
        # Construir URL
        url = f"{BASE_URL}/country/{PAIS}/indicator/{codigo_indicador}"
        params = {
            'format': 'json',
            'date': f"{fecha_inicio}:{fecha_fin}",
            'per_page': 1000
        }

        print(f"📥 Descargando: {INDICADORES.get(codigo_indicador, codigo_indicador)}")

        # Realizar petición
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        # Verificar que hay datos
        if len(data) < 2 or not data[1]:
            print(f"   ⚠️  No hay datos disponibles para este indicador")
            return None

        # Extraer datos
        registros = []
        for item in data[1]:
            registros.append({
                'pais': item['country']['value'],
                'codigo_pais': item['countryiso3code'],
                'indicador': item['indicator']['value'],
                'codigo_indicador': item['indicator']['id'],
                'año': int(item['date']) if item['date'] else None,
                'valor': float(item['value']) if item['value'] else None
            })

        df = pd.DataFrame(registros)
        print(f"   ✅ Descargado: {len(df)} registros")

        return df

    except requests.exceptions.RequestException as e:
        print(f"   ❌ Error de conexión: {e}")
        return None
    except Exception as e:
        print(f"   ❌ Error inesperado: {e}")
        return None


def descargar_todos_indicadores():
    """
    Descarga todos los indicadores relevantes y los combina en un DataFrame
    """
    print("\n" + "="*70)
    print("🌍 DESCARGANDO INDICADORES DEL BANCO MUNDIAL")
    print("="*70)
    print(f"País: Panamá (PA)")
    print(f"Período: 2018-2024")
    print(f"Total indicadores: {len(INDICADORES)}")

    todos_datos = []

    for codigo, nombre in INDICADORES.items():
        df = descargar_indicador(codigo)
        if df is not None and not df.empty:
            todos_datos.append(df)

    if not todos_datos:
        print("\n❌ No se pudo descargar ningún indicador")
        return None

    # Combinar todos los dataframes
    df_completo = pd.concat(todos_datos, ignore_index=True)

    print(f"\n✅ Total de registros descargados: {len(df_completo)}")

    return df_completo


def crear_dataset_ancho(df):
    """
    Convierte el dataset largo a formato ancho (años como columnas)

    Args:
        df (pd.DataFrame): DataFrame en formato largo

    Returns:
        pd.DataFrame: DataFrame en formato ancho
    """
    if df is None or df.empty:
        return None

    # Crear tabla pivote
    df_ancho = df.pivot_table(
        index='codigo_indicador',
        columns='año',
        values='valor',
        aggfunc='first'
    )

    # Agregar nombre del indicador
    df_ancho['nombre_indicador'] = df_ancho.index.map(INDICADORES)

    # Reordenar columnas
    cols = ['nombre_indicador'] + [col for col in df_ancho.columns if col != 'nombre_indicador']
    df_ancho = df_ancho[cols]

    return df_ancho


def guardar_datos(df_largo, df_ancho):
    """
    Guarda los datos en archivos CSV y Excel

    Args:
        df_largo (pd.DataFrame): Datos en formato largo
        df_ancho (pd.DataFrame): Datos en formato ancho
    """
    print("\n" + "="*70)
    print("💾 GUARDANDO DATOS")
    print("="*70)

    timestamp = datetime.now().strftime('%Y%m%d')

    # Guardar formato largo
    if df_largo is not None:
        archivo_largo_csv = RAW_DATA_DIR / f"banco_mundial_panama_largo_{timestamp}.csv"
        archivo_largo_xlsx = RAW_DATA_DIR / f"banco_mundial_panama_largo_{timestamp}.xlsx"

        df_largo.to_csv(archivo_largo_csv, index=False, encoding='utf-8-sig')
        df_largo.to_excel(archivo_largo_xlsx, index=False, engine='openpyxl')

        print(f"✅ Formato largo guardado:")
        print(f"   CSV:  {archivo_largo_csv}")
        print(f"   Excel: {archivo_largo_xlsx}")

    # Guardar formato ancho
    if df_ancho is not None:
        archivo_ancho_csv = RAW_DATA_DIR / f"banco_mundial_panama_ancho_{timestamp}.csv"
        archivo_ancho_xlsx = RAW_DATA_DIR / f"banco_mundial_panama_ancho_{timestamp}.xlsx"

        df_ancho.to_csv(archivo_ancho_csv, encoding='utf-8-sig')
        df_ancho.to_excel(archivo_ancho_xlsx, engine='openpyxl')

        print(f"\n✅ Formato ancho guardado:")
        print(f"   CSV:  {archivo_ancho_csv}")
        print(f"   Excel: {archivo_ancho_xlsx}")


def crear_resumen_datos(df):
    """
    Crea un resumen de los datos descargados

    Args:
        df (pd.DataFrame): DataFrame con los datos
    """
    if df is None or df.empty:
        return

    print("\n" + "="*70)
    print("📊 RESUMEN DE DATOS DESCARGADOS")
    print("="*70)

    # Años disponibles
    años = sorted(df['año'].dropna().unique())
    print(f"\n📅 Años disponibles: {', '.join(map(str, años))}")

    # Indicadores por año
    print(f"\n📈 Indicadores por año:")
    for año in años:
        n_indicadores = df[df['año'] == año]['codigo_indicador'].nunique()
        print(f"   {año}: {n_indicadores} indicadores")

    # Indicadores con datos completos
    df_completo = df.groupby('codigo_indicador')['valor'].count()
    indicadores_completos = df_completo[df_completo == len(años)]

    print(f"\n✅ Indicadores con datos completos (todos los años): {len(indicadores_completos)}")
    if len(indicadores_completos) > 0:
        for codigo in indicadores_completos.index:
            print(f"   - {INDICADORES[codigo]}")

    # Indicadores con datos faltantes
    indicadores_incompletos = df_completo[df_completo < len(años)]
    if len(indicadores_incompletos) > 0:
        print(f"\n⚠️  Indicadores con datos faltantes: {len(indicadores_incompletos)}")
        for codigo in indicadores_incompletos.index:
            años_disponibles = df_completo[codigo]
            print(f"   - {INDICADORES[codigo]}: {años_disponibles}/{len(años)} años")


def main():
    """
    Función principal
    """
    print("\n" + "="*70)
    print("🌍 DESCARGADOR DE DATOS DEL BANCO MUNDIAL - PANAMÁ")
    print("="*70)
    print(f"📁 Carpeta de destino: {RAW_DATA_DIR}")
    print(f"📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Descargar datos
    df_largo = descargar_todos_indicadores()

    if df_largo is not None and not df_largo.empty:
        # Crear formato ancho
        df_ancho = crear_dataset_ancho(df_largo)

        # Guardar datos
        guardar_datos(df_largo, df_ancho)

        # Crear resumen
        crear_resumen_datos(df_largo)

        print("\n" + "="*70)
        print("✅ PROCESO COMPLETADO EXITOSAMENTE")
        print("="*70)
        print(f"\n📂 Revisa los archivos en: {RAW_DATA_DIR}")
        print("\n💡 TIP: Los datos están disponibles en dos formatos:")
        print("   - Largo: Una fila por indicador-año")
        print("   - Ancho: Una fila por indicador, columnas por año")

    else:
        print("\n❌ No se pudieron descargar datos")

    print("\n📖 Fuente: Banco Mundial - https://data.worldbank.org")
    print("📄 Licencia: CC BY 4.0")


if __name__ == "__main__":
    main()
