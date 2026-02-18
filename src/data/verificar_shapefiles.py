"""
Script para verificar y explorar los shapefiles de Panamá descargados

Este script:
- Carga los shapefiles
- Muestra información básica
- Genera visualizaciones simples
- Verifica que todo esté correcto

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

import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path

# Configuración
ROOT_DIR = Path(__file__).parent.parent.parent
SHAPEFILES_DIR = ROOT_DIR / "data" / "shapefiles" / "HDX" / "shapefile"
OUTPUTS_DIR = ROOT_DIR / "outputs" / "figures"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Archivos a verificar
ARCHIVOS = {
    'pais': 'pan_admin0.shp',
    'provincias': 'pan_admin1.shp',
    'distritos': 'pan_admin2.shp',
    'corregimientos': 'pan_admin3.shp'
}


def cargar_y_analizar(nombre, archivo):
    """
    Carga y analiza un shapefile

    Args:
        nombre (str): Nombre descriptivo
        archivo (str): Nombre del archivo

    Returns:
        GeoDataFrame: Datos cargados
    """
    print(f"\n{'='*70}")
    print(f"📊 {nombre.upper()}")
    print(f"{'='*70}")

    ruta = SHAPEFILES_DIR / archivo

    if not ruta.exists():
        print(f"❌ Archivo no encontrado: {ruta}")
        return None

    try:
        # Cargar shapefile
        print(f"📂 Cargando: {archivo}")
        gdf = gpd.read_file(ruta)

        # Información básica
        print(f"\n✅ Cargado exitosamente")
        print(f"   Total de registros: {len(gdf)}")
        print(f"   Sistema de coordenadas: {gdf.crs}")
        print(f"   Bounds: {gdf.total_bounds}")

        # Columnas disponibles
        print(f"\n📋 Columnas disponibles ({len(gdf.columns)}):")
        for col in gdf.columns:
            # Mostrar tipo y ejemplo
            tipo = gdf[col].dtype
            if col != 'geometry':
                ejemplo = gdf[col].iloc[0] if len(gdf) > 0 else 'N/A'
                print(f"   • {col}: {tipo}")
                if len(str(ejemplo)) < 50:
                    print(f"      Ejemplo: {ejemplo}")

        # Primeras filas (sin geometría para mejor visualización)
        print(f"\n📄 Primeros 3 registros:")
        cols_sin_geom = [col for col in gdf.columns if col != 'geometry']
        print(gdf[cols_sin_geom].head(3).to_string())

        # Estadísticas de áreas (si hay geometría)
        if 'geometry' in gdf.columns:
            gdf_proyectado = gdf.to_crs(epsg=32617)  # UTM Zone 17N para Panamá
            areas_km2 = gdf_proyectado.geometry.area / 1_000_000
            print(f"\n📏 Estadísticas de área (km²):")
            print(f"   Mínima: {areas_km2.min():.2f} km²")
            print(f"   Máxima: {areas_km2.max():.2f} km²")
            print(f"   Media: {areas_km2.mean():.2f} km²")
            print(f"   Total: {areas_km2.sum():.2f} km²")

        return gdf

    except Exception as e:
        print(f"❌ Error al cargar: {e}")
        return None


def crear_visualizacion(gdf, nombre, archivo_salida):
    """
    Crea una visualización del shapefile

    Args:
        gdf (GeoDataFrame): Datos geoespaciales
        nombre (str): Nombre para el título
        archivo_salida (str): Nombre del archivo PNG
    """
    try:
        print(f"\n🎨 Creando visualización...")

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        # Plotear
        gdf.plot(ax=ax,
                 edgecolor='black',
                 facecolor='lightblue',
                 linewidth=0.5,
                 alpha=0.7)

        # Título y labels
        ax.set_title(f'{nombre.title()} de Panamá',
                     fontsize=16,
                     fontweight='bold')
        ax.set_xlabel('Longitud')
        ax.set_ylabel('Latitud')

        # Agregar grid
        ax.grid(True, alpha=0.3)

        # Agregar texto con número de regiones
        ax.text(0.02, 0.98, f'Total: {len(gdf)} regiones',
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Guardar
        ruta_salida = OUTPUTS_DIR / archivo_salida
        plt.tight_layout()
        plt.savefig(ruta_salida, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"   ✅ Guardado: {ruta_salida}")

    except Exception as e:
        print(f"   ❌ Error al crear visualización: {e}")


def generar_reporte():
    """
    Genera un reporte completo de los shapefiles
    """
    print("\n" + "="*70)
    print("🗺️  VERIFICACIÓN DE SHAPEFILES DE PANAMÁ")
    print("="*70)
    print(f"Carpeta: {SHAPEFILES_DIR}")
    print(f"Total de archivos: {len(ARCHIVOS)}")

    resultados = {}

    # Cargar y analizar cada shapefile
    for nombre, archivo in ARCHIVOS.items():
        gdf = cargar_y_analizar(nombre, archivo)
        if gdf is not None:
            resultados[nombre] = gdf

            # Crear visualización
            crear_visualizacion(gdf, nombre, f"mapa_{nombre}.png")

    # Resumen final
    print(f"\n{'='*70}")
    print("📊 RESUMEN DE VERIFICACIÓN")
    print(f"{'='*70}")

    for nombre, gdf in resultados.items():
        print(f"\n✅ {nombre.upper()}")
        print(f"   Registros: {len(gdf)}")
        print(f"   Archivo: {ARCHIVOS[nombre]}")

        # Identificar columna de nombre si existe
        cols_nombre = [col for col in gdf.columns if 'ES' in col.upper() or 'NAME' in col.upper()]
        if cols_nombre:
            print(f"   Columna de nombre: {cols_nombre[0]}")
            print(f"   Ejemplos: {', '.join(gdf[cols_nombre[0]].head(5).tolist())}")

    print(f"\n{'='*70}")
    print("✅ VERIFICACIÓN COMPLETADA")
    print(f"{'='*70}")

    print(f"\n📂 Visualizaciones guardadas en: {OUTPUTS_DIR}")

    print("\n💡 PRÓXIMOS PASOS:")
    print("   1. Usar 'corregimientos' (pan_admin3.shp) para el análisis principal")
    print("   2. Unir con datos de desempleo usando código de corregimiento")
    print("   3. Crear mapas de coropletas coloreados por tasa de desempleo")
    print("   4. Identificar hotspots y coldspots")

    print("\n🎯 EJEMPLO DE USO:")
    print("""
    import geopandas as gpd
    import pandas as pd

    # Cargar shapefile de corregimientos
    corregimientos = gpd.read_file('data/shapefiles/HDX/shapefile/pan_admin3.shp')

    # Cargar datos de desempleo (cuando estén procesados)
    desempleo = pd.read_csv('data/processed/desempleo_por_corregimiento.csv')

    # Unir
    mapa_desempleo = corregimientos.merge(
        desempleo,
        left_on='ADM3_PCODE',  # Código del corregimiento
        right_on='codigo_corregimiento'
    )

    # Visualizar
    mapa_desempleo.plot(column='tasa_desempleo',
                        cmap='RdYlGn_r',  # Rojo=alto, Verde=bajo
                        legend=True,
                        figsize=(15, 12))
    """)


if __name__ == "__main__":
    generar_reporte()
