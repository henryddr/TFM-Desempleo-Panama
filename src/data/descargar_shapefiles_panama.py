"""
Script para descargar shapefiles de Panamá

Fuentes:
1. Humanitarian Data Exchange (HDX) - Actualizado Diciembre 2024
   - 13 provincias (admin level 1)
   - 76 distritos (admin level 2)
   - 594 corregimientos (admin level 3)

2. STRI (Smithsonian Tropical Research Institute) - 2024
   - 699 corregimientos
   - 82 distritos
   - Provincias

3. DIVA-GIS - Alternativa

Licencia: CC BY-IGO (Creative Commons Attribution for Intergovernmental Organisations)

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
import zipfile
import time

# Configuración
ROOT_DIR = Path(__file__).parent.parent.parent
SHAPEFILES_DIR = ROOT_DIR / "data" / "shapefiles"
SHAPEFILES_DIR.mkdir(parents=True, exist_ok=True)

# URLs de descarga
FUENTES = {
    'HDX': {
        'nombre': 'Humanitarian Data Exchange (HDX)',
        'descripcion': 'Límites administrativos oficiales de Panamá (2024)',
        'archivos': {
            'shapefile': {
                'url': 'https://data.humdata.org/dataset/19b220b3-2802-432b-b93d-b1a2f2ee347a/resource/f31f539a-45fb-47fc-b7a6-d899085713b8/download/pan_admin_boundaries.shp.zip',
                'nombre': 'pan_admin_boundaries.shp.zip',
                'tamaño': '6.6 MB',
                'contenido': [
                    '13 provincias (admin level 1)',
                    '76 distritos (admin level 2)',
                    '594 corregimientos (admin level 3)'
                ]
            },
            'geojson': {
                'url': 'https://data.humdata.org/dataset/19b220b3-2802-432b-b93d-b1a2f2ee347a/resource/64d9b04e-4964-41a3-8f9d-bd00eab6c1e4/download/pan_admin_boundaries.geojson.zip',
                'nombre': 'pan_admin_boundaries.geojson.zip',
                'tamaño': '9.0 MB',
                'contenido': [
                    'Formato GeoJSON (alternativa)',
                    'Mismo contenido que shapefile'
                ]
            }
        },
        'licencia': 'CC BY-IGO',
        'actualizado': 'Diciembre 2024'
    },
    'DIVA-GIS': {
        'nombre': 'DIVA-GIS',
        'descripcion': 'Límites administrativos globales',
        'archivos': {
            'admin': {
                'url': 'https://biogeo.ucdavis.edu/data/diva/adm/PAN_adm.zip',
                'nombre': 'PAN_adm.zip',
                'tamaño': '~3 MB',
                'contenido': [
                    'Provincias (adm1)',
                    'Distritos (adm2)',
                    'Corregimientos (adm3)'
                ]
            }
        },
        'licencia': 'Libre para uso académico',
        'actualizado': '2023'
    }
}


def descargar_archivo(url, ruta_destino, nombre_display):
    """
    Descarga un archivo desde una URL

    Args:
        url (str): URL del archivo
        ruta_destino (Path): Ruta donde guardar
        nombre_display (str): Nombre para mostrar

    Returns:
        bool: True si exitoso
    """
    try:
        print(f"   📥 Descargando: {nombre_display}")
        print(f"      URL: {url}")

        # Realizar petición con stream
        response = requests.get(url, timeout=120, stream=True)
        response.raise_for_status()

        # Obtener tamaño total
        total_size = int(response.headers.get('content-length', 0))

        # Descargar con barra de progreso simple
        downloaded = 0
        with open(ruta_destino, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    # Mostrar progreso cada 1 MB
                    if downloaded % (1024 * 1024) == 0:
                        mb_downloaded = downloaded / (1024 * 1024)
                        if total_size > 0:
                            porcentaje = (downloaded / total_size) * 100
                            print(f"      {porcentaje:.1f}% ({mb_downloaded:.1f} MB)", end='\r')
                        else:
                            print(f"      {mb_downloaded:.1f} MB descargados", end='\r')

        tamaño_final = ruta_destino.stat().st_size / (1024 * 1024)
        print(f"\n      ✅ Completado: {tamaño_final:.2f} MB")

        return True

    except requests.exceptions.RequestException as e:
        print(f"\n      ❌ Error de descarga: {e}")
        return False
    except Exception as e:
        print(f"\n      ❌ Error inesperado: {e}")
        return False


def descomprimir_zip(ruta_zip, carpeta_destino):
    """
    Descomprime un archivo ZIP

    Args:
        ruta_zip (Path): Ruta del archivo ZIP
        carpeta_destino (Path): Carpeta donde descomprimir

    Returns:
        bool: True si exitoso
    """
    try:
        print(f"   📦 Descomprimiendo: {ruta_zip.name}")

        carpeta_destino.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(ruta_zip, 'r') as zip_ref:
            zip_ref.extractall(carpeta_destino)

        # Contar archivos extraídos
        archivos = list(carpeta_destino.rglob('*'))
        archivos_no_dir = [f for f in archivos if f.is_file()]

        print(f"      ✅ Extraídos {len(archivos_no_dir)} archivos")

        # Listar shapefiles principales
        shapefiles = list(carpeta_destino.rglob('*.shp'))
        if shapefiles:
            print(f"      📊 Shapefiles encontrados:")
            for shp in shapefiles:
                tamaño_kb = shp.stat().st_size / 1024
                print(f"         - {shp.name} ({tamaño_kb:.1f} KB)")

        return True

    except zipfile.BadZipFile:
        print(f"      ❌ Error: Archivo ZIP corrupto")
        return False
    except Exception as e:
        print(f"      ❌ Error al descomprimir: {e}")
        return False


def descargar_fuente(nombre_fuente, info_fuente):
    """
    Descarga todos los archivos de una fuente

    Args:
        nombre_fuente (str): Nombre de la fuente
        info_fuente (dict): Información de la fuente

    Returns:
        tuple: (exitosos, fallidos)
    """
    print(f"\n{'='*70}")
    print(f"📍 {info_fuente['nombre']}")
    print(f"{'='*70}")
    print(f"Descripción: {info_fuente['descripcion']}")
    print(f"Licencia: {info_fuente['licencia']}")
    print(f"Actualizado: {info_fuente['actualizado']}")
    print(f"Total de archivos: {len(info_fuente['archivos'])}")

    exitosos = 0
    fallidos = 0

    for tipo, info_archivo in info_fuente['archivos'].items():
        print(f"\n--- {tipo.upper()} ---")

        # Ruta de descarga
        ruta_zip = SHAPEFILES_DIR / info_archivo['nombre']
        carpeta_extraccion = SHAPEFILES_DIR / nombre_fuente / tipo

        # Verificar si ya existe
        if carpeta_extraccion.exists() and any(carpeta_extraccion.iterdir()):
            print(f"   ⏭️  Ya existe, saltando...")
            exitosos += 1
            continue

        # Mostrar información
        print(f"   Tamaño aproximado: {info_archivo['tamaño']}")
        print(f"   Contenido:")
        for item in info_archivo['contenido']:
            print(f"      • {item}")

        # Descargar
        if descargar_archivo(info_archivo['url'], ruta_zip, info_archivo['nombre']):
            # Descomprimir
            if descomprimir_zip(ruta_zip, carpeta_extraccion):
                exitosos += 1

                # Eliminar ZIP para ahorrar espacio (opcional)
                # ruta_zip.unlink()
                # print(f"      🗑️  ZIP eliminado (archivos extraídos)")
            else:
                fallidos += 1
        else:
            fallidos += 1

        time.sleep(1)  # Pausa entre descargas

    return exitosos, fallidos


def verificar_shapefiles():
    """
    Verifica y lista todos los shapefiles descargados
    """
    print(f"\n{'='*70}")
    print("📊 VERIFICACIÓN DE SHAPEFILES")
    print(f"{'='*70}")

    shapefiles = list(SHAPEFILES_DIR.rglob('*.shp'))

    if not shapefiles:
        print("\n⚠️  No se encontraron shapefiles")
        return

    print(f"\n✅ Total de shapefiles encontrados: {len(shapefiles)}")
    print("\n📁 Estructura de archivos:")

    # Agrupar por fuente
    por_fuente = {}
    for shp in shapefiles:
        partes = shp.relative_to(SHAPEFILES_DIR).parts
        if len(partes) > 0:
            fuente = partes[0] if len(partes) > 1 else 'raiz'
            if fuente not in por_fuente:
                por_fuente[fuente] = []
            por_fuente[fuente].append(shp)

    for fuente, archivos in por_fuente.items():
        print(f"\n   📂 {fuente}/")
        for shp in sorted(archivos):
            tamaño_mb = shp.stat().st_size / (1024 * 1024)
            nombre_rel = shp.relative_to(SHAPEFILES_DIR / fuente)
            print(f"      • {nombre_rel} ({tamaño_mb:.2f} MB)")


def crear_guia_uso():
    """
    Crea un archivo de guía de uso de los shapefiles
    """
    guia_path = SHAPEFILES_DIR / "README_SHAPEFILES.md"

    contenido = """# Shapefiles de Panamá

## Archivos Descargados

### 1. HDX (Humanitarian Data Exchange)
**Fuente:** https://data.humdata.org/dataset/cod-ab-pan
**Actualizado:** Diciembre 2024
**Licencia:** CC BY-IGO

**Contiene:**
- 13 provincias (admin level 1)
- 76 distritos (admin level 2)
- 594 corregimientos (admin level 3)

**Archivos principales:**
- `pan_adm1.shp` - Provincias
- `pan_adm2.shp` - Distritos
- `pan_adm3.shp` - Corregimientos

### 2. DIVA-GIS
**Fuente:** https://www.diva-gis.org/gdata
**Actualizado:** 2023
**Licencia:** Libre para uso académico

**Contiene:**
- Límites administrativos en múltiples niveles
- Compatible con la mayoría de software GIS

**Archivos principales:**
- `PAN_adm1.shp` - Provincias
- `PAN_adm2.shp` - Distritos
- `PAN_adm3.shp` - Corregimientos

## Cómo Usar

### En Python con GeoPandas

```python
import geopandas as gpd

# Cargar provincias
provincias = gpd.read_file('data/shapefiles/HDX/shapefile/pan_adm1.shp')

# Cargar distritos
distritos = gpd.read_file('data/shapefiles/HDX/shapefile/pan_adm2.shp')

# Cargar corregimientos
corregimientos = gpd.read_file('data/shapefiles/HDX/shapefile/pan_adm3.shp')

# Ver información
print(corregimientos.head())
print(f"Total corregimientos: {len(corregimientos)}")

# Visualizar
corregimientos.plot()
```

### Columnas Importantes

**Provincias (adm1):**
- `ADM1_ES`: Nombre de la provincia
- `ADM1_PCODE`: Código de la provincia
- `geometry`: Geometría del polígono

**Distritos (adm2):**
- `ADM1_ES`: Provincia
- `ADM2_ES`: Nombre del distrito
- `ADM2_PCODE`: Código del distrito
- `geometry`: Geometría del polígono

**Corregimientos (adm3):**
- `ADM1_ES`: Provincia
- `ADM2_ES`: Distrito
- `ADM3_ES`: Nombre del corregimiento
- `ADM3_PCODE`: Código del corregimiento
- `geometry`: Geometría del polígono

## Sistema de Coordenadas

**Proyección:** WGS84 (EPSG:4326)
**Unidades:** Grados decimales

## Atribución Requerida

Al usar estos datos en tu TFM, incluye:

```
Fuentes Geoespaciales:
- Humanitarian Data Exchange (HDX). (2024). Panama - Subnational Administrative Boundaries.
  https://data.humdata.org/dataset/cod-ab-pan
  Licencia: CC BY-IGO

- DIVA-GIS. (2023). Panama Administrative Boundaries.
  https://www.diva-gis.org/gdata
  Libre para uso académico
```

## Problemas Comunes

### Error al cargar shapefile
```python
# Asegúrate de que todos los archivos auxiliares estén presentes:
# .shp, .shx, .dbf, .prj
```

### Encoding de caracteres
```python
# Si hay problemas con acentos:
gdf = gpd.read_file('archivo.shp', encoding='utf-8')
```

### Reproyectar
```python
# Cambiar proyección si es necesario
gdf_utm = gdf.to_crs(epsg=32617)  # UTM Zone 17N para Panamá
```

## Próximos Pasos

1. Cargar shapefiles en Python
2. Unir con datos de desempleo por corregimiento
3. Crear mapas de coropletas
4. Análisis de hotspots
5. Mapas interactivos con Folium

---

**Proyecto:** Predicción de Desempleo Regional en Panamá
**Autor:** Henry De Gracia
**Fecha:** Enero 2025
"""

    with open(guia_path, 'w', encoding='utf-8') as f:
        f.write(contenido)

    print(f"\n📝 Guía de uso creada: {guia_path}")


def main():
    """
    Función principal
    """
    print("\n" + "="*70)
    print("🗺️  DESCARGADOR DE SHAPEFILES DE PANAMÁ")
    print("="*70)
    print(f"Carpeta destino: {SHAPEFILES_DIR}")
    print(f"Fuentes disponibles: {len(FUENTES)}")
    print("="*70)

    total_exitosos = 0
    total_fallidos = 0

    # Descargar de cada fuente
    for nombre, info in FUENTES.items():
        exitosos, fallidos = descargar_fuente(nombre, info)
        total_exitosos += exitosos
        total_fallidos += fallidos

    # Verificar archivos descargados
    verificar_shapefiles()

    # Crear guía de uso
    crear_guia_uso()

    # Resumen final
    print(f"\n{'='*70}")
    print("📊 RESUMEN FINAL")
    print(f"{'='*70}")
    print(f"✅ Descargas exitosas: {total_exitosos}")
    if total_fallidos > 0:
        print(f"❌ Descargas fallidas: {total_fallidos}")

    print(f"\n📂 Archivos guardados en: {SHAPEFILES_DIR}")
    print(f"\n{'='*70}")
    print("✅ PROCESO COMPLETADO")
    print(f"{'='*70}")

    print("\n💡 PRÓXIMOS PASOS:")
    print("   1. Verifica los shapefiles en: data/shapefiles/")
    print("   2. Lee la guía: data/shapefiles/README_SHAPEFILES.md")
    print("   3. Prueba cargar los shapefiles en Python:")
    print("      ")
    print("      import geopandas as gpd")
    print("      gdf = gpd.read_file('data/shapefiles/HDX/shapefile/pan_adm3.shp')")
    print("      print(gdf.head())")
    print("      gdf.plot()")

    print("\n🎯 Para el proyecto TFM:")
    print("   - Usa los corregimientos (adm3) para el análisis")
    print("   - Únelos con datos de desempleo por código")
    print("   - Crea mapas de coropletas por tasa de desempleo")


if __name__ == "__main__":
    main()
