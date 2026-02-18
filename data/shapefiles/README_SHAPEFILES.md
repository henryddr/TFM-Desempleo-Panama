# Shapefiles de Panama

## Fuente

**HDX (Humanitarian Data Exchange)**
- URL: https://data.humdata.org/dataset/cod-ab-pan
- Licencia: CC BY-IGO
- Nivel utilizado: Administrativo 1 (provincias y comarcas)

## Archivos incluidos

### shapefile/
- `pan_admin1.shp` - Geometrias de 13 provincias y comarcas
- `pan_admin1.shx` - Indice espacial
- `pan_admin1.dbf` - Atributos
- `pan_admin1.prj` - Proyeccion (WGS84 / EPSG:4326)
- `pan_admin1.cpg` - Codificacion de caracteres

### geojson/
- `pan_admin1.geojson` - Formato GeoJSON (usado por Folium para mapas interactivos)

## Uso en el proyecto

```python
import geopandas as gpd

# Cargar provincias
provincias = gpd.read_file('data/shapefiles/HDX/shapefile/pan_admin1.shp')
```

### Columnas principales
- `ADM1_ES`: Nombre de la provincia/comarca
- `ADM1_PCODE`: Codigo de la provincia
- `geometry`: Geometria del poligono

## Atribucion

```
Humanitarian Data Exchange (HDX). (2024). Panama - Subnational Administrative Boundaries.
https://data.humdata.org/dataset/cod-ab-pan
Licencia: CC BY-IGO
```

**Proyecto:** Aplicacion de Machine Learning para la Prediccion de Desempleo Regional en Panama
**Autor:** Henry De Gracia
**Fecha:** Febrero 2026
