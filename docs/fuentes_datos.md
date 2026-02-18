# Fuentes de Datos del Proyecto

**Proyecto:** Aplicacion de Machine Learning para la Prediccion de Desempleo Regional en Panama
**Autor:** Henry De Gracia
**Fecha:** Febrero 2026

---

## 1. INEC - Instituto Nacional de Estadistica y Censo de Panama

**URL:** https://www.inec.gob.pa/publicaciones/Default2.aspx?ID_CATEGORIA=5&ID_SUBCATEGORIA=38
**Tipo:** Encuesta de Mercado Laboral (EML)
**Frecuencia:** Semestral (marzo/agosto u octubre)
**Cobertura:** Nacional, provincial, urbano/rural, por sexo
**Periodo utilizado:** 2012-2024 (14 periodos, excluyendo 2020)

### Datos descargados
- **560 archivos Excel** descargados automaticamente via scripts (`src/data/descarga_inec.py`)
- **59 cuadros procesados** de 10 tipos diferentes (ver `docs/catalogo_cuadros_inec.md`)

### Variables extraidas
| Cuadro | Variable |
|--------|----------|
| 2 | Tasa de desempleo por provincia (variable objetivo) |
| 1A | Tendencia temporal nacional |
| 4 | Estructura demografica laboral |
| 6 | Tasa de participacion laboral |
| 13 | Estructura sectorial (primario, secundario, terciario) |
| 16 | Nivel educativo (sin educacion, secundaria, universitaria) |
| 19 | Subempleo (horas trabajadas) |
| 22 | Estructura empresarial (tamano de empresa) |
| 25 | Mediana salarial por provincia |
| 39 | Tasa de informalidad |

---

## 2. Banco Mundial

**URL:** https://data.worldbank.org/country/panama
**API:** https://api.worldbank.org/v2
**Descarga:** Automatica via `src/data/descargar_banco_mundial.py`

### Indicadores utilizados (8 de 13 descargados)

| Indicador | Codigo | Uso en el modelo |
|-----------|--------|------------------|
| PIB per capita (US$ constantes) | NY.GDP.PCAP.KD | Contexto macroeconomico |
| Crecimiento del PIB (%) | NY.GDP.MKTP.KD.ZG | Ciclo economico |
| Inflacion (% anual) | FP.CPI.TOTL.ZG | Contexto de precios |
| Desempleo juvenil (%) | SL.UEM.1524.ZS | Presion laboral juvenil |
| Indice de Gini | SI.POV.GINI | Desigualdad de ingresos |
| Poblacion total | SP.POP.TOTL | Escala demografica |
| PIB (US$ constantes) | NY.GDP.MKTP.KD | Tamano de la economia |
| Exportaciones (% del PIB) | NE.EXP.GNFS.ZS | Apertura comercial |

Estos indicadores son nacionales y se asignan uniformemente a todas las provincias en cada ano, funcionando como contexto macroeconomico.

---

## 3. Datos Geoespaciales - HDX

**URL:** https://data.humdata.org/dataset/cod-ab-pan
**Fuente:** Humanitarian Data Exchange (United Nations OCHA)
**Licencia:** CC BY-IGO
**Nivel utilizado:** Administrativo 1 (provincias y comarcas)

### Archivos utilizados
- `pan_admin1.shp` (.shp, .shx, .dbf, .prj, .cpg) - Geometrias de 13 provincias
- `pan_admin1.geojson` - Formato GeoJSON para mapas interactivos (Folium)

---

## Licencias y Atribucion

| Fuente | Licencia |
|--------|----------|
| INEC Panama | Datos publicos del gobierno de Panama |
| Banco Mundial | CC BY 4.0 |
| HDX | CC BY-IGO |

### Atribucion requerida
```
- INEC Panama (2024). Encuesta de Mercado Laboral. https://www.inec.gob.pa
- Banco Mundial (2024). World Development Indicators - Panama. https://data.worldbank.org/country/panama
- HDX (2024). Panama - Subnational Administrative Boundaries. https://data.humdata.org
```
