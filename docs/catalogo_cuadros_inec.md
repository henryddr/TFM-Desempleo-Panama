# Catalogo de Cuadros INEC - Encuesta de Mercado Laboral

**Proyecto:** Aplicacion de Machine Learning para la Prediccion de Desempleo Regional en Panama
**Autor:** Henry De Gracia
**Fecha:** Febrero 2026
**Fuente:** INEC - Contraloria General de la Republica de Panama

---

## Resumen

De los 560 archivos Excel descargados del INEC, se procesaron **59 cuadros** correspondientes a **14 periodos** (2012-2024, excluyendo 2020 por encuesta telefonica COVID). Se utilizan **10 tipos de cuadro** que contienen las variables relevantes para el modelo.

## Periodos y Cuadros Incluidos

### Periodos antiguos (solo Cuadro 2 disponible)

| Periodo | Archivo | Cuadro |
|---------|---------|--------|
| Agosto 2012 | `P4851441-02.xls` | Cuadro 2 |
| Agosto 2013 | `P5571441-02.xls` | Cuadro 2 |
| Agosto 2014 | `P6361441-02.xls` | Cuadro 2 |
| Marzo 2015 | `P6821441-01.xls` | Cuadro 2 |
| Agosto 2015 | `P7171441-02.xls` | Cuadro 2 |
| Marzo 2016 | `P7511441-01.xls` | Cuadro 2 |
| Agosto 2016 | `P7841441-02.xls` | Cuadro 2 |
| Marzo 2017 | `P8181441-02.xls` | Cuadro 2 |
| Agosto 2017 | `P8561Cuadro 2.xls` | Cuadro 2 |

### Periodos completos (10 cuadros cada uno)

| Periodo | Cuadros procesados |
|---------|-------------------|
| Agosto 2018 | 2, 1A, 4, 6, 13, 16, 19, 22, 25, 39 |
| Agosto 2019 | 2, 1A, 4, 6, 13, 16, 19, 22, 25, 39 |
| Octubre 2021 | 2, 1A, 4, 6, 13, 16, 19, 22, 25, 39 |
| Agosto 2023 | 2, 1A, 4, 6, 13, 16, 19, 22, 25, 39 |
| Octubre 2024 | 2, 1A, 4, 6, 13, 16, 19, 22, 25, 39 |

**Nota:** Septiembre 2020 fue excluido (encuesta telefonica reducida por COVID-19, solo 5 cuadros basicos).

---

## Descripcion de los 10 Cuadros Utilizados

| Cuadro | Contenido | Variable extraida |
|--------|-----------|-------------------|
| **2** | Poblacion 15+ por sexo, provincia, area y condicion de actividad | Tasa de desempleo por provincia (variable objetivo) |
| **1A** | Serie historica de condicion de actividad economica (total nacional) | Tendencia temporal nacional |
| **4** | Condicion de actividad por provincia, sexo y grupos de edad | Estructura demografica laboral |
| **6** | Tasas de actividad economica por provincia, sexo y area | Tasa de participacion laboral |
| **13** | Ocupados por provincia y categoria de actividad (sector economico) | Estructura sectorial (primario, secundario, terciario) |
| **16** | Ocupados no agricolas por nivel de instruccion y provincia | Nivel educativo (sin educacion, secundaria, universitaria) |
| **19** | Empleados por horas semanales trabajadas y provincia | Subempleo (horas trabajadas) |
| **22** | Empleados por tamano de empresa y provincia | Estructura empresarial (micro, pequena, grande) |
| **25** | Mediana de salario mensual por provincia y actividad | Mediana salarial por provincia |
| **39** | Empleo informal por sector, segun provincia | Tasa de informalidad |

---

## Provincias y Comarcas (13 regiones)

| Nombre | Tipo |
|--------|------|
| Bocas del Toro | Provincia |
| Cocle | Provincia |
| Colon | Provincia |
| Chiriqui | Provincia |
| Darien | Provincia |
| Herrera | Provincia |
| Los Santos | Provincia |
| Panama | Provincia |
| Panama Oeste | Provincia |
| Veraguas | Provincia |
| Guna Yala | Comarca Indigena |
| Embera-Wounaan | Comarca Indigena |
| Ngabe-Bugle | Comarca Indigena |

---

## Notas Tecnicas

1. **Encoding:** Los archivos usan encoding latin-1/cp1252.
2. **Celdas combinadas:** Los Excel usan celdas combinadas en encabezados. Al leerlos con pandas, se convierten en NaN excepto en la primera celda.
3. **Filas de encabezado:** Se saltan las primeras 8-12 filas para llegar a los datos.
4. **Jerarquia en filas:** Datos organizados como Total > Urbana/Rural > Provincia > Desglose por sexo.
5. **Procesamiento:** Implementado en `src/data/procesar_datos_inec.py`.
