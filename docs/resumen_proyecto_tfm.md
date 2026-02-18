# Resumen Ejecutivo del Proyecto TFM

**Titulo:** Aplicacion de Machine Learning para la Prediccion de Desempleo Regional en Panama: Un Enfoque de Politica Publica Basado en Datos y Analisis Geoespacial
**Autor:** Henry De Gracia
**Master:** Máster Data Science, Big Data & Business Analytics - Universidad Complutense de Madrid
**Tutores:** Carlos Ortega y Santiago Mota
**Fecha:** Febrero 2026

---

## Problema

Panama presenta desigualdades significativas en su mercado laboral a nivel regional. Mientras provincias como Herrera y Los Santos mantienen tasas de desempleo inferiores al 5%, Colon y las comarcas indigenas superan el 10% de forma recurrente. La pandemia de COVID-19 agravo estas brechas, y la recuperacion ha sido desigual entre regiones. Actualmente no existen herramientas predictivas para anticipar crisis de desempleo a nivel provincial.

## Objetivo

Construir un modelo predictivo de la tasa de desempleo a nivel provincial en Panama, capaz de clasificar niveles de riesgo laboral y productivizado en un dashboard interactivo para apoyar la toma de decisiones en politica publica.

---

## Datos

| Fuente | Descripcion | Volumen |
|--------|-------------|---------|
| INEC Panama | Encuesta de Mercado Laboral | 560 archivos descargados, 59 cuadros procesados |
| Banco Mundial | 8 indicadores macroeconomicos via API | 2018-2024 |
| HDX | Shapefiles de provincias (nivel admin 1) | 13 regiones |

**Dataset final:** 1,605 observaciones (13 provincias x 17 periodos x area x sexo) con 31 features.

---

## Metodologia

### Feature Engineering
31 variables creadas en 7 categorias: retardos temporales, variaciones, brechas de genero, ratios educativos, indicadores post-COVID, interacciones y dummies de provincia.

### Modelos Evaluados
5 algoritmos estudiados durante el master, comparados con validacion cruzada temporal Leave-One-Period-Out (LOPO):

| Modelo | MAE | RMSE | R-cuadrado |
|--------|-----|------|------------|
| Ridge | 1.39 | 1.83 | 0.62 |
| Lasso | 1.38 | 1.82 | 0.62 |
| Random Forest | 1.20 | 1.61 | 0.72 |
| Gradient Boosting | 1.16 | 1.57 | 0.75 |
| **XGBoost** | **1.13** | **1.53** | **0.77** |

### Modelo Seleccionado: XGBoost
- **R-cuadrado = 0.77**: explica el 77% de la variabilidad del desempleo regional
- **MAE = 1.13 pp**: error medio de 1.13 puntos porcentuales
- **Precision de clasificacion de riesgo = 79.4%** (bajo/moderado/critico)

---

## Resultados Principales

### Variables mas determinantes
1. Subempleo del periodo anterior (principal predictor)
2. Tasa de desempleo del periodo anterior (inercia)
3. Estructura empresarial (tamano de empresa)
4. Nivel educativo
5. Interaccion desempleo previo x post-COVID

### Hallazgos clave
- El subempleo es el mejor predictor del desempleo futuro
- La inercia del desempleo justifica intervenciones tempranas
- La brecha de genero persiste (+2.0 pp mujeres vs hombres)
- El efecto COVID tiene cola larga (aun relevante en 2024)
- Las comarcas indigenas y Colon presentan riesgo critico recurrente

---

## Productivizacion

Dashboard interactivo con Streamlit (8 secciones):
- Resumen del modelo y KPIs
- Mapa interactivo de riesgo (Folium)
- Evolucion temporal con graficos dinamicos (Plotly)
- Predicciones vs realidad
- Analisis de riesgo por provincia
- Rendimiento del modelo (metricas CV)
- Interpretabilidad SHAP
- Predictor interactivo (analisis what-if)

---

## Analisis Geoespacial

Mapas coropleticos a nivel provincial:
- Mapa de riesgo del ultimo periodo (Octubre 2024)
- Comparacion temporal 2018 vs 2024
- Mapa real vs prediccion del modelo
- Mapa interactivo con Folium (hover por provincia)

---

## Infraestructura Tecnica

- **Lenguaje:** Python 3.11
- **ML:** scikit-learn, XGBoost, SHAP
- **Visualizacion:** Plotly, Matplotlib, Folium, GeoPandas
- **Dashboard:** Streamlit
- **Testing:** 80 tests automatizados (pytest)
- **Pipeline:** Ejecutable con `python main.py`

---
