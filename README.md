# Aplicacion de Machine Learning para la Prediccion de Desempleo Regional en Panama

## Un Enfoque de Politica Publica Basado en Datos y Analisis Geoespacial

**Trabajo Fin de Master** - Henry De Gracia

**Master:** Data Science, Big Data & Business Analytics - Universidad Complutense de Madrid (UCM)
**Tutores:** Carlos Ortega y Santiago Mota
**Fecha:** Febrero 2026

---

## Resumen del proyecto

### Problema

Predecir la tasa de desempleo a nivel provincial en Panama utilizando datos oficiales
del INEC y Banco Mundial (2011-2024), clasificando el riesgo laboral por region.

### Datos

- **1,605 observaciones** (13 provincias x 17 periodos x area x sexo)
- **31 features** (lag, delta, contraste, interacciones, temporales, categoricas)
- Fuentes: INEC Panama (560 archivos Excel, 59 cuadros procesados de 14 periodos), Banco Mundial (API), HDX (shapefiles)

### Modelo

- **XGBoost** seleccionado entre 5 modelos evaluados (Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost)
- Validacion cruzada: Leave-One-Period-Out (LOPO)
- R-cuadrado: **0.77** | MAE: **1.13 pp** | RMSE: **1.53 pp**
- Precision de clasificacion de riesgo: **79.4%**

### Productivizacion

Dashboard Streamlit con 8 secciones interactivas y predictor what-if en tiempo real.

---

## Contenido del repositorio

| Archivo / Carpeta | Descripcion |
|-------------------|-------------|
| `notebooks/TFM_Memoria_Henry_De_Gracia.ipynb` | **Memoria del TFM** (~20 paginas al exportar) |
| `notebooks/01_analisis_exploratorio.ipynb` | Analisis exploratorio detallado (Anexo) |
| `app/app.py` | Dashboard interactivo Streamlit |
| `src/` | Codigo fuente completo (datos, features, modelo, visualizaciones) |
| `tests/` | 80 tests automatizados |
| `main.py` | Pipeline completo (ejecuta todo el proyecto) |
| `data/raw/` | 60 archivos originales (59 INEC + 1 Banco Mundial) |
| `data/processed/` | Datasets procesados listos para usar |
| `models/` | Modelo XGBoost entrenado + scaler + features |
| `outputs/` | Mapas generados (22 PNG + 1 HTML interactivo) y reportes |
| `docs/` | Documentacion complementaria |
| `requirements.txt` | Dependencias de Python |

---

## Instrucciones de instalacion y ejecucion

### Requisitos previos

- Python 3.10 o superior (desarrollado con Python 3.11.9)

### Paso 1: Clonar el repositorio y crear entorno virtual

```bash
git clone https://github.com/henryddr/TFM-Desempleo-Panama.git
cd TFM-Desempleo-Panama
python -m venv venv
```

Activar el entorno virtual:

- **Windows:** `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

Instalar dependencias:

```bash
pip install -r requirements.txt
```

**Nota sobre SHAP:** Si la instalacion de `shap` falla en Windows, se puede omitir sin afectar
el funcionamiento del pipeline ni del dashboard. La seccion de interpretabilidad SHAP
simplemente no se mostrara. Para instalar SHAP con Anaconda: `conda install -c conda-forge shap`

### Paso 2: Ejecutar los tests (verificar que todo funciona)

```bash
python -m pytest tests/ -v
```

Resultado esperado: **80 tests passed** (puede haber algunos "skipped" si SHAP no esta instalado).

### Paso 3: Ver la memoria del TFM

```bash
jupyter notebook notebooks/TFM_Memoria_Henry_De_Gracia.ipynb
```

Ejecutar todas las celdas (Kernel > Restart & Run All). El notebook genera los graficos
interactivos Plotly y muestra los mapas incrustados.

Para exportar a HTML:

```bash
jupyter nbconvert --to html notebooks/TFM_Memoria_Henry_De_Gracia.ipynb
```

### Paso 4: Ejecutar el dashboard interactivo

```bash
streamlit run app/app.py
```

Si el comando anterior falla en Windows, usar:

```bash
python -m streamlit run app/app.py
```

Se abre en el navegador en `http://localhost:8501`. El dashboard tiene 8 secciones:

1. **Resumen del Modelo** - KPIs y comparacion de modelos
2. **Mapa Interactivo** - Mapa Folium por provincia
3. **Evolucion Temporal** - Graficos de tendencia y animacion
4. **Predicciones vs Real** - Barras, scatter y residuos
5. **Analisis de Riesgo** - Clasificacion por provincia
6. **Rendimiento del Modelo** - Metricas CV detalladas
7. **Interpretabilidad SHAP** - Explicacion de predicciones con SHAP
8. **Predictor Interactivo** - Ajustar variables y predecir en tiempo real

### Paso 5 (opcional): Re-ejecutar el pipeline completo

```bash
python main.py
```

Esto regenera los datos procesados, re-entrena el modelo y genera los mapas.
No es necesario ejecutarlo ya que todos los artefactos estan incluidos.

---

## Estructura del codigo

```
TFM-Desempleo-Panama/
├── README.md                                # Este archivo
├── LEEME.md                                 # Instrucciones en formato entrega
├── main.py                                  # Pipeline completo
├── requirements.txt                         # Dependencias
├── app/
│   └── app.py                              # Dashboard Streamlit (8 secciones)
├── data/
│   ├── raw/                                # 60 archivos originales (59 INEC + 1 BM)
│   ├── processed/                          # Datasets procesados
│   │   ├── desempleo_por_provincia.csv     # Dataset principal (1,605 filas)
│   │   ├── features_desempleo.csv          # Con features engineered
│   │   └── serie_historica_nacional.csv    # Serie nacional
│   └── shapefiles/HDX/                     # Geometrias provinciales (admin1)
├── models/
│   ├── modelo_desempleo.pkl                # Modelo XGBoost entrenado
│   ├── scaler.pkl                          # StandardScaler
│   └── feature_cols.pkl                    # Lista de 31 features
├── notebooks/
│   ├── TFM_Memoria_Henry_De_Gracia.ipynb   # MEMORIA DEL TFM
│   └── 01_analisis_exploratorio.ipynb      # EDA detallado (Anexo)
├── outputs/
│   ├── figures/                            # 22 mapas PNG + 1 HTML interactivo
│   └── reports/                            # Metricas, predicciones, CV, features
├── src/
│   ├── config.py                           # Configuracion global
│   ├── data/                               # Descarga y procesamiento
│   ├── features/crear_features.py          # Feature engineering
│   ├── models/entrenar_modelo.py           # Entrenamiento con LOPO CV
│   └── visualization/
│       ├── graficos_interactivos.py        # 11 funciones Plotly/Seaborn
│       └── mapas.py                        # Mapas con Folium/Geopandas
├── tests/                                  # 80 tests (pytest)
│   ├── conftest.py                         # Fixtures compartidas
│   ├── test_config.py                      # Tests de configuracion
│   ├── test_datos.py                       # Tests de calidad de datos
│   ├── test_features.py                    # Tests de feature engineering
│   ├── test_modelo.py                      # Tests del modelo
│   └── test_visualizaciones.py             # Tests de visualizaciones
└── docs/                                   # Documentacion complementaria
```
