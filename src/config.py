"""
Configuración global del proyecto TFM
"""
import os
from pathlib import Path

# Rutas del proyecto
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SHAPEFILES_DIR = DATA_DIR / "shapefiles"

MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# Parámetros del modelo
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Provincias de Panamá
PROVINCIAS = [
    "Bocas del Toro",
    "Coclé",
    "Colón",
    "Chiriquí",
    "Darién",
    "Herrera",
    "Los Santos",
    "Panamá",
    "Veraguas",
    "Panamá Oeste"
]

# Niveles de riesgo de desempleo
RIESGO_BAJO = 6.0      # < 6%
RIESGO_MODERADO = 8.0  # 6-8%
RIESGO_CRITICO = 8.0   # > 8%

# Colores para visualizaciones
COLOR_MAP = {
    'bajo': '#2ecc71',      # Verde
    'moderado': '#f39c12',  # Naranja
    'critico': '#e74c3c'    # Rojo
}

# Configuración de la aplicación web
APP_TITLE = "Aplicación de Machine Learning para la Predicción de Desempleo Regional en Panamá"
APP_ICON = "\U0001f1f5\U0001f1e6"  # Bandera de Panama
APP_LAYOUT = "wide"
