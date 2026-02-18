"""
Fixtures compartidas para los tests del proyecto TFM.
"""
import pandas as pd
import numpy as np
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.config import PROCESSED_DATA_DIR, REPORTS_DIR, MODELS_DIR


@pytest.fixture
def df_desempleo():
    """Carga el dataset procesado de desempleo por provincia."""
    path = PROCESSED_DATA_DIR / "desempleo_por_provincia.csv"
    if not path.exists():
        pytest.skip(f"Archivo no encontrado: {path}")
    return pd.read_csv(path)


@pytest.fixture
def df_features():
    """Carga el dataset con features engineered."""
    path = PROCESSED_DATA_DIR / "features_desempleo.csv"
    if not path.exists():
        pytest.skip(f"Archivo no encontrado: {path}")
    return pd.read_csv(path)


@pytest.fixture
def df_predicciones():
    """Carga las predicciones del modelo."""
    path = REPORTS_DIR / "predicciones_modelo.csv"
    if not path.exists():
        pytest.skip(f"Archivo no encontrado: {path}")
    return pd.read_csv(path)


@pytest.fixture
def df_cv_resultados():
    """Carga los resultados de cross-validation."""
    path = REPORTS_DIR / "cv_resultados.csv"
    if not path.exists():
        pytest.skip(f"Archivo no encontrado: {path}")
    return pd.read_csv(path)


@pytest.fixture
def df_feature_importance():
    """Carga la importancia de features."""
    path = REPORTS_DIR / "feature_importance.csv"
    if not path.exists():
        pytest.skip(f"Archivo no encontrado: {path}")
    return pd.read_csv(path, index_col=0)


@pytest.fixture
def resumen_modelo():
    """Carga el resumen del modelo desde JSON."""
    import json
    path = REPORTS_DIR / "resumen_modelo.json"
    if not path.exists():
        pytest.skip(f"Archivo no encontrado: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


@pytest.fixture
def df_sintetico():
    """Crea un DataFrame sintetico para tests que no dependen de archivos."""
    np.random.seed(42)
    n = 100
    provincias = ['Panama', 'Colon', 'Chiriqui', 'Cocle', 'Veraguas']
    periodos = ['2020-08', '2021-10', '2022-08', '2023-08']
    areas = ['total', 'urbana', 'rural']
    sexos = ['total', 'hombres', 'mujeres']

    rows = []
    for prov in provincias:
        for per in periodos:
            for area in areas:
                for sexo in sexos:
                    rows.append({
                        'provincia': prov,
                        'periodo': per,
                        'anio': int(per[:4]),
                        'area': area,
                        'sexo': sexo,
                        'tasa_desempleo': np.random.uniform(2, 15),
                        'tasa_participacion': np.random.uniform(40, 80),
                        'pea': np.random.randint(5000, 100000),
                        'pct_subempleo': np.random.uniform(5, 30),
                        'empleo_informal_pct': np.random.uniform(30, 80),
                        'pct_universitaria': np.random.uniform(5, 45),
                        'pct_sin_educacion': np.random.uniform(0, 20),
                        'pct_sector_primario': np.random.uniform(5, 40),
                        'pct_sector_secundario': np.random.uniform(10, 30),
                        'pct_sector_terciario': np.random.uniform(30, 70),
                        'pct_empresa_grande': np.random.uniform(20, 70),
                        'mediana_salario': np.random.uniform(200, 800),
                        'pct_secundaria': np.random.uniform(20, 40),
                        'pct_microempresa': np.random.uniform(10, 50),
                        'pib_per_capita': np.random.uniform(8000, 20000),
                    })
    return pd.DataFrame(rows)
