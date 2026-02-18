"""Tests para el modelo de prediccion y sus resultados."""
import pandas as pd
import numpy as np
import joblib
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import MODELS_DIR, RIESGO_BAJO, RIESGO_CRITICO
from src.models.entrenar_modelo import clasificar_riesgo, obtener_feature_cols


class TestClasificarRiesgo:
    def test_riesgo_bajo(self):
        assert clasificar_riesgo(3.0) == 'bajo'
        assert clasificar_riesgo(5.9) == 'bajo'

    def test_riesgo_moderado(self):
        assert clasificar_riesgo(RIESGO_BAJO) == 'moderado'
        assert clasificar_riesgo(7.0) == 'moderado'

    def test_riesgo_critico(self):
        assert clasificar_riesgo(RIESGO_CRITICO) == 'critico'
        assert clasificar_riesgo(15.0) == 'critico'

    def test_riesgo_limite_bajo(self):
        assert clasificar_riesgo(RIESGO_BAJO - 0.01) == 'bajo'

    def test_riesgo_limite_critico(self):
        assert clasificar_riesgo(RIESGO_CRITICO - 0.01) == 'moderado'

    def test_riesgo_cero(self):
        assert clasificar_riesgo(0.0) == 'bajo'


class TestModeloGuardado:
    def test_modelo_pkl_existe(self):
        assert (MODELS_DIR / "modelo_desempleo.pkl").exists()

    def test_scaler_pkl_existe(self):
        assert (MODELS_DIR / "scaler.pkl").exists()

    def test_feature_cols_pkl_existe(self):
        assert (MODELS_DIR / "feature_cols.pkl").exists()

    def test_modelo_puede_cargarse(self):
        modelo = joblib.load(MODELS_DIR / "modelo_desempleo.pkl")
        assert hasattr(modelo, 'predict')

    def test_scaler_puede_cargarse(self):
        scaler = joblib.load(MODELS_DIR / "scaler.pkl")
        assert hasattr(scaler, 'transform')

    def test_feature_cols_es_lista(self):
        feature_cols = joblib.load(MODELS_DIR / "feature_cols.pkl")
        assert isinstance(feature_cols, list)
        assert len(feature_cols) > 0

    def test_prediccion_forma_correcta(self):
        modelo = joblib.load(MODELS_DIR / "modelo_desempleo.pkl")
        scaler = joblib.load(MODELS_DIR / "scaler.pkl")
        feature_cols = joblib.load(MODELS_DIR / "feature_cols.pkl")

        X_dummy = np.zeros((1, len(feature_cols)))
        X_scaled = scaler.transform(X_dummy)
        pred = modelo.predict(X_scaled)
        assert pred.shape == (1,)
        assert np.isfinite(pred[0])


class TestResultadosModelo:
    def test_resumen_tiene_mejor_modelo(self, resumen_modelo):
        assert 'mejor_modelo' in resumen_modelo

    def test_resumen_tiene_metricas(self, resumen_modelo):
        metricas = resumen_modelo['metricas_cv']
        assert 'mae' in metricas
        assert 'rmse' in metricas
        assert 'r2' in metricas

    def test_r2_positivo(self, resumen_modelo):
        assert resumen_modelo['metricas_cv']['r2'] > 0

    def test_rmse_razonable(self, resumen_modelo):
        assert resumen_modelo['metricas_cv']['rmse'] < 5.0

    def test_precision_riesgo_superior_a_azar(self, resumen_modelo):
        # Con 3 clases, azar = 33%. El modelo debe ser mejor.
        assert resumen_modelo['precision_riesgo_cv'] > 0.5

    def test_predicciones_tienen_columnas(self, df_predicciones):
        requeridas = ['provincia', 'periodo', 'tasa_desempleo',
                      'prediccion', 'riesgo_real', 'riesgo_predicho']
        for col in requeridas:
            assert col in df_predicciones.columns

    def test_predicciones_no_negativas(self, df_predicciones):
        assert (df_predicciones['prediccion'] >= 0).all()

    def test_cv_tiene_todos_modelos(self, df_cv_resultados):
        modelos = df_cv_resultados['modelo'].unique()
        assert len(modelos) >= 4
