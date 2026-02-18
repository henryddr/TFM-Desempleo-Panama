"""
Tests para el modulo de interpretabilidad SHAP.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Verificar si SHAP esta disponible
try:
    import shap
    SHAP_INSTALLED = True
except ImportError:
    SHAP_INSTALLED = False


class TestSHAPAnalysis:
    """Tests para las funciones de analisis SHAP."""

    def test_import_interpretabilidad_module(self):
        """Verificar que el modulo de interpretabilidad se puede importar."""
        from src.interpretabilidad import SHAP_AVAILABLE
        # El modulo debe importarse, SHAP_AVAILABLE indica si SHAP esta instalado
        assert isinstance(SHAP_AVAILABLE, bool)

    @pytest.mark.skipif(not SHAP_INSTALLED, reason="SHAP no instalado")
    def test_import_shap_functions(self):
        """Verificar que las funciones SHAP se pueden importar."""
        from src.interpretabilidad import (
            calcular_shap_values,
            fig_shap_summary,
            fig_shap_waterfall,
            fig_shap_bar,
            fig_shap_dependence,
        )
        assert calcular_shap_values is not None
        assert fig_shap_summary is not None
        assert fig_shap_waterfall is not None
        assert fig_shap_bar is not None
        assert fig_shap_dependence is not None

    @pytest.mark.skipif(not SHAP_INSTALLED, reason="SHAP no instalado")
    def test_calcular_shap_values_con_modelo_simple(self):
        """Probar calculo de SHAP con un modelo XGBoost simple."""
        from xgboost import XGBRegressor
        from src.interpretabilidad import calcular_shap_values

        # Crear datos de prueba
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100),
        })
        y = X['feature1'] * 2 + X['feature2'] + np.random.randn(100) * 0.1

        # Entrenar modelo simple
        modelo = XGBRegressor(n_estimators=10, random_state=42)
        modelo.fit(X, y)

        # Calcular SHAP
        shap_values = calcular_shap_values(modelo, X, X.columns.tolist())

        # Verificaciones
        assert shap_values is not None
        assert len(shap_values) == len(X)
        assert shap_values.values.shape == (100, 3)

    @pytest.mark.skipif(not SHAP_INSTALLED, reason="SHAP no instalado")
    def test_fig_shap_bar_retorna_figura(self):
        """Verificar que fig_shap_bar retorna una figura matplotlib."""
        from xgboost import XGBRegressor
        from src.interpretabilidad import calcular_shap_values, fig_shap_bar
        import matplotlib.pyplot as plt

        # Crear datos y modelo
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.randn(50),
            'feature2': np.random.randn(50),
        })
        y = X['feature1'] + np.random.randn(50) * 0.1

        modelo = XGBRegressor(n_estimators=5, random_state=42)
        modelo.fit(X, y)

        shap_values = calcular_shap_values(modelo, X, X.columns.tolist())
        fig = fig_shap_bar(shap_values, max_display=5)

        assert fig is not None
        assert hasattr(fig, 'savefig')  # Es una figura matplotlib
        plt.close(fig)

    @pytest.mark.skipif(not SHAP_INSTALLED, reason="SHAP no instalado")
    def test_generar_explicacion_texto(self):
        """Verificar que se genera explicacion en texto correctamente."""
        from xgboost import XGBRegressor
        from src.interpretabilidad.shap_analysis import (
            calcular_shap_values,
            generar_explicacion_texto
        )

        # Crear datos y modelo
        np.random.seed(42)
        X = pd.DataFrame({
            'tasa_lag': np.random.randn(30),
            'subempleo': np.random.randn(30),
        })
        y = X['tasa_lag'] * 2 + np.random.randn(30) * 0.1

        modelo = XGBRegressor(n_estimators=5, random_state=42)
        modelo.fit(X, y)

        shap_values = calcular_shap_values(modelo, X, X.columns.tolist())
        explicacion = generar_explicacion_texto(
            shap_values, idx=0, feature_names=X.columns.tolist(), top_n=2
        )

        assert isinstance(explicacion, str)
        assert "Prediccion" in explicacion
        assert "Valor base" in explicacion

    @pytest.mark.skipif(not SHAP_INSTALLED, reason="SHAP no instalado")
    def test_obtener_top_features_shap(self):
        """Verificar que obtener_top_features_shap retorna DataFrame correcto."""
        from xgboost import XGBRegressor
        from src.interpretabilidad.shap_analysis import (
            calcular_shap_values,
            obtener_top_features_shap
        )

        # Crear datos y modelo
        np.random.seed(42)
        X = pd.DataFrame({
            'feat1': np.random.randn(50),
            'feat2': np.random.randn(50),
            'feat3': np.random.randn(50),
        })
        y = X['feat1'] * 3 + X['feat2'] + np.random.randn(50) * 0.1

        modelo = XGBRegressor(n_estimators=10, random_state=42)
        modelo.fit(X, y)

        shap_values = calcular_shap_values(modelo, X, X.columns.tolist())
        df_top = obtener_top_features_shap(shap_values, X.columns.tolist(), top_n=2)

        assert isinstance(df_top, pd.DataFrame)
        assert 'feature' in df_top.columns
        assert 'importancia_shap' in df_top.columns
        assert len(df_top) == 2
        # feat1 deberia ser la mas importante
        assert df_top.iloc[0]['feature'] == 'feat1'

    def test_shap_not_available_raises_error(self):
        """Verificar que se lanza error si SHAP no esta disponible."""
        from src.interpretabilidad.shap_analysis import SHAP_AVAILABLE

        if SHAP_AVAILABLE:
            pytest.skip("SHAP esta instalado, no aplica este test")

        # Si SHAP no esta instalado, importar calcular_shap_values deberia
        # funcionar pero llamarlo deberia lanzar ImportError
        from src.interpretabilidad.shap_analysis import calcular_shap_values

        with pytest.raises(ImportError):
            calcular_shap_values(None, None)


@pytest.mark.skipif(not SHAP_INSTALLED, reason="SHAP no instalado")
class TestIntegracionConModelo:
    """Tests de integracion con el modelo entrenado del proyecto."""

    def test_shap_con_modelo_proyecto(self, df_features):
        """Probar SHAP con el modelo real del proyecto."""
        import joblib
        from src.interpretabilidad import calcular_shap_values
        from src.config import MODELS_DIR

        # Cargar modelo
        modelo = joblib.load(MODELS_DIR / "modelo_desempleo.pkl")
        feature_cols = joblib.load(MODELS_DIR / "feature_cols.pkl")

        # Preparar datos
        df_sample = df_features[
            (df_features['area'] == 'total') &
            (df_features['sexo'] == 'total')
        ].head(50).copy()

        X_sample = df_sample[feature_cols].fillna(0)

        # Calcular SHAP
        shap_values = calcular_shap_values(modelo, X_sample, feature_cols)

        assert shap_values is not None
        assert len(shap_values) == len(X_sample)
