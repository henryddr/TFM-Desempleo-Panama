"""Tests para las funciones de visualizacion."""
import pandas as pd
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.visualization.graficos_interactivos import (
    fig_evolucion_temporal,
    fig_heatmap_provincia_periodo,
    fig_real_vs_prediccion_barras,
    fig_scatter_prediccion_vs_real,
    fig_boxplot_desempleo,
    fig_feature_importance,
    fig_cv_metricas_por_periodo,
    fig_comparacion_modelos,
    fig_distribucion_residuos,
    fig_evolucion_animada,
    fig_correlacion_seaborn,
)
import plotly.graph_objects as go
import matplotlib.figure


class TestGraficosPlotly:
    """Tests que verifican que cada funcion retorna un Figure valido."""

    def test_evolucion_temporal(self, df_predicciones):
        fig = fig_evolucion_temporal(df_predicciones)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_evolucion_temporal_con_filtros(self, df_predicciones):
        provincias = df_predicciones['provincia'].unique()[:3].tolist()
        fig = fig_evolucion_temporal(
            df_predicciones, provincias=provincias,
            area='total', sexo='total'
        )
        assert isinstance(fig, go.Figure)

    def test_heatmap(self, df_predicciones):
        fig = fig_heatmap_provincia_periodo(df_predicciones)
        assert isinstance(fig, go.Figure)

    def test_barras_real_vs_pred(self, df_predicciones):
        periodo = df_predicciones['periodo'].max()
        fig = fig_real_vs_prediccion_barras(df_predicciones, periodo)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # Real + Prediccion

    def test_scatter(self, df_predicciones):
        fig = fig_scatter_prediccion_vs_real(df_predicciones)
        assert isinstance(fig, go.Figure)

    def test_boxplot(self, df_predicciones):
        fig = fig_boxplot_desempleo(df_predicciones)
        assert isinstance(fig, go.Figure)

    def test_feature_importance(self, df_feature_importance):
        fig = fig_feature_importance(df_feature_importance, top_n=10)
        assert isinstance(fig, go.Figure)

    def test_cv_metricas(self, df_cv_resultados):
        for metrica in ['rmse', 'mae', 'r2']:
            fig = fig_cv_metricas_por_periodo(df_cv_resultados, metrica=metrica)
            assert isinstance(fig, go.Figure)

    def test_comparacion_modelos(self, resumen_modelo):
        fig = fig_comparacion_modelos(resumen_modelo['todos_modelos_cv'])
        assert isinstance(fig, go.Figure)

    def test_distribucion_residuos(self, df_predicciones):
        fig = fig_distribucion_residuos(df_predicciones)
        assert isinstance(fig, go.Figure)

    def test_evolucion_animada(self, df_predicciones):
        fig = fig_evolucion_animada(df_predicciones)
        assert isinstance(fig, go.Figure)
        assert fig.layout.updatemenus is not None  # Tiene boton Play


class TestSeaborn:
    def test_correlacion_retorna_matplotlib_figure(self, df_features):
        cols = ['tasa_desempleo', 'tasa_participacion', 'pct_subempleo']
        cols = [c for c in cols if c in df_features.columns]
        fig = fig_correlacion_seaborn(df_features, cols)
        assert isinstance(fig, matplotlib.figure.Figure)
