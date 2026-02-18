"""Tests para la calidad y estructura de los datos procesados."""
import pandas as pd
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import PROCESSED_DATA_DIR


class TestDesempleoPorProvincia:
    """Tests sobre el dataset procesado principal."""

    def test_archivo_existe(self):
        assert (PROCESSED_DATA_DIR / "desempleo_por_provincia.csv").exists()

    def test_columnas_requeridas(self, df_desempleo):
        requeridas = ['provincia', 'periodo', 'anio', 'tasa_desempleo', 'pea']
        for col in requeridas:
            assert col in df_desempleo.columns, f"Falta columna: {col}"

    def test_no_esta_vacio(self, df_desempleo):
        assert len(df_desempleo) > 0

    def test_provincias_minimas(self, df_desempleo):
        assert df_desempleo['provincia'].nunique() >= 10

    def test_periodos_multiples(self, df_desempleo):
        assert df_desempleo['periodo'].nunique() >= 5

    def test_tasa_desempleo_rango_valido(self, df_desempleo):
        tasas = df_desempleo['tasa_desempleo'].dropna()
        assert tasas.min() >= 0, "Tasa de desempleo negativa"
        assert tasas.max() <= 100, "Tasa de desempleo mayor a 100%"

    def test_areas_esperadas(self, df_desempleo):
        if 'area' in df_desempleo.columns:
            areas = set(df_desempleo['area'].unique())
            assert 'total' in areas

    def test_sexos_esperados(self, df_desempleo):
        if 'sexo' in df_desempleo.columns:
            sexos = set(df_desempleo['sexo'].unique())
            assert 'total' in sexos

    def test_pea_positiva(self, df_desempleo):
        pea = df_desempleo['pea'].dropna()
        assert (pea >= 0).all(), "PEA con valores negativos"


class TestFeaturesDesempleo:
    """Tests sobre el dataset con features engineered."""

    def test_archivo_existe(self):
        assert (PROCESSED_DATA_DIR / "features_desempleo.csv").exists()

    def test_tiene_mas_columnas_que_base(self, df_features, df_desempleo):
        assert df_features.shape[1] > df_desempleo.shape[1]

    def test_tiene_lag_features(self, df_features):
        lag_cols = [c for c in df_features.columns if '_lag1' in c]
        assert len(lag_cols) > 0, "No hay features de lag"

    def test_tiene_features_interaccion(self, df_features):
        inter_cols = [c for c in df_features.columns if 'lag_x_' in c]
        assert len(inter_cols) > 0, "No hay features de interaccion"

    def test_post_covid_es_binario(self, df_features):
        if 'post_covid' in df_features.columns:
            valores = df_features['post_covid'].dropna().unique()
            assert set(valores).issubset({0, 1})

    def test_dummies_provincia_existen(self, df_features):
        prov_cols = [c for c in df_features.columns if c.startswith('prov_')]
        assert len(prov_cols) > 0, "No hay dummies de provincia"
