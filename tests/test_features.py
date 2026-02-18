"""Tests para el modulo de feature engineering."""
import pandas as pd
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.features.crear_features import (
    crear_features_contraste,
    crear_features_temporales,
    crear_lag_features,
    crear_delta_features,
    crear_ratios,
    crear_features_interaccion,
    codificar_categoricas,
    imputar_valores,
)


class TestFeaturesContraste:
    def test_crea_brecha_urbano_rural(self, df_sintetico):
        df = crear_features_contraste(df_sintetico)
        assert 'brecha_desempleo_urb_rur' in df.columns

    def test_crea_brecha_genero(self, df_sintetico):
        df = crear_features_contraste(df_sintetico)
        assert 'brecha_desempleo_genero' in df.columns

    def test_crea_ratio_pea_urbana(self, df_sintetico):
        df = crear_features_contraste(df_sintetico)
        assert 'ratio_pea_urbana' in df.columns

    def test_no_pierde_filas(self, df_sintetico):
        n_original = len(df_sintetico)
        df = crear_features_contraste(df_sintetico)
        assert len(df) == n_original


class TestFeaturesTemporales:
    def test_crea_post_covid(self, df_sintetico):
        df = crear_features_temporales(df_sintetico)
        assert 'post_covid' in df.columns

    def test_post_covid_correcto(self, df_sintetico):
        df = crear_features_temporales(df_sintetico)
        for _, row in df.iterrows():
            esperado = 1 if row['anio'] >= 2021 else 0
            assert row['post_covid'] == esperado

    def test_crea_anio_norm(self, df_sintetico):
        df = crear_features_temporales(df_sintetico)
        assert 'anio_norm' in df.columns
        assert df['anio_norm'].min() >= 0
        assert df['anio_norm'].max() <= 1


class TestLagFeatures:
    def test_crea_lag_desempleo(self, df_sintetico):
        df = crear_lag_features(df_sintetico)
        assert 'tasa_desempleo_lag1' in df.columns

    def test_lag_tiene_nulos_primer_periodo(self, df_sintetico):
        df = crear_lag_features(df_sintetico)
        primer_periodo = df['periodo'].min()
        lag_primer = df[df['periodo'] == primer_periodo]['tasa_desempleo_lag1']
        assert lag_primer.isna().all()


class TestDeltaFeatures:
    def test_crea_delta_participacion(self, df_sintetico):
        df = crear_delta_features(df_sintetico)
        assert 'tasa_participacion_delta' in df.columns

    def test_crea_delta_informalidad(self, df_sintetico):
        df = crear_delta_features(df_sintetico)
        assert 'empleo_informal_pct_delta' in df.columns


class TestRatios:
    def test_crea_educacion_alta(self, df_sintetico):
        df = crear_ratios(df_sintetico)
        assert 'educacion_alta' in df.columns

    def test_educacion_alta_es_suma(self, df_sintetico):
        df = crear_ratios(df_sintetico)
        esperado = df_sintetico['pct_universitaria'] + df_sintetico['pct_secundaria']
        pd.testing.assert_series_equal(
            df['educacion_alta'].reset_index(drop=True),
            esperado.reset_index(drop=True),
            check_names=False
        )

    def test_crea_ratio_terciario_primario(self, df_sintetico):
        df = crear_ratios(df_sintetico)
        assert 'ratio_terciario_primario' in df.columns


class TestFeaturesInteraccion:
    def test_crea_lag_x_post_covid(self, df_sintetico):
        df = crear_features_temporales(df_sintetico)
        df = crear_lag_features(df)
        df = crear_features_interaccion(df)
        assert 'lag_x_post_covid' in df.columns

    def test_interaccion_es_producto(self, df_sintetico):
        df = crear_features_temporales(df_sintetico)
        df = crear_lag_features(df)
        df = crear_features_interaccion(df)
        mask = df['tasa_desempleo_lag1'].notna()
        esperado = df.loc[mask, 'tasa_desempleo_lag1'] * df.loc[mask, 'post_covid']
        resultado = df.loc[mask, 'lag_x_post_covid']
        pd.testing.assert_series_equal(resultado, esperado, check_names=False)


class TestCodificarCategoricas:
    def test_crea_dummies_provincia(self, df_sintetico):
        df = codificar_categoricas(df_sintetico)
        prov_cols = [c for c in df.columns if c.startswith('prov_')]
        assert len(prov_cols) > 0

    def test_drop_first(self, df_sintetico):
        n_provincias = df_sintetico['provincia'].nunique()
        df = codificar_categoricas(df_sintetico)
        prov_cols = [c for c in df.columns if c.startswith('prov_')]
        assert len(prov_cols) == n_provincias - 1


class TestImputacion:
    def test_reduce_nulos(self, df_sintetico):
        df = df_sintetico.copy()
        df.loc[5, 'tasa_desempleo'] = np.nan
        df.loc[10, 'pct_subempleo'] = np.nan
        nulos_antes = df.isna().sum().sum()
        df_imp = imputar_valores(df)
        nulos_despues = df_imp.isna().sum().sum()
        assert nulos_despues <= nulos_antes
