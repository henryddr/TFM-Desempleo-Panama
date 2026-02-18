"""
Feature engineering para prediccion de desempleo regional en Panama.
Genera features derivados a partir del dataset procesado del INEC.

Estrategia: se mantienen area (total/urbana/rural) y sexo
(total/hombres/mujeres) como dimensiones de fila. Features de
contraste urbano-rural y de brecha de genero se computan a nivel
provincia-periodo y se fusionan a todas las filas.
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import PROCESSED_DATA_DIR


def cargar_datos():
    """Carga el dataset procesado."""
    return pd.read_csv(PROCESSED_DATA_DIR / "desempleo_por_provincia.csv")


# ---------------------------------------------------------------------------
# Features de contraste (urbano/rural, genero)
# ---------------------------------------------------------------------------

def crear_features_contraste(df):
    """Crea features de contraste urbano/rural y hombres/mujeres.

    Para cada provincia-periodo, calcula:
      - brecha_desempleo_urb_rur: tasa urbana - tasa rural
      - brecha_desempleo_genero: tasa mujeres - tasa hombres
      - ratio_pea_urbana: PEA urbana / PEA total

    Estos features se fusionan a todas las filas (area x sexo).
    """
    merge_keys = ['provincia', 'periodo']
    features = []

    # --- Brecha urbano-rural (desde sexo='total') ---
    total_sexo = df[df['sexo'] == 'total'].copy()

    urbana = total_sexo[total_sexo['area'] == 'urbana'][
        merge_keys + ['tasa_desempleo', 'pea']
    ].rename(columns={'tasa_desempleo': 'tasa_urb', 'pea': 'pea_urb'})

    rural = total_sexo[total_sexo['area'] == 'rural'][
        merge_keys + ['tasa_desempleo']
    ].rename(columns={'tasa_desempleo': 'tasa_rur'})

    total_area = total_sexo[total_sexo['area'] == 'total'][
        merge_keys + ['pea']
    ].rename(columns={'pea': 'pea_total'})

    brecha_area = urbana.merge(rural, on=merge_keys, how='outer')
    brecha_area = brecha_area.merge(total_area, on=merge_keys, how='left')
    brecha_area['brecha_desempleo_urb_rur'] = brecha_area['tasa_urb'] - brecha_area['tasa_rur']
    brecha_area['ratio_pea_urbana'] = (
        brecha_area['pea_urb'] / brecha_area['pea_total'].replace(0, np.nan)
    )
    features.append(brecha_area[merge_keys + ['brecha_desempleo_urb_rur', 'ratio_pea_urbana']])

    # --- Brecha de genero (desde area='total') ---
    total_area_df = df[df['area'] == 'total'].copy()

    hombres = total_area_df[total_area_df['sexo'] == 'hombres'][
        merge_keys + ['tasa_desempleo']
    ].rename(columns={'tasa_desempleo': 'tasa_hombres'})

    mujeres = total_area_df[total_area_df['sexo'] == 'mujeres'][
        merge_keys + ['tasa_desempleo']
    ].rename(columns={'tasa_desempleo': 'tasa_mujeres'})

    brecha_gen = hombres.merge(mujeres, on=merge_keys, how='outer')
    brecha_gen['brecha_desempleo_genero'] = brecha_gen['tasa_mujeres'] - brecha_gen['tasa_hombres']
    features.append(brecha_gen[merge_keys + ['brecha_desempleo_genero']])

    # Fusionar todos los features de contraste
    for feat_df in features:
        df = df.merge(feat_df, on=merge_keys, how='left')

    return df


# ---------------------------------------------------------------------------
# Imputacion
# ---------------------------------------------------------------------------

def imputar_valores(df, excluir_sufijos=None):
    """Imputa valores faltantes con forward/backward fill por grupo.

    Args:
        excluir_sufijos: lista de sufijos de columna a excluir de la
            imputación (ej. ['_lag1', '_delta'] para no rellenar lags).
    """
    group_cols = ['provincia']
    if 'area' in df.columns:
        group_cols.append('area')
    if 'sexo' in df.columns:
        group_cols.append('sexo')

    df = df.sort_values(group_cols + ['periodo'])
    numericas = df.select_dtypes(include=[np.number]).columns
    for col in numericas:
        if excluir_sufijos and any(col.endswith(s) for s in excluir_sufijos):
            continue
        if df[col].isna().any():
            df[col] = df.groupby(group_cols)[col].transform(
                lambda x: x.ffill().bfill()
            )
    return df


# ---------------------------------------------------------------------------
# Features temporales
# ---------------------------------------------------------------------------

def crear_features_temporales(df):
    """Crea features basados en el tiempo."""
    anio_min, anio_max = df['anio'].min(), df['anio'].max()
    df['anio_norm'] = (df['anio'] - anio_min) / max(anio_max - anio_min, 1)
    df['post_covid'] = (df['anio'] >= 2021).astype(int)
    return df


# ---------------------------------------------------------------------------
# Lag / delta
# ---------------------------------------------------------------------------

def crear_lag_features(df):
    """Crea features de lag (periodo anterior) por provincia-area-sexo."""
    group_cols = ['provincia']
    if 'area' in df.columns:
        group_cols.append('area')
    if 'sexo' in df.columns:
        group_cols.append('sexo')

    df = df.sort_values(group_cols + ['periodo'])
    cols_lag = [
        'tasa_desempleo', 'tasa_participacion', 'empleo_informal_pct',
        'mediana_salario', 'pct_subempleo'
    ]
    for col in cols_lag:
        if col in df.columns:
            df[f'{col}_lag1'] = df.groupby(group_cols)[col].shift(1)
    return df


def crear_delta_features(df):
    """Crea features de cambio respecto al periodo anterior."""
    group_cols = ['provincia']
    if 'area' in df.columns:
        group_cols.append('area')
    if 'sexo' in df.columns:
        group_cols.append('sexo')

    df = df.sort_values(group_cols + ['periodo'])
    cols_delta = ['tasa_participacion', 'empleo_informal_pct', 'mediana_salario']
    for col in cols_delta:
        if col in df.columns:
            df[f'{col}_delta'] = df.groupby(group_cols)[col].diff()
    return df


# ---------------------------------------------------------------------------
# Ratios
# ---------------------------------------------------------------------------

def crear_ratios(df):
    """Crea ratios e interacciones entre variables."""
    if 'empleo_informal_pct' in df.columns:
        df['empleo_formal_pct'] = 100 - df['empleo_informal_pct']

    if 'pct_universitaria' in df.columns and 'pct_secundaria' in df.columns:
        df['educacion_alta'] = df['pct_universitaria'] + df['pct_secundaria']

    if 'pct_sector_terciario' in df.columns and 'pct_sector_primario' in df.columns:
        df['ratio_terciario_primario'] = (
            df['pct_sector_terciario'] /
            df['pct_sector_primario'].replace(0, np.nan)
        )

    if 'pct_microempresa' in df.columns and 'pct_empresa_grande' in df.columns:
        df['ratio_micro_grande'] = (
            df['pct_microempresa'] /
            df['pct_empresa_grande'].replace(0, np.nan)
        )

    if 'mediana_salario' in df.columns and 'pib_per_capita' in df.columns:
        df['salario_rel_pib'] = df['mediana_salario'] / (df['pib_per_capita'] / 12)

    return df


# ---------------------------------------------------------------------------
# Features de interacción
# ---------------------------------------------------------------------------

def crear_features_interaccion(df):
    """Crea interacciones entre lag y variables contextuales.

    Permite al modelo capturar que el lag temporal tiene efectos
    diferentes según contexto (urbano/rural, género, COVID).
    """
    if 'tasa_desempleo_lag1' in df.columns:
        # Lag × post-COVID
        if 'post_covid' in df.columns:
            df['lag_x_post_covid'] = (
                df['tasa_desempleo_lag1'] * df['post_covid']
            )

        # Lag × sexo mujeres
        if 'sexo' in df.columns:
            df['lag_x_sexo_mujeres'] = (
                df['tasa_desempleo_lag1']
                * (df['sexo'] == 'mujeres').astype(int)
            )

        # Lag × area urbana
        if 'area' in df.columns:
            df['lag_x_area_urbana'] = (
                df['tasa_desempleo_lag1']
                * (df['area'] == 'urbana').astype(int)
            )

    return df


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def codificar_categoricas(df):
    """One-hot encoding de provincia, area y sexo."""
    # Provincia
    dummies_prov = pd.get_dummies(
        df['provincia'], prefix='prov', drop_first=True
    ).astype(int)
    df = pd.concat([df, dummies_prov], axis=1)

    # Area
    if 'area' in df.columns:
        dummies_area = pd.get_dummies(
            df['area'], prefix='area', drop_first=True
        ).astype(int)
        df = pd.concat([df, dummies_area], axis=1)

    # Sexo
    if 'sexo' in df.columns:
        dummies_sexo = pd.get_dummies(
            df['sexo'], prefix='sexo', drop_first=True
        ).astype(int)
        df = pd.concat([df, dummies_sexo], axis=1)

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Pipeline principal de feature engineering."""
    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    # 1. Cargar datos
    df = cargar_datos()
    print(f"Dataset cargado: {df.shape[0]} filas x {df.shape[1]} columnas")
    if 'area' in df.columns:
        print(f"  Areas: {df['area'].value_counts().to_dict()}")
    if 'sexo' in df.columns:
        print(f"  Sexos: {df['sexo'].value_counts().to_dict()}")
    print(f"Nulos iniciales: {df.isna().sum().sum()}")

    n_cols_base = df.shape[1]

    # 2. Features de contraste (urbano/rural, genero)
    df = crear_features_contraste(df)
    print(f"\nTras features contraste: {df.shape[0]} filas x {df.shape[1]} columnas")

    # 3. Imputar valores faltantes
    df = imputar_valores(df)
    print(f"Nulos tras imputacion: {df.isna().sum().sum()}")

    # 4. Features temporales
    df = crear_features_temporales(df)

    # 5. Lag features
    df = crear_lag_features(df)

    # 6. Delta features
    df = crear_delta_features(df)

    # 7. Ratios
    df = crear_ratios(df)

    # 7b. Features de interacción (lag × contexto)
    df = crear_features_interaccion(df)

    # 8. Codificar categoricas
    df = codificar_categoricas(df)

    # 9. Segunda imputacion para features derivados (sin rellenar lag/delta)
    df = imputar_valores(df, excluir_sufijos=['_lag1', '_delta'])

    # 10. Guardar
    output = PROCESSED_DATA_DIR / "features_desempleo.csv"
    df.to_csv(output, index=False)

    # Resumen
    nuevas = df.shape[1] - n_cols_base
    print(f"\nDataset final: {df.shape[0]} filas x {df.shape[1]} columnas")
    print(f"Features nuevos: {nuevas}")
    print(f"Nulos restantes: {df.isna().sum().sum()}")

    print(f"\nGuardado en: {output}")
    return df


if __name__ == "__main__":
    main()
