"""
Entrenamiento y evaluación de modelos de predicción de desempleo regional.
Implementa validación Leave-One-Period-Out y clasificación de riesgo.
"""
import pandas as pd
import numpy as np
import json
import joblib
import sys
from pathlib import Path

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR,
    RANDOM_STATE, RIESGO_BAJO, RIESGO_CRITICO
)

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

TARGET = 'tasa_desempleo'

# Features seleccionados: ~30 variables para esquema area x sexo
# Dataset expandido a 1605 filas (17 periodos x 13 entidades x 3 areas x 3 sexos)
FEATURES_SELECCIONADOS = [
    # Lag temporal (1)
    'tasa_desempleo_lag1',
    # Mercado laboral (3)
    'tasa_participacion', 'pct_subempleo', 'empleo_informal_pct',
    # Demografía y educación (4)
    'pct_sin_educacion', 'pct_universitaria',
    'pct_jovenes_pea', 'pct_mayores_pea',
    # Estructura económica (3)
    'pct_sector_primario', 'pct_sector_secundario', 'pct_empresa_grande',
    # Contrastes urbano/rural y género (3)
    'brecha_desempleo_urb_rur', 'brecha_desempleo_genero', 'ratio_pea_urbana',
    # Dimensiones area y sexo (4)
    'area_total', 'area_urbana', 'sexo_mujeres', 'sexo_total',
    # Dinámica temporal (3)
    'pct_subempleo_lag1', 'empleo_informal_pct_delta', 'post_covid',
    # Efectos fijos provinciales (2)
    'prov_Col\u00f3n', 'prov_Comarca Kuna Yala',
    # Interacciones lag × contexto (3)
    'lag_x_post_covid', 'lag_x_sexo_mujeres', 'lag_x_area_urbana',
    # Features macro (3)
    'pib_crecimiento', 'desempleo_juvenil_nacional', 'mediana_salario',
    # Features estructurales (2)
    'ratio_terciario_primario', 'educacion_alta',
]


def cargar_datos():
    """Carga el dataset con features."""
    return pd.read_csv(PROCESSED_DATA_DIR / "features_desempleo.csv")


def obtener_feature_cols(df):
    """Retorna las features seleccionadas que existen en el DataFrame."""
    return [c for c in FEATURES_SELECCIONADOS if c in df.columns]


def clasificar_riesgo(tasa):
    """Clasifica nivel de riesgo según umbrales de config.py."""
    if tasa < RIESGO_BAJO:
        return 'bajo'
    elif tasa < RIESGO_CRITICO:
        return 'moderado'
    else:
        return 'critico'


def definir_modelos():
    """Define los modelos a evaluar."""
    modelos = {
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1, max_iter=10000),
        'RandomForest': RandomForestRegressor(
            n_estimators=200, max_depth=5, min_samples_leaf=4,
            random_state=RANDOM_STATE
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=150, max_depth=3, min_samples_leaf=4,
            learning_rate=0.05, random_state=RANDOM_STATE
        ),
    }
    if HAS_XGBOOST:
        modelos['XGBoost'] = XGBRegressor(
            n_estimators=150, max_depth=3, min_child_weight=4,
            learning_rate=0.05, random_state=RANDOM_STATE,
            verbosity=0
        )
    return modelos


PARAM_GRIDS = {
    'GradientBoosting': {
        'learning_rate': [0.01, 0.03, 0.05, 0.08],
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5],
        'min_samples_leaf': [3, 4, 6],
        'subsample': [0.8, 0.9, 1.0],
    },
}
if HAS_XGBOOST:
    PARAM_GRIDS['XGBoost'] = {
        'learning_rate': [0.01, 0.03, 0.05, 0.08],
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5],
        'min_child_weight': [2, 3, 4],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0],
    }


def tuning_hiperparametros(nombre, modelo, X_train, y_train):
    """Ajusta hiperparámetros con RandomizedSearchCV interno (5-fold).

    Solo aplica tuning a GradientBoosting y XGBoost.
    Para otros modelos, retorna el modelo sin cambios.
    """
    if nombre not in PARAM_GRIDS:
        return modelo

    search = RandomizedSearchCV(
        modelo,
        PARAM_GRIDS[nombre],
        n_iter=30,
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        refit=True,
        random_state=RANDOM_STATE,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_


def lopo_cv(df, feature_cols, modelos):
    """Validación cruzada Leave-One-Period-Out."""
    periodos = sorted(df['periodo'].unique())
    resultados = {nombre: [] for nombre in modelos}
    predicciones_cv = []

    for periodo_test in periodos:
        train = df[df['periodo'] != periodo_test]
        test = df[df['periodo'] == periodo_test]

        X_train = train[feature_cols].copy()
        y_train = train[TARGET]
        X_test = test[feature_cols].copy()
        y_test = test[TARGET]

        # Imputar NaN con mediana del entrenamiento
        medians = X_train.median()
        X_train = X_train.fillna(medians)
        X_test = X_test.fillna(medians)

        # Escalar
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)

        for nombre, modelo in modelos.items():
            modelo_tuned = tuning_hiperparametros(nombre, modelo, X_tr, y_train)
            if modelo_tuned is modelo:
                modelo.fit(X_tr, y_train)
            y_pred = np.clip(modelo_tuned.predict(X_te), 0, None)

            resultados[nombre].append({
                'periodo': periodo_test,
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred) if len(y_test) > 1 else 0.0,
            })

            for i, idx in enumerate(test.index):
                predicciones_cv.append({
                    'modelo': nombre,
                    'provincia': test.loc[idx, 'provincia'],
                    'periodo': periodo_test,
                    'real': y_test.loc[idx],
                    'predicho': y_pred[i],
                })

    return resultados, predicciones_cv


def entrenar_final(df, feature_cols, nombre_modelo, modelos):
    """Entrena el modelo final con todos los datos."""
    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df[TARGET]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    modelo = modelos[nombre_modelo]
    modelo = tuning_hiperparametros(nombre_modelo, modelo, X_scaled, y)

    # Feature importance
    if hasattr(modelo, 'feature_importances_'):
        importances = pd.Series(modelo.feature_importances_, index=feature_cols)
    elif hasattr(modelo, 'coef_'):
        importances = pd.Series(np.abs(modelo.coef_), index=feature_cols)
    else:
        importances = pd.Series(dtype=float)
    importances = importances.sort_values(ascending=False)

    # Predicciones + clasificación de riesgo
    y_pred = modelo.predict(X_scaled)
    id_cols = ['provincia', 'periodo', 'anio']
    if 'area' in df.columns:
        id_cols.append('area')
    if 'sexo' in df.columns:
        id_cols.append('sexo')
    resultado = df[id_cols + [TARGET]].copy()
    resultado['prediccion'] = np.clip(y_pred, 0, None)
    resultado['residuo'] = resultado[TARGET] - y_pred
    resultado['riesgo_real'] = resultado[TARGET].apply(clasificar_riesgo)
    resultado['riesgo_predicho'] = pd.Series(
        y_pred, index=resultado.index
    ).apply(clasificar_riesgo)

    return modelo, scaler, importances, resultado


def main():
    print("=" * 60)
    print("ENTRENAMIENTO DE MODELOS")
    print("=" * 60)

    # 1. Cargar datos
    df = cargar_datos()
    feature_cols = obtener_feature_cols(df)
    print(f"Dataset: {df.shape[0]} filas, {len(feature_cols)} features")
    print(f"Periodos: {sorted(df['periodo'].unique())}")
    print(f"Provincias: {df['provincia'].nunique()}")

    # 2. Definir modelos
    modelos = definir_modelos()
    print(f"Modelos: {list(modelos.keys())}")

    # 3. Validación Leave-One-Period-Out
    print("\nValidacion Leave-One-Period-Out...")
    resultados_cv, predicciones_cv = lopo_cv(df, feature_cols, modelos)

    # Resumen CV
    print(f"\n{'Modelo':<22} {'MAE':>7} {'RMSE':>7} {'R2':>7}")
    print("-" * 45)
    resumen = {}
    for nombre, res in resultados_cv.items():
        mae = np.mean([r['mae'] for r in res])
        rmse = np.mean([r['rmse'] for r in res])
        r2 = np.mean([r['r2'] for r in res])
        print(f"{nombre:<22} {mae:>7.3f} {rmse:>7.3f} {r2:>7.3f}")
        resumen[nombre] = {'mae': mae, 'rmse': rmse, 'r2': r2}

    mejor = min(resumen, key=lambda x: resumen[x]['rmse'])
    print(f"\nMejor modelo: {mejor} (RMSE={resumen[mejor]['rmse']:.3f})")

    # Detalle por periodo del mejor modelo
    print(f"\nDetalle por periodo ({mejor}):")
    for r in resultados_cv[mejor]:
        print(f"  {r['periodo']}: MAE={r['mae']:.3f} RMSE={r['rmse']:.3f} R2={r['r2']:.3f}")

    # 4. Entrenar modelo final
    print(f"\nEntrenando {mejor} en todo el dataset...")
    modelo_final, scaler, importances, df_pred = entrenar_final(
        df, feature_cols, mejor, modelos
    )

    # 5. Feature importance
    print("\nTop 15 features:")
    for i, (feat, imp) in enumerate(importances.head(15).items()):
        print(f"  {i+1:2d}. {feat:<35} {imp:.4f}")

    # 6. Precisión de clasificación de riesgo (en entrenamiento)
    acc_train = (df_pred['riesgo_real'] == df_pred['riesgo_predicho']).mean()
    print(f"\nClasificacion de riesgo (entrenamiento): {acc_train:.1%}")
    for nivel in ['bajo', 'moderado', 'critico']:
        mask = df_pred['riesgo_real'] == nivel
        if mask.sum() > 0:
            a = (df_pred.loc[mask, 'riesgo_real'] ==
                 df_pred.loc[mask, 'riesgo_predicho']).mean()
            print(f"  {nivel}: {a:.1%} ({mask.sum()} casos)")

    # 7. Precisión de riesgo en CV (out-of-sample)
    print("\nClasificacion de riesgo en CV (out-of-sample):")
    df_cv = pd.DataFrame(predicciones_cv)
    for nombre in modelos:
        sub = df_cv[df_cv['modelo'] == nombre].copy()
        sub['riesgo_real'] = sub['real'].apply(clasificar_riesgo)
        sub['riesgo_pred'] = sub['predicho'].apply(clasificar_riesgo)
        acc_cv = (sub['riesgo_real'] == sub['riesgo_pred']).mean()
        print(f"  {nombre}: {acc_cv:.1%}")

    # 8. Predicciones del último periodo (solo total/total para resumen)
    ultimo_periodo = df_pred['periodo'].max()
    print(f"\nPredicciones {ultimo_periodo} (area=total, sexo=total):")
    pred_ultimo = df_pred[df_pred['periodo'] == ultimo_periodo].copy()
    if 'area' in pred_ultimo.columns:
        pred_ultimo = pred_ultimo[pred_ultimo['area'] == 'total']
    if 'sexo' in pred_ultimo.columns:
        pred_ultimo = pred_ultimo[pred_ultimo['sexo'] == 'total']
    pred_ultimo = pred_ultimo.sort_values('prediccion', ascending=False)
    print(f"  {'Provincia':<25} {'Real':>7} {'Pred':>7} {'Riesgo':>10}")
    print("  " + "-" * 52)
    for _, row in pred_ultimo.iterrows():
        print(f"  {row['provincia']:<25} {row[TARGET]:>7.2f} "
              f"{row['prediccion']:>7.2f} {row['riesgo_real']:>10}")

    # 9. Guardar todo
    print("\nGuardando modelo y resultados...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Modelo, scaler, features
    joblib.dump(modelo_final, MODELS_DIR / "modelo_desempleo.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(feature_cols, MODELS_DIR / "feature_cols.pkl")

    # Resultados CV detallados
    cv_detail = []
    for nombre, res in resultados_cv.items():
        for r in res:
            cv_detail.append({'modelo': nombre, **r})
    pd.DataFrame(cv_detail).to_csv(
        REPORTS_DIR / "cv_resultados.csv", index=False
    )

    # Predicciones del modelo final
    df_pred.to_csv(REPORTS_DIR / "predicciones_modelo.csv", index=False)

    # Predicciones CV (out-of-sample)
    df_cv.to_csv(REPORTS_DIR / "predicciones_cv.csv", index=False)

    # Feature importance
    importances.to_frame('importancia').to_csv(
        REPORTS_DIR / "feature_importance.csv"
    )

    # Resumen JSON
    # Calcular precisión riesgo CV del mejor modelo
    sub_mejor = df_cv[df_cv['modelo'] == mejor].copy()
    sub_mejor['riesgo_real'] = sub_mejor['real'].apply(clasificar_riesgo)
    sub_mejor['riesgo_pred'] = sub_mejor['predicho'].apply(clasificar_riesgo)
    acc_cv_mejor = float(
        (sub_mejor['riesgo_real'] == sub_mejor['riesgo_pred']).mean()
    )

    resumen_final = {
        'mejor_modelo': mejor,
        'metricas_cv': {k: round(v, 4) for k, v in resumen[mejor].items()},
        'todos_modelos_cv': {
            n: {k: round(v, 4) for k, v in m.items()}
            for n, m in resumen.items()
        },
        'n_features': len(feature_cols),
        'n_filas': len(df),
        'n_periodos': df['periodo'].nunique(),
        'n_provincias': df['provincia'].nunique(),
        'precision_riesgo_train': round(float(acc_train), 4),
        'precision_riesgo_cv': round(acc_cv_mejor, 4),
        'features_top10': list(importances.head(10).index),
    }
    with open(REPORTS_DIR / "resumen_modelo.json", 'w', encoding='utf-8') as f:
        json.dump(resumen_final, f, indent=2, ensure_ascii=False)

    print(f"\nModelo: {MODELS_DIR / 'modelo_desempleo.pkl'}")
    print(f"Reportes: {REPORTS_DIR}")

    print("\n" + "=" * 60)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 60)

    return modelo_final


if __name__ == "__main__":
    main()
