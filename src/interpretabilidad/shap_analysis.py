"""
Analisis de interpretabilidad con SHAP (SHapley Additive exPlanations).
Permite explicar predicciones individuales y entender el comportamiento global del modelo.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None


def calcular_shap_values(modelo, X, feature_names=None):
    """
    Calcula los valores SHAP para un modelo XGBoost.

    Args:
        modelo: Modelo XGBoost entrenado
        X: Datos de entrada (array o DataFrame)
        feature_names: Lista de nombres de features (opcional si X es DataFrame)

    Returns:
        shap.Explanation: Objeto con valores SHAP

    Raises:
        ImportError: Si SHAP no esta instalado
    """
    if not SHAP_AVAILABLE:
        raise ImportError(
            "SHAP no esta instalado. Instala con: pip install shap"
        )

    if isinstance(X, pd.DataFrame):
        feature_names = feature_names or X.columns.tolist()
        X_array = X.values
    else:
        X_array = X

    # TreeExplainer es eficiente para modelos basados en arboles
    explainer = shap.TreeExplainer(modelo)
    shap_values = explainer(X_array)

    # Asignar nombres de features si estan disponibles
    if feature_names is not None:
        shap_values.feature_names = feature_names

    return shap_values


def fig_shap_summary(shap_values, X=None, max_display=20):
    """
    Genera un summary plot de SHAP (beeswarm plot).
    Muestra la importancia de cada feature y como afecta las predicciones.

    Args:
        shap_values: Valores SHAP calculados
        X: Datos originales (opcional, para colorear por valor)
        max_display: Numero maximo de features a mostrar

    Returns:
        matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    if X is not None:
        shap.summary_plot(
            shap_values, X,
            max_display=max_display,
            show=False,
            plot_size=None
        )
    else:
        shap.summary_plot(
            shap_values,
            max_display=max_display,
            show=False,
            plot_size=None
        )

    plt.title('SHAP Summary Plot - Impacto de Features en las Predicciones',
              fontsize=14, pad=20)
    plt.tight_layout()

    return plt.gcf()


def fig_shap_waterfall(shap_values, idx=0, max_display=15):
    """
    Genera un waterfall plot para una prediccion individual.
    Muestra como cada feature contribuye a la prediccion final.

    Args:
        shap_values: Valores SHAP calculados
        idx: Indice de la observacion a explicar
        max_display: Numero maximo de features a mostrar

    Returns:
        matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=(12, 8))

    shap.plots.waterfall(shap_values[idx], max_display=max_display, show=False)

    plt.title(f'SHAP Waterfall - Explicacion de Prediccion Individual (obs. {idx})',
              fontsize=14, pad=20)
    plt.tight_layout()

    return plt.gcf()


def fig_shap_bar(shap_values, max_display=20):
    """
    Genera un bar plot de importancia media de features segun SHAP.

    Args:
        shap_values: Valores SHAP calculados
        max_display: Numero maximo de features a mostrar

    Returns:
        matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=(10, 8))

    shap.plots.bar(shap_values, max_display=max_display, show=False)

    plt.title('SHAP Feature Importance - Importancia Media Absoluta',
              fontsize=14, pad=20)
    plt.tight_layout()

    return plt.gcf()


def fig_shap_dependence(shap_values, feature, X=None, interaction_feature=None):
    """
    Genera un dependence plot para una feature especifica.
    Muestra como el valor de la feature afecta la prediccion.

    Args:
        shap_values: Valores SHAP calculados
        feature: Nombre o indice de la feature a analizar
        X: Datos originales (para el eje X)
        interaction_feature: Feature para colorear (detectar interacciones)

    Returns:
        matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    if X is not None:
        shap.dependence_plot(
            feature,
            shap_values.values if hasattr(shap_values, 'values') else shap_values,
            X,
            interaction_index=interaction_feature,
            ax=ax,
            show=False
        )
    else:
        shap.dependence_plot(
            feature,
            shap_values.values if hasattr(shap_values, 'values') else shap_values,
            interaction_index=interaction_feature,
            ax=ax,
            show=False
        )

    plt.title(f'SHAP Dependence Plot - {feature}', fontsize=14, pad=20)
    plt.tight_layout()

    return fig


def generar_explicacion_texto(shap_values, idx, feature_names, top_n=5):
    """
    Genera una explicacion en texto de una prediccion individual.

    Args:
        shap_values: Valores SHAP calculados
        idx: Indice de la observacion
        feature_names: Lista de nombres de features
        top_n: Numero de features mas importantes a incluir

    Returns:
        str: Explicacion en texto
    """
    valores = shap_values[idx].values
    base_value = shap_values[idx].base_values
    prediccion = base_value + valores.sum()

    # Ordenar por valor absoluto
    indices_ordenados = np.argsort(np.abs(valores))[::-1][:top_n]

    explicacion = []
    explicacion.append(f"Prediccion: {prediccion:.2f}%")
    explicacion.append(f"Valor base (promedio): {base_value:.2f}%")
    explicacion.append("")
    explicacion.append(f"Top {top_n} factores que influyen en esta prediccion:")

    for i, idx_feat in enumerate(indices_ordenados, 1):
        nombre = feature_names[idx_feat] if idx_feat < len(feature_names) else f"Feature {idx_feat}"
        valor_shap = valores[idx_feat]
        direccion = "aumenta" if valor_shap > 0 else "disminuye"
        explicacion.append(
            f"  {i}. {nombre}: {direccion} la prediccion en {abs(valor_shap):.2f} pp"
        )

    return "\n".join(explicacion)


def obtener_top_features_shap(shap_values, feature_names, top_n=10):
    """
    Obtiene las features mas importantes segun SHAP.

    Args:
        shap_values: Valores SHAP calculados
        feature_names: Lista de nombres de features
        top_n: Numero de features a retornar

    Returns:
        pd.DataFrame: DataFrame con feature e importancia
    """
    # Calcular importancia media absoluta
    importancias = np.abs(shap_values.values).mean(axis=0)

    df = pd.DataFrame({
        'feature': feature_names,
        'importancia_shap': importancias
    })

    df = df.sort_values('importancia_shap', ascending=False).head(top_n)
    df = df.reset_index(drop=True)

    return df
