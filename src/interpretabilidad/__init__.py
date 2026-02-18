"""Modulo de interpretabilidad del modelo."""
try:
    from .shap_analysis import (
        calcular_shap_values,
        fig_shap_summary,
        fig_shap_waterfall,
        fig_shap_bar,
        fig_shap_dependence,
        SHAP_AVAILABLE,
    )

    __all__ = [
        'calcular_shap_values',
        'fig_shap_summary',
        'fig_shap_waterfall',
        'fig_shap_bar',
        'fig_shap_dependence',
        'SHAP_AVAILABLE',
    ]
except ImportError:
    SHAP_AVAILABLE = False
    __all__ = ['SHAP_AVAILABLE']
