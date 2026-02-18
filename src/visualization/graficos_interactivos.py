"""
Visualizaciones interactivas con Plotly y Seaborn para el proyecto de desempleo regional.
Funciones Plotly retornan plotly.graph_objects.Figure.
Funciones Seaborn retornan matplotlib.figure.Figure.
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import COLOR_MAP, RIESGO_BAJO, RIESGO_CRITICO


def _filtrar(df, area="total", sexo="total", provincias=None):
    """Filtra DataFrame por area, sexo y provincias."""
    mask = pd.Series(True, index=df.index)
    if 'area' in df.columns:
        mask &= df['area'] == area
    if 'sexo' in df.columns:
        mask &= df['sexo'] == sexo
    if provincias:
        mask &= df['provincia'].isin(provincias)
    return df[mask].copy()


def fig_evolucion_temporal(df, provincias=None, area="total", sexo="total"):
    """Line chart: tasa de desempleo por provincia a lo largo del tiempo."""
    df_f = _filtrar(df, area, sexo, provincias)
    df_f = df_f.sort_values('periodo')

    fig = px.line(
        df_f, x='periodo', y='tasa_desempleo',
        color='provincia', markers=True,
        labels={
            'periodo': 'Periodo',
            'tasa_desempleo': 'Tasa de Desempleo (%)',
            'provincia': 'Provincia'
        },
        title='Evolucion Temporal del Desempleo por Provincia'
    )

    fig.add_hline(y=RIESGO_BAJO, line_dash="dash", line_color=COLOR_MAP['moderado'],
                  annotation_text=f"Umbral bajo ({RIESGO_BAJO}%)")
    fig.add_hline(y=RIESGO_CRITICO, line_dash="dash", line_color=COLOR_MAP['critico'],
                  annotation_text=f"Umbral critico ({RIESGO_CRITICO}%)")

    fig.update_layout(
        xaxis_tickangle=-45,
        legend=dict(orientation="h", yanchor="bottom", y=-0.4, xanchor="center", x=0.5),
        height=550,
        template="plotly_white"
    )
    return fig


def fig_heatmap_provincia_periodo(df, area="total", sexo="total"):
    """Heatmap: provincias (y) x periodos (x), coloreado por tasa de desempleo."""
    df_f = _filtrar(df, area, sexo)
    pivot = df_f.pivot_table(
        index='provincia', columns='periodo',
        values='tasa_desempleo', aggfunc='mean'
    )
    pivot = pivot.sort_index()

    fig = px.imshow(
        pivot,
        text_auto=".1f",
        color_continuous_scale="RdYlGn_r",
        labels=dict(x="Periodo", y="Provincia", color="Tasa (%)"),
        title="Mapa de Calor: Desempleo por Provincia y Periodo",
        aspect="auto"
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        height=500,
        template="plotly_white"
    )
    return fig


def fig_real_vs_prediccion_barras(df, periodo, area="total", sexo="total"):
    """Barras agrupadas: valor real vs prediccion para un periodo."""
    df_f = _filtrar(df, area, sexo)
    df_f = df_f[df_f['periodo'] == periodo].sort_values('tasa_desempleo', ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df_f['provincia'], x=df_f['tasa_desempleo'],
        name='Real', orientation='h',
        marker_color='#3498db'
    ))
    fig.add_trace(go.Bar(
        y=df_f['provincia'], x=df_f['prediccion'],
        name='Prediccion', orientation='h',
        marker_color='#e67e22'
    ))

    fig.update_layout(
        barmode='group',
        title=f'Real vs Prediccion - {periodo}',
        xaxis_title='Tasa de Desempleo (%)',
        yaxis_title='Provincia',
        height=500,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    return fig


def fig_scatter_prediccion_vs_real(df, area="total", sexo="total"):
    """Scatter: prediccion vs real con linea de referencia 45 grados."""
    df_f = _filtrar(df, area, sexo)

    fig = px.scatter(
        df_f, x='tasa_desempleo', y='prediccion',
        color='provincia', hover_data=['periodo'],
        labels={
            'tasa_desempleo': 'Tasa Real (%)',
            'prediccion': 'Prediccion (%)',
            'provincia': 'Provincia'
        },
        title='Prediccion vs Valor Real'
    )

    val_min = min(df_f['tasa_desempleo'].min(), df_f['prediccion'].min()) - 1
    val_max = max(df_f['tasa_desempleo'].max(), df_f['prediccion'].max()) + 1
    fig.add_shape(
        type="line", x0=val_min, y0=val_min, x1=val_max, y1=val_max,
        line=dict(color="gray", width=2, dash="dash")
    )

    fig.update_layout(
        height=550,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )
    return fig


def fig_boxplot_desempleo(df, area="total", sexo="total"):
    """Box plot: distribucion de desempleo por provincia."""
    df_f = _filtrar(df, area, sexo)

    medianas = df_f.groupby('provincia')['tasa_desempleo'].median().sort_values(ascending=False)
    orden = medianas.index.tolist()

    fig = px.box(
        df_f, x='provincia', y='tasa_desempleo',
        category_orders={'provincia': orden},
        labels={
            'provincia': 'Provincia',
            'tasa_desempleo': 'Tasa de Desempleo (%)'
        },
        title='Distribucion de Desempleo por Provincia'
    )

    fig.add_hline(y=RIESGO_BAJO, line_dash="dash", line_color=COLOR_MAP['moderado'])
    fig.add_hline(y=RIESGO_CRITICO, line_dash="dash", line_color=COLOR_MAP['critico'])

    fig.update_layout(
        xaxis_tickangle=-45,
        height=500,
        template="plotly_white"
    )
    return fig


def fig_feature_importance(df_importance, top_n=15):
    """Barras horizontales de importancia de features."""
    df_top = df_importance.head(top_n).copy()
    df_top = df_top.sort_values('importancia', ascending=True)

    fig = go.Figure(go.Bar(
        x=df_top['importancia'],
        y=df_top.index,
        orientation='h',
        marker_color='#27ae60'
    ))

    fig.update_layout(
        title=f'Top {top_n} Features mas Importantes',
        xaxis_title='Importancia',
        yaxis_title='',
        height=max(400, top_n * 30),
        template="plotly_white"
    )
    return fig


def fig_cv_metricas_por_periodo(df_cv, metrica="rmse"):
    """Lineas: metrica CV por periodo, una linea por modelo."""
    fig = px.line(
        df_cv, x='periodo', y=metrica,
        color='modelo', markers=True,
        labels={
            'periodo': 'Periodo',
            metrica: metrica.upper(),
            'modelo': 'Modelo'
        },
        title=f'Validacion Cruzada: {metrica.upper()} por Periodo'
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        height=500,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.4, xanchor="center", x=0.5)
    )
    return fig


def fig_comparacion_modelos(todos_modelos_cv):
    """Barras: comparacion de modelos en MAE, RMSE, R2."""
    df = pd.DataFrame(todos_modelos_cv).T
    df.index.name = 'modelo'
    df = df.reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(name='MAE', x=df['modelo'], y=df['mae'],
                         marker_color='#3498db'))
    fig.add_trace(go.Bar(name='RMSE', x=df['modelo'], y=df['rmse'],
                         marker_color='#e74c3c'))
    fig.add_trace(go.Bar(name='R\u00b2', x=df['modelo'], y=df['r2'],
                         marker_color='#2ecc71'))

    fig.update_layout(
        barmode='group',
        title='Comparacion de Modelos - Metricas CV',
        xaxis_title='Modelo',
        yaxis_title='Valor',
        height=450,
        template="plotly_white"
    )
    return fig


def fig_distribucion_residuos(df, area="total", sexo="total"):
    """Histograma de residuos del modelo."""
    df_f = _filtrar(df, area, sexo)

    fig = px.histogram(
        df_f, x='residuo', nbins=40,
        labels={'residuo': 'Residuo (Real - Prediccion)', 'count': 'Frecuencia'},
        title='Distribucion de Residuos del Modelo',
        color_discrete_sequence=['#3498db']
    )

    media = df_f['residuo'].mean()
    fig.add_vline(x=0, line_dash="solid", line_color="black", line_width=2)
    fig.add_vline(x=media, line_dash="dash", line_color="red",
                  annotation_text=f"Media: {media:.3f}")

    fig.update_layout(
        height=400,
        template="plotly_white"
    )
    return fig


def fig_evolucion_animada(df, area="total", sexo="total"):
    """Bar chart animado: desempleo por provincia con boton play."""
    df_f = _filtrar(df, area, sexo)
    df_f = df_f.sort_values(['periodo', 'tasa_desempleo'], ascending=[True, False])

    fig = px.bar(
        df_f, x='tasa_desempleo', y='provincia',
        orientation='h',
        animation_frame='periodo',
        range_x=[0, df_f['tasa_desempleo'].max() * 1.15],
        color='tasa_desempleo',
        color_continuous_scale='RdYlGn_r',
        labels={
            'tasa_desempleo': 'Tasa de Desempleo (%)',
            'provincia': 'Provincia',
            'periodo': 'Periodo'
        },
        title='Evolucion del Desempleo por Provincia (Animado)'
    )

    fig.add_vline(x=RIESGO_BAJO, line_dash="dash", line_color=COLOR_MAP['moderado'],
                  line_width=1)
    fig.add_vline(x=RIESGO_CRITICO, line_dash="dash", line_color=COLOR_MAP['critico'],
                  line_width=1)

    fig.update_layout(
        height=550,
        template="plotly_white",
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False
    )
    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 800
    fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 400
    return fig


def fig_correlacion_seaborn(df, feature_cols, target='tasa_desempleo'):
    """Heatmap de correlacion con Seaborn (retorna matplotlib Figure)."""
    cols = [c for c in feature_cols if c in df.columns]
    if target in df.columns and target not in cols:
        cols.append(target)
    df_num = df[cols].dropna()

    corr = df_num.corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap='RdBu_r', center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.5,
        cbar_kws={'shrink': 0.8, 'label': 'Correlacion'},
        ax=ax
    )
    ax.set_title('Matriz de Correlacion de Features', fontsize=16, pad=20)
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
    return fig
