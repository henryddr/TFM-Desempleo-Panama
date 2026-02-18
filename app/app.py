"""
Dashboard interactivo: Prediccion de Desempleo Regional - Panama
Trabajo Final de Master (TFM)
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import sys
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    APP_TITLE, APP_ICON, APP_LAYOUT,
    MODELS_DIR, REPORTS_DIR, FIGURES_DIR, SHAPEFILES_DIR, PROCESSED_DATA_DIR,
    COLOR_MAP, RIESGO_BAJO, RIESGO_CRITICO
)
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

# ── Configuracion de pagina ──────────────────────────────────────────────────
st.set_page_config(page_title=APP_TITLE, page_icon="\U0001f1f5\U0001f1e6", layout=APP_LAYOUT)

# ── Tema Panama: rojo, blanco y azul ─────────────────────────────────────────
AZUL_PA = "#003DA5"
ROJO_PA = "#D21034"
BLANCO_PA = "#FFFFFF"
AZUL_CLARO_PA = "#E8EFF9"

st.markdown(f"""
<style>
    /* Header y titulo */
    .stApp > header {{
        background-color: {AZUL_PA};
    }}
    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {AZUL_PA};
    }}
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {{
        color: {BLANCO_PA} !important;
    }}
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
        border-bottom-color: {ROJO_PA};
        color: {ROJO_PA};
    }}
    /* Barra superior decorativa */
    .panama-header {{
        background: linear-gradient(90deg, {AZUL_PA} 33%, {BLANCO_PA} 33%, {BLANCO_PA} 66%, {ROJO_PA} 66%);
        height: 6px;
        border-radius: 3px;
        margin-bottom: 1rem;
    }}
</style>
""", unsafe_allow_html=True)


# ── Carga de datos ───────────────────────────────────────────────────────────
@st.cache_data
def cargar_datos():
    with open(REPORTS_DIR / "resumen_modelo.json", "r", encoding="utf-8") as f:
        resumen = json.load(f)

    df_pred = pd.read_csv(REPORTS_DIR / "predicciones_modelo.csv")
    df_cv = pd.read_csv(REPORTS_DIR / "cv_resultados.csv")
    df_feat = pd.read_csv(REPORTS_DIR / "feature_importance.csv", index_col=0)
    df_features = pd.read_csv(PROCESSED_DATA_DIR / "features_desempleo.csv")

    return resumen, df_pred, df_cv, df_feat, df_features


@st.cache_resource
def cargar_modelo():
    modelo = joblib.load(MODELS_DIR / "modelo_desempleo.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    feature_cols = joblib.load(MODELS_DIR / "feature_cols.pkl")
    return modelo, scaler, feature_cols


resumen, df_pred, df_cv, df_feat, df_features = cargar_datos()
modelo, scaler, feature_cols = cargar_modelo()


def clasificar_riesgo(tasa):
    if tasa < RIESGO_BAJO:
        return 'bajo'
    elif tasa < RIESGO_CRITICO:
        return 'moderado'
    else:
        return 'critico'


# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("Navegacion")

seccion = st.sidebar.radio(
    "Ir a seccion:",
    [
        "Resumen del Modelo",
        "Mapa Interactivo",
        "Evolucion Temporal",
        "Predicciones vs Real",
        "Analisis de Riesgo",
        "Rendimiento del Modelo",
        "Interpretabilidad SHAP",
        "Predictor Interactivo",
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filtros de Datos")

periodos = sorted(df_pred['periodo'].unique())
provincias_all = sorted(df_pred['provincia'].unique())

periodo_sel = st.sidebar.selectbox(
    "Periodo", periodos, index=len(periodos) - 1
)
area_sel = st.sidebar.selectbox("Area", ["total", "urbana", "rural"], index=0)
sexo_sel = st.sidebar.selectbox("Sexo", ["total", "hombres", "mujeres"], index=0)
provincias_sel = st.sidebar.multiselect(
    "Provincias", provincias_all, default=provincias_all
)

st.sidebar.markdown("---")
st.sidebar.caption("TFM - Aplicacion de Machine Learning para la Prediccion de Desempleo Regional en Panama")


# ══════════════════════════════════════════════════════════════════════════════
# SECCION 1: RESUMEN DEL MODELO
# ══════════════════════════════════════════════════════════════════════════════
if seccion == "Resumen del Modelo":
    st.title(APP_TITLE)
    st.markdown('<div class="panama-header"></div>', unsafe_allow_html=True)
    st.markdown("### Resumen del Modelo Predictivo")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mejor Modelo", resumen['mejor_modelo'])
    with col2:
        st.metric("R\u00b2 (CV)", f"{resumen['metricas_cv']['r2']:.4f}")
    with col3:
        st.metric("RMSE (CV)", f"{resumen['metricas_cv']['rmse']:.4f}")
    with col4:
        st.metric("MAE (CV)", f"{resumen['metricas_cv']['mae']:.4f}")

    col5, col6, col7 = st.columns(3)
    with col5:
        st.metric("Precision Riesgo (CV)",
                   f"{resumen['precision_riesgo_cv'] * 100:.1f}%")
    with col6:
        st.metric("Features", resumen['n_features'])
    with col7:
        st.metric("Observaciones", f"{resumen['n_filas']:,}")

    st.markdown("---")

    st.markdown("#### Descripcion del Proyecto")
    st.markdown(f"""
    Este modelo predice la **tasa de desempleo regional** en las
    {resumen['n_provincias']} provincias y comarcas de Panama,
    utilizando datos de **{resumen['n_periodos']} periodos** (2011-2024).

    Se emplea validacion cruzada **Leave-One-Period-Out (LOPO)** para
    evaluar el rendimiento predictivo de forma temporal.
    Los niveles de riesgo se clasifican como:
    **bajo** (<{RIESGO_BAJO}%), **moderado** ({RIESGO_BAJO}-{RIESGO_CRITICO}%),
    y **critico** (>{RIESGO_CRITICO}%).
    """)

    st.markdown("#### Comparacion de Modelos")
    df_modelos = pd.DataFrame(resumen['todos_modelos_cv']).T
    df_modelos.columns = ['MAE', 'RMSE', 'R\u00b2']
    df_modelos = df_modelos.sort_values('R\u00b2', ascending=False)
    st.table(
        df_modelos.style
        .highlight_max(axis=0, subset=['R\u00b2'],
                       props='background-color: #003DA5; color: white;')
        .highlight_min(axis=0, subset=['MAE', 'RMSE'],
                       props='background-color: #003DA5; color: white;')
        .format("{:.4f}")
    )

    st.markdown("#### Top 10 Features mas Importantes")
    fig = fig_feature_importance(df_feat, top_n=10)
    st.plotly_chart(fig, use_container_width=True, key="feat_resumen")


# ══════════════════════════════════════════════════════════════════════════════
# SECCION 2: MAPA INTERACTIVO
# ══════════════════════════════════════════════════════════════════════════════
elif seccion == "Mapa Interactivo":
    st.title("Mapa Interactivo de Desempleo")
    st.markdown(f"**Periodo seleccionado:** {periodo_sel}")

    try:
        from src.visualization.mapas import (
            cargar_shapefile, cargar_datos_desempleo,
            cargar_predicciones, mapa_interactivo
        )
        import streamlit.components.v1 as components

        gdf = cargar_shapefile()
        df_desempleo = cargar_datos_desempleo()
        df_pred_total = cargar_predicciones()

        m = mapa_interactivo(gdf, df_desempleo, periodo_sel,
                             df_pred=df_pred_total)
        components.html(m._repr_html_(), height=600, scrolling=True)

    except FileNotFoundError as e:
        st.error(f"No se encontro el archivo geoespacial: {e}")
        st.info("Verifique que los shapefiles estan en data/shapefiles/HDX/")
    except Exception as e:
        st.error(f"Error al generar el mapa: {e}")

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f'<div style="background-color:{COLOR_MAP["bajo"]};padding:10px;'
            f'border-radius:5px;text-align:center;color:white;">'
            f'<b>Bajo</b> (&lt;{RIESGO_BAJO}%)</div>',
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f'<div style="background-color:{COLOR_MAP["moderado"]};padding:10px;'
            f'border-radius:5px;text-align:center;color:white;">'
            f'<b>Moderado</b> ({RIESGO_BAJO}-{RIESGO_CRITICO}%)</div>',
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f'<div style="background-color:{COLOR_MAP["critico"]};padding:10px;'
            f'border-radius:5px;text-align:center;color:white;">'
            f'<b>Critico</b> (&gt;{RIESGO_CRITICO}%)</div>',
            unsafe_allow_html=True
        )


# ══════════════════════════════════════════════════════════════════════════════
# SECCION 3: EVOLUCION TEMPORAL
# ══════════════════════════════════════════════════════════════════════════════
elif seccion == "Evolucion Temporal":
    st.title("Evolucion Temporal del Desempleo")

    tab_linea, tab_animado = st.tabs(["Lineas por Provincia", "Animacion Temporal"])

    with tab_linea:
        fig = fig_evolucion_temporal(df_pred, provincias=provincias_sel,
                                     area=area_sel, sexo=sexo_sel)
        st.plotly_chart(fig, use_container_width=True)

    with tab_animado:
        st.markdown("Presiona **Play** para ver la evolucion del desempleo:")
        fig_anim = fig_evolucion_animada(df_pred, area=area_sel, sexo=sexo_sel)
        st.plotly_chart(fig_anim, use_container_width=True)

    st.markdown("### Mapa de Calor: Provincia x Periodo")
    fig_hm = fig_heatmap_provincia_periodo(df_pred, area=area_sel, sexo=sexo_sel)
    st.plotly_chart(fig_hm, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECCION 4: PREDICCIONES VS REAL
# ══════════════════════════════════════════════════════════════════════════════
elif seccion == "Predicciones vs Real":
    st.title("Predicciones vs Valores Reales")

    tab1, tab2 = st.tabs(["Por Periodo", "Todas las Predicciones"])

    with tab1:
        fig_bar = fig_real_vs_prediccion_barras(
            df_pred, periodo_sel, area=area_sel, sexo=sexo_sel
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab2:
        fig_scatter = fig_scatter_prediccion_vs_real(
            df_pred, area=area_sel, sexo=sexo_sel
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("### Distribucion de Residuos")
    fig_res = fig_distribucion_residuos(df_pred, area=area_sel, sexo=sexo_sel)
    st.plotly_chart(fig_res, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECCION 5: ANALISIS DE RIESGO
# ══════════════════════════════════════════════════════════════════════════════
elif seccion == "Analisis de Riesgo":
    st.title("Analisis de Riesgo de Desempleo")

    mask = (
        (df_pred['periodo'] == periodo_sel) &
        (df_pred['area'] == area_sel) &
        (df_pred['sexo'] == sexo_sel)
    )
    df_f = df_pred[mask].copy()

    st.markdown(f"### Clasificacion de Riesgo - {periodo_sel}")

    n_bajo = (df_f['riesgo_real'] == 'bajo').sum()
    n_moderado = (df_f['riesgo_real'] == 'moderado').sum()
    n_critico = (df_f['riesgo_real'] == 'critico').sum()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Riesgo Bajo", n_bajo,
                   help=f"Tasa < {RIESGO_BAJO}%")
    with col2:
        st.metric("Riesgo Moderado", n_moderado,
                   help=f"Tasa {RIESGO_BAJO}-{RIESGO_CRITICO}%")
    with col3:
        st.metric("Riesgo Critico", n_critico,
                   help=f"Tasa > {RIESGO_CRITICO}%")

    if len(df_f) > 0:
        aciertos = (df_f['riesgo_real'] == df_f['riesgo_predicho']).sum()
        st.metric("Precision Clasificacion",
                   f"{aciertos / len(df_f) * 100:.1f}%")

    st.markdown("### Detalle por Provincia")
    if len(df_f) > 0:
        df_display = df_f[['provincia', 'tasa_desempleo', 'prediccion',
                            'residuo', 'riesgo_real', 'riesgo_predicho']].copy()
        df_display.columns = ['Provincia', 'Tasa Real (%)', 'Prediccion (%)',
                              'Residuo', 'Riesgo Real', 'Riesgo Predicho']
        df_display = df_display.sort_values('Tasa Real (%)', ascending=False)
        st.dataframe(
            df_display.style.format({
                'Tasa Real (%)': '{:.2f}',
                'Prediccion (%)': '{:.2f}',
                'Residuo': '{:.2f}'
            }),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("No hay datos para la combinacion de filtros seleccionada.")

    st.markdown("### Distribucion por Provincia (todos los periodos)")
    fig_box = fig_boxplot_desempleo(df_pred, area=area_sel, sexo=sexo_sel)
    st.plotly_chart(fig_box, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECCION 6: RENDIMIENTO DEL MODELO
# ══════════════════════════════════════════════════════════════════════════════
elif seccion == "Rendimiento del Modelo":
    st.title("Rendimiento del Modelo - Validacion Cruzada")

    st.markdown("### Comparacion de Modelos")
    fig_comp = fig_comparacion_modelos(resumen['todos_modelos_cv'])
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("### Metricas por Periodo")
    metrica_sel = st.selectbox(
        "Seleccionar metrica",
        ["rmse", "mae", "r2"],
        format_func=lambda x: x.upper()
    )
    fig_cv_fig = fig_cv_metricas_por_periodo(df_cv, metrica=metrica_sel)
    st.plotly_chart(fig_cv_fig, use_container_width=True)

    st.markdown("### Importancia de Features")
    fig_feat_fig = fig_feature_importance(df_feat, top_n=20)
    st.plotly_chart(fig_feat_fig, use_container_width=True)

    st.markdown("### Matriz de Correlacion (Seaborn)")
    corr_features = [
        'tasa_desempleo', 'tasa_desempleo_lag1', 'tasa_participacion',
        'pct_subempleo', 'empleo_informal_pct', 'pct_universitaria',
        'pct_empresa_grande', 'mediana_salario', 'pib_crecimiento',
        'brecha_desempleo_genero', 'ratio_terciario_primario',
        'educacion_alta', 'desempleo_juvenil_nacional',
    ]
    fig_corr = fig_correlacion_seaborn(df_features, corr_features)
    st.pyplot(fig_corr)

    with st.expander("Ver resultados detallados de CV"):
        pivot_cv = df_cv.pivot_table(
            index='periodo', columns='modelo',
            values=['rmse', 'mae', 'r2']
        ).round(4)
        st.dataframe(pivot_cv, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECCION 7: INTERPRETABILIDAD SHAP
# ══════════════════════════════════════════════════════════════════════════════
elif seccion == "Interpretabilidad SHAP":
    st.title("Interpretabilidad del Modelo con SHAP")
    st.markdown("""
    **SHAP (SHapley Additive exPlanations)** permite entender como el modelo
    toma decisiones, tanto a nivel global como para predicciones individuales.

    A diferencia de la importancia de features tradicional, SHAP muestra:
    - **Direccion del efecto**: si una variable aumenta o disminuye la prediccion
    - **Magnitud del impacto**: cuantos puntos porcentuales aporta cada variable
    - **Explicaciones individuales**: por que una provincia tiene alta/baja prediccion
    """)

    try:
        from src.interpretabilidad.shap_analysis import (
            calcular_shap_values,
            fig_shap_summary,
            fig_shap_waterfall,
            fig_shap_bar,
            generar_explicacion_texto,
            obtener_top_features_shap
        )

        # Preparar datos para SHAP (muestra para eficiencia)
        @st.cache_data
        def preparar_datos_shap():
            # Filtrar datos para calcular SHAP
            df_sample = df_features[
                (df_features['area'] == 'total') &
                (df_features['sexo'] == 'total')
            ].copy()

            # Tomar una muestra si hay muchos datos
            if len(df_sample) > 500:
                df_sample = df_sample.sample(n=500, random_state=42)

            X_sample = df_sample[feature_cols].fillna(0)
            return df_sample, X_sample

        @st.cache_resource
        def calcular_shap_cached(_modelo, X_sample, feature_cols):
            return calcular_shap_values(_modelo, X_sample, feature_cols)

        df_sample, X_sample = preparar_datos_shap()

        with st.spinner("Calculando valores SHAP (esto puede tardar unos segundos)..."):
            shap_values = calcular_shap_cached(modelo, X_sample, feature_cols)

        # Tabs para diferentes visualizaciones
        tab_global, tab_individual, tab_comparar = st.tabs([
            "Importancia Global",
            "Explicacion Individual",
            "Comparar Features"
        ])

        with tab_global:
            st.markdown("### Importancia Global de Features (SHAP)")
            st.markdown("""
            Este grafico muestra como cada feature contribuye a las predicciones
            en promedio. El color indica el valor de la feature (rojo=alto, azul=bajo).
            """)

            fig_summary = fig_shap_summary(shap_values, X_sample, max_display=15)
            st.pyplot(fig_summary)
            plt.close()

            st.markdown("### Importancia Media Absoluta")
            fig_bar_shap = fig_shap_bar(shap_values, max_display=15)
            st.pyplot(fig_bar_shap)
            plt.close()

            # Tabla de top features
            st.markdown("### Top 10 Features segun SHAP")
            df_top = obtener_top_features_shap(shap_values, feature_cols, top_n=10)
            st.dataframe(
                df_top.style.format({'importancia_shap': '{:.4f}'}),
                use_container_width=True,
                hide_index=True
            )

        with tab_individual:
            st.markdown("### Explicacion de Prediccion Individual")
            st.markdown("""
            Selecciona una observacion para ver como cada feature contribuye
            a esa prediccion especifica. El waterfall plot muestra el camino
            desde el valor base hasta la prediccion final.
            """)

            # Selector de observacion
            provincias_sample = df_sample['provincia'].tolist()
            periodos_sample = df_sample['periodo'].tolist()

            opciones = [
                f"{prov} - {per}"
                for prov, per in zip(provincias_sample, periodos_sample)
            ]

            seleccion = st.selectbox(
                "Seleccionar observacion",
                range(len(opciones)),
                format_func=lambda x: opciones[x]
            )

            # Mostrar waterfall
            fig_water = fig_shap_waterfall(shap_values, idx=seleccion, max_display=12)
            st.pyplot(fig_water)
            plt.close()

            # Explicacion en texto
            st.markdown("### Explicacion en Texto")
            explicacion = generar_explicacion_texto(
                shap_values, seleccion, feature_cols, top_n=5
            )
            st.code(explicacion, language=None)

            # Datos de la observacion
            with st.expander("Ver datos de la observacion seleccionada"):
                row = df_sample.iloc[seleccion]
                cols_mostrar = ['provincia', 'periodo', 'tasa_desempleo'] + [
                    c for c in feature_cols[:10] if c in df_sample.columns
                ]
                st.dataframe(
                    row[cols_mostrar].to_frame().T,
                    use_container_width=True,
                    hide_index=True
                )

        with tab_comparar:
            st.markdown("### Comparar Importancia: SHAP vs Feature Importance")
            st.markdown("""
            Comparacion entre la importancia de XGBoost (basada en ganancia)
            y la importancia SHAP (basada en contribucion a predicciones).
            """)

            # Obtener ambas importancias
            df_shap_top = obtener_top_features_shap(shap_values, feature_cols, top_n=15)
            df_xgb_top = df_feat.head(15).copy()
            df_xgb_top = df_xgb_top.reset_index()
            df_xgb_top.columns = ['feature', 'importancia_xgb']

            # Merge
            df_comp = pd.merge(
                df_shap_top, df_xgb_top,
                on='feature', how='outer'
            ).fillna(0)

            # Normalizar para comparar
            df_comp['shap_norm'] = df_comp['importancia_shap'] / df_comp['importancia_shap'].max()
            df_comp['xgb_norm'] = df_comp['importancia_xgb'] / df_comp['importancia_xgb'].max()

            # Grafico comparativo
            import plotly.graph_objects as go

            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(
                name='SHAP',
                y=df_comp['feature'],
                x=df_comp['shap_norm'],
                orientation='h',
                marker_color='#3498db'
            ))
            fig_comp.add_trace(go.Bar(
                name='XGBoost',
                y=df_comp['feature'],
                x=df_comp['xgb_norm'],
                orientation='h',
                marker_color='#e67e22'
            ))
            fig_comp.update_layout(
                barmode='group',
                title='Comparacion de Importancia (Normalizada)',
                xaxis_title='Importancia Relativa',
                height=500,
                template='plotly_white'
            )
            st.plotly_chart(fig_comp, use_container_width=True)

            st.markdown("""
            **Interpretacion:**
            - Si ambos metodos coinciden, la feature es robustamente importante
            - SHAP captura mejor las interacciones y efectos no lineales
            - XGBoost feature importance puede sobrevalorar features con muchas categorias
            """)

    except ImportError as e:
        st.error(f"Error al importar SHAP: {e}")
        st.info("""
        Para usar esta seccion, instala SHAP:
        ```
        pip install shap
        ```
        """)
    except Exception as e:
        st.error(f"Error al calcular SHAP: {e}")
        st.info("Intenta recargar la pagina o contacta al desarrollador.")


# ══════════════════════════════════════════════════════════════════════════════
# SECCION 8: PREDICTOR INTERACTIVO
# ══════════════════════════════════════════════════════════════════════════════
elif seccion == "Predictor Interactivo":
    st.title("Predictor Interactivo de Desempleo")
    st.markdown("""
    Ajusta las variables clave para generar una prediccion con el modelo
    **{modelo}** entrenado. Los valores iniciales corresponden a la ultima
    observacion disponible para la provincia seleccionada.
    """.format(modelo=resumen['mejor_modelo']))

    # Seleccion de provincia, area, sexo
    col_prov, col_area, col_sexo = st.columns(3)
    with col_prov:
        prov_pred = st.selectbox("Provincia", provincias_all,
                                  index=provincias_all.index("Panama")
                                  if "Panama" in provincias_all
                                  else 0,
                                  key="pred_prov")
    with col_area:
        area_pred = st.selectbox("Area", ["total", "urbana", "rural"],
                                  key="pred_area")
    with col_sexo:
        sexo_pred = st.selectbox("Sexo", ["total", "hombres", "mujeres"],
                                  key="pred_sexo")

    # Obtener datos base para esta combinacion
    ultimo_periodo = df_features['periodo'].max()
    mask_base = (
        (df_features['provincia'] == prov_pred) &
        (df_features['periodo'] == ultimo_periodo)
    )
    if 'area' in df_features.columns:
        mask_base &= df_features['area'] == area_pred
    if 'sexo' in df_features.columns:
        mask_base &= df_features['sexo'] == sexo_pred

    df_base = df_features[mask_base]

    if len(df_base) == 0:
        st.warning("No hay datos para esta combinacion. Selecciona otra provincia/area/sexo.")
    else:
        row_base = df_base.iloc[0]
        tasa_real = row_base.get('tasa_desempleo', None)

        st.markdown(f"**Datos base:** {prov_pred} - {ultimo_periodo} "
                    f"(area={area_pred}, sexo={sexo_pred})")
        if tasa_real is not None:
            st.markdown(f"**Tasa de desempleo real:** {tasa_real:.2f}%")

        st.markdown("---")
        st.markdown("#### Ajustar Variables")

        # Sliders para las variables mas importantes que el usuario puede modificar
        slider_vars = {
            'tasa_desempleo_lag1': ('Desempleo periodo anterior (%)', 0.0, 20.0),
            'tasa_participacion': ('Tasa de participacion (%)', 25.0, 95.0),
            'pct_subempleo': ('Subempleo (%)', 0.0, 45.0),
            'empleo_informal_pct': ('Empleo informal (%)', 30.0, 95.0),
            'pct_universitaria': ('Educacion universitaria (%)', 0.0, 50.0),
            'pct_empresa_grande': ('Empresa grande (%)', 25.0, 70.0),
            'mediana_salario': ('Mediana salarial (B/.)', 100.0, 900.0),
            'pib_crecimiento': ('Crecimiento PIB (%)', -20.0, 20.0),
        }

        valores_ajustados = {}
        cols = st.columns(2)
        for i, (var, (label, vmin, vmax)) in enumerate(slider_vars.items()):
            val_actual = float(row_base.get(var, (vmin + vmax) / 2))
            val_actual = max(vmin, min(vmax, val_actual))
            with cols[i % 2]:
                valores_ajustados[var] = st.slider(
                    label, vmin, vmax, val_actual, 0.1, key=f"slider_{var}"
                )

        # Construir vector de features
        X_pred = pd.DataFrame([row_base[feature_cols].values],
                              columns=feature_cols)
        for var, val in valores_ajustados.items():
            if var in X_pred.columns:
                X_pred[var] = val

        # Recalcular interacciones que dependen de los sliders
        if 'lag_x_post_covid' in X_pred.columns and 'post_covid' in X_pred.columns:
            X_pred['lag_x_post_covid'] = (
                X_pred['tasa_desempleo_lag1'] * X_pred['post_covid']
            )
        if 'lag_x_sexo_mujeres' in X_pred.columns and 'sexo_mujeres' in X_pred.columns:
            X_pred['lag_x_sexo_mujeres'] = (
                X_pred['tasa_desempleo_lag1'] * X_pred['sexo_mujeres']
            )
        if 'lag_x_area_urbana' in X_pred.columns and 'area_urbana' in X_pred.columns:
            X_pred['lag_x_area_urbana'] = (
                X_pred['tasa_desempleo_lag1'] * X_pred['area_urbana']
            )
        if 'pct_subempleo_lag1' in X_pred.columns:
            X_pred['pct_subempleo_lag1'] = valores_ajustados.get(
                'pct_subempleo', X_pred['pct_subempleo_lag1'].values[0]
            )

        # Imputar NaN y escalar
        X_pred = X_pred.fillna(0)
        X_scaled = scaler.transform(X_pred)

        # Predecir
        prediccion = float(modelo.predict(X_scaled)[0])
        riesgo = clasificar_riesgo(prediccion)

        # Mostrar resultado
        st.markdown("---")
        st.markdown("### Resultado de la Prediccion")

        col_res1, col_res2, col_res3 = st.columns(3)
        with col_res1:
            delta = None
            if tasa_real is not None:
                delta = f"{prediccion - tasa_real:+.2f} vs real"
            st.metric("Prediccion", f"{prediccion:.2f}%", delta=delta)
        with col_res2:
            color = COLOR_MAP.get(riesgo, '#999')
            st.markdown(
                f'<div style="background-color:{color};padding:15px;'
                f'border-radius:8px;text-align:center;color:white;'
                f'font-size:1.3em;margin-top:10px;">'
                f'<b>Riesgo: {riesgo.upper()}</b></div>',
                unsafe_allow_html=True
            )
        with col_res3:
            if tasa_real is not None:
                st.metric("Tasa Real", f"{tasa_real:.2f}%")

        # Tabla de valores usados
        with st.expander("Ver todos los valores de features usados"):
            df_show = pd.DataFrame({
                'Feature': feature_cols,
                'Valor': X_pred.values[0]
            }).sort_values('Feature')
            st.dataframe(df_show, use_container_width=True, hide_index=True)
