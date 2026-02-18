"""
Visualización geoespacial del desempleo regional en Panamá.
Genera mapas coropléticos estáticos (matplotlib) e interactivos (folium).
"""
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import folium
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    PROCESSED_DATA_DIR, SHAPEFILES_DIR, FIGURES_DIR, REPORTS_DIR,
    RIESGO_BAJO, RIESGO_CRITICO, COLOR_MAP
)

# Mapeo de nombres: CSV → Shapefile
# Solo las 3 comarcas difieren (prefijo "Comarca " + variante ortográfica)
NOMBRE_CSV_A_SHP = {
    'Comarca Ember\u00e1': 'Ember\u00e1',
    'Comarca Kuna Yala': 'Kuna Yala',
    'Comarca Ng\u00e4be Bugl\u00e9': 'Ng\u00f6be Bugl\u00e9',
}

# Inverso: Shapefile → CSV
NOMBRE_SHP_A_CSV = {v: k for k, v in NOMBRE_CSV_A_SHP.items()}


def cargar_shapefile():
    """Carga el shapefile de provincias de Panamá."""
    # Intentar geojson primero (más portable)
    geojson_path = SHAPEFILES_DIR / "HDX" / "geojson" / "pan_admin1.geojson"
    shp_path = SHAPEFILES_DIR / "HDX" / "shapefile" / "pan_admin1.shp"

    for path in [geojson_path, shp_path]:
        if path.exists():
            try:
                return gpd.read_file(str(path))
            except Exception as e:
                print(f"  AVISO: No se pudo leer {path.name}: {e}")
                continue

    raise FileNotFoundError(
        f"No se encontró shapefile ni geojson en {SHAPEFILES_DIR / 'HDX'}\n"
        f"  Verificar: geojson={geojson_path.exists()}, shp={shp_path.exists()}"
    )


def cargar_datos_desempleo():
    """Carga datos de desempleo procesados, filtrando area='total' y sexo='total'."""
    df = pd.read_csv(PROCESSED_DATA_DIR / "desempleo_por_provincia.csv")
    if 'area' in df.columns:
        df = df[df['area'] == 'total'].copy()
    if 'sexo' in df.columns:
        df = df[df['sexo'] == 'total'].copy()
    return df


def cargar_predicciones():
    """Carga predicciones del modelo, filtrando area='total' y sexo='total'."""
    path = REPORTS_DIR / "predicciones_modelo.csv"
    if path.exists():
        df = pd.read_csv(path)
        if 'area' in df.columns:
            df = df[df['area'] == 'total'].copy()
        if 'sexo' in df.columns:
            df = df[df['sexo'] == 'total'].copy()
        return df
    return None


def unir_datos_geo(gdf, df_datos, periodo=None):
    """Une datos de desempleo con geometrías del shapefile."""
    if periodo:
        df_datos = df_datos[df_datos['periodo'] == periodo].copy()

    # Crear columna de mapeo en los datos
    # Para las comarcas usamos el diccionario, para el resto el nombre es idéntico
    df_datos = df_datos.copy()
    df_datos['nombre_shp'] = df_datos['provincia'].map(
        lambda x: NOMBRE_CSV_A_SHP.get(x, x)
    )

    # Unir con shapefile
    gdf_merged = gdf.merge(
        df_datos,
        left_on='adm1_name',
        right_on='nombre_shp',
        how='left'
    )
    return gdf_merged


def clasificar_riesgo(tasa):
    """Clasifica nivel de riesgo."""
    if pd.isna(tasa):
        return 'sin datos'
    if tasa < RIESGO_BAJO:
        return 'bajo'
    elif tasa < RIESGO_CRITICO:
        return 'moderado'
    else:
        return 'critico'


# ========== MAPAS ESTÁTICOS (matplotlib) ==========

def mapa_desempleo_estatico(gdf, df_datos, periodo, output_path=None):
    """Mapa coroplético de tasa de desempleo por provincia."""
    gdf_m = unir_datos_geo(gdf, df_datos, periodo)

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # Colormap
    vmin = 0
    vmax = max(15, gdf_m['tasa_desempleo'].max() * 1.1)
    cmap = plt.cm.RdYlGn_r  # Rojo=alto, Verde=bajo

    gdf_m.plot(
        column='tasa_desempleo',
        ax=ax,
        legend=False,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolor='black',
        linewidth=0.8,
        missing_kwds={'color': 'lightgray', 'label': 'Sin datos'}
    )

    # Colorbar
    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('Tasa de Desempleo (%)', fontsize=12)

    # Etiquetas de provincias
    for _, row in gdf_m.iterrows():
        if pd.notna(row.get('tasa_desempleo')):
            centroid = row.geometry.centroid
            nombre_corto = row.get('provincia', row['adm1_name'])
            if nombre_corto and len(nombre_corto) > 15:
                nombre_corto = nombre_corto[:12] + '...'
            ax.annotate(
                f"{nombre_corto}\n{row['tasa_desempleo']:.1f}%",
                xy=(centroid.x, centroid.y),
                ha='center', va='center',
                fontsize=7, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          alpha=0.7, edgecolor='none')
            )

    ax.set_title(
        f'Tasa de Desempleo por Provincia - {periodo}',
        fontsize=16, fontweight='bold', pad=20
    )
    ax.set_axis_off()

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Guardado: {output_path}")
    plt.close(fig)
    return fig


def mapa_riesgo_estatico(gdf, df_datos, periodo, output_path=None):
    """Mapa de clasificación de riesgo por provincia."""
    gdf_m = unir_datos_geo(gdf, df_datos, periodo)
    gdf_m['riesgo'] = gdf_m['tasa_desempleo'].apply(clasificar_riesgo)

    # Asignar colores por riesgo
    color_riesgo = {
        'bajo': COLOR_MAP['bajo'],
        'moderado': COLOR_MAP['moderado'],
        'critico': COLOR_MAP['critico'],
        'sin datos': '#cccccc'
    }
    gdf_m['color'] = gdf_m['riesgo'].map(color_riesgo)

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    gdf_m.plot(
        ax=ax,
        color=gdf_m['color'],
        edgecolor='black',
        linewidth=0.8
    )

    # Etiquetas
    for _, row in gdf_m.iterrows():
        if pd.notna(row.get('tasa_desempleo')):
            centroid = row.geometry.centroid
            nombre = row.get('provincia', row['adm1_name'])
            if nombre and len(nombre) > 15:
                nombre = nombre[:12] + '...'
            ax.annotate(
                f"{nombre}\n{row['tasa_desempleo']:.1f}%",
                xy=(centroid.x, centroid.y),
                ha='center', va='center',
                fontsize=7, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          alpha=0.7, edgecolor='none')
            )

    # Leyenda
    legend_elements = [
        Patch(facecolor=COLOR_MAP['bajo'], edgecolor='black',
              label=f'Bajo (<{RIESGO_BAJO}%)'),
        Patch(facecolor=COLOR_MAP['moderado'], edgecolor='black',
              label=f'Moderado ({RIESGO_BAJO}-{RIESGO_CRITICO}%)'),
        Patch(facecolor=COLOR_MAP['critico'], edgecolor='black',
              label=f'Critico (>{RIESGO_CRITICO}%)'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=11,
              title='Nivel de Riesgo', title_fontsize=12)

    ax.set_title(
        f'Clasificacion de Riesgo de Desempleo - {periodo}',
        fontsize=16, fontweight='bold', pad=20
    )
    ax.set_axis_off()

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Guardado: {output_path}")
    plt.close(fig)
    return fig


def mapa_comparativo_estatico(gdf, df_datos, periodo1, periodo2, output_path=None):
    """Mapa comparativo de dos periodos lado a lado."""
    gdf_m1 = unir_datos_geo(gdf, df_datos, periodo1)
    gdf_m2 = unir_datos_geo(gdf, df_datos, periodo2)

    vmin = 0
    vmax = max(
        15,
        gdf_m1['tasa_desempleo'].max() * 1.1,
        gdf_m2['tasa_desempleo'].max() * 1.1
    )
    cmap = plt.cm.RdYlGn_r

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))

    for ax, gdf_m, periodo in [(ax1, gdf_m1, periodo1), (ax2, gdf_m2, periodo2)]:
        gdf_m.plot(
            column='tasa_desempleo', ax=ax, cmap=cmap,
            vmin=vmin, vmax=vmax, edgecolor='black', linewidth=0.8,
            missing_kwds={'color': 'lightgray'}
        )
        for _, row in gdf_m.iterrows():
            if pd.notna(row.get('tasa_desempleo')):
                centroid = row.geometry.centroid
                ax.annotate(
                    f"{row['tasa_desempleo']:.1f}%",
                    xy=(centroid.x, centroid.y),
                    ha='center', va='center', fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              alpha=0.7, edgecolor='none')
                )
        ax.set_title(periodo, fontsize=14, fontweight='bold')
        ax.set_axis_off()

    # Colorbar compartida
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax1, ax2], shrink=0.6, pad=0.02)
    cbar.set_label('Tasa de Desempleo (%)', fontsize=12)

    fig.suptitle(
        'Evolucion del Desempleo por Provincia',
        fontsize=18, fontweight='bold', y=0.98
    )

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Guardado: {output_path}")
    plt.close(fig)
    return fig


# ========== MAPAS INTERACTIVOS (folium) ==========

def mapa_interactivo(gdf, df_datos, periodo, df_pred=None, output_path=None):
    """Mapa interactivo con folium para Streamlit.
    Incluye datos reales y predicciones del modelo en el tooltip.
    """
    gdf_m = unir_datos_geo(gdf, df_datos, periodo)
    gdf_m['riesgo'] = gdf_m['tasa_desempleo'].apply(clasificar_riesgo)

    # Integrar predicciones si están disponibles
    if df_pred is not None:
        df_pred_p = df_pred[df_pred['periodo'] == periodo].copy()
        df_pred_p['nombre_shp'] = df_pred_p['provincia'].map(
            lambda x: NOMBRE_CSV_A_SHP.get(x, x)
        )
        pred_map = df_pred_p.set_index('nombre_shp')['prediccion'].to_dict()
        riesgo_pred_map = {
            k: clasificar_riesgo(v) for k, v in pred_map.items()
        }
        gdf_m['prediccion'] = gdf_m['adm1_name'].map(pred_map)
        gdf_m['riesgo_predicho'] = gdf_m['adm1_name'].map(riesgo_pred_map)
    else:
        gdf_m['prediccion'] = np.nan
        gdf_m['riesgo_predicho'] = 'sin modelo'

    # Centro de Panamá
    m = folium.Map(location=[8.5, -80.0], zoom_start=7, tiles='cartodbpositron')

    # Estilo por riesgo
    def estilo(feature):
        riesgo = feature['properties'].get('riesgo', 'sin datos')
        color = {
            'bajo': COLOR_MAP['bajo'],
            'moderado': COLOR_MAP['moderado'],
            'critico': COLOR_MAP['critico'],
            'sin datos': '#cccccc'
        }.get(riesgo, '#cccccc')
        return {
            'fillColor': color,
            'color': 'black',
            'weight': 1.5,
            'fillOpacity': 0.7
        }

    # Preparar columnas para export
    export_cols = ['geometry', 'adm1_name', 'provincia',
                   'tasa_desempleo', 'riesgo',
                   'prediccion', 'riesgo_predicho']
    gdf_export = gdf_m[export_cols].copy()
    gdf_export['provincia'] = gdf_export['provincia'].fillna(
        gdf_export['adm1_name'].map(lambda x: NOMBRE_SHP_A_CSV.get(x, x))
    )
    gdf_export['tasa_desempleo'] = gdf_export['tasa_desempleo'].round(2).fillna(0)
    gdf_export['prediccion'] = gdf_export['prediccion'].round(2).fillna(0)
    gdf_export['riesgo'] = gdf_export['riesgo'].fillna('sin datos')
    gdf_export['riesgo_predicho'] = gdf_export['riesgo_predicho'].fillna('sin modelo')

    # Tooltip con datos reales + predicciones
    tooltip_fields = ['provincia', 'tasa_desempleo', 'riesgo',
                      'prediccion', 'riesgo_predicho']
    tooltip_aliases = ['Provincia:', 'Desempleo Real (%):', 'Riesgo Real:',
                       'Prediccion (%):', 'Riesgo Predicho:']

    geojson = folium.GeoJson(
        gdf_export.to_json(),
        style_function=estilo,
        tooltip=folium.GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=tooltip_aliases,
            localize=True,
            sticky=True,
            labels=True,
            style="""
                background-color: white;
                border: 2px solid black;
                border-radius: 3px;
                box-shadow: 3px;
                font-size: 13px;
                padding: 10px;
            """
        ),
        highlight_function=lambda x: {
            'weight': 3,
            'color': '#666',
            'fillOpacity': 0.9
        }
    )
    geojson.add_to(m)

    # Leyenda
    legend_html = f"""
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000;
                background-color: white; padding: 15px; border-radius: 5px;
                border: 2px solid grey; font-size: 14px;">
        <b>Nivel de Riesgo - {periodo}</b><br>
        <i style="background:{COLOR_MAP['bajo']};width:18px;height:18px;
           display:inline-block;margin-right:5px;border:1px solid black;"></i>
        Bajo (&lt;{RIESGO_BAJO}%)<br>
        <i style="background:{COLOR_MAP['moderado']};width:18px;height:18px;
           display:inline-block;margin-right:5px;border:1px solid black;"></i>
        Moderado ({RIESGO_BAJO}-{RIESGO_CRITICO}%)<br>
        <i style="background:{COLOR_MAP['critico']};width:18px;height:18px;
           display:inline-block;margin-right:5px;border:1px solid black;"></i>
        Critico (&gt;{RIESGO_CRITICO}%)<br>
        <br><i>Color = riesgo real</i><br>
        <i>Hover para ver prediccion</i>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    if output_path:
        m.save(str(output_path))
        print(f"  Guardado: {output_path}")

    return m


# ========== MAIN ==========

def main():
    print("=" * 60)
    print("GENERACION DE MAPAS")
    print("=" * 60)

    # Crear directorio de figuras
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Cargar datos
    print("\n1. Cargando datos...")
    gdf = cargar_shapefile()
    df = cargar_datos_desempleo()
    df_pred = cargar_predicciones()
    print(f"   Shapefile: {len(gdf)} provincias")
    print(f"   Datos: {len(df)} filas")
    periodos = sorted(df['periodo'].unique())
    print(f"   Periodos: {periodos}")

    ultimo = periodos[-1]
    primero_completo = '2018-08'  # Primer periodo con datos completos

    # 2. Mapa de desempleo - último periodo
    print(f"\n2. Mapa de desempleo ({ultimo})...")
    mapa_desempleo_estatico(
        gdf, df, ultimo,
        FIGURES_DIR / f"mapa_desempleo_{ultimo.replace('-', '_')}.png"
    )

    # 3. Mapa de riesgo - último periodo
    print(f"\n3. Mapa de riesgo ({ultimo})...")
    mapa_riesgo_estatico(
        gdf, df, ultimo,
        FIGURES_DIR / f"mapa_riesgo_{ultimo.replace('-', '_')}.png"
    )

    # 4. Mapa comparativo
    print(f"\n4. Mapa comparativo ({primero_completo} vs {ultimo})...")
    mapa_comparativo_estatico(
        gdf, df, primero_completo, ultimo,
        FIGURES_DIR / f"mapa_comparativo_{primero_completo.replace('-', '_')}_vs_{ultimo.replace('-', '_')}.png"
    )

    # 5. Mapa interactivo (con predicciones si están disponibles)
    print(f"\n5. Mapa interactivo ({ultimo})...")
    mapa_interactivo(
        gdf, df, ultimo, df_pred=df_pred,
        output_path=FIGURES_DIR / f"mapa_interactivo_{ultimo.replace('-', '_')}.html"
    )

    # 6. Mapas de riesgo para todos los periodos
    print("\n6. Mapas de riesgo por periodo...")
    for periodo in periodos:
        mapa_riesgo_estatico(
            gdf, df, periodo,
            FIGURES_DIR / f"mapa_riesgo_{periodo.replace('-', '_')}.png"
        )

    # 7. Mapa de predicciones vs real (si hay modelo)
    if df_pred is not None:
        print(f"\n7. Mapa predicciones vs real ({ultimo})...")
        df_pred_ultimo = df_pred[df_pred['periodo'] == ultimo].copy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))

        gdf_real = unir_datos_geo(gdf, df, ultimo)
        gdf_pred_m = gdf.merge(
            df_pred_ultimo.assign(
                nombre_shp=df_pred_ultimo['provincia'].map(
                    lambda x: NOMBRE_CSV_A_SHP.get(x, x)
                )
            ),
            left_on='adm1_name', right_on='nombre_shp', how='left'
        )

        vmin, vmax = 0, 15
        cmap = plt.cm.RdYlGn_r

        for ax, gdf_m, titulo, col in [
            (ax1, gdf_real, f'Real - {ultimo}', 'tasa_desempleo'),
            (ax2, gdf_pred_m, f'Prediccion - {ultimo}', 'prediccion')
        ]:
            gdf_m.plot(
                column=col, ax=ax, cmap=cmap,
                vmin=vmin, vmax=vmax, edgecolor='black', linewidth=0.8,
                missing_kwds={'color': 'lightgray'}
            )
            for _, row in gdf_m.iterrows():
                val = row.get(col)
                if pd.notna(val):
                    centroid = row.geometry.centroid
                    ax.annotate(
                        f"{val:.1f}%",
                        xy=(centroid.x, centroid.y),
                        ha='center', va='center', fontsize=8,
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2',
                                  facecolor='white', alpha=0.7,
                                  edgecolor='none')
                    )
            ax.set_title(titulo, fontsize=14, fontweight='bold')
            ax.set_axis_off()

        sm = plt.cm.ScalarMappable(
            cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax)
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=[ax1, ax2], shrink=0.6, pad=0.02)
        cbar.set_label('Tasa de Desempleo (%)', fontsize=12)

        fig.suptitle(
            'Desempleo Real vs Prediccion del Modelo',
            fontsize=18, fontweight='bold', y=0.98
        )
        plt.tight_layout()
        path_pred = FIGURES_DIR / f"mapa_real_vs_prediccion_{ultimo.replace('-', '_')}.png"
        fig.savefig(path_pred, dpi=150, bbox_inches='tight')
        print(f"  Guardado: {path_pred}")
        plt.close(fig)

        # 8. Mapa de riesgo basado en predicciones
        print(f"\n8. Mapa riesgo predicho vs real ({ultimo})...")
        gdf_riesgo_real = unir_datos_geo(gdf, df, ultimo)
        gdf_riesgo_real['riesgo'] = gdf_riesgo_real['tasa_desempleo'].apply(
            clasificar_riesgo
        )
        gdf_riesgo_real['color'] = gdf_riesgo_real['riesgo'].map(
            {**{k: v for k, v in COLOR_MAP.items()}, 'sin datos': '#cccccc'}
        )

        gdf_riesgo_pred = gdf.merge(
            df_pred_ultimo.assign(
                nombre_shp=df_pred_ultimo['provincia'].map(
                    lambda x: NOMBRE_CSV_A_SHP.get(x, x)
                )
            ),
            left_on='adm1_name', right_on='nombre_shp', how='left'
        )
        gdf_riesgo_pred['riesgo'] = gdf_riesgo_pred['prediccion'].apply(
            clasificar_riesgo
        )
        gdf_riesgo_pred['color'] = gdf_riesgo_pred['riesgo'].map(
            {**{k: v for k, v in COLOR_MAP.items()}, 'sin datos': '#cccccc'}
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))

        for ax, gdf_m, titulo, col_tasa in [
            (ax1, gdf_riesgo_real, f'Riesgo Real - {ultimo}', 'tasa_desempleo'),
            (ax2, gdf_riesgo_pred, f'Riesgo Predicho - {ultimo}', 'prediccion')
        ]:
            gdf_m.plot(
                ax=ax, color=gdf_m['color'],
                edgecolor='black', linewidth=0.8
            )
            for _, row in gdf_m.iterrows():
                val = row.get(col_tasa)
                if pd.notna(val):
                    centroid = row.geometry.centroid
                    nombre = row.get('provincia', row['adm1_name'])
                    if nombre and len(nombre) > 15:
                        nombre = nombre[:12] + '...'
                    ax.annotate(
                        f"{nombre}\n{val:.1f}%",
                        xy=(centroid.x, centroid.y),
                        ha='center', va='center', fontsize=7,
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2',
                                  facecolor='white', alpha=0.7,
                                  edgecolor='none')
                    )
            ax.set_title(titulo, fontsize=14, fontweight='bold')
            ax.set_axis_off()

        legend_elements = [
            Patch(facecolor=COLOR_MAP['bajo'], edgecolor='black',
                  label=f'Bajo (<{RIESGO_BAJO}%)'),
            Patch(facecolor=COLOR_MAP['moderado'], edgecolor='black',
                  label=f'Moderado ({RIESGO_BAJO}-{RIESGO_CRITICO}%)'),
            Patch(facecolor=COLOR_MAP['critico'], edgecolor='black',
                  label=f'Critico (>{RIESGO_CRITICO}%)'),
        ]
        ax1.legend(handles=legend_elements, loc='lower left', fontsize=10,
                   title='Nivel de Riesgo', title_fontsize=11)

        fig.suptitle(
            'Clasificacion de Riesgo: Real vs Prediccion del Modelo',
            fontsize=18, fontweight='bold', y=0.98
        )
        plt.tight_layout()
        path_riesgo_pred = FIGURES_DIR / f"mapa_riesgo_predicho_{ultimo.replace('-', '_')}.png"
        fig.savefig(path_riesgo_pred, dpi=150, bbox_inches='tight')
        print(f"  Guardado: {path_riesgo_pred}")
        plt.close(fig)

    # Resumen
    print("\n" + "=" * 60)
    print("MAPAS GENERADOS")
    print("=" * 60)
    pngs = list(FIGURES_DIR.glob("mapa_*.png"))
    htmls = list(FIGURES_DIR.glob("mapa_*.html"))
    print(f"  Estaticos (PNG): {len(pngs)}")
    for p in sorted(pngs):
        print(f"    - {p.name}")
    print(f"  Interactivos (HTML): {len(htmls)}")
    for h in sorted(htmls):
        print(f"    - {h.name}")


if __name__ == "__main__":
    main()
