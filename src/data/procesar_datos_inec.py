"""
Script de procesamiento y consolidación de datos INEC.

Extrae datos clave de los Excel del INEC (Cuadros 2, 4, 6, 13, 16, 19, 22,
25, 39, 1A) de todos los periodos (2018-2024), integra indicadores del Banco
Mundial, y consolida en un dataset unificado.

Salida:
    - data/processed/desempleo_por_provincia.csv
    - data/processed/serie_historica_nacional.csv
"""

import os
import sys
import re
import warnings

import numpy as np
import pandas as pd

# Agregar raíz del proyecto al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, PROVINCIAS

warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

# ---------------------------------------------------------------------------
# Provincias y comarcas indígenas tal como aparecen en los Excel del INEC
# ---------------------------------------------------------------------------
COMARCAS = [
    "Comarca Kuna Yala",
    "Comarca Emberá",
    "Comarca Ngäbe Buglé",
]

# Nombres canónicos de salida
NOMBRES_CANONICOS = {
    "Bocas del Toro": "Bocas del Toro",
    "Coclé": "Coclé",
    "Colón": "Colón",
    "Chiriquí": "Chiriquí",
    "Darién": "Darién",
    "Herrera": "Herrera",
    "Los Santos": "Los Santos",
    "Panamá": "Panamá",
    "Panamá Oeste": "Panamá Oeste",
    "Veraguas": "Veraguas",
    "Comarca Kuna Yala": "Comarca Kuna Yala",
    "Comarca Emberá": "Comarca Emberá",
    "Comarca Ngäbe Buglé": "Comarca Ngäbe Buglé",
}

# Patrones para encontrar nombres de provincia/comarca en texto con encoding variado
PATRONES_PROVINCIA = [
    ("Bocas del Toro", "Bocas del Toro"),
    ("Coclé", "Coclé"),
    ("Cocl", "Coclé"),
    ("Colón", "Colón"),
    ("Col", "Colón"),           # por encoding
    ("Chiriquí", "Chiriquí"),
    ("Chiriqu", "Chiriquí"),
    ("Darién", "Darién"),
    ("Dari", "Darién"),
    ("Herrera", "Herrera"),
    ("Los Santos", "Los Santos"),
    ("Panamá Oeste", "Panamá Oeste"),   # debe ir ANTES de "Panamá"
    ("Panama Oeste", "Panamá Oeste"),
    ("Panamá", "Panamá"),
    ("Panama", "Panamá"),
    ("Veraguas", "Veraguas"),
    ("Comarca Kuna Yala", "Comarca Kuna Yala"),
    ("Kuna Yala", "Comarca Kuna Yala"),
    ("Com. Kuna Yala", "Comarca Kuna Yala"),
    ("Comarca Emberá", "Comarca Emberá"),
    ("Ember", "Comarca Emberá"),
    ("Com. Ember", "Comarca Emberá"),
    ("Comarca Ngäbe", "Comarca Ngäbe Buglé"),
    ("Ng", "Comarca Ngäbe Buglé"),      # Ngäbe con encoding variado
    ("Com. Ng", "Comarca Ngäbe Buglé"),
]

# Definición de periodos a procesar
PERIODOS = {
    # ===================================================================
    # Periodos antiguos 2011-2014
    # ===================================================================
    "inec_agosto_2012": {
        "carpeta": "inec_agosto_2012",
        "periodos_cuadro2": [
            {"periodo": "2011-08", "anio": 2011, "cols": (1, 2, 3)},
            {"periodo": "2012-08", "anio": 2012, "cols": (4, 5, 6)},
        ],
        "formato_cuadro2": "expanded",
        "archivo_cuadro2": "02",
        "periodo_actual": {"periodo": "2012-08", "anio": 2012},
        "tiene_cuadros_completos": False,
    },
    "inec_agosto_2013": {
        "carpeta": "inec_agosto_2013",
        "periodos_cuadro2": [
            {"periodo": "2012-08", "anio": 2012, "cols": (1, 2, 3)},
            {"periodo": "2013-08", "anio": 2013, "cols": (4, 5, 6)},
        ],
        "formato_cuadro2": "expanded",
        "archivo_cuadro2": "02",
        "periodo_actual": {"periodo": "2013-08", "anio": 2013},
        "tiene_cuadros_completos": False,
    },
    "inec_agosto_2014": {
        "carpeta": "inec_agosto_2014",
        "periodos_cuadro2": [
            {"periodo": "2013-08", "anio": 2013, "cols": (1, 2, 3)},
            {"periodo": "2014-08", "anio": 2014, "cols": (4, 5, 6)},
        ],
        "formato_cuadro2": "expanded",
        "archivo_cuadro2": "02",
        "periodo_actual": {"periodo": "2014-08", "anio": 2014},
        "tiene_cuadros_completos": False,
    },
    # ===================================================================
    # Periodos antiguos 2015-2017
    # ===================================================================
    "inec_marzo_2015": {
        "carpeta": "inec_marzo_2015",
        "periodos_cuadro2": [
            {"periodo": "2014-03", "anio": 2014, "cols": (1, 2, 3)},
            {"periodo": "2015-03", "anio": 2015, "cols": (4, 5, 6)},
        ],
        "formato_cuadro2": "expanded",
        # Mapeo numérico: Cuadro -> sufijo de archivo
        "archivo_cuadro2": "01",
        "periodo_actual": {"periodo": "2015-03", "anio": 2015},
        "tiene_cuadros_completos": False,  # Solo Cuadro 2
    },
    "inec_agosto_2015": {
        "carpeta": "inec_agosto_2015",
        "periodos_cuadro2": [
            {"periodo": "2014-08", "anio": 2014, "cols": (1, 2, 3)},
            {"periodo": "2015-08", "anio": 2015, "cols": (4, 5, 6)},
        ],
        "formato_cuadro2": "expanded",
        "archivo_cuadro2": "02",
        "periodo_actual": {"periodo": "2015-08", "anio": 2015},
        "tiene_cuadros_completos": False,
    },
    "inec_marzo_2016": {
        "carpeta": "inec_marzo_2016",
        "periodos_cuadro2": [
            {"periodo": "2015-03", "anio": 2015, "cols": (1, 2, 3)},
            {"periodo": "2016-03", "anio": 2016, "cols": (4, 5, 6)},
        ],
        "formato_cuadro2": "expanded",
        "archivo_cuadro2": "01",
        "periodo_actual": {"periodo": "2016-03", "anio": 2016},
        "tiene_cuadros_completos": False,
    },
    "inec_agosto_2016": {
        "carpeta": "inec_agosto_2016",
        "periodos_cuadro2": [
            {"periodo": "2015-08", "anio": 2015, "cols": (1, 2, 3)},
            {"periodo": "2016-08", "anio": 2016, "cols": (4, 5, 6)},
        ],
        "formato_cuadro2": "expanded",
        "archivo_cuadro2": "02",
        "periodo_actual": {"periodo": "2016-08", "anio": 2016},
        "tiene_cuadros_completos": False,
    },
    "inec_marzo_2017": {
        "carpeta": "inec_marzo_2017",
        "periodos_cuadro2": [
            {"periodo": "2016-03", "anio": 2016, "cols": (1, 2, 3)},
            {"periodo": "2017-03", "anio": 2017, "cols": (4, 5, 6)},
        ],
        "formato_cuadro2": "format_b",
        "archivo_cuadro2": "02",
        "periodo_actual": {"periodo": "2017-03", "anio": 2017},
        "tiene_cuadros_completos": False,
    },
    "inec_agosto_2017": {
        "carpeta": "inec_agosto_2017",
        "periodos_cuadro2": [
            {"periodo": "2016-08", "anio": 2016, "cols": (1, 2, 3)},
            {"periodo": "2017-08", "anio": 2017, "cols": (4, 5, 6)},
        ],
        "formato_cuadro2": "compact",
        "periodo_actual": {"periodo": "2017-08", "anio": 2017},
        "tiene_cuadros_completos": False,
    },
    # ===================================================================
    # Periodos originales 2018-2024
    # ===================================================================
    "inec_agosto_2018": {
        "carpeta": "inec_agosto_2018",
        "periodos_cuadro2": [
            {"periodo": "2017-08", "anio": 2017, "cols": (1, 2, 3)},
            {"periodo": "2018-08", "anio": 2018, "cols": (4, 5, 6)},
        ],
        "formato_cuadro2": "expanded",
        "periodo_cuadro6": {"periodo": "2018-08", "anio": 2018},
        "periodos_cuadro39": [
            {"periodo": "2017-08", "anio": 2017, "col_pob_ocup": 2, "col_informal_pct": 7},
            {"periodo": "2018-08", "anio": 2018, "col_pob_ocup": 8, "col_informal_pct": 13},
        ],
        # Cuadros de 1 solo periodo (el "actual" del archivo)
        "periodo_actual": {"periodo": "2018-08", "anio": 2018},
        # Cuadro 13 tiene 2 periodos
        "periodos_cuadro13": [
            {"periodo": "2017-08", "anio": 2017, "col_num": 3, "col_pct": 4},
            {"periodo": "2018-08", "anio": 2018, "col_num": 5, "col_pct": 6},
        ],
        "tiene_cuadros_completos": True,
    },
    "inec_agosto_2019": {
        "carpeta": "inec_agosto_2019",
        "periodos_cuadro2": [
            {"periodo": "2018-08", "anio": 2018, "cols": (1, 2, 3)},
            {"periodo": "2019-08", "anio": 2019, "cols": (4, 5, 6)},
        ],
        "formato_cuadro2": "expanded",
        "periodo_cuadro6": {"periodo": "2019-08", "anio": 2019},
        "periodos_cuadro39": [
            {"periodo": "2018-08", "anio": 2018, "col_pob_ocup": 2, "col_informal_pct": 7},
            {"periodo": "2019-08", "anio": 2019, "col_pob_ocup": 8, "col_informal_pct": 13},
        ],
        "periodo_actual": {"periodo": "2019-08", "anio": 2019},
        "periodos_cuadro13": [
            {"periodo": "2018-08", "anio": 2018, "col_num": 3, "col_pct": 4},
            {"periodo": "2019-08", "anio": 2019, "col_num": 5, "col_pct": 6},
        ],
        "tiene_cuadros_completos": True,
    },
    "inec_septiembre_2020": {
        "carpeta": "inec_septiembre_2020",
        "tiene_cuadros_completos": False,
    },
    "inec_octubre_2021": {
        "carpeta": "inec_octubre_2021",
        "periodos_cuadro2": [
            {"periodo": "2019-08", "anio": 2019, "cols": (1, 2, 3)},
            {"periodo": "2021-10", "anio": 2021, "cols": (4, 5, 6)},
        ],
        "formato_cuadro2": "expanded",
        "periodo_cuadro6": {"periodo": "2021-10", "anio": 2021},
        "periodos_cuadro39": [
            {"periodo": "2019-08", "anio": 2019, "col_pob_ocup": 2, "col_informal_pct": 7},
            {"periodo": "2021-10", "anio": 2021, "col_pob_ocup": 8, "col_informal_pct": 13},
        ],
        "periodo_actual": {"periodo": "2021-10", "anio": 2021},
        "periodos_cuadro13": [
            {"periodo": "2019-08", "anio": 2019, "col_num": 3, "col_pct": 4},
            {"periodo": "2021-10", "anio": 2021, "col_num": 5, "col_pct": 6},
        ],
        "tiene_cuadros_completos": True,
    },
    "inec_agosto_2023": {
        "carpeta": "inec_agosto_2023",
        "periodos_cuadro2": [
            {"periodo": "2022-04", "anio": 2022, "cols": (1, 2, 3)},
            {"periodo": "2023-08", "anio": 2023, "cols": (4, 5, 6)},
        ],
        "formato_cuadro2": "expanded",
        "periodo_cuadro6": {"periodo": "2023-08", "anio": 2023},
        "periodos_cuadro39": [
            {"periodo": "2022-04", "anio": 2022, "col_pob_ocup": 2, "col_informal_pct": 7},
            {"periodo": "2023-08", "anio": 2023, "col_pob_ocup": 8, "col_informal_pct": 13},
        ],
        "periodo_actual": {"periodo": "2023-08", "anio": 2023},
        "periodos_cuadro13": [
            {"periodo": "2022-04", "anio": 2022, "col_num": 3, "col_pct": 4},
            {"periodo": "2023-08", "anio": 2023, "col_num": 5, "col_pct": 6},
        ],
        "tiene_cuadros_completos": True,
    },
    "inec_octubre_2024": {
        "carpeta": "inec_octubre_2024",
        "periodos_cuadro2": [
            {"periodo": "2023-08", "anio": 2023, "cols": (1, 2, 3)},
            {"periodo": "2024-10", "anio": 2024, "cols": (4, 5, 6)},
        ],
        "formato_cuadro2": "expanded",
        "periodo_cuadro6": {"periodo": "2024-10", "anio": 2024},
        "periodos_cuadro39": [
            {"periodo": "2023-08", "anio": 2023, "col_pob_ocup": 2, "col_informal_pct": 7},
            {"periodo": "2024-10", "anio": 2024, "col_pob_ocup": 8, "col_informal_pct": 13},
        ],
        "periodo_actual": {"periodo": "2024-10", "anio": 2024},
        "periodos_cuadro13": [
            {"periodo": "2023-08", "anio": 2023, "col_num": 3, "col_pct": 4},
            {"periodo": "2024-10", "anio": 2024, "col_num": 5, "col_pct": 6},
        ],
        "tiene_cuadros_completos": True,
    },
}


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def encontrar_archivo(carpeta, cuadro, numero_archivo=None):
    """Busca el archivo Excel que contiene 'Cuadro X' en el nombre.

    Para archivos antiguos (2015-2016) que usan nombres numéricos como
    P{ID}1441-{NN}.xls, se puede pasar ``numero_archivo`` (ej. '02') para
    buscar por sufijo.
    """
    ruta_carpeta = RAW_DATA_DIR / carpeta
    if not ruta_carpeta.exists():
        print(f"  [WARN] Carpeta no encontrada: {ruta_carpeta}")
        return None

    extensiones = ('.xlsx', '.XLSX', '.xls', '.XLS')

    # --- Búsqueda por nombre "Cuadro X" ---
    patron = f"Cuadro {cuadro}."
    for f in os.listdir(ruta_carpeta):
        if patron in f and f.endswith(extensiones):
            return ruta_carpeta / f

    # Intento secundario con espacio extra
    patron2 = f"Cuadro {cuadro} ."
    for f in os.listdir(ruta_carpeta):
        if patron2 in f and f.endswith(extensiones):
            return ruta_carpeta / f

    # Intento con case-insensitive
    patron_lower = patron.lower()
    for f in os.listdir(ruta_carpeta):
        if patron_lower in f.lower() and f.endswith(extensiones):
            return ruta_carpeta / f

    # --- Búsqueda por número de archivo (archivos antiguos P{ID}1441-{NN}.xls) ---
    if numero_archivo is not None:
        sufijo = f"-{numero_archivo}."
        for f in os.listdir(ruta_carpeta):
            if sufijo in f and f.endswith(extensiones):
                return ruta_carpeta / f

    return None


def _limpiar_texto(val):
    """Limpia puntos suspensivos, espacios extra, y caracteres decorativos."""
    if pd.isna(val):
        return ""
    s = str(val).strip()
    # Eliminar puntos suspensivos y puntos decorativos al final
    s = re.sub(r'[\.…·]+\s*$', '', s)
    # Eliminar puntos decorativos intermedios (secuencias de 3+ puntos)
    s = re.sub(r'\.{3,}', '', s)
    return s.strip()


def _identificar_provincia(texto):
    """Identifica el nombre canónico de provincia/comarca dado un texto de celda.

    Requiere que el texto limpio sea corto (< 40 chars) para evitar falsos
    positivos con frases como 'República de Panamá' en encabezados.
    """
    texto_limpio = _limpiar_texto(texto)
    if not texto_limpio:
        return None

    # Descartar textos largos (encabezados, títulos, notas al pie)
    if len(texto_limpio) > 40:
        return None

    # Buscar de más específico a más general
    # Primero "Panamá Oeste" antes de "Panamá"
    for patron, nombre_canonico in PATRONES_PROVINCIA:
        if patron.lower() in texto_limpio.lower():
            return nombre_canonico
    return None


def _safe_float(val):
    """Convierte un valor a float de forma segura."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().replace(',', '').replace('..', '')
    if not s or s == '-':
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _es_fila_provincia_valida(texto):
    """Verifica que el texto no sea una subsección (Urbana, Rural, TOTAL, etc.)."""
    t = _limpiar_texto(texto).lower()
    if t in ('urbana', 'rural', 'hombres', 'mujeres'):
        return False
    if 'total' in t:
        return False
    if 'no ind' in t:
        return False
    if re.match(r'\d', t):
        return False
    return True


# ---------------------------------------------------------------------------
# Cuadro 2 — Datos principales por provincia
# ---------------------------------------------------------------------------

def _extraer_bloque_c2(df, fila_inicio, col_total, offsets):
    """Extrae un bloque de datos del Cuadro 2 según offsets dados.

    Returns:
        dict con pob_15_mas, pea, ocupados, desocupados, tasa_desempleo,
        tasa_participacion.
    """
    n = len(df)

    def _val(offset):
        idx = fila_inicio + offset
        return _safe_float(df.iloc[idx, col_total]) if idx < n else np.nan

    pob_15_mas = _val(offsets['pob_15_mas'])
    pea = _val(offsets['pea'])
    ocupados = _val(offsets['ocupados'])
    desocupados = _val(offsets['desocupados'])
    tasa_desempleo = _val(offsets['tasa_desempleo'])

    tasa_participacion = np.nan
    if pob_15_mas and pob_15_mas > 0 and not np.isnan(pea):
        tasa_participacion = (pea / pob_15_mas) * 100

    return {
        "poblacion_15_mas": pob_15_mas,
        "pea": pea,
        "ocupados": ocupados,
        "desocupados": desocupados,
        "tasa_desempleo": tasa_desempleo,
        "tasa_participacion": tasa_participacion,
    }


# Offsets dentro de cada sub-bloque según formato
OFFSETS_EXPANDED = {  # 2015-2016, 2018+ (22 filas por sub-bloque)
    'pob_15_mas': 0, 'pea': 2, 'ocupados': 7, 'desocupados': 8,
    'tasa_desempleo': 11,
}
OFFSETS_COMPACT = {  # Agosto 2017 (10 filas por sub-bloque)
    'pob_15_mas': 0, 'pea': 1, 'ocupados': 3, 'desocupados': 4,
    'tasa_desempleo': 5,
}
OFFSETS_FORMAT_B = {  # Marzo 2017 (54 filas, sin urbano/rural por provincia)
    'pob_15_mas': 0, 'pea': 2, 'ocupados': 7, 'desocupados': 12,
    'tasa_desempleo': 15,
}


def extraer_cuadro2(filepath, periodos_info, formato='expanded'):
    """
    Parsea Cuadro 2 y retorna DataFrame con datos por provincia para
    ambos periodos contenidos en el archivo.

    Extrae tres filas por provincia (total/urbana/rural) para formatos
    que lo soporten, y una sola fila (total) para comarcas y formato B.

    Args:
        filepath: Ruta al archivo Excel.
        periodos_info: Lista de dicts con 'periodo', 'anio', 'cols'.
        formato: 'expanded' (2015-2016, 2018+), 'compact' (ago-2017),
                 'format_b' (mar-2017, sin urbano/rural por provincia).
    """
    df = pd.read_excel(filepath, header=None)

    # Eliminar filas de salto de página "(Continuación)" que rompen offsets fijos
    mask_continuacion = df[0].apply(
        lambda v: 'ontinuaci' in str(v).lower() if pd.notna(v) else False
    )
    if mask_continuacion.any():
        n_removed = mask_continuacion.sum()
        df = df[~mask_continuacion].reset_index(drop=True)
        print(f"    [INFO] Eliminadas {n_removed} filas de salto de página")

    resultados = []

    if formato == 'expanded':
        offsets = OFFSETS_EXPANDED
        sub_bloque = 22
        bloque_provincia = 66   # 3 x 22 (total + urbana + rural)
        bloque_comarca = 22     # solo total
        tiene_urbano_rural = True
    elif formato == 'compact':
        offsets = OFFSETS_COMPACT
        sub_bloque = 10
        bloque_provincia = 30   # 3 x 10
        bloque_comarca = 10
        tiene_urbano_rural = True
    else:  # format_b
        offsets = OFFSETS_FORMAT_B
        sub_bloque = 54
        bloque_provincia = 54   # sin urbano/rural
        bloque_comarca = 54
        tiene_urbano_rural = False

    i = 0
    while i < len(df):
        val = df.iloc[i, 0]
        provincia = _identificar_provincia(val)

        if provincia is None:
            i += 1
            continue

        if pd.isna(df.iloc[i, 1]):
            i += 1
            continue

        if not _es_fila_provincia_valida(val):
            i += 1
            continue

        es_comarca = provincia in COMARCAS

        for p_info in periodos_info:
            col_total = p_info["cols"][0]
            col_hombres = p_info["cols"][1]
            col_mujeres = p_info["cols"][2]

            # Lista de (fila_inicio, area) a procesar
            bloques_area = [("total", i)]
            if tiene_urbano_rural and not es_comarca:
                bloques_area.append(("urbana", i + sub_bloque))
                bloques_area.append(("rural", i + 2 * sub_bloque))

            for area, fila_area in bloques_area:
                # Extraer total, hombres, mujeres para esta area
                for sexo, col in [("total", col_total),
                                  ("hombres", col_hombres),
                                  ("mujeres", col_mujeres)]:
                    datos = _extraer_bloque_c2(df, fila_area, col, offsets)
                    resultados.append({
                        "provincia": provincia,
                        "periodo": p_info["periodo"],
                        "anio": p_info["anio"],
                        "area": area,
                        "sexo": sexo,
                        **datos,
                    })

        # Avanzar al siguiente bloque
        if es_comarca:
            i += bloque_comarca
        else:
            i += bloque_provincia

    return pd.DataFrame(resultados)


# ---------------------------------------------------------------------------
# Cuadro 4 — Estructura demográfica (% jóvenes y mayores en PEA)
# ---------------------------------------------------------------------------

def extraer_cuadro4(filepath, periodo_info):
    """
    Parsea Cuadro 4 y retorna DataFrame con indicadores demográficos.

    Estructura por provincia (offsets desde fila del nombre):
        +0: provincia total -> col1=pob_total, col2=pea_total
        +2: 15-19 -> col2=pea_15_19
        +3: 20-24 -> col2=pea_20_24
        +8: 60-69 -> col2=pea_60_69
        +9: 70+   -> col2=pea_70_mas

    Extrae: pct_jovenes_pea (15-24/PEA), pct_mayores_pea (60+/PEA)
    """
    df = pd.read_excel(filepath, header=None)
    resultados = []

    i = 0
    while i < len(df):
        val = df.iloc[i, 0]
        provincia = _identificar_provincia(val)

        if provincia is None:
            i += 1
            continue

        if pd.isna(df.iloc[i, 1]):
            i += 1
            continue

        if not _es_fila_provincia_valida(val):
            i += 1
            continue

        pea_total = _safe_float(df.iloc[i, 2])
        pea_15_19 = _safe_float(df.iloc[i + 2, 2]) if i + 2 < len(df) else np.nan
        pea_20_24 = _safe_float(df.iloc[i + 3, 2]) if i + 3 < len(df) else np.nan
        pea_60_69 = _safe_float(df.iloc[i + 8, 2]) if i + 8 < len(df) else np.nan
        pea_70_mas = _safe_float(df.iloc[i + 9, 2]) if i + 9 < len(df) else np.nan

        pct_jovenes_pea = np.nan
        pct_mayores_pea = np.nan
        if pea_total and pea_total > 0:
            jovenes = (pea_15_19 or 0) + (pea_20_24 or 0)
            mayores = (pea_60_69 or 0) + (pea_70_mas or 0)
            pct_jovenes_pea = (jovenes / pea_total) * 100
            pct_mayores_pea = (mayores / pea_total) * 100

        resultados.append({
            "provincia": provincia,
            "periodo": periodo_info["periodo"],
            "anio": periodo_info["anio"],
            "pct_jovenes_pea": pct_jovenes_pea,
            "pct_mayores_pea": pct_mayores_pea,
        })

        # Saltar bloque provincia (total + hombres + mujeres = ~30 filas)
        i += 30

    return pd.DataFrame(resultados)


# ---------------------------------------------------------------------------
# Cuadro 6 — Tasas de actividad económica
# ---------------------------------------------------------------------------

def extraer_cuadro6(filepath, periodo_info):
    """
    Parsea Cuadro 6 y retorna DataFrame con tasas de actividad
    (total, urbana, rural) por provincia.
    """
    df = pd.read_excel(filepath, header=None)
    resultados = []

    for i in range(10, len(df)):
        val = df.iloc[i, 0]
        provincia = _identificar_provincia(val)

        if provincia is None:
            continue

        if not _es_fila_provincia_valida(val):
            continue

        tasa_total = _safe_float(df.iloc[i, 1])
        if np.isnan(tasa_total):
            continue

        tasa_urbana = _safe_float(df.iloc[i, 2])
        tasa_rural = _safe_float(df.iloc[i, 3])

        resultados.append({
            "provincia": provincia,
            "periodo": periodo_info["periodo"],
            "anio": periodo_info["anio"],
            "tasa_actividad_total": tasa_total,
            "tasa_actividad_urbana": tasa_urbana,
            "tasa_actividad_rural": tasa_rural,
        })

    df_result = pd.DataFrame(resultados)
    if not df_result.empty:
        df_result = df_result.drop_duplicates(subset=["provincia"], keep="first")
    return df_result


# ---------------------------------------------------------------------------
# Cuadro 13 — Composición sectorial (primario / secundario / terciario)
# ---------------------------------------------------------------------------

def extraer_cuadro13(filepath, periodos_info):
    """
    Parsea Cuadro 13 y retorna DataFrame con % empleo por sector.

    Estructura por provincia: busca fila de provincia, luego lee los 3 subtotales:
        "Sector Primario"   -> col1, con porcentaje en col_pct
        "Sector Secundario" -> col1, con porcentaje en col_pct
        "Sector Terciario"  -> col1, con porcentaje en col_pct
    """
    df = pd.read_excel(filepath, header=None)
    resultados = []

    i = 10
    while i < len(df):
        val = df.iloc[i, 0]
        provincia = _identificar_provincia(val)

        if provincia is None:
            i += 1
            continue

        if pd.isna(df.iloc[i, 3]):
            i += 1
            continue

        if not _es_fila_provincia_valida(val):
            i += 1
            continue

        # Buscar los 3 sectores en las ~45 filas siguientes
        sectores = {}
        for j in range(1, 45):
            if i + j >= len(df):
                break
            val1 = str(df.iloc[i + j, 1]).strip() if not pd.isna(df.iloc[i + j, 1]) else ''
            if 'Sector Primario' in val1:
                sectores['primario'] = i + j
            elif 'Sector Secundario' in val1:
                sectores['secundario'] = i + j
            elif 'Sector Terciario' in val1:
                sectores['terciario'] = i + j
            # Si encontramos otra provincia, parar
            next_prov = _identificar_provincia(df.iloc[i + j, 0])
            if next_prov and next_prov != provincia and _es_fila_provincia_valida(df.iloc[i + j, 0]):
                break

        for p_info in periodos_info:
            col_pct = p_info["col_pct"]
            pct_primario = _safe_float(df.iloc[sectores['primario'], col_pct]) if 'primario' in sectores else np.nan
            pct_secundario = _safe_float(df.iloc[sectores['secundario'], col_pct]) if 'secundario' in sectores else np.nan
            pct_terciario = _safe_float(df.iloc[sectores['terciario'], col_pct]) if 'terciario' in sectores else np.nan

            resultados.append({
                "provincia": provincia,
                "periodo": p_info["periodo"],
                "anio": p_info["anio"],
                "pct_sector_primario": pct_primario,
                "pct_sector_secundario": pct_secundario,
                "pct_sector_terciario": pct_terciario,
            })

        i += 40

    return pd.DataFrame(resultados)


# ---------------------------------------------------------------------------
# Cuadro 16 — Nivel educativo de la población ocupada
# ---------------------------------------------------------------------------

def extraer_cuadro16(filepath, periodo_info):
    """
    Parsea Cuadro 16 y retorna DataFrame con % por nivel educativo.

    Estructura: fila de provincia tiene totales por nivel educativo.
    Cols: 3=Total, 4=Ningún grado, 5=Primaria 1-3, 6=Primaria 4-6,
          7=Secundaria 1-3, 8=Secundaria 4-6, 9=Vocacional,
          10=Universitaria, 11=No universitaria
    """
    df = pd.read_excel(filepath, header=None)
    resultados = []

    for i in range(15, len(df)):
        val = df.iloc[i, 0]
        provincia = _identificar_provincia(val)

        if provincia is None:
            continue

        if pd.isna(df.iloc[i, 3]):
            continue

        if not _es_fila_provincia_valida(val):
            continue

        total = _safe_float(df.iloc[i, 3])
        if np.isnan(total) or total == 0:
            continue

        ningun_grado = _safe_float(df.iloc[i, 4]) or 0
        primaria_1_3 = _safe_float(df.iloc[i, 5]) or 0
        primaria_4_6 = _safe_float(df.iloc[i, 6]) or 0
        secundaria_1_3 = _safe_float(df.iloc[i, 7]) or 0
        secundaria_4_6 = _safe_float(df.iloc[i, 8]) or 0
        vocacional = _safe_float(df.iloc[i, 9]) or 0
        universitaria = _safe_float(df.iloc[i, 10]) or 0

        pct_sin_educacion = (ningun_grado / total) * 100
        pct_primaria = ((primaria_1_3 + primaria_4_6) / total) * 100
        pct_secundaria = ((secundaria_1_3 + secundaria_4_6 + vocacional) / total) * 100
        pct_universitaria = (universitaria / total) * 100

        resultados.append({
            "provincia": provincia,
            "periodo": periodo_info["periodo"],
            "anio": periodo_info["anio"],
            "pct_sin_educacion": pct_sin_educacion,
            "pct_primaria": pct_primaria,
            "pct_secundaria": pct_secundaria,
            "pct_universitaria": pct_universitaria,
        })

    df_result = pd.DataFrame(resultados)
    if not df_result.empty:
        df_result = df_result.drop_duplicates(subset=["provincia"], keep="first")
    return df_result


# ---------------------------------------------------------------------------
# Cuadro 19 — Subempleo (horas semanales trabajadas)
# ---------------------------------------------------------------------------

def extraer_cuadro19(filepath, periodo_info):
    """
    Parsea Cuadro 19 y retorna DataFrame con % de subempleo.

    Estructura: fila de provincia tiene totales por rango de horas.
    Cols: 1=Total, 2=Menos de 25, 3=25-34, 4=35-39, 5=40+, 6=No declaradas
    """
    df = pd.read_excel(filepath, header=None)
    resultados = []

    for i in range(15, len(df)):
        val = df.iloc[i, 0]
        provincia = _identificar_provincia(val)

        if provincia is None:
            continue

        if pd.isna(df.iloc[i, 1]):
            continue

        if not _es_fila_provincia_valida(val):
            continue

        total = _safe_float(df.iloc[i, 1])
        menos_25 = _safe_float(df.iloc[i, 2])

        if np.isnan(total) or total == 0:
            continue

        pct_subempleo = (menos_25 / total) * 100 if not np.isnan(menos_25) else np.nan

        resultados.append({
            "provincia": provincia,
            "periodo": periodo_info["periodo"],
            "anio": periodo_info["anio"],
            "pct_subempleo": pct_subempleo,
        })

    df_result = pd.DataFrame(resultados)
    if not df_result.empty:
        df_result = df_result.drop_duplicates(subset=["provincia"], keep="first")
    return df_result


# ---------------------------------------------------------------------------
# Cuadro 22 — Tamaño de empresa
# ---------------------------------------------------------------------------

def extraer_cuadro22(filepath, periodo_info):
    """
    Parsea Cuadro 22 y retorna DataFrame con distribución por tamaño de empresa.

    Estructura: fila de provincia tiene totales por tamaño.
    Cols: 3=Total, 4=Menos de 5, 5=5-10, 6=11-19, 7=20-49, 8=50+
    """
    df = pd.read_excel(filepath, header=None)
    resultados = []

    for i in range(12, len(df)):
        val = df.iloc[i, 0]
        provincia = _identificar_provincia(val)

        if provincia is None:
            continue

        if pd.isna(df.iloc[i, 3]):
            continue

        if not _es_fila_provincia_valida(val):
            continue

        total = _safe_float(df.iloc[i, 3])
        menos_5 = _safe_float(df.iloc[i, 4])
        mas_50 = _safe_float(df.iloc[i, 8])

        if np.isnan(total) or total == 0:
            continue

        pct_microempresa = (menos_5 / total) * 100 if not np.isnan(menos_5) else np.nan
        pct_empresa_grande = (mas_50 / total) * 100 if not np.isnan(mas_50) else np.nan

        resultados.append({
            "provincia": provincia,
            "periodo": periodo_info["periodo"],
            "anio": periodo_info["anio"],
            "pct_microempresa": pct_microempresa,
            "pct_empresa_grande": pct_empresa_grande,
        })

    df_result = pd.DataFrame(resultados)
    if not df_result.empty:
        df_result = df_result.drop_duplicates(subset=["provincia"], keep="first")
    return df_result


# ---------------------------------------------------------------------------
# Cuadro 25 — Mediana salarial
# ---------------------------------------------------------------------------

def extraer_cuadro25(filepath, periodo_info):
    """
    Parsea Cuadro 25 y retorna DataFrame con mediana salarial por provincia.

    Estructura: fila de provincia tiene mediana en col 3.
    """
    df = pd.read_excel(filepath, header=None)
    resultados = []

    for i in range(15, len(df)):
        val = df.iloc[i, 0]
        provincia = _identificar_provincia(val)

        if provincia is None:
            continue

        if pd.isna(df.iloc[i, 3]):
            continue

        if not _es_fila_provincia_valida(val):
            continue

        mediana = _safe_float(df.iloc[i, 3])
        if np.isnan(mediana):
            continue

        resultados.append({
            "provincia": provincia,
            "periodo": periodo_info["periodo"],
            "anio": periodo_info["anio"],
            "mediana_salario": mediana,
        })

    df_result = pd.DataFrame(resultados)
    if not df_result.empty:
        df_result = df_result.drop_duplicates(subset=["provincia"], keep="first")
    return df_result


# ---------------------------------------------------------------------------
# Cuadro 39 — Empleo informal
# ---------------------------------------------------------------------------

def extraer_cuadro39(filepath, periodos_info):
    """
    Parsea Cuadro 39 y retorna DataFrame con datos de informalidad por provincia.
    """
    df = pd.read_excel(filepath, header=None)
    resultados = []

    for i in range(10, min(30, len(df))):
        val = df.iloc[i, 1]
        provincia = _identificar_provincia(val)

        if provincia is None:
            continue

        for p_info in periodos_info:
            pct_col = p_info["col_informal_pct"]
            empleo_informal_pct = _safe_float(df.iloc[i, pct_col])

            resultados.append({
                "provincia": provincia,
                "periodo": p_info["periodo"],
                "anio": p_info["anio"],
                "empleo_informal_pct": empleo_informal_pct,
            })

    return pd.DataFrame(resultados)


# ---------------------------------------------------------------------------
# Cuadro 1A — Serie histórica nacional
# ---------------------------------------------------------------------------

def extraer_cuadro1a(filepath):
    """
    Parsea Cuadro 1A y retorna DataFrame con serie histórica nacional.
    """
    df = pd.read_excel(filepath, header=None)
    resultados = []

    for i in range(13, len(df)):
        anio_val = df.iloc[i, 1]
        if pd.isna(anio_val):
            continue

        anio_str = str(anio_val).strip()
        match = re.search(r'(\d{4})', anio_str)
        if not match:
            continue
        anio = int(match.group(1))

        periodo = f"{anio}-08"
        if 'octubre' in anio_str.lower() or 'oct' in anio_str.lower():
            periodo = f"{anio}-10"
        elif 'abril' in anio_str.lower() or 'abr' in anio_str.lower():
            periodo = f"{anio}-04"

        resultados.append({
            "anio": anio,
            "periodo": periodo,
            "etiqueta_original": anio_str,
            "poblacion_15_mas": _safe_float(df.iloc[i, 2]),
            "pea": _safe_float(df.iloc[i, 3]),
            "tasa_participacion": _safe_float(df.iloc[i, 4]),
            "ocupados": _safe_float(df.iloc[i, 5]),
            "desocupados": _safe_float(df.iloc[i, 6]),
            "tasa_desempleo": _safe_float(df.iloc[i, 7]),
            "no_economicamente_activa": _safe_float(df.iloc[i, 8]),
        })

    return pd.DataFrame(resultados)


# ---------------------------------------------------------------------------
# Banco Mundial — Indicadores macroeconómicos
# ---------------------------------------------------------------------------

def cargar_banco_mundial():
    """
    Carga indicadores del Banco Mundial y retorna DataFrame pivoteado por año.

    Lee el CSV en formato ancho y transpone a una fila por año con columnas
    para cada indicador macroeconómico relevante.
    """
    ruta = RAW_DATA_DIR / "banco_mundial_panama_ancho_20260115.csv"
    if not ruta.exists():
        print("  [WARN] No se encontró archivo del Banco Mundial")
        return None

    df = pd.read_csv(ruta)

    # Mapeo de indicadores a nombres de columna
    indicadores = {
        "NY.GDP.MKTP.KD.ZG": "pib_crecimiento",
        "FP.CPI.TOTL.ZG": "inflacion",
        "SI.POV.GINI": "indice_gini",
        "SE.XPD.TOTL.GD.ZS": "gasto_educacion_pib",
        "SL.UEM.1524.ZS": "desempleo_juvenil_nacional",
        "NE.EXP.GNFS.ZS": "exportaciones_pib",
        "NY.GDP.PCAP.CD": "pib_per_capita",
        "SP.POP.TOTL": "poblacion_total_nacional",
    }

    resultados = []
    anios = [c for c in df.columns if c not in ('codigo_indicador', 'nombre_indicador')]

    for anio_str in anios:
        anio = int(anio_str)
        fila = {"anio": anio}
        for codigo, nombre_col in indicadores.items():
            row = df[df["codigo_indicador"] == codigo]
            if not row.empty:
                fila[nombre_col] = _safe_float(row[anio_str].values[0])
            else:
                fila[nombre_col] = np.nan
        resultados.append(fila)

    return pd.DataFrame(resultados)


# ---------------------------------------------------------------------------
# Consolidación
# ---------------------------------------------------------------------------

def _concat_dedup(lista_dfs, subset_cols):
    """Concatena lista de DataFrames y elimina duplicados quedándose con el último."""
    if not lista_dfs:
        return pd.DataFrame()
    df = pd.concat(lista_dfs, ignore_index=True)
    df = df.drop_duplicates(subset=subset_cols, keep="last")
    return df


def consolidar_datos():
    """Itera todos los periodos, une los datos y genera CSVs."""
    todos_cuadro2 = []
    todos_cuadro4 = []
    todos_cuadro6 = []
    todos_cuadro13 = []
    todos_cuadro16 = []
    todos_cuadro19 = []
    todos_cuadro22 = []
    todos_cuadro25 = []
    todos_cuadro39 = []
    serie_historica = None

    for nombre_periodo, config in PERIODOS.items():
        carpeta = config["carpeta"]
        print(f"\n{'='*60}")
        print(f"Procesando: {nombre_periodo}")
        print(f"{'='*60}")

        tiene_cuadros_completos = config.get("tiene_cuadros_completos", False)
        tiene_cuadro2 = "periodos_cuadro2" in config

        if not tiene_cuadros_completos and not tiene_cuadro2:
            print(f"  -> Periodo con datos limitados (COVID), se omite.")
            continue

        # --- Cuadro 2 (siempre se intenta) ---
        if tiene_cuadro2:
            num_archivo = config.get("archivo_cuadro2")
            formato = config.get("formato_cuadro2", "expanded")
            fp = encontrar_archivo(carpeta, "2", numero_archivo=num_archivo)
            if fp:
                print(f"  Cuadro 2: {fp.name} (formato={formato})")
                df_tmp = extraer_cuadro2(
                    fp, config["periodos_cuadro2"], formato=formato
                )
                print(f"    -> {len(df_tmp)} registros")
                todos_cuadro2.append(df_tmp)
            else:
                print(f"  [WARN] Cuadro 2 no encontrado en {carpeta}")

        # --- Cuadros adicionales (solo para periodos completos) ---
        if not tiene_cuadros_completos:
            continue

        periodo_actual = config["periodo_actual"]

        # --- Cuadro 4 (demografía, 1 periodo) ---
        fp = encontrar_archivo(carpeta, "4")
        if fp:
            print(f"  Cuadro 4: {fp.name}")
            df_tmp = extraer_cuadro4(fp, periodo_actual)
            print(f"    -> {len(df_tmp)} registros")
            todos_cuadro4.append(df_tmp)

        # --- Cuadro 6 (actividad, 1 periodo) ---
        fp = encontrar_archivo(carpeta, "6")
        if fp:
            print(f"  Cuadro 6: {fp.name}")
            df_tmp = extraer_cuadro6(fp, config["periodo_cuadro6"])
            print(f"    -> {len(df_tmp)} registros")
            todos_cuadro6.append(df_tmp)

        # --- Cuadro 13 (sectores, 2 periodos) ---
        fp = encontrar_archivo(carpeta, "13")
        if fp:
            print(f"  Cuadro 13: {fp.name}")
            df_tmp = extraer_cuadro13(fp, config["periodos_cuadro13"])
            print(f"    -> {len(df_tmp)} registros")
            todos_cuadro13.append(df_tmp)

        # --- Cuadro 16 (educación, 1 periodo) ---
        fp = encontrar_archivo(carpeta, "16")
        if fp:
            print(f"  Cuadro 16: {fp.name}")
            df_tmp = extraer_cuadro16(fp, periodo_actual)
            print(f"    -> {len(df_tmp)} registros")
            todos_cuadro16.append(df_tmp)

        # --- Cuadro 19 (subempleo, 1 periodo) ---
        fp = encontrar_archivo(carpeta, "19")
        if fp:
            print(f"  Cuadro 19: {fp.name}")
            df_tmp = extraer_cuadro19(fp, periodo_actual)
            print(f"    -> {len(df_tmp)} registros")
            todos_cuadro19.append(df_tmp)

        # --- Cuadro 22 (tamaño empresa, 1 periodo) ---
        fp = encontrar_archivo(carpeta, "22")
        if fp:
            print(f"  Cuadro 22: {fp.name}")
            df_tmp = extraer_cuadro22(fp, periodo_actual)
            print(f"    -> {len(df_tmp)} registros")
            todos_cuadro22.append(df_tmp)

        # --- Cuadro 25 (salario, 1 periodo) ---
        fp = encontrar_archivo(carpeta, "25")
        if fp:
            print(f"  Cuadro 25: {fp.name}")
            df_tmp = extraer_cuadro25(fp, periodo_actual)
            print(f"    -> {len(df_tmp)} registros")
            todos_cuadro25.append(df_tmp)

        # --- Cuadro 39 (informalidad, 2 periodos) ---
        fp = encontrar_archivo(carpeta, "39")
        if fp:
            print(f"  Cuadro 39: {fp.name}")
            df_tmp = extraer_cuadro39(fp, config["periodos_cuadro39"])
            print(f"    -> {len(df_tmp)} registros")
            todos_cuadro39.append(df_tmp)

        # --- Cuadro 1A (serie histórica, se sobreescribe con la más reciente) ---
        fp = encontrar_archivo(carpeta, "1A")
        if fp:
            print(f"  Cuadro 1A: {fp.name}")
            serie_historica = extraer_cuadro1a(fp)
            print(f"    -> {len(serie_historica)} registros")

    # ===================================================================
    # Consolidación
    # ===================================================================
    print(f"\n{'='*60}")
    print("Consolidando datos...")
    print(f"{'='*60}")

    # --- Base: Cuadro 2 (con columna 'area') ---
    if not todos_cuadro2:
        print("  [ERROR] No se pudo extraer datos del Cuadro 2")
        return
    df_principal = _concat_dedup(
        todos_cuadro2, ["provincia", "periodo", "area", "sexo"]
    )
    print(f"  Cuadro 2 base: {len(df_principal)} registros")
    n_areas = df_principal.groupby('area').size()
    for area, count in n_areas.items():
        print(f"    area={area}: {count} filas")
    n_sexos = df_principal.groupby('sexo').size()
    for sexo, count in n_sexos.items():
        print(f"    sexo={sexo}: {count} filas")

    # --- Merge helper ---
    # Los cuadros adicionales (4, 6, 13, ...) tienen datos a nivel
    # provincia-total. Al hacer merge por (provincia, periodo), las filas
    # urbana/rural heredan los mismos valores provinciales.
    def _merge_cuadro(lista, nombre, cols_merge):
        nonlocal df_principal
        df_all = _concat_dedup(lista, ["provincia", "periodo"])
        if df_all.empty:
            return
        df_principal = df_principal.merge(
            df_all[["provincia", "periodo"] + cols_merge],
            on=["provincia", "periodo"],
            how="left",
        )
        n_filled = df_principal[cols_merge[0]].notna().sum()
        print(f"  {nombre} unido: {n_filled}/{len(df_principal)} filas con datos")

    # --- Cuadro 4: demografía ---
    _merge_cuadro(todos_cuadro4, "Cuadro 4 (demografía)",
                  ["pct_jovenes_pea", "pct_mayores_pea"])

    # --- Cuadro 6: actividad ---
    _merge_cuadro(todos_cuadro6, "Cuadro 6 (actividad)",
                  ["tasa_actividad_total", "tasa_actividad_urbana", "tasa_actividad_rural"])

    # --- Cuadro 13: sectores ---
    _merge_cuadro(todos_cuadro13, "Cuadro 13 (sectores)",
                  ["pct_sector_primario", "pct_sector_secundario", "pct_sector_terciario"])

    # --- Cuadro 16: educación ---
    _merge_cuadro(todos_cuadro16, "Cuadro 16 (educación)",
                  ["pct_sin_educacion", "pct_primaria", "pct_secundaria", "pct_universitaria"])

    # --- Cuadro 19: subempleo ---
    _merge_cuadro(todos_cuadro19, "Cuadro 19 (subempleo)",
                  ["pct_subempleo"])

    # --- Cuadro 22: tamaño empresa ---
    _merge_cuadro(todos_cuadro22, "Cuadro 22 (empresas)",
                  ["pct_microempresa", "pct_empresa_grande"])

    # --- Cuadro 25: salario ---
    _merge_cuadro(todos_cuadro25, "Cuadro 25 (salario)",
                  ["mediana_salario"])

    # --- Cuadro 39: informalidad ---
    _merge_cuadro(todos_cuadro39, "Cuadro 39 (informalidad)",
                  ["empleo_informal_pct"])

    # --- Banco Mundial ---
    df_bm = cargar_banco_mundial()
    if df_bm is not None:
        df_principal = df_principal.merge(df_bm, on="anio", how="left")
        n_filled = df_principal["pib_crecimiento"].notna().sum()
        print(f"  Banco Mundial unido: {n_filled}/{len(df_principal)} filas con datos")

    # --- Ordenar ---
    df_principal = df_principal.sort_values(
        ["periodo", "provincia", "area", "sexo"]
    ).reset_index(drop=True)

    # --- Columnas en orden lógico ---
    columnas_orden = [
        # Identificación
        "provincia", "periodo", "anio", "area", "sexo",
        # Mercado laboral (Cuadro 2)
        "poblacion_15_mas", "pea", "ocupados", "desocupados",
        "tasa_desempleo", "tasa_participacion",
        # Actividad (Cuadro 6)
        "tasa_actividad_total", "tasa_actividad_urbana", "tasa_actividad_rural",
        # Demografía (Cuadro 4)
        "pct_jovenes_pea", "pct_mayores_pea",
        # Sectores (Cuadro 13)
        "pct_sector_primario", "pct_sector_secundario", "pct_sector_terciario",
        # Educación (Cuadro 16)
        "pct_sin_educacion", "pct_primaria", "pct_secundaria", "pct_universitaria",
        # Subempleo (Cuadro 19)
        "pct_subempleo",
        # Empresas (Cuadro 22)
        "pct_microempresa", "pct_empresa_grande",
        # Salario (Cuadro 25)
        "mediana_salario",
        # Informalidad (Cuadro 39)
        "empleo_informal_pct",
        # Banco Mundial (macro)
        "pib_crecimiento", "inflacion", "indice_gini",
        "gasto_educacion_pib", "desempleo_juvenil_nacional",
        "exportaciones_pib", "pib_per_capita", "poblacion_total_nacional",
    ]
    # Solo incluir columnas que existen
    columnas_final = [c for c in columnas_orden if c in df_principal.columns]
    df_principal = df_principal[columnas_final]

    # --- Guardar ---
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    ruta_provincia = PROCESSED_DATA_DIR / "desempleo_por_provincia.csv"
    df_principal.to_csv(ruta_provincia, index=False, encoding="utf-8")
    print(f"\n  Guardado: {ruta_provincia}")
    print(f"    Filas: {len(df_principal)}")
    print(f"    Columnas: {len(df_principal.columns)}")
    print(f"    Provincias: {df_principal['provincia'].nunique()}")
    print(f"    Periodos: {sorted(df_principal['periodo'].unique())}")
    print(f"    Areas: {sorted(df_principal['area'].unique())}")

    if serie_historica is not None:
        ruta_historica = PROCESSED_DATA_DIR / "serie_historica_nacional.csv"
        serie_historica = serie_historica.sort_values("anio").reset_index(drop=True)
        serie_historica.to_csv(ruta_historica, index=False, encoding="utf-8")
        print(f"\n  Guardado: {ruta_historica}")
        print(f"    Filas: {len(serie_historica)}")
        print(f"    Rango: {serie_historica['anio'].min()}-{serie_historica['anio'].max()}")

    return df_principal, serie_historica


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("PROCESAMIENTO DE DATOS INEC - PANAMÁ")
    print("=" * 60)

    resultado = consolidar_datos()
    if resultado is None:
        print("\n[ERROR] No se pudieron procesar los datos.")
        return

    df_prov, df_hist = resultado

    print(f"\n{'='*60}")
    print("RESUMEN FINAL")
    print(f"{'='*60}")

    print(f"\n--- Desempleo por Provincia ---")
    print(f"  Registros: {len(df_prov)}")
    print(f"  Columnas: {len(df_prov.columns)}")
    print(f"  Provincias/Comarcas: {df_prov['provincia'].nunique()}")
    print(f"  Periodos: {df_prov['periodo'].nunique()}")

    print(f"\n  Cobertura de datos (% filas no nulas):")
    for col in df_prov.columns:
        if col in ("provincia", "periodo", "anio"):
            continue
        pct = df_prov[col].notna().mean() * 100
        print(f"    {col:.<40s} {pct:5.1f}%")

    print(f"  Areas: {sorted(df_prov['area'].unique())}")
    if 'sexo' in df_prov.columns:
        print(f"  Sexos: {sorted(df_prov['sexo'].unique())}")

    print(f"\n  Muestra Oct-2024 (Panamá provincia, total):")
    mask = (
        (df_prov["provincia"] == "Panamá")
        & (df_prov["periodo"] == "2024-10")
        & (df_prov["area"] == "total")
    )
    if 'sexo' in df_prov.columns:
        mask = mask & (df_prov["sexo"] == "total")
    fila = df_prov[mask]
    if not fila.empty:
        for col in df_prov.columns:
            if col in ("provincia", "periodo"):
                continue
            val = fila[col].values[0]
            if isinstance(val, float) and not np.isnan(val):
                print(f"    {col}: {val:,.2f}")

    if df_hist is not None:
        print(f"\n--- Serie Histórica Nacional ---")
        print(f"  Años: {len(df_hist)} ({df_hist['anio'].min()}-{df_hist['anio'].max()})")
        ultimo = df_hist.iloc[-1]
        print(f"  Último: {ultimo['etiqueta_original']} -> desempleo {ultimo['tasa_desempleo']:.2f}%")

    print(f"\nArchivos generados en: {PROCESSED_DATA_DIR}")
    print("Procesamiento completado.")


if __name__ == "__main__":
    main()
