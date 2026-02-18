"""Tests para la configuracion del proyecto."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    ROOT_DIR, DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR,
    FIGURES_DIR, SHAPEFILES_DIR, RANDOM_STATE, RIESGO_BAJO, RIESGO_CRITICO,
    COLOR_MAP, PROVINCIAS
)


class TestDirectorios:
    def test_root_dir_existe(self):
        assert ROOT_DIR.exists()

    def test_data_dir_existe(self):
        assert DATA_DIR.exists()

    def test_processed_data_dir_existe(self):
        assert PROCESSED_DATA_DIR.exists()

    def test_models_dir_existe(self):
        assert MODELS_DIR.exists()

    def test_shapefiles_dir_existe(self):
        assert SHAPEFILES_DIR.exists()


class TestParametros:
    def test_random_state_es_entero(self):
        assert isinstance(RANDOM_STATE, int)

    def test_umbrales_riesgo_orden(self):
        assert RIESGO_BAJO <= RIESGO_CRITICO

    def test_umbrales_riesgo_positivos(self):
        assert RIESGO_BAJO > 0
        assert RIESGO_CRITICO > 0

    def test_color_map_tiene_tres_niveles(self):
        assert 'bajo' in COLOR_MAP
        assert 'moderado' in COLOR_MAP
        assert 'critico' in COLOR_MAP

    def test_provincias_no_vacia(self):
        assert len(PROVINCIAS) > 0

    def test_provincias_contiene_panama(self):
        assert 'Panamá' in PROVINCIAS
