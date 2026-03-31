import pandas as pd
from pandera.pandas import DataFrameSchema, Column, Check
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"


@pytest.fixture
def datos_creditos():
    """Carga los datos de créditos desde un archivo CSV."""
    ruta = DATA_DIR / "datos_creditos.csv"
    return pd.read_csv(ruta, sep=";")


@pytest.fixture
def datos_tarjetas():
    """Carga los datos de tarjetas desde un archivo CSV."""
    ruta = DATA_DIR / "datos_tarjetas.csv"
    return pd.read_csv(ruta, sep=";")


def test_esquema_datos_creditos(datos_creditos):
    """Prueba para validar el esquema de los datos de créditos."""
    esquema = DataFrameSchema({
        "id_cliente": Column(int, nullable=False),
        "edad": Column(int, Check.greater_than_or_equal_to(18)),
        "ingresos": Column(float, Check.greater_than_or_equal_to(0)),
        "falta_pago": Column(int, nullable=False)
    })
    esquema.validate(datos_creditos)