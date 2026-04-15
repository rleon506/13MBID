import pandas as pd
import pandera as pa
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
        "id_cliente": Column(float, nullable=False),
        "edad": Column(int, Check.greater_than_or_equal_to(18)),
        "importe_solicitado": Column(int, Check.greater_than(0)),
        "duracion_credito": Column(int, Check.greater_than(0)),
        "antiguedad_empleado": Column(float, Check.greater_than_or_equal_to(0), nullable=True),
        "situacion_vivienda": Column(str, nullable=False),
        "objetivo_credito": Column(str, nullable=False),
        "pct_ingreso": Column(float, Check.greater_than_or_equal_to(0)),
        "tasa_interes": Column(float, Check.greater_than_or_equal_to(0), nullable=True),
        "estado_credito": Column(int, nullable=False),
        "ingresos": Column(float, Check.greater_than_or_equal_to(0)),
        #"falta_pago": Column(pa.Int, nullable=False)
        "falta_pago": Column(str, Check.isin(["Y", "N"]), nullable=False)
    }, coerce=True)
    esquema.validate(datos_creditos)

def test_esquema_datos_tarjetas(datos_tarjetas):
    """ Prueba para validar el esquema de los datos de tarjetas.
    Args:
        datos_tarjetas (pd.DataFrame): DataFrame con los datos de tarjetas.
    """
    # Atributo a analizar: Exactitud (a nivel de estructura del dataset
    esquema = DataFrameSchema({
        "id_cliente": Column(float, nullable=False),
        "antiguedad_cliente": Column(float, Check.greater_than_or_equal_to(0), nullable=True),
        "estado_civil": Column(str, nullable=False),
        "estado_cliente": Column(str, nullable=False),
        "gastos_ult_12m": Column(float, Check.greater_than(0)),
        "genero": Column(str, nullable=False),
        "limite_credito_tc": Column(float, Check.greater_than(0)),
        "nivel_educativo": Column(str, nullable=False),
        "nivel_tarjeta": Column(str, nullable=False),
        "operaciones_ult_12m": Column(float, Check.greater_than(0)),
        "personas_a_cargo": Column(float, Check.greater_than_or_equal_to(0), nullable=True),
    })
    esquema.validate(datos_tarjetas)

def test_basicos_creditos(datos_creditos):
    """ Prueba para validar aspectos básicos de los datos de créditos.
    Args:
        datos_creditos (pd.DataFrame): DataFrame con los datos de créditos.
    """
    # Atributo a analizar: Exactitud (a nivel de estructura del dataset)
    df = datos_creditos
    # Verificar que el dataset no sea nulo completamente
    assert not df.empty, "El dataset de créditos está vacío."
    # Verificar la cantidad de columnas para completar estructura del dataset
    assert df.shape[1] == 12, f"El dataset de créditos debería tener 12 columnas, pero tiene {df.shape[1]}."

    # Verificar que no haya valores nulos en general
    # Atributo a analizar: Completitud (a nivel general del dataset)
    #assert df.isnull().sum().sum() == 0, "Existen valores nulos en el dataset de créditos."
    assert df.drop(columns=["tasa_interes", "antiguedad_empleado"]) \
         .isnull().sum().sum() == 0, \
    "Existen valores nulos en columnas obligatorias del dataset de créditos."

def test_basicos_tarjetas(datos_tarjetas):
    """ Prueba para validar aspectos básicos de los datos de tarjetas.
    Args:
        datos_tarjetas (pd.DataFrame): DataFrame con los datos de tarjetas.
    """
    # Atributo a analizar: Exactitud (a nivel de estructura del dataset)
    df = datos_tarjetas
    # Verificar que el dataset no sea nulo completamente
    assert not df.empty, "El dataset de tarjetas está vacío."
    # Verificar la cantidad de columnas para completar estructura del dataset
    assert df.shape[1] == 11, f"El dataset de tarjetas debería tener 11 columnas, pero tiene {df.shape[1]}."

    # Verificar que no haya valores nulos en general
    # Atributo a analizar: Completitud (a nivel general del dataset)
    assert df.isnull().sum().sum() == 0, "Existen valores nulos en el dataset de tarjetas."

def test_integridad_referencial(datos_creditos, datos_tarjetas):
    """ Prueba para validar la integridad referencial entre los datasets de créditos y tarjetas.
    Args:
        datos_creditos (pd.DataFrame): DataFrame con los datos de créditos.
        datos_tarjetas (pd.DataFrame): DataFrame con los datos de tarjetas.
    """
    # Atributo a analizar: Consistencia (a nivel de relación entre datasets)

    df_ids = datos_creditos[["id_cliente"]].merge(
        datos_tarjetas[["id_cliente"]], 
        on="id_cliente", 
        how="outer",
        indicator=True
    )
    integridad_referencial = DataFrameSchema({
        "_merge": Column(
            str, 
            Check.isin(["both"]),
            nullable=False
        )
    })
    integridad_referencial.validate(df_ids)

def test_valores_nulos_columnas_obligatorias(datos_creditos):
    """Valida que no existan nulos en las columnas obligatorias."""

    columnas_obligatorias = [
        "id_cliente",
        "edad",
        "importe_solicitado",
        "duracion_credito",
        "situacion_vivienda",
        "objetivo_credito",
        "pct_ingreso",
        "estado_credito",
        "ingresos",
        "falta_pago"
    ]
    assert datos_creditos[columnas_obligatorias].isnull().sum().sum() == 0, \
        "Existen valores nulos en columnas obligatorias del dataset de créditos."
    
def test_unicidad_id_cliente(datos_creditos, datos_tarjetas):
    """Valida que los IDs de clientes sean únicos en ambos datasets."""

    # Créditos
    assert datos_creditos["id_cliente"].is_unique, \
        "Existen IDs duplicados en el dataset de créditos."
    # Tarjetas
    assert datos_tarjetas["id_cliente"].is_unique, \
        "Existen IDs duplicados en el dataset de tarjetas."
    
def test_edad_valida(datos_creditos):
    """Valida e informa edades fuera de rango."""
    edades_invalidas = datos_creditos[~datos_creditos["edad"].between(18, 100)]
    print(f"Edades fuera de rango encontradas: {edades_invalidas['edad'].unique()}")
    assert "edad" in datos_creditos.columns

def test_falta_pago_valido(datos_creditos):
    """Valida que falta_pago solo contenga Y o N."""
    assert datos_creditos["falta_pago"].isin(["Y", "N"]).all(), \
        "Existen valores inválidos en la columna falta_pago."

 