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
    """Prueba para validar aspectos básicos de los datos de créditos."""

    df = datos_creditos
    assert not df.empty, "El dataset de créditos está vacío."
    assert df.shape[1] == 12, (
        f"El dataset de créditos debería tener 12 columnas, pero tiene {df.shape[1]}."
    )

    nulos_por_columna = df.isnull().sum()
    columnas_con_nulos = nulos_por_columna[nulos_por_columna > 0]

    print("Columnas con valores nulos:")
    print(columnas_con_nulos)

    total_nulos = columnas_con_nulos.sum()
    print(f"Total de valores nulos: {total_nulos}")

    assert columnas_con_nulos.empty, (
        f"Existen valores nulos en el dataset:\n{columnas_con_nulos.to_dict()}"
    )

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

 # Atributo a analizar: Completitud (a nivel de relación entre datasets)
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

 # Atributo a analizar: Consistencia (a nivel de relación entre datasets)    
def test_unicidad_id_cliente(datos_creditos, datos_tarjetas):
    """Valida que los IDs de clientes sean únicos en ambos datasets."""

    # Créditos
    assert datos_creditos["id_cliente"].is_unique, \
        "Existen IDs duplicados en el dataset de créditos."
    # Tarjetas
    assert datos_tarjetas["id_cliente"].is_unique, \
        "Existen IDs duplicados en el dataset de tarjetas."


def test_edad_valida(datos_creditos):
    """Valida que las edades estén entre 18 y 100."""
    assert "edad" in datos_creditos.columns, "La columna 'edad' no existe."
    edades_invalidas = datos_creditos.loc[
        ~datos_creditos["edad"].between(18, 100), "edad"
    ]

    cantidad_invalidos = len(edades_invalidas)
    print(f"Cantidad de edades fuera de rango: {cantidad_invalidos}")
    print(f"Valores encontrados: {edades_invalidas.unique().tolist()}")

    assert cantidad_invalidos == 0, (
        f"Se encontraron {cantidad_invalidos} registros con edades fuera de rango: "
        f"{edades_invalidas.unique().tolist()}"
    )

def test_falta_pago_valido(datos_creditos):
    """Valida que falta_pago solo contenga Y o N."""
    assert datos_creditos["falta_pago"].isin(["Y", "N"]).all(), \
        "Existen valores inválidos en la columna falta_pago."

 