import pandas as pd
from pathlib import Path
import pytest
import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*`Number` field should not be instantiated.*",
)

import great_expectations as ge


pytestmark = [
    pytest.mark.filterwarnings("ignore:.*Number.*should not be instantiated.*"),
    pytest.mark.filterwarnings("ignore:.*result_format.*Validator-level.*"),
    pytest.mark.filterwarnings("ignore:.*result_format.*Expectation-level.*"),
]


# Paths
PROJECT_DIR = Path(".").resolve()
DATA_DIR = PROJECT_DIR / "data"


def test_great_expectations():
    """ Prueba para validar la calidad de los datos utilizando Great Expectations.
    """
    # Cargar los datos de créditos y tarjetas
    df_creditos = pd.read_csv(DATA_DIR / "raw/datos_creditos.csv", sep=";")
    df_tarjetas = pd.read_csv(DATA_DIR / "raw/datos_tarjetas.csv", sep=";")

    results = {
        "success": True,
        "expectations": [],
        "statistics": {"success_count": 0, "total_count": 0}
    }

    def add_expectation(expectation_name, condition, message=""):
        results["statistics"]["total_count"] += 1
        if condition:
            results["statistics"]["success_count"] += 1
            results["expectations"].append({
                "expectation": expectation_name,
                "success": True
            })
        else:
            results["success"] = False
            results["expectations"].append({
                "expectation": expectation_name,
                "success": False,
                "message": message
            })

    
    # Atributo a analizar: Exactitud (rangos de valores en datos)
    # Validación 1: Rango de edad (18-100 años)
    edad_valida = df_creditos["edad"].between(18, 100).all()
    mensaje_edad = ""
    if not edad_valida:
        edades_fuera = df_creditos[(df_creditos["edad"] < 18) | (df_creditos["edad"] > 100)]["edad"].unique()
        mensaje_edad = f"Edades fuera de rango encontradas: {sorted(edades_fuera)}"
    add_expectation(
        "rango_edad",
        edad_valida,
        f"La edad debe estar entre 18 y 100 años. {mensaje_edad}"
    )
   
    # Validación 2: Rango de valores para situación de vivienda (ALQUILER, PROPIA, OTROS, HIPOTECA)
    vivienda_valida = df_creditos["situacion_vivienda"].isin(["ALQUILER", "PROPIA", "OTROS", "HIPOTECA"]).all()
    mensaje_vivienda = ""
    if not vivienda_valida:
        viviendas_fuera = df_creditos[~df_creditos["situacion_vivienda"].isin(["ALQUILER", "PROPIA", "OTROS", "HIPOTECA"])]["situacion_vivienda"].unique()
        mensaje_vivienda = f"Situaciones de vivienda no válidas encontradas: {sorted(viviendas_fuera)}" 
    add_expectation(
        "situacion_vivienda",
        vivienda_valida,
        f"La situación de vivienda no se encuentra en el rango válido. {mensaje_vivienda}"
    )

    # Validación 3: Validar que el límite de rango de crédito sea mayor a 0
    limites_invalidos = df_tarjetas["limite_credito_tc"] <= 0
    valido_limite_credito = ~limites_invalidos.any()
    mensaje_limite_credito = ""
    if not valido_limite_credito:
        cantidad = limites_invalidos.sum()
        valores_limite_credito = df_tarjetas.loc[limites_invalidos, "limite_credito_tc"].unique().tolist()
        mensaje_limite_credito = f"Se encontraron {cantidad} registros inválidos. Valores: {valores_limite_credito}"
    add_expectation(
        "limite_credito_tc_positivo",
        valido_limite_credito,
        f"El límite de crédito debe ser mayor a 0. {mensaje_limite_credito}"
    )

    # Validación 4: Validar que el género sea válido
    valores_validos = ["M", "F"]
    mask_invalidos = ~df_tarjetas["genero"].isin(valores_validos)
    valido_genero = ~mask_invalidos.any()
    mensaje_genero = ""
    if not valido_genero:
        cantidad = mask_invalidos.sum()
        valores_genero = df_tarjetas.loc[mask_invalidos, "genero"].unique().tolist()
        mensaje_genero = f"Se encontraron {cantidad} registros inválidos. Valores: {valores_genero}"
    add_expectation(
        "genero_valido",
        valido_genero,
        f"El género debe ser M o F. {mensaje_genero}"
    )

    # Validación 5: Validar que nivel educativo sea válido
    valores_validos = ["SECUNDARIA", "UNIVERSITARIO", "POSTGRADO", "OTROS", 
                       "UNIVERSITARIO_COMPLETO", "SECUNDARIO_COMPLETO", 
                       "UNIVERSITARIO_INCOMPLETO", "POSGRADO_INCOMPLETO", 
                       "POSGRADO_COMPLETO"]
    mask_invalidos = ~df_tarjetas["nivel_educativo"].isin(valores_validos)
    valido_nivel_edu = ~mask_invalidos.any()
    mensaje_nivel_edu = ""
    if not valido_nivel_edu:
        cantidad = mask_invalidos.sum()
        valores_nivel_edu = df_tarjetas.loc[mask_invalidos, "nivel_educativo"].unique().tolist()
        mensaje_nivel_edu = f"Se encontraron {cantidad} registros inválidos. Valores: {valores_nivel_edu}"
    add_expectation(
        "nivel_educativo_valido",
        valido_nivel_edu,
        f"El nivel educativo contiene valores inválidos. {mensaje_nivel_edu}"
    )

    # Validación 6: Validar que las transacciones de los últimos 12 meses sean mayor a 0
    mask_invalidos = df_tarjetas["operaciones_ult_12m"] <= 0
    valido_ope_ult_12m = ~mask_invalidos.any()
    mensaje_ope_ult_12m = ""
    if not valido_ope_ult_12m:
        cantidad = mask_invalidos.sum()
        valores = df_tarjetas.loc[mask_invalidos, "operaciones_ult_12m"].unique().tolist()
        mensaje_ope_ult_12m = f"Se encontraron {cantidad} registros inválidos. Valores: {valores}"
    add_expectation(
        "operaciones_positivas",
        valido_ope_ult_12m,
        f"Las operaciones deben ser mayores a 0. {mensaje_ope_ult_12m}"
    )


        # Resumen y validación final
    print("\n" + "="*70)
    print("RESUMEN DE VALIDACIONES")
    print("="*70)
    for exp in results["expectations"]:
        status = "✓ PASS" if exp["success"] else "✗ FAIL"
        print(f"{status}: {exp['expectation']}")
        if not exp["success"] and "message" in exp:
            print(f"       Detalle: {exp['message']}")
    print(f"\nTotal: {results['statistics']['success_count']}/{results['statistics']['total_count']} validaciones pasaron")
    print("="*70 + "\n")

    # El test falla si alguna validación no pasó
    assert results["success"], f"Se encontraron {results['statistics']['total_count'] - results['statistics']['success_count']} validaciones fallidas"