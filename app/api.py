from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
from typing import Dict

app = FastAPI(
    title="Modelo de Predicción de Mora en Créditos",
    description="Una API para predecir la probabilidad de mora en créditos utilizando un modelo de machine learning entrenado con datos históricos.",
    version="1.0.0"
)

class PredictionRequest(BaseModel):
    #################################################################################
    # TODO: adaptar el modelo según los resultados de la fase de preparación de datos
    ##################################################################################
    edad: int = Field(..., description="Edad del cliente")
    antiguedad_empleado: float = Field(..., description="Antigüedad del empleado")
    situacion_vivienda: str = Field(..., description="Situación de la vivienda")
    ingresos: int = Field(..., description="Ingresos del cliente")
    objetivo_credito: str = Field(..., description="Objetivo del crédito")
    pct_ingreso: float = Field(..., description="Porcentaje de ingreso")
    tasa_interes: float = Field(..., description="Tasa de interés")
    estado_credito: int = Field(..., description="Estado del crédito")
    antiguedad_cliente: float = Field(..., description="Antigüedad del cliente")
    estado_civil: str = Field(..., description="Estado civil")
    estado_cliente: str = Field(..., description="Estado del cliente")
    gastos_ult_12m: float = Field(..., description="Gastos de los últimos 12 meses")
    genero: str = Field(..., description="Género")
    limite_credito_tc: float = Field(..., description="Límite de crédito de la tarjeta de crédito")
    nivel_educativo: str = Field(..., description="Nivel educativo")
    personas_a_cargo: float = Field(..., description="Personas a cargo")
    capacidad_pago: float = Field(..., description="Capacidad de pago")
    operaciones_mensuales: float = Field(..., description="Operaciones mensuales")
    presion_financiera: float = Field(..., description="Presión financiera")
    gasto_promedio_operacion: float = Field(..., description="Gasto promedio por operación")
    operaciones_mensuales_tc: float = Field(..., description="Operaciones mensuales con tarjeta de crédito")
    estabilidad_laboral: float = Field(..., description="Estabilidad laboral")

    class Config:
        json_schema_extra = {
            "example": {
                "edad": 21,
                "antiguedad_empleado": 5.0,
                "situacion_vivienda": "PROPIA",
                "ingresos": 9600,
                "objetivo_credito": "EDUCACIÓN",
                "pct_ingreso": 0.1,
                "tasa_interes": 11.14,
                "estado_credito": 0,
                "antiguedad_cliente": 39.0,
                "estado_civil": "CASADO",
                "estado_cliente": "ACTIVO",
                "gastos_ult_12m": 1144.0,
                "genero": "M",
                "limite_credito_tc": 12691.0,
                "nivel_educativo": "SECUNDARIO_COMPLETO",
                "personas_a_cargo": 3.0,
                "capacidad_pago": 0.104167,
                "operaciones_mensuales": 3.5,
                "presion_financiera": 0.17125,
                "gasto_promedio_operacion": 27.238095,
                "operaciones_mensuales_tc": 3.5,
                "estabilidad_laboral": 0.238095
            }
        }

class PredictionResponse(BaseModel):
    prediction: str
    probability: Dict[str, float]
    class_labels: Dict[str, str]
    model_info: Dict[str, str]

# Cargar el modelo entrenado
MODEL_PATH = "models/prod_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    print("Modelo cargado exitosamente.")
except FileNotFoundError:
    print(f"Error: No se encontró el modelo en la ruta {MODEL_PATH}. Asegúrate de que el modelo esté entrenado y guardado correctamente.")
    model = None
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    model = None

@app.get("/")
def read_root():
    return {
        "message": "Bienvenido a la API de Predicción de Mora en Créditos",
        "endpoints": {
            "/predict": "POST - Realiza una predicción de mora en créditos",
            "/docs": "GET - Documentación interactiva de la API",
            "/health": "GET - Verifica el estado de la API"
        }
    }

@app.get("/health")
def health_check():
    if model is not None:
        return {"status": "ok", "message": "La API está funcionando correctamente."}
    else:
        return {"status": "error", "message": "El modelo no está cargado. Verifica el estado del modelo."}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="El modelo no está disponible. Intenta nuevamente más tarde.")
    
    try:
        # Convertir la solicitud a un DataFrame
        input_data = pd.DataFrame([request.dict()])
        
        # Realizar la predicción
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        # Mapear las etiquetas de clase a descripciones legibles
        class_labels = model.named_steps['model'].classes_
        probability_dict = {
            # El dict de probabilidades se construye dinámicamente para que se adapte a cualquier número de clases
            # CAMBIO: no estaba hecha la conversión a str y ese era el error en SP6
            str(class_labels[i]): float(probability[i]) for i in range(len(class_labels))
        }
        model_info = {
            "model_version": "1.0.0",
            "model_type": type(model.named_steps["model"]).__name__, # Para que el nombre se complete automáticamente según el modelo cargado
        }
        return PredictionResponse(
            prediction=str(prediction),
            probability=probability_dict,
            class_labels={
                "0": "No entra en mora (N)",
                "1": "Entra en mora (Y)"
            },
            model_info=model_info
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Error al realizar la predicción: {e}")