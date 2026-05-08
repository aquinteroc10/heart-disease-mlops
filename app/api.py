from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd


model = joblib.load("model.joblib")

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predicción de falla cardíaca usando RandomForest",
    version="1.0"
)


class PacienteInput(BaseModel):
    Age: int
    RestingBP: int
    Cholesterol: int
    FastingBS: int
    MaxHR: int
    Oldpeak: float
    Sex_F: bool
    Sex_M: bool
    ChestPainType_ASY: bool
    ChestPainType_ATA: bool
    ChestPainType_NAP: bool
    ChestPainType_TA: bool
    RestingECG_LVH: bool
    RestingECG_Normal: bool
    RestingECG_ST: bool
    ExerciseAngina_N: bool
    ExerciseAngina_Y: bool
    ST_Slope_Down: bool
    ST_Slope_Flat: bool
    ST_Slope_Up: bool


@app.get("/")
def inicio():
    return {"mensaje": "API de predicción de falla cardíaca activa"}


@app.post("/predict")
def predecir(paciente: PacienteInput):
    datos = pd.DataFrame([paciente.dict()])
    probabilidad = model.predict_proba(datos)[0][1]
    prediccion = int(probabilidad > 0.5)
    return {
        "prediccion": prediccion,
        "probabilidad_enfermedad": round(float(probabilidad), 3),
        "resultado": "Con enfermedad" if prediccion == 1 else "Sin enfermedad"
    }


@app.get("/health")
def health():
    return {"status": "ok"}