from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import mlflow
import pandas as pd

app = FastAPI(title="Insurance Fraud Detection API")


# --- Input schema ---
class ClaimInput(BaseModel):
    premium_amount: float
    claim_amount: float
    age: int
    tenure: int
    no_of_family_members: int
    incident_hour_of_the_day: int
    incident_severity: str
    any_injury: int
    police_report_available: int
    risk_segmentation: str
    loss_dt: str          # "2024-01-15"
    report_dt: str        # "2024-01-18"
    txn_date_time: str    # "2024-01-20"
    policy_eff_dt: str    # "2020-06-01"


# --- Output schema ---
class PredictionOutput(BaseModel):
    prediction: int
    probability: float
    risk_level: str


def create_features_from_input(data: ClaimInput) -> pd.DataFrame:
    """Transforma datos crudos en features para el modelo"""
    # Parsear fechas
    loss_dt = pd.to_datetime(data.loss_dt)
    report_dt = pd.to_datetime(data.report_dt)
    txn_date_time = pd.to_datetime(data.txn_date_time)
    policy_eff_dt = pd.to_datetime(data.policy_eff_dt)

    features = {
        "PREMIUM_AMOUNT": data.premium_amount,
        "CLAIM_AMOUNT": data.claim_amount,
        "AGE": data.age,
        "TENURE": data.tenure,
        "NO_OF_FAMILY_MEMBERS": data.no_of_family_members,
        "INCIDENT_HOUR_OF_THE_DAY": data.incident_hour_of_the_day,
        "claim_to_premium_ratio": data.claim_amount / max(data.premium_amount, 1),
        "is_high_claim": int(data.claim_amount > 30000),
        "days_loss_to_report": (report_dt - loss_dt).days,
        "days_report_to_txn": (txn_date_time - report_dt).days,
        "policy_age_days": (loss_dt - policy_eff_dt).days,
        "is_night_incident": int(data.incident_hour_of_the_day in [0, 1, 2, 3, 4, 5]),
        "is_major_loss": int(data.incident_severity == "Major Loss"),
        "no_police_report": int(data.police_report_available == 0),
        "no_injury_high_claim": int(data.any_injury == 0 and data.claim_amount > 30000),
        "risk_encoded": {"L": 0, "M": 1, "H": 2}.get(data.risk_segmentation, 0),
    }
    return pd.DataFrame([features])


@app.get("/health")
def health():
    return {"status": "healthy"}

# --- Cargar modelo al iniciar la API ---
MODEL_NAME = "insurance-fraud-model"
model = mlflow.xgboost.load_model(f"models:/{MODEL_NAME}/latest")
print(f"✅ Model loaded: {MODEL_NAME}")


@app.post("/predict", response_model=PredictionOutput)
def predict(claim: ClaimInput):
    # 1. Crear features desde datos crudos
    features_df = create_features_from_input(claim)

    # 2. Predecir
    prediction = int(model.predict(features_df)[0])
    probability = float(model.predict_proba(features_df)[0][1])

    # 3. Asignar nivel de riesgo
    if probability >= 0.75:
        risk_level = "HIGH"
    elif probability >= 0.5:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return PredictionOutput(
        prediction=prediction,
        probability=round(probability, 4),
        risk_level=risk_level,
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)