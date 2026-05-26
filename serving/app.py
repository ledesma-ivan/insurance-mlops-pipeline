import os
import sys

import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, os.getenv("PROJECT_ROOT", "."))

app = FastAPI(
    title="Insurance Fraud Detection API",
    description="Fraud detection + RAG explainability layer for insurance claims.",
)


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
    loss_dt: str
    report_dt: str
    txn_date_time: str
    policy_eff_dt: str


# --- Output schemas ---
class PredictionOutput(BaseModel):
    prediction: int
    probability: float
    risk_level: str


class ExplainOutput(BaseModel):
    prediction: int
    probability: float
    risk_level: str
    explanation: str


def create_features_from_input(data: ClaimInput) -> pd.DataFrame:
    """Transforma datos crudos en features para el modelo"""
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


# --- Cargar modelo al iniciar ---
MODEL_PATH = os.getenv("MODEL_PATH", "models/latest/model.json")
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)
print(f"✅ Model loaded from {MODEL_PATH}")

# --- Cargar RAG explainer (lazy: se inicializa en el primer /explain) ---
_explainer = None


def _get_explainer():
    global _explainer
    if _explainer is None:
        try:
            from explainability.knowledge_base import load_knowledge_base
            from explainability.rag_explainer import ClaimExplainer
            vectorstore = load_knowledge_base()
            _explainer = ClaimExplainer(vectorstore)
            print("✅ RAG explainer ready")
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Explainer unavailable: {exc}")
    return _explainer


@app.get("/health")
def health():
    return {"status": "healthy", "model_path": MODEL_PATH}


@app.post("/predict", response_model=PredictionOutput)
def predict(claim: ClaimInput):
    features_df = create_features_from_input(claim)

    prediction = int(model.predict(features_df)[0])
    probability = float(model.predict_proba(features_df)[0][1])

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


@app.post("/explain", response_model=ExplainOutput)
def explain(claim: ClaimInput):
    """
    Predicts fraud risk and explains the decision using RAG over MLflow logs
    and feature importance knowledge base.
    Requires ANTHROPIC_API_KEY for LLM-generated explanations; falls back to
    rule-based text if the key is absent.
    """
    features_df = create_features_from_input(claim)
    features_dict = features_df.iloc[0].to_dict()

    prediction = int(model.predict(features_df)[0])
    probability = float(model.predict_proba(features_df)[0][1])

    if probability >= 0.75:
        risk_level = "HIGH"
    elif probability >= 0.5:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    explainer = _get_explainer()
    explanation = explainer.explain(features_dict, risk_level, probability)

    return ExplainOutput(
        prediction=prediction,
        probability=round(probability, 4),
        risk_level=risk_level,
        explanation=explanation,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
