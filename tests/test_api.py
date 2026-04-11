import pytest
from fastapi.testclient import TestClient
from serving.app import app


client = TestClient(app)


def test_health():
    """Verifica que el health check funcione"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_returns_200():
    """Verifica que /predict responda correctamente"""
    payload = {
        "premium_amount": 150.0,
        "claim_amount": 50000,
        "age": 35,
        "tenure": 24,
        "no_of_family_members": 3,
        "incident_hour_of_the_day": 2,
        "incident_severity": "Major Loss",
        "any_injury": 0,
        "police_report_available": 0,
        "risk_segmentation": "H",
        "loss_dt": "2024-01-15",
        "report_dt": "2024-01-25",
        "txn_date_time": "2024-01-28",
        "policy_eff_dt": "2020-06-01",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


def test_predict_output_format():
    """Verifica que el output tenga el formato correcto"""
    payload = {
        "premium_amount": 150.0,
        "claim_amount": 50000,
        "age": 35,
        "tenure": 24,
        "no_of_family_members": 3,
        "incident_hour_of_the_day": 2,
        "incident_severity": "Major Loss",
        "any_injury": 0,
        "police_report_available": 0,
        "risk_segmentation": "H",
        "loss_dt": "2024-01-15",
        "report_dt": "2024-01-25",
        "txn_date_time": "2024-01-28",
        "policy_eff_dt": "2020-06-01",
    }
    response = client.post("/predict", json=payload)
    data = response.json()

    assert "prediction" in data
    assert "probability" in data
    assert "risk_level" in data
    assert data["prediction"] in [0, 1]
    assert 0 <= data["probability"] <= 1
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]


def test_predict_low_risk():
    """Verifica un caso de bajo riesgo"""
    payload = {
        "premium_amount": 100.0,
        "claim_amount": 500,
        "age": 40,
        "tenure": 60,
        "no_of_family_members": 2,
        "incident_hour_of_the_day": 14,
        "incident_severity": "Minor Loss",
        "any_injury": 0,
        "police_report_available": 1,
        "risk_segmentation": "L",
        "loss_dt": "2024-01-15",
        "report_dt": "2024-01-16",
        "txn_date_time": "2024-01-17",
        "policy_eff_dt": "2018-06-01",
    }
    response = client.post("/predict", json=payload)
    data = response.json()
    assert data["risk_level"] == "LOW"