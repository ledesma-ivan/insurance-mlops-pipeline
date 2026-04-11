import pytest
import pandas as pd
from feature_store.feature_engineering import (
    load_raw_data,
    create_features,
    select_model_features,
    prepare_dataset,
)


def test_load_raw_data():
    """Verifica que los datos se cargan correctamente"""
    df = load_raw_data()
    assert df.shape[0] == 10000, "Debe tener 10000 filas"
    assert df.shape[1] > 38, "Debe tener más columnas después del join"


def test_create_features():
    """Verifica que las features se crean correctamente"""
    df = load_raw_data()
    df = create_features(df)

    expected_features = [
        "claim_to_premium_ratio",
        "is_high_claim",
        "days_loss_to_report",
        "days_report_to_txn",
        "policy_age_days",
        "is_night_incident",
        "is_major_loss",
        "no_police_report",
        "no_injury_high_claim",
        "risk_encoded",
        "target",
    ]

    for feature in expected_features:
        assert feature in df.columns, f"Falta feature: {feature}"


def test_target_is_binary():
    """Verifica que el target sea 0 o 1"""
    df = load_raw_data()
    df = create_features(df)
    assert set(df["target"].unique()) == {0, 1}, "Target debe ser binario"


def test_prepare_dataset_shapes():
    """Verifica que el split sea correcto"""
    X_train, X_test, y_train, y_test = prepare_dataset()

    assert len(X_train) == 8000, "Train debe tener 8000 filas"
    assert len(X_test) == 2000, "Test debe tener 2000 filas"
    assert X_train.shape[1] == 16, "Debe tener 16 features"


def test_no_nulls_in_features():
    """Verifica que no haya nulls en las features del modelo"""
    X_train, X_test, _, _ = prepare_dataset()
    assert X_train.isnull().sum().sum() == 0, "No debe haber nulls en train"
    assert X_test.isnull().sum().sum() == 0, "No debe haber nulls en test"