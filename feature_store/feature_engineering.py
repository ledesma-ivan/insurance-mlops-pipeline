import pandas as pd
import numpy as np


def load_raw_data(
        # Cargamos los datos desde los CSVs (en un proyecto real, podrían venir de una base de datos o data lake)
    insurance_path: str = "data/raw/insurance_data.csv",
    employee_path: str = "data/raw/employee_data.csv",
    vendor_path: str = "data/raw/vendor_data.csv",
) -> pd.DataFrame:
    """Load and join all tables"""
    insurance = pd.read_csv(insurance_path)
    employee = pd.read_csv(employee_path)
    vendor = pd.read_csv(vendor_path)

    # Join tables
    df = insurance.merge(employee, on="AGENT_ID", how="left", suffixes=("", "_agent"))
    df = df.merge(vendor, on="VENDOR_ID", how="left", suffixes=("", "_vendor"))

    print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features for fraud detection"""
    features = df.copy()

    # Parse dates
    features["TXN_DATE_TIME"] = pd.to_datetime(features["TXN_DATE_TIME"])
    features["LOSS_DT"] = pd.to_datetime(features["LOSS_DT"])
    features["REPORT_DT"] = pd.to_datetime(features["REPORT_DT"])
    features["POLICY_EFF_DT"] = pd.to_datetime(features["POLICY_EFF_DT"])

    # === Features de montos ===
    # Ratio claim vs premium (indicador clásico de fraude)
    features["claim_to_premium_ratio"] = (
        features["CLAIM_AMOUNT"] /
        features["PREMIUM_AMOUNT"].replace(0, 1)
    )

    # Claim amount alto (flag)
    features["is_high_claim"] = (
        features["CLAIM_AMOUNT"] > features["CLAIM_AMOUNT"].quantile(0.75)
    ).astype(int)

    # === Features temporales ===
    # Días entre pérdida y reporte (fraude suele reportar tarde)
    features["days_loss_to_report"] = (
        features["REPORT_DT"] - features["LOSS_DT"]
    ).dt.days

    # Días entre reporte y transacción
    features["days_report_to_txn"] = (
        features["TXN_DATE_TIME"] - features["REPORT_DT"]
    ).dt.days

    # Antigüedad de la póliza al momento del claim
    features["policy_age_days"] = (
        features["LOSS_DT"] - features["POLICY_EFF_DT"]
    ).dt.days

    # Hora del incidente (madrugada = sospechoso)
    features["is_night_incident"] = (
        features["INCIDENT_HOUR_OF_THE_DAY"].isin([0, 1, 2, 3, 4, 5])
    ).astype(int)

    # === Features de riesgo ===
    # Severidad alta
    features["is_major_loss"] = (
        features["INCIDENT_SEVERITY"] == "Major Loss"
    ).astype(int)

    # Sin reporte policial (sospechoso)
    features["no_police_report"] = (
        features["POLICE_REPORT_AVAILABLE"] == 0
    ).astype(int)

    # Sin lesiones pero claim alto
    features["no_injury_high_claim"] = (
        (features["ANY_INJURY"] == 0) &
        (features["is_high_claim"] == 1)
    ).astype(int)

    # === Features socioeconómicas ===
    # Encoding de risk segmentation
    risk_map = {"L": 0, "M": 1, "H": 2}
    features["risk_encoded"] = features["RISK_SEGMENTATION"].map(risk_map).fillna(0)

    # === Target ===
    # CLAIM_STATUS: A = Approved, D = Denied (posible fraude)
    features["target"] = (features["CLAIM_STATUS"] == "D").astype(int)

    print(f"✅ Features created: {features.shape[1]} columns")
    return features


def select_model_features(df: pd.DataFrame) -> list:
    """Define which features the model uses"""
    numeric_features = [
        # Originales
        "PREMIUM_AMOUNT",
        "CLAIM_AMOUNT",
        "AGE",
        "TENURE",
        "NO_OF_FAMILY_MEMBERS",
        "INCIDENT_HOUR_OF_THE_DAY",
        # Creadas
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
    ]
    return numeric_features


def prepare_dataset(
    insurance_path: str = "data/raw/insurance_data.csv",
    employee_path: str = "data/raw/employee_data.csv",
    vendor_path: str = "data/raw/vendor_data.csv",
) -> tuple:
    """Full pipeline: load → join → create features → select → split"""
    from sklearn.model_selection import train_test_split

    df = load_raw_data(insurance_path, employee_path, vendor_path)
    df = create_features(df)

    feature_cols = select_model_features(df)
    X = df[feature_cols].fillna(0)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"✅ Dataset split: train={len(X_train)}, test={len(X_test)}")
    print(f"   Target distribution: {y.value_counts().to_dict()}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_dataset()