import numpy as np
import pandas as pd


def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Calcula Population Stability Index (PSI) entre dos distribuciones.
    expected: datos de entrenamiento
    actual: datos nuevos (producción)
    """
    # 1. Crear buckets basados en la distribución de entrenamiento
    breakpoints = np.quantile(expected, np.linspace(0, 1, bins + 1))
    breakpoints = np.unique(breakpoints)

    # 2. Calcular porcentaje en cada bucket
    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    # 3. Convertir a proporciones (evitar zeros)
    expected_pct = (expected_counts + 1) / (len(expected) + len(breakpoints))
    actual_pct = (actual_counts + 1) / (len(actual) + len(breakpoints))

    # 4. Calcular PSI
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))

    return float(psi)


def detect_drift(
    train_df: pd.DataFrame,
    new_df: pd.DataFrame,
    features: list,
    threshold: float = 0.25,
) -> dict:
    """
    Detecta drift en cada feature comparando datos de training vs nuevos.
    """
    results = {}

    for feature in features:
        psi_value = calculate_psi(
            train_df[feature].values,
            new_df[feature].values,
        )

        if psi_value > threshold:
            status = "🔴 DRIFT"
        elif psi_value > 0.1:
            status = "🟡 WARNING"
        else:
            status = "🟢 OK"

        results[feature] = {
            "psi": round(psi_value, 4),
            "status": status,
        }

    return results


if __name__ == "__main__":
    from feature_store.feature_engineering import prepare_dataset

    X_train, X_test, y_train, y_test = prepare_dataset()

    # Simular drift: modificar datos de test como si fueran "datos nuevos"
    X_new = X_test.copy()
    X_new["CLAIM_AMOUNT"] = X_new["CLAIM_AMOUNT"] * 2  # Inflación duplicó claims
    X_new["AGE"] = X_new["AGE"] + 10  # Población envejeció

    print("=" * 60)
    print("📊 DRIFT DETECTION REPORT")
    print("=" * 60)

    results = detect_drift(X_train, X_new, X_train.columns.tolist())

    for feature, info in results.items():
        print(f"   {feature:35s} PSI={info['psi']:.4f}  {info['status']}")

    drifted = [f for f, info in results.items() if "DRIFT" in info["status"]]
    print(f"\n⚠️ Features with drift: {len(drifted)}")
    if drifted:
        print(f"   {drifted}")
