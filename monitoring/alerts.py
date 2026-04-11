import json
from datetime import datetime

from monitoring.drift import detect_drift


def check_and_alert(
    train_df,
    new_df,
    features: list,
    threshold: float = 0.25,
) -> dict:
    """
    Corre drift detection y decide si hay que reentrenar.
    """
    # 1. Detectar drift
    results = detect_drift(train_df, new_df, features, threshold)

    # 2. Identificar features con drift
    drifted = [f for f, info in results.items() if "DRIFT" in info["status"]]

    # 3. Crear alerta
    alert = {
        "timestamp": datetime.now().isoformat(),
        "total_features": len(features),
        "drifted_features": len(drifted),
        "drifted_names": drifted,
        "needs_retraining": len(drifted) > 0,
        "details": results,
    }

    # 4. Loguear alerta
    log_alert(alert)

    # 5. Trigger retraining si es necesario
    if alert["needs_retraining"]:
        trigger_retraining(alert)

    return alert


def log_alert(alert: dict):
    """Guarda la alerta en un archivo JSON"""
    print("\n" + "=" * 60)
    print("🚨 DRIFT ALERT REPORT")
    print("=" * 60)
    print(f"   Timestamp: {alert['timestamp']}")
    print(f"   Features checked: {alert['total_features']}")
    print(f"   Features drifted: {alert['drifted_features']}")
    print(f"   Needs retraining: {alert['needs_retraining']}")

    if alert["drifted_names"]:
        print(f"   Drifted: {alert['drifted_names']}")

    # Guardar a archivo
    with open("monitoring/alerts_log.json", "a") as f:
        f.write(json.dumps(alert, default=str) + "\n")
    print("\n   📁 Alert saved to monitoring/alerts_log.json")


def trigger_retraining(alert: dict):
    """Dispara reentrenamiento automático"""
    print("\n🔄 TRIGGERING AUTOMATIC RETRAINING...")
    print(f"   Reason: drift detected in {alert['drifted_names']}")

    from training.train import train

    train()

    print("✅ Retraining complete!")


if __name__ == "__main__":
    from feature_store.feature_engineering import prepare_dataset

    X_train, X_test, y_train, y_test = prepare_dataset()

    # Simular datos nuevos con drift
    X_new = X_test.copy()
    X_new["CLAIM_AMOUNT"] = X_new["CLAIM_AMOUNT"] * 2
    X_new["AGE"] = X_new["AGE"] + 10

    alert = check_and_alert(X_train, X_new, X_train.columns.tolist())
