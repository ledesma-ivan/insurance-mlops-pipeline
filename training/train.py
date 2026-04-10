import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from feature_store.feature_engineering import prepare_dataset


def train():
    # 1. Cargar datos
    X_train, X_test, y_train, y_test = prepare_dataset()

    # 2. Calcular scale_pos_weight
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"⚖️ scale_pos_weight: {scale_pos_weight:.2f}")

    # 3. Definir hiperparámetros
    params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "aucpr",
        "random_state": 42,
    }

    # 4. Entrenar con MLflow tracking
    mlflow.set_experiment("insurance-fraud-detection")

    with mlflow.start_run(run_name="xgboost-baseline"):
        # Log parameters
        mlflow.log_params(params)

        # Train
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Calculate metrics
        metrics = {
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred),
        }

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log model
        mlflow.xgboost.log_model(model, artifact_path="model")

        # Register model in MLflow Registry
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"
        mlflow.register_model(model_uri, "insurance-fraud-model")
        print(f"📦 Model registered as 'insurance-fraud-model'")

        # Print results
        print("\n📊 Metrics:")
        for name, value in metrics.items():
            print(f"   {name}: {value:.4f}")

        print(f"\n📝 Classification Report:\n")
        print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

        print(f"✅ Run logged to MLflow: {mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    train()