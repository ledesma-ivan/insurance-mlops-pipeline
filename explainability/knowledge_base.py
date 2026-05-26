"""
Builds and persists the FAISS knowledge base for the RAG explainer.

Sources:
  1. Static domain docs (feature descriptions + fraud patterns)
  2. Dynamic feature importances loaded from the MLflow Production model
"""

import os
from typing import Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from explainability.domain_docs import FEATURE_DOCS

INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "explainability/faiss_index")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _load_feature_importance_docs(
    mlflow_tracking_uri: str, model_name: str
) -> list[Document]:
    """Loads feature importances from the Production model in MLflow Registry."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(mlflow_tracking_uri)
        client = MlflowClient()

        versions = client.get_latest_versions(model_name, stages=["Production"])
        if not versions:
            versions = client.get_latest_versions(model_name)
        if not versions:
            print("No MLflow model versions found — skipping importance docs.")
            return []

        run_id = versions[0].run_id
        run = client.get_run(run_id)
        metrics = run.data.metrics

        importance_lines = []
        feature_metrics = {k: v for k, v in metrics.items() if k.startswith("importance_")}

        if feature_metrics:
            ranked = sorted(feature_metrics.items(), key=lambda x: x[1], reverse=True)
            importance_lines = [
                f"  {rank + 1}. {k.replace('importance_', '')}: {v:.4f}"
                for rank, (k, v) in enumerate(ranked)
            ]
        else:
            importance_lines = [
                "  (importances not logged — run training/train.py to populate)"
            ]

        model_metrics = {
            k: v for k, v in metrics.items() if not k.startswith("importance_")
        }
        metrics_summary = ", ".join(
            f"{k}={v:.4f}" for k, v in model_metrics.items()
        )

        return [
            Document(
                page_content=(
                    f"Feature importances del modelo en producción (run_id={run_id}):\n"
                    + "\n".join(importance_lines)
                    + f"\n\nMétricas del modelo: {metrics_summary}"
                ),
                metadata={"type": "feature_importances", "run_id": run_id},
            )
        ]
    except Exception as exc:
        print(f"Could not load MLflow importances: {exc}")
        return []


def build_knowledge_base(
    mlflow_tracking_uri: str = "http://localhost:5000",
    model_name: str = "insurance-fraud-model",
    save: bool = True,
) -> FAISS:
    """Builds the FAISS index from domain docs + MLflow importances."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    docs = FEATURE_DOCS + _load_feature_importance_docs(mlflow_tracking_uri, model_name)
    print(f"Building FAISS index from {len(docs)} documents...")

    vectorstore = FAISS.from_documents(docs, embeddings)

    if save:
        os.makedirs(INDEX_PATH, exist_ok=True)
        vectorstore.save_local(INDEX_PATH)
        print(f"FAISS index saved to {INDEX_PATH}")

    return vectorstore


def load_knowledge_base(
    mlflow_tracking_uri: Optional[str] = None,
    model_name: str = "insurance-fraud-model",
) -> FAISS:
    """Loads existing FAISS index, or builds it if not found."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if os.path.exists(os.path.join(INDEX_PATH, "index.faiss")):
        print(f"Loading FAISS index from {INDEX_PATH}")
        return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

    uri = mlflow_tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    return build_knowledge_base(mlflow_tracking_uri=uri, model_name=model_name, save=True)


if __name__ == "__main__":
    import sys
    uri = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000"
    build_knowledge_base(mlflow_tracking_uri=uri)
    print("Knowledge base built successfully.")
