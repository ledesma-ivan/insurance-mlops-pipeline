import pandas as pd
import os

FEATURE_STORE_PATH = "data/processed/feature_store.parquet"


def save_features(df: pd.DataFrame, path: str = FEATURE_STORE_PATH):
    """Guarda features procesadas en el Feature Store (Parquet)"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"✅ Features saved to Feature Store: {path} ({df.shape[0]} rows, {df.shape[1]} cols)")


def load_features(path: str = FEATURE_STORE_PATH) -> pd.DataFrame:
    """Carga features desde el Feature Store"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature Store not found at {path}. Run feature engineering first.")
    df = pd.read_parquet(path)
    print(f"✅ Features loaded from Feature Store: {df.shape[0]} rows, {df.shape[1]} cols")
    return df


def get_feature_metadata(path: str = FEATURE_STORE_PATH) -> dict:
    """Devuelve metadata del Feature Store"""
    df = pd.read_parquet(path)
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "feature_names": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "file_size_mb": round(os.path.getsize(path) / (1024 * 1024), 2),
    }


if __name__ == "__main__":
    from feature_store.feature_engineering import load_raw_data, create_features

    # 1. Crear features
    df = load_raw_data()
    df = create_features(df)

    # 2. Guardar en Feature Store
    save_features(df)

    # 3. Verificar que se puede leer
    df_loaded = load_features()

    # 4. Mostrar metadata
    metadata = get_feature_metadata()
    print(f"\n📋 Feature Store Metadata:")
    for key, value in metadata.items():
        if key != "dtypes":
            print(f"   {key}: {value}")