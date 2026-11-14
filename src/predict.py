"""Predicting people the model has never seen (a file without the target column)."""
import os

import pandas as pd

from src.data import load_data
from src.preprocessing import add_engineered_features, clean_dataset, split_features_target


def predict_new_rows(pipeline, config: dict, out_path: str, threshold: float = 0.5) -> str:
    """
    Predicts every row of `config["data"]["predict_path"]` with the fitted `pipeline` and saves
    `<id>, probability, prediction` to `out_path`. Same cleaning, engineered features and
    feature split as the training data, but no duplicate removal and no row is ever dropped:
    rows out = rows in. Returns the path.
    """
    data_config = config["data"]
    id_column = config["diagnostics"]["id_column"]
    raw = load_data(data_config["predict_path"])

    cleaned = clean_dataset(raw, config["diagnostics"])
    cleaned = add_engineered_features(cleaned, config["preprocessing"].get("engineered_features", []))
    X_new, _, _ = split_features_target(cleaned, data_config,
                                        config["preprocessing"].get("mnar_indicator_sources", []))
    probability = pipeline.predict_proba(X_new)[:, 1]

    out = pd.DataFrame({id_column: raw[id_column].values, "probability": probability,
                        "prediction": (probability >= threshold).astype(int)})
    if len(out) != len(raw):
        raise RuntimeError(f"{len(raw)} rows in, {len(out)} rows out")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out.to_csv(out_path, index=False)
    return out_path