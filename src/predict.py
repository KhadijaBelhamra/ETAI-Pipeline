"""
Predicts new data with the model saved by `main.py`:

    python predict.py data/compas_new_people.csv predictions.csv
    python predict.py data/adult_new.csv adult_predictions.csv --config config_adult.yaml

new CSV -> the same cleaning as training (no rows dropped) -> engineered features -> the saved
pipeline -> one row per input row: id, probability, prediction.
"""
import argparse

import joblib
import pandas as pd
import yaml

from src.preprocessing import add_engineered_features, clean_dataset, split_features_target


def predict(df: pd.DataFrame, pipeline, config: dict, threshold: float = 0.5) -> pd.DataFrame:
    """
    Returns the id, the probability of the positive class and the prediction (at `threshold`) for
    every row of `df`. Rows are never dropped (no duplicate removal): rows out = rows in.
    """
    clean = clean_dataset(df, config["diagnostics"])
    clean = add_engineered_features(clean, config["preprocessing"].get("engineered_features", []))
    X, _, _ = split_features_target(clean, config["data"], config["preprocessing"].get("mnar_indicator_sources", []))
    probability = pipeline.predict_proba(X)[:, 1]
    predictions = pd.DataFrame({"id": df[config["diagnostics"]["id_column"]].values,
                                "probability": probability,
                                "prediction": (probability >= threshold).astype(int)})
    if len(predictions) != len(df):
        raise RuntimeError(f"{len(df)} rows in, {len(predictions)} rows out")
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a CSV with the model saved by main.py.")
    parser.add_argument("input", help="CSV with the same columns as the training data (no target needed)")
    parser.add_argument("output", nargs="?", default="predictions.csv", help="where to write the predictions")
    parser.add_argument("--config", default="config.yaml", help="the config the model was trained with")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    pipeline = joblib.load(config["output"]["model_path"])
    predictions = predict(pd.read_csv(args.input), pipeline, config)
    predictions.to_csv(args.output, index=False)
    print(f"{len(predictions)} rows predicted -> {args.output}")