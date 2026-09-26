"""
Entry point for the baseline predictive pipeline.

Run with:
    python main.py

This orchestrates the pipeline:
    load config -> load data -> clean dataset -> preprocess -> split
    -> train -> evaluate (train & test) -> save results

Week 3 introduces dataset cleaning before the existing baseline
preprocessing stage, allowing us to compare model performance
before and after cleaning.

How it works: (Week 3)
    Load configuration
            ↓
    Load raw dataset
            ↓
    clean_dataset()             ← Week 3 addition
            ↓
    preprocess()                ← Existing Week 2 function
            ↓
    Train/test split
            ↓
    Train model
            ↓
    Evaluate performance
            ↓
    Evaluate fairness
            ↓
    Save results
"""

import yaml

from src.data import load_data
from src.preprocessing import clean_dataset, preprocess
from src.model import build_model
from src.evaluate import evaluate, fairness_report
from src.results import save_run


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    # Load raw dataset
    df_raw = load_data(config["data"]["path"])

    # Week 3: clean the dataset before preprocessing
    df_clean = clean_dataset(df_raw, config["diagnostics"])

    # Baseline preprocessing and train/test split
    X_train, X_test, y_train, y_test, extras_test = preprocess(
        df_clean,
        target=config["data"]["target"],
        sensitive_attr=config["data"]["sensitive_attr"],
        drop_columns=config["data"]["drop_columns"],
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
    )

    # Build and train model
    model = build_model(config["model"])
    model.fit(X_train, y_train)

    # Predict on both splits to monitor overfitting
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Evaluate predictive performance
    report = evaluate(y_train, y_train_pred, y_test, y_test_pred)

    # Evaluate fairness on the test set
    report += "\n" + fairness_report(
        y_test,
        y_test_pred,
        extras_test,
        sensitive_attr=config["data"]["sensitive_attr"],
    )

    # Save results
    model_name = model.__class__.__name__.lower()

    if "tree" in model_name:
        model_type = "tree"
    elif "logistic" in model_name:
        model_type = "lr"

    results_dir = config.get("output", {}).get("results_dir", "results")
    results_dir = f"{results_dir}/{model_type}"

    path = save_run(results_dir, config, report)

    print(f"Full results saved to {path}")

if __name__ == "__main__":
    main()