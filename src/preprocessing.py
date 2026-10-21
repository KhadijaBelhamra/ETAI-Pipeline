"""
Preprocessing: from raw data to model-ready data.

Two kinds of steps, kept apart on purpose:
  - Learn-nothing steps (`clean_dataset`, `split_features_target`): safe on the whole
    dataset, and they work on label-free inference data (`y` is then `None`).
  - Learned steps (imputation, encoding, scaling): live inside `build_preprocessor`'s
    ColumnTransformer, which sits in the model Pipeline. They are re-fitted on each CV
    fold's training part and never see the validation fold or the locked test set.

`sensitive_attr` (race) is kept out of the model features and returned separately,
so it can be used for fairness auditing (see src/evaluate.py).
"""
import numpy as np
import pandas as pd
from category_encoders import CountEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler, TargetEncoder,
)


def flag_invalid_values(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """
    Apply `{column: {"min": ..., "max": ...}}` domain rules (either bound optional) and
    set violations to NaN **in place** on `df`.

    Returns a small report with the number of violations per column.
    """
    report_rows = []
    for column, bounds in rules.items():
        if column not in df.columns:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        lower_ok = numeric >= bounds["min"] if "min" in bounds else pd.Series(True, index=numeric.index)
        upper_ok = numeric <= bounds["max"] if "max" in bounds else pd.Series(True, index=numeric.index)
        violations = numeric.notna() & ~(lower_ok & upper_ok)
        report_rows.append({"column": column, "rule": bounds, "violations": int(violations.sum())})
        df.loc[violations, column] = np.nan
    return pd.DataFrame(report_rows)


def _canonicalize_categories(df: pd.DataFrame, columns_and_maps: dict, placeholder_tokens: set) -> pd.DataFrame:
    """Strip, lowercase-match and map category labels; placeholder tokens become NaN."""
    out = df.copy()
    for col, mapping in columns_and_maps.items():
        if col not in out.columns:
            continue
        cleaned = out[col].astype(str).str.strip()
        lowered = cleaned.str.lower()
        out[col] = lowered.map(mapping).fillna(cleaned)
        out.loc[out[col].astype(str).str.strip().isin(placeholder_tokens), col] = np.nan
    return out


def clean_dataset(df: pd.DataFrame, diagnostics_config: dict) -> pd.DataFrame:
    """
    Config-driven cleaning: numeric-text conversion, domain-rule violations -> NaN,
    category standardisation, and redundant-column removal.

    Row-preserving (every input row comes out, in order) and target-agnostic, so it is
    safe on inference data. Duplicate removal is training-only: see `drop_duplicate_rows`.
    """
    out = df.copy()
    placeholder_tokens = set(diagnostics_config.get("placeholder_tokens", []))

    for col in diagnostics_config.get("numeric_text_columns", []):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col].replace(list(placeholder_tokens), np.nan), errors="coerce")

    flag_invalid_values(out, diagnostics_config.get("validity_rules", {}))

    out = _canonicalize_categories(out, diagnostics_config.get("canonical_categories", {}), placeholder_tokens)

    columns_to_drop = [c for c in diagnostics_config.get("redundant_columns", []) if c in out.columns]
    return out.drop(columns=columns_to_drop)


def drop_duplicate_rows(df: pd.DataFrame, id_column: str = None) -> pd.DataFrame:
    """
    TRAINING DATA ONLY. Drop exact duplicate rows and repeated ids (keeping the first) so
    the same person is not counted twice or split across dev and test. Run before
    `split_dev_test`; never on data you must predict, where every row needs a prediction.
    """
    out = df.drop_duplicates()
    if id_column and id_column in out.columns:
        out = out.drop_duplicates(subset=id_column, keep="first")
    return out


def add_missingness_indicators(df: pd.DataFrame, mnar_indicator_sources: list) -> pd.DataFrame:
    """Add a `<col>_was_missing` flag for each listed column, before it is imputed."""
    out = df.copy()
    for col in mnar_indicator_sources:
        if col in out.columns:
            out[f"{col}_was_missing"] = out[col].isna().astype(int)
    return out


def _flag(condition: pd.Series, *sources: pd.Series) -> pd.Series:
    """0/1 flag that stays NaN wherever any of the source values is NaN (so it gets imputed)."""
    known = pd.concat(sources, axis=1).notna().all(axis=1)
    return condition.astype(float).where(known)


# Engineered features: each one is a row-wise function of the cleaned columns. Nothing is
# learned from the data (no means, no quantiles), so they are leak-safe on the whole dataset
# and also work on label-free inference data. NaN inputs give NaN outputs; the numeric
# pipeline imputes them later, inside each CV fold.
_ENGINEERED = {
    # recidivism is strongly skewed towards few priors: log compresses the long tail
    "log_priors": lambda d: np.log1p(d["priors_count"]),
    # any prior offence at all (separates "none" from "some" better than the raw count)
    "has_priors": lambda d: _flag(d["priors_count"] > 0, d["priors_count"]),
    # offence rate: priors per year of adult life (age 18 = 1, so no division by zero)
    "priors_per_adult_year": lambda d: d["priors_count"] / (d["age"] - 17).clip(lower=1),
    # any juvenile record, and a juvenile felony specifically
    "has_juvenile_record": lambda d: _flag(
        d[["juv_fel_count", "juv_misd_count", "juv_other_count"]].sum(axis=1, min_count=3) > 0,
        d[["juv_fel_count", "juv_misd_count", "juv_other_count"]].sum(axis=1, min_count=3)),
    "has_juvenile_felony": lambda d: _flag(d["juv_fel_count"] > 0, d["juv_fel_count"]),
    # young AND already several priors: an interaction that linear models cannot form alone
    "young_with_priors": lambda d: _flag((d["age"] < 25) & (d["priors_count"] >= 2),
                                         d["age"], d["priors_count"]),
}


def add_engineered_features(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """
    Add the engineered features named in config.yaml's `preprocessing.engineered_features`.
    Run after `clean_dataset` (it needs clean numeric columns). `build_preprocessor` adds the
    same names to the numeric features, so one list in the YAML switches them on or off.
    """
    unknown = [n for n in feature_names if n not in _ENGINEERED]
    if unknown:
        raise ValueError(f"Unknown engineered features: {unknown}. Options: {list(_ENGINEERED)}")
    out = df.copy()
    for name in feature_names:
        out[name] = _ENGINEERED[name](out)
    return out


def split_features_target(df: pd.DataFrame, data_config: dict, mnar_indicator_sources: list):
    """
    Return (X, y, extras). `extras` holds the sensitive attribute and COMPAS's score_text
    for the fairness report. `y` is `None` when the target column is absent.
    """
    target = data_config["target"]
    sensitive_attr = data_config["sensitive_attr"]
    drop_columns = data_config.get("drop_columns", [])

    df = add_missingness_indicators(df, mnar_indicator_sources)
    y = df[target] if target in df.columns else None

    extras_cols = [c for c in [sensitive_attr, "score_text"] if c in df.columns]
    extras = df[extras_cols].copy() if extras_cols else None

    always_drop = set(drop_columns) | {target, sensitive_attr}
    feature_cols = [c for c in df.columns if c not in always_drop]
    X = df[feature_cols]
    return X, y, extras


_SCALERS = {"none": "passthrough", "standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}

# Each factory takes a seed (only the target encoder uses it). sklearn's TargetEncoder
# cross-fits during fit, so a row's own label never leaks into its own encoded value.
_ENCODERS = {
    "onehot": lambda seed: OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    "ordinal": lambda seed: OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
    "count": lambda seed: CountEncoder(handle_unknown=0, handle_missing=0),
    "target": lambda seed: TargetEncoder(target_type="binary", cv=5, random_state=seed),
}


def build_preprocessor(preprocessing_config: dict) -> ColumnTransformer:
    """
    Build an unfitted ColumnTransformer (impute -> scale numerics, impute -> encode
    categoricals, pass missingness flags through) from config.yaml's `preprocessing`
    section. Fitting happens inside the model Pipeline, on training data only.
    """
    encoder_name = preprocessing_config["encoder"]
    scaler_name = preprocessing_config["scaler"]
    numeric_features = list(preprocessing_config["numeric_features"])
    numeric_features += [f for f in preprocessing_config.get("engineered_features", [])
                         if f not in numeric_features]
    categorical_features = preprocessing_config["categorical_features"]
    mnar_indicator_sources = preprocessing_config.get("mnar_indicator_sources", [])
    imputation = preprocessing_config.get("imputation", {})

    scaler_factory = _SCALERS[scaler_name]
    scaler = scaler_factory() if callable(scaler_factory) else scaler_factory
    encoder = _ENCODERS[encoder_name](preprocessing_config.get("random_state"))

    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy=imputation.get("numeric_strategy", "median"))),
        ("scale", scaler),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy=imputation.get("categorical_strategy", "most_frequent"))),
        ("encode", encoder),
    ])

    indicator_cols = [f"{c}_was_missing" for c in mnar_indicator_sources]

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
        ("indicators", "passthrough", indicator_cols),
    ])


def split_dev_test(X, y, extras, test_size: float, random_state: int):
    """
    Stratified split into a development set and a locked test set, keeping X, y and
    `extras` row-aligned. The test set is never used to fit, tune or choose anything;
    its size and seed live in config.yaml's `test_set` section.
    """
    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = train_test_split(
        X, y, extras, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_dev, X_test, y_dev, y_test, extras_dev, extras_test