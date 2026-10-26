# Import Libraries
from pathlib import Path

import numpy as np
import pandas as pd
from evidently import BinaryClassification, DataDefinition, Dataset, Report
from evidently.metrics import Accuracy, F1Score, Precision, Recall
from evidently.presets import DataDriftPreset
from evidently.tests import Reference, gte
from skops.io import load

from generate_data import generate_transactions

# Load pre-trained model
MODEL_PATH = Path("models") / "mlflow_model.skops"

# Types skops cannot vouch for its own
# Review once with get_untrusted_types(file=MODEL_PATH)
# The load fails if the file ever contans more
TRUSTED_TYPES = ["sklearn.tree._tree.Tree"]

loaded = load(MODEL_PATH, trusted=TRUSTED_TYPES)
model = loaded["model"]
encoder = loaded["encoder"]
feature_cols = loaded["feature_cols"]

NUMERIC = ["amount", "hour", "day_of_week"]
CATEGORICAL = ["merchant_category"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Raw Columns -> exactly the columns (and order)
    the model was trained on.
    """
    encoded = pd.DataFrame(
        encoder.transform(df[CATEGORICAL]),
        columns=encoder.get_feature_names_out(),
        index=df.index,
    )
    return pd.concat(objs=[df[NUMERIC], encoded], axis=1)[feature_cols]


def add_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score a labelled batch with the same model
    and encoder the API uses.
    """
    out = df.copy()
    X = build_features(out)
    out["fraud_probability"] = model.predict_proba(X)[:, 1]
    out["prediction"] = model.predict(X)
    return out


perf_schema = DataDefinition(
    classification=[
        BinaryClassification(
            target="is_fraud",
            prediction_labels="prediction",
            prediction_probas="fraud_probability",
        )
    ],
    numerical_columns=NUMERIC + ["fraud_probability"],
    categorical_columns=["merchant_category", "is_fraud", "prediction"],
)


def to_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(data=df, data_definition=perf_schema)


def build_performance_report() -> Report:
    # One-sided: Fail only if the metric falls below 50% of the its
    # referenc value
    return Report(
        [
            Recall(conf_matrix=True, tests=[gte(Reference(relative=0.5))]),
            Precision(conf_matrix=True, tests=[gte(Reference(relative=0.5))]),
            F1Score(conf_matrix=True, tests=[gte(Reference(relative=0.5))]),
            Accuracy(),  # Reported, deliberately not tested: it hides the problem
        ]
    )


def summarize_tests(snapshot) -> dict:  # same helper as Step 4
    by_status = {}
    for t in snapshot.tests_results:
        by_status.setdefault(t.status.value, []).append(t)
    return {
        "passed": not by_status.get("FAIL") and not by_status.get("ERROR"),
        "counts": {k: len(v) for k, v in by_status.items()},
        "failures": [(t.name, t.description) for t in by_status.get("FAIL", [])],
    }


def metric_values(snapshot) -> dict:
    return {
        m["metric_name"].split("(")[0]: round(float(m["value"]), 4)
        for m in snapshot.dict()["metrics"]
    }


def generate_adapted_fraud(n_samples=10000, fraud_ratio=0.02, seed=42) -> pd.DataFrame:
    """
    Concept drift: legit traffic unchanged
    fraudsters switch to smaller amounts
    daytime hours and everyday merchants
    """
    np.random.seed(seed)
    cats = ["grocery", "restaurant", "retail", "online", "travel"]
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud
    legit = pd.DataFrame(
        {
            "amount": np.random.lognormal(3.5, 1.2, n_legit),
            "hour": np.random.randint(0, 24, n_legit),
            "day_of_week": np.random.randint(0, 7, n_legit),
            "merchant_category": np.random.choice(
                cats, n_legit, p=[0.30, 0.25, 0.25, 0.15, 0.05]
            ),
            "is_fraud": 0,
        }
    )
    fraud = pd.DataFrame(
        {
            "amount": np.random.lognormal(4.0, 1.2, n_fraud),
            "hour": np.random.randint(8, 22, n_fraud),
            "day_of_week": np.random.randint(0, 7, n_fraud),
            "merchant_category": np.random.choice(
                cats, n_fraud, p=[0.25, 0.25, 0.25, 0.15, 0.10]
            ),
            "is_fraud": 1,
        }
    )
    return (
        pd.concat(
            objs=[legit, fraud],
            ignore_index=True,
        )
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    reference_df = add_predictions(pd.read_csv("data/test.csv"))
    clean_df = add_predictions(
        generate_transactions(n_samples=10000, fraud_ratio=0.02, seed=101)
    )
    adapted_df = add_predictions(generate_adapted_fraud())
    reference = to_dataset(reference_df)

    for name, df in [("clean", clean_df), ("adapted", adapted_df)]:
        snap = build_performance_report().run(to_dataset(df), reference_data=reference)
        s = summarize_tests(snapshot=snap)
        print(f"{name:8s} passed={s['passed']} counts={s['counts']}")
        print("     ", metric_values(snap))
        for _, d in s["failures"]:
            print("     FAIL:", d)

    # The same batch through the Step 3/4 drift gate:
    cols = NUMERIC + CATEGORICAL
    drift_schema = DataDefinition(
        numerical_columns=NUMERIC, categorical_columns=CATEGORICAL
    )
    drift = Report(
        [
            DataDriftPreset(
                drift_share=0.25, num_method="wasserstein", cat_method="jensenshannon"
            )
        ],
        include_tests=True,
    ).run(
        Dataset.from_pandas(data=adapted_df[cols], data_definition=drift_schema),
        Dataset.from_pandas(data=reference_df[cols], data_definition=drift_schema),
    )
    print("\nadapted batch, DRIFT gate passed:", summarize_tests(drift)["passed"])
