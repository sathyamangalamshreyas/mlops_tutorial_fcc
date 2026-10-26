# Import Libraries
from pathlib import Path

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


def make_adapted_fraud(batch: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Concept drift that is independent of how the data was generated:
    legitimate rows are untouched; every fraud row gets the amount, hour and merchant
    of a randomly chosen legitimate row from the same batch (fraud now looks legitimate)."""
    out = batch.copy()
    fraud = out["is_fraud"] == 1
    donors = out[~fraud].sample(n=int(fraud.sum()), replace=True, random_state=seed)
    for col in ["amount", "hour", "merchant_category"]:
        out.loc[fraud, col] = donors[col].values
    return out


if __name__ == "__main__":
    reference_df = add_predictions(pd.read_csv("data/test.csv"))
    clean_df = add_predictions(
        generate_transactions(n_samples=10000, fraud_ratio=0.02, seed=101)
    )
    adapted_df = add_predictions(
        make_adapted_fraud(
            generate_transactions(n_samples=10000, fraud_ratio=0.02, seed=202)
        )
    )
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

#     def drift_check(df, reference_df):
#         cols = NUMERIC + CATEGORICAL
#         schema = DataDefinition(
#             numerical_columns=NUMERIC, categorical_columns=CATEGORICAL
#         )
#         snap = Report(
#             [
#                 DataDriftPreset(
#                     drift_share=0.25,
#                     num_method="wasserstein",
#                     cat_method="jensenshannon",
#                 )
#             ],
#             include_tests=True,
#         ).run(
#             Dataset.from_pandas(df[cols], data_definition=schema),
#             Dataset.from_pandas(reference_df[cols], data_definition=schema),
#         )
#         scores = {
#             m["config"]["column"]: round(m["value"], 3)
#             for m in snap.dict()["metrics"]
#             if "column" in m["config"]
#         }
#         return summarize_tests(snap)["passed"], scores
#
#     print(
#         "reference rows:",
#         len(reference_df),
#         "| frauds:",
#         int(reference_df.is_fraud.sum()),
#         "| amount std:",
#         round(reference_df.amount.std(), 1),
#     )
#     for name, df in [("clean", clean_df), ("adapted", adapted_df)]:
#         passed, scores = drift_check(df, reference_df)
#         print(f"{name:8s} drift gate passed={passed}  scores={scores}")
