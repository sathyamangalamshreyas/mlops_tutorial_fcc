# Import Libraries
from pathlib import Path

import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.metrics import MaxValue, MinValue, MissingValueCount
from evidently.presets import DataDriftPreset
from evidently.tests import eq, gt, gte, lt, lte

# Create a Reports folder
project_root = Path(__file__).resolve().parent.parent

reports_path = project_root / "reports"

if not reports_path.exists():
    reports_path.mkdir(parents=True, exist_ok=True)

# Import data
reference_df = pd.read_csv("data/train.csv")
current_df = pd.read_csv("data/test.csv")

# Describe the columns once, reuse for both datasets
schema = DataDefinition(
    numerical_columns=["amount", "hour", "day_of_week"],
    categorical_columns=["merchant_category"],
)

reference = Dataset.from_pandas(data=reference_df, data_definition=schema)
current = Dataset.from_pandas(data=current_df, data_definition=schema)

report = Report([DataDriftPreset()])

# NOTE: The Order current first and reference second
my_eval = report.run(current_data=current, reference_data=reference)

my_eval.save_html("reports/drift_report_step1.html")


def parse_drift(result):
    """
    Flatten a drift snapshot into plain Python values
    """
    out = {"dataset": None, "columns": {}}
    for m in result.dict()["metrics"]:
        cfg = m["config"]
        kind = cfg["type"].split(":")[-1]
        if kind == "DriftedColumnsCount":
            out["dataset"] = {**m["value"], "threshold": cfg["drift_share"]}
        elif kind == "ValueDrift":
            out["columns"][cfg["column"]] = {
                "score": m["value"],
                "method": cfg["method"],
                "threshold": cfg["threshold"],
            }
    return out


print(parse_drift(result=my_eval))

# Prepare dataframe with inflated numbers
inflated_df = current_df.copy(deep=True)
inflated_df["amount"] *= 2

night_df = current_df.copy(deep=True)
night_df["hour"] = np.random.choice([0, 1, 2, 3, 22, 23], size=len(night_df))

for name, df in [("inflated", inflated_df), ("night", night_df)]:
    cur = Dataset.from_pandas(data=df, data_definition=schema)
    res = parse_drift(
        Report([DataDriftPreset()]).run(current_data=cur, reference_data=reference)
    )
    print(name, res["dataset"])
    for col, r in res["columns"].items():
        print(
            f"    {col:18s} score={r['score']:.3f} thr={r['threshold']} {r['method']}"
        )

DRIFT_SHARE = 0.25  # With 4 features: any single drifted columns raises the flag


# pin the drift methods and lower the dataset-level share
def build_drift_report() -> Report:
    return Report(
        [
            DataDriftPreset(
                drift_share=DRIFT_SHARE,
                num_method="wasserstein",
                cat_method="jensenshannon",
            )
        ],
        include_tests=True,
    )


# Evaluate the drift
def evaluate_drift(snapshot) -> dict:
    parsed = parse_drift(snapshot)
    ds = parsed["dataset"]
    # Valid because both pinned methods are distance-based
    # (drift = score >= threshold)
    drifted = [c for c, r in parsed["columns"].items() if r["score"] >= r["threshold"]]
    assert ds["count"] == len(drifted), (
        "column check disagrees with Evidently's own count"
    )
    return {
        "drifted_columns": drifted,
        "n_drifted": len(drifted),
        "n_features": len(parsed["columns"]),
        "drift_share": ds["share"],
        "dataset_drift": ds["share"] >= ds["threshold"],
    }


for name, curr_df in [
    ("clean", current_df),
    ("inflated", inflated_df),
    ("night", night_df),
]:
    cur = Dataset.from_pandas(curr_df, data_definition=schema)
    print(
        name,
        evaluate_drift(
            build_drift_report().run(current_data=cur, reference_data=reference)
        ),
    )


def build_quality_report() -> Report:
    return Report(
        [
            MinValue(column="amount", tests=[gt(0)]),  # ty: ignore[missing-argument, invalid-argument-type]
            MaxValue(column="amount", tests=[lt(50000)]),
            MinValue(column="hour", tests=[gte(0)]),
            MaxValue(column="hour", tests=[lte(23)]),
            MinValue(column="day_of_week", tests=[gte(0)]),
            MaxValue(column="day_of_week", tests=[lte(6)]),
            MissingValueCount(column="amount", tests=[eq(0, is_critical=False)]),
        ]
    )


def summarize_tests(snapshot) -> dict:
    by_status = {}
    for t in snapshot.tests_results:
        by_status.setdefault(t.status.value, []).append(t)
    return {
        "passed": not by_status.get("FAIL") and not by_status.get("ERROR"),
        "counts": {k: len(v) for k, v in by_status.items()},
        "failures": [(t.name, t.description) for t in by_status.get("FAIL", [])],
        "warnings": [(t.name, t.description) for t in by_status.get("WARNING", [])],
    }


# A deliberately corrupted batch: bad hours, a negative amount, some nulls
bad_df = current_df.copy()
bad_df.loc[bad_df.index[:50], "hour"] = 25
bad_df.loc[bad_df.index[:5], "amount"] = -10.0
bad_df.loc[bad_df.index[10:20], "amount"] = np.nan

for name, cur_df in [("clean", current_df), ("night", night_df), ("bad", bad_df)]:
    cur = Dataset.from_pandas(cur_df, data_definition=schema)
    for label, rep in [
        ("drift", build_drift_report()),
        ("quality", build_quality_report()),
    ]:
        s = summarize_tests(rep.run(current_data=cur, reference_data=reference))
        print(f"{name:6s} {label:8s} passed={s['passed']} counts={s['counts']}")
        for _, d in s["failures"]:
            print("     FAIL:", d)
        for _, d in s["warnings"]:
            print("     WARN:", d)
