from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset
from evidently.sdk.models import PanelMetric
from evidently.sdk.panels import DashboardPanelPlot
from evidently.ui.workspace import Workspace

from evidently_step_2 import (
    CATEGORICAL,
    NUMERIC,
    add_predictions,
    build_performance_report,
    make_adapted_fraud,
    to_dataset,
)
from generate_data import generate_transactions

WORKSPACE_PATH = "evidently_workspace"
PROJECT_NAME = "fraud-detection-monitoring"
COLUMNS = NUMERIC + CATEGORICAL
drift_schema = DataDefinition(
    numerical_columns=NUMERIC, categorical_columns=CATEGORICAL
)


def build_drift_report() -> Report:
    return Report(
        [
            DataDriftPreset(
                drift_share=0.25, num_method="wasserstein", cat_method="jensenshannon"
            )
        ],
        include_tests=True,
    )


def drift_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(df[COLUMNS], data_definition=drift_schema)


def simulate_batch(day: int) -> pd.DataFrame:
    """Days 5-6: 30% of transactions shift to night hours (data drift).
    Days 7-10: fraudsters now look legitimate (concept drift)."""
    batch = generate_transactions(n_samples=5000, fraud_ratio=0.02, seed=100 + day)
    if day in (5, 6):
        rng = np.random.default_rng(day)
        moved = rng.random(len(batch)) < 0.3
        batch.loc[moved, "hour"] = rng.choice([0, 1, 2, 3, 22, 23], int(moved.sum()))
    if day in (7, 8, 9, 10):
        batch = make_adapted_fraud(batch, seed=day)
    return add_predictions(batch)


def fresh_project(ws: Workspace):
    """Recreate the project so the script can be re-run from scratch."""
    for old in ws.search_project(PROJECT_NAME):
        ws.delete_project(old.id)
    return ws.create_project(
        PROJECT_NAME, description="Fraud model: data drift + performance"
    )


def log_days(ws: Workspace, project, n_days: int = 10) -> None:
    reference_df = add_predictions(pd.read_csv("data/test.csv"))
    ref_perf, ref_drift = to_dataset(reference_df), drift_dataset(reference_df)
    start = datetime(2026, 10, 1)
    for day in range(1, n_days + 1):
        ts = start + timedelta(days=day)
        batch = simulate_batch(day)
        meta: dict[str, str | dict[str, str] | list[str]] = {"day": str(day)}
        drift = build_drift_report().run(
            drift_dataset(batch),
            ref_drift,
            timestamp=ts,
            tags=["drift"],
            metadata=meta,
            name=f"drift-day-{day:02d}",
        )
        perf = build_performance_report().run(
            to_dataset(batch),
            ref_perf,
            timestamp=ts,
            tags=["performance"],
            metadata=meta,
            name=f"perf-day-{day:02d}",
        )
        ws.add_run(project.id, drift)
        ws.add_run(project.id, perf)
        print(f"day {day:2d}: logged drift + performance runs")


def build_dashboard(project) -> None:
    project.dashboard.clear_dashboard()
    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Share of drifted columns",
            subtitle="Alert level 0.25 = any one of the 4 columns",
            size="half",
            values=[
                PanelMetric(
                    legend="share",
                    metric="DriftedColumnsCount",
                    metric_labels={"value_type": "share"},
                )
            ],
            plot_params={"plot_type": "line"},
        )
    )
    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Drift score per column",
            subtitle="Drift when the score is >= 0.1",
            size="half",
            values=[
                PanelMetric(legend=c, metric="ValueDrift", metric_labels={"column": c})
                for c in COLUMNS
            ],
            plot_params={"plot_type": "line"},
        )
    )
    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Recall, precision, F1",
            subtitle="Fraud caught (recall) and alert quality (precision)",
            size="half",
            values=[
                PanelMetric(legend="recall", metric="Recall"),
                PanelMetric(legend="precision", metric="Precision"),
                PanelMetric(legend="f1", metric="F1Score"),
            ],
            plot_params={"plot_type": "line"},
        )
    )
    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Accuracy",
            subtitle="Check the y-axis range: it barely moves",
            size="half",
            values=[PanelMetric(legend="accuracy", metric="Accuracy")],
            plot_params={"plot_type": "line"},
        )
    )
    project.save()


if __name__ == "__main__":
    ws = Workspace.create(WORKSPACE_PATH)
    project = fresh_project(ws)
    log_days(ws, project)
    build_dashboard(project)
    print(f"\nProject id: {project.id}")
    print(f"Start the UI with:  evidently ui --workspace {WORKSPACE_PATH}")
