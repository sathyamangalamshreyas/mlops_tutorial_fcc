"""
Train fraud detection model with MLFlow experiment tracking

This script demonstrates proper ML experiment tracking:
    - Log all hyperparameters
    - Log all metrics (train and test)
    - Log the trained model as an artifact
    - Register the model in the Model Registry
"""

# Import Libraries
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder
from skops.io import dump

# Configure MLFlow to use the tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Create the experiment
# All runs will be grouped under this experiment name
mlflow.set_experiment("fraud_detection")


# Function to load and pre-process data
def load_and_preprocess_data():
    """
    Load and preprocess training and test data
    """
    print("Loading data...")
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    # Encode categorical features
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )
    train_encoded = encoder.fit_transform(train_df[["merchant_category"]])
    test_encoded = encoder.transform(test_df[["merchant_category"]])

    # Get the generated feature names
    encoded_cols = encoder.get_feature_names_out(["merchant_category"])

    # Convert encoded arrays into DataFrames
    train_encoded_df = pd.DataFrame(
        data=train_encoded,
        columns=encoded_cols,
        index=train_df.index,
    )
    test_encoded_df = pd.DataFrame(
        data=test_encoded,
        columns=encoded_cols,
        index=test_df.index,
    )
    # Remove the original categorical columns and add the encoded columns
    train_df = pd.concat(
        objs=[train_df.drop(columns=["merchant_category"]), train_encoded_df], axis=1
    )
    test_df = pd.concat(
        objs=[test_df.drop(columns=["merchant_category"]), test_encoded_df], axis=1
    )
    # Prepare features and labels
    target = "is_fraud"
    feature_cols = [col for col in train_df.columns if col != target]
    X_train = train_df[feature_cols]
    y_train = train_df[target]

    X_test = test_df[feature_cols]
    y_test = test_df[target]

    return X_train, y_train, X_test, y_test, encoder, feature_cols


def train_and_log_model(
    n_estimators: int = 100,
    max_depth: int = 10,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
):
    """
    Train a model and log everything to MLFlow

    Args:
        - n_estimators: Number of trees in the forest
        - max_depth: Maximum depth of each tree
        - min_samples_split: Minimum samples required to split a node
        - min_samples_lead: Minimum samples required at a leaf node
    """
    X_train, y_train, X_test, y_test, encoder, feature_cols = load_and_preprocess_data()

    # Start an MLFlow run - everything logged will be associated with this run
    with mlflow.start_run():
        # Add a description run name
        run_name = (
            f"rf_est{n_estimators}_depth{max_depth}_{datetime.now().strftime('%H%M%S')}"
        )
        mlflow.set_tag("mlflow.runName", run_name)

        # Log all hyperparameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("min_samples_leaf", min_samples_leaf)
        mlflow.log_param("model_type", "RandomForestClassifier")

        # Log Data information
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("fraud_ratio", float(y_train.mean()))
        mlflow.log_param("n_features", X_train.shape[1])

        # Train the model
        print(
            f"\nTraining the model: n_estimators={n_estimators}, max_depth={max_depth}"
        )
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        # Evaluate and log metrics for train and test sets
        for dataset_name, X, y in [
            ("train", X_train, y_train),
            ("test", X_test, y_test),
        ]:
            y_pred = model.predict(X)
            y_prob = model.predict_proba(X)[:, 1]

            # Calculate all metrics
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred)
            recall = recall_score(y, y_pred, zero_division=0)
            f1 = f1_score(y, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y, y_prob)

            # Log metrics with dataset prefix
            mlflow.log_metric(f"{dataset_name}_accuracy", accuracy)
            mlflow.log_metric(f"{dataset_name}_precision", precision)
            mlflow.log_metric(f"{dataset_name}_recall", recall)
            mlflow.log_metric(f"{dataset_name}_f1", f1)
            mlflow.log_metric(f"{dataset_name}_roc_auc", roc_auc)

            print(
                f"   {dataset_name.upper()} - Accuracy: {accuracy:.4f}, f1: {f1:.4f}, roc_auc: {roc_auc:.4f}"
            )

        # Log feature importance
        for feature, importance in sorted(
            zip(X_train.columns, model.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        ):
            mlflow.log_metric(f"importance_{feature}", importance)

        # Log the model to MLFlow and register it in the Model Registry
        # This created a new verion of the model automatically
        print(f"\nRegistering the model in the MLFlow Model Registry...")
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name="fraud-detection-model",
            input_example=X_train.iloc[:5],  # Example input for demonstration
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        # Save the model, encoder, and feature columns
        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)
        # Define model file path
        model_path = models_dir / "mlflow_model.skops"
        dump(
            {"model": model, "encoder": encoder, "feature_cols": feature_cols},
            model_path,
        )
        mlflow.log_artifact(str(model_path))

        # Get the run ID for reference
        run_id = mlflow.active_run().info.run_id
        print(f"\nMLFlow Run ID: {run_id}")
        print(f"View this run: http://127.0.0.1:5000/#/experiments/1/runs/{run_id}")

        return model, encoder


def run_experiment_sweep():
    """
    Run multiple experiments with different with
    different hyperparameters

    This demonstrates how MLFlow helps compare different configurations
    """

    print("=" * 60)
    print("Running Hyperparameter Experiment Sweep")
    print("=" * 60)

    # Define different configurations to try
    experiments = [
        {"n_estimators": 50, "max_depth": 5},
        {"n_estimators": 100, "max_depth": 10},
        {"n_estimators": 100, "max_depth": 15},
        {"n_estimators": 200, "max_depth": 10},
        {"n_estimators": 200, "max_depth": 20},
    ]

    for i, params in enumerate(experiments, 1):
        print(f"\n--- Experiment {i}/{len(experiments)} ---")
        train_and_log_model(**params)

    print("\n" + "=" * 60)
    print("Experiment Sweep Complete!")
    print("=" * 60)
    print(f"\nView all experiments at: http://127.0.0.1:5000")
    print("Compare runs to find the best hyperparameters")


if __name__ == "__main__":
    run_experiment_sweep()
