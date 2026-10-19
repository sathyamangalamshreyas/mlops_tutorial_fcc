"""
Train a fraud detection model - Naive Version

This script demonstrates the "quick and dirty" approach to ML:
    - No Experiment Tracking
    - No Model Versioning
    - Just train and save to a pickle file
"""

# Import Libraries
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import OneHotEncoder
from skops.io import dump, get_untrusted_types


# Function to train the RandomForestClassifier
def naive_rf_clf():
    """
    Function to feature engineer train and validate a
    RandomForest Classifier
    """
    # Load data
    print("Loading Data...")
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    # Print basic statistics
    print(f"Training Samples: {len(train_df):,}")
    print(f"Test Samples: {len(test_df):,}")
    print(f"Fraud Ratio in Train set: {train_df['is_fraud'].mean():2f}")
    print(f"train data types: {train_df.dtypes}")

    # Encode Categorical Features
    # We need to save the encoder to use the same mapping during inference
    print(f"\nEncoding categorical features...")
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
        train_encoded,
        columns=encoded_cols,
        index=train_df.index,
    )

    test_encoded_df = pd.DataFrame(
        test_encoded,
        columns=encoded_cols,
        index=test_df.index,
    )

    # Remove the original categorical columns and add the encoded columns
    train_df = pd.concat(
        [train_df.drop(columns=["merchant_category"]), train_encoded_df],
        axis=1,
    )

    test_df = pd.concat(
        [test_df.drop(columns=["merchant_category"]), test_encoded_df],
        axis=1,
    )

    print(f"Encoded merchant categories: {list(encoded_cols)}")

    # Prepare features and labels
    target = "is_fraud"
    feature_cols = [col for col in train_df.columns if col != target]
    X_train = train_df[feature_cols]
    y_train = train_df[target]

    X_test = test_df[feature_cols]
    y_test = test_df[target]

    # Train a Random Forest Classifier
    naive_model = RandomForestClassifier(
        n_estimators=100,  # Number of trees
        max_depth=10,  # Maximum depth of each tree
        random_state=42,  # For reproduceability
        n_jobs=-1,  # Use all available CPU cores
    )
    naive_model.fit(X_train, y_train)
    print("Training Complete!")

    # Evaluate on the test data
    print("\n" + "=" * 50)
    print("Model Evaluation")
    print("=" * 50)

    y_pred = naive_model.predict(X_test)
    y_prob = naive_model.predict_proba(X_test)[:, 1]

    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"\nPrecison: {precision_score(y_test, y_pred):.4f}")
    print(f"\nRecall:   {recall_score(y_test, y_pred):.4f}")
    print(f"\nF1-score: {f1_score(y_test, y_pred):.4f}")

    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"    True Negatives: {cm[0][0]:,} (correctly identified legit transactions)")
    print(f"    False Positives: {cm[0][1]:,} (legit transactions flagged as fraud)")
    print(
        f"    False Negatives: {cm[1][0]:,} (Fraud transactions flagged as legit - DANGEROUS)"
    )
    print(f"    True Positives: {cm[1][1]:,} (Correctly Identified Fraud Transactions)")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))

    # Feature Importance
    print(f"\nFeature Importance:")
    for name, importance in sorted(
        zip(feature_cols, naive_model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    ):
        print(f"    {name}: {importance:.4f}")

    # Save the model and encoder together
    print(f"\nSaving model to models/naive_model.skops")
    # Define the models directory relative to root
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    # Define model file path
    model_path = models_dir / "naive_model.skops"
    dump(
        {"model": naive_model, "encoder": encoder, "feature_cols": feature_cols},
        model_path,
    )

    unknown_types = get_untrusted_types(file=model_path)
    print(f"\nSkops unknown files types: \n{unknown_types}")

    print(f"\nModel trained and saved successfully!")


if __name__ == "__main__":
    naive_rf_clf()
