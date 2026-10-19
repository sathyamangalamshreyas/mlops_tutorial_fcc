"""
Feast feature retrieval for training and serving

This module provides functions to retrieve features from Feast:
    - get_training_features(): For offline training (historical feratures)
    - get_online_features(): For real-time serving (low-latency)

Important: Both functions use the SAME feature definitions, ensuring
consistancy between training and serving
"""

from datetime import datetime

import pandas as pd
from feast import FeatureStore

# Initialize Feast store (points to our feature_repo)
store = FeatureStore(repo_path="feature_repo")


def get_training_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get features for training using Feast's offline store

    Uses point-in-time correct joins to prevent data leakage
    This means features are looked up as of the time each transaction
    occured, not as of "now" - preventing you from accidentally using
    feature data

    Args:
        df: DataFrame with at least 'merchant_category' column

    Return:
        DataFrame with original columns plus Feast features
    """
    print("Retrieving training features from Feast offline store...")

    # Prepare entity dataframe with timestamps
    # Each row needs: entity key(s) + event_timestamp
    entity_df = df[["merchant_category"]].copy()
    entity_df["event_timestamp"] = datetime.now()  # See note below
    entity_df = entity_df.drop_duplicates()

    # ⚠️ Simplification: For clarity, we use the current timestamp here.
    # In real systems, this would be the actual event time of each transaction.

    # Retrieve historical features
    # Feast handles the point-in-time join automatically
    training_data = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "merchant_stats:avg_amount",
            "merchant_stats:transaction_count",
            "merchant_stats:fraud_rate",
        ],
    ).to_df()

    # Merge features back with original dataframe
    result = df.merge(
        training_data[
            ["merchant_category", "avg_amount", "transaction_count", "fraud_rate"]
        ],
        on="merchant_category",
        how="left",
    )

    print(f"Retrieved features for {len(entity_df)} unique merchants")
    return result


def get_online_features(merchant_category: str) -> dict:
    """
    Get features for real-time serving using Feast's online store
    This is optimized for low-latency retrieval (milliseconds)
    Use this in your prediction API for real-time inference

    Args:
        merchant_category: The merchant category to look up

    Returns:
        Dictionary with feature names and values
    """
    # Retrieve from online-store (low-latency)
    feature_vector = store.get_online_features(
        features=[
            "merchant_stats:avg_amount",
            "merchant_stats:transaction_count",
            "merchant_stats:fraud_rate",
        ],
        entity_rows=[{"merchant_category": merchant_category}],
    ).to_dict()

    # Format the response
    return {
        "merchant_avg_amount": feature_vector["avg_amount"][0],
        "merchant_tx_count": feature_vector["transaction_count"][0],
        "merchant_fraud_rate": feature_vector["fraud_rate"][0],
    }


def get_online_features_batch(merchant_categories: list) -> pd.DataFrame:
    """
    Get features for multiple merchants at once (batch serving)
    More efficient than calling get_online_features() in a loop

    Args:
        merchant_categories: List of merchant categories to look up

    Returns:
        DataFrame with features for each merchant
    """
    feature_vector = store.get_online_features(
        features=[
            "merchant_stats:avg_amount",
            "merchant_stats:transaction_count",
            "merchant_stats:fraud_rate",
        ],
        entity_rows=[{"merchant_category": mc} for mc in merchant_categories],
    ).to_df()

    return feature_vector


if __name__ == "__main__":
    # Test the feature retrieval functions
    print("=" * 60)
    print("Testing Feast Feature Retrieval")
    print("=" * 60)

    # Test Offline retrieval (for training)
    print("\n1. Testing Offline feature retrieval (for training)...")
    train_df = pd.read_csv("data/train.csv").head(10)
    enriched = get_training_features(train_df)
    print("\n   Sample enriched training data:")
    print(enriched[["amount", "merchant_category", "avg_amount", "fraud_rate"]].head())

    # Test Online retrieval (for serving)
    print("\n2. Testing Online feature retrieval (for serving)... ")
    for category in ["online", "grocery", "travel", "restaurant", "retail"]:
        features = get_online_features(category)
        print(
            f"    {category}: avg_amount=${features['merchant_avg_amount']:.2f}, "
            f"fraud_rate={features['merchant_fraud_rate']:.2%}"
        )

    # Test batch retrieval
    print("\n3. Testing batch online retrieval...")
    batch_features = get_online_features_batch(["online", "grocery", "travel"])
    print(batch_features)

    print("\n" + "=" * 60)
    print("Feast Feature Retrieval Test Complete!")
    print("\n" + "=" * 60)
