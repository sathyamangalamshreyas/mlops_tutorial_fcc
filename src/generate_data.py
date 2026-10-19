"""
Generate synthetic data using faker
The script below creates realistic-looking transaction data where fraudulent
transactions have different patterns than legitimate ones:
    - Fraud tends to have higher amounts
    - Fraud tends to occur at night
    - Fraud is more common for online and travel merchants
"""

import os

import numpy as np
import pandas as pd
from faker import Faker


def generate_transactions(
    n_samples=10000,
    fraud_ratio=0.02,
    seed=42,
) -> pd.DataFrame:
    """Generate synthetic fraud detection dataset."""
    fake = Faker()
    fake.seed_instance(seed)
    np.random.seed(seed=seed)

    # Compute legit and fraud sample lengths
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = int(n_samples - n_fraud)

    # --- Pre-calculate NumPy distributions to completely avoid dictionary loops ---
    hours_pool = np.arange(24)
    categories = ["grocery", "restaurant", "retail", "online", "travel"]

    # Legit probabilities
    legit_hour_probs = np.array([0.05 if 8 <= h <= 21 else 0.02 for h in hours_pool])
    legit_hour_probs /= legit_hour_probs.sum()
    legit_cat_probs = [0.30, 0.25, 0.25, 0.15, 0.05]

    # Fraud probabilities
    fraud_hour_probs = np.array(
        [0.15 if (h >= 23 or h <= 4) else 0.01 for h in hours_pool]
    )
    fraud_hour_probs /= fraud_hour_probs.sum()
    fraud_cat_probs = [0.02, 0.03, 0.15, 0.55, 0.25]

    # 1. Generate Legit Transactions
    legit_data = []
    # Vectorized fast sampling using numpy
    legit_hours_sampled = np.random.choice(hours_pool, size=n_legit, p=legit_hour_probs)
    legit_cats_sampled = np.random.choice(categories, size=n_legit, p=legit_cat_probs)
    legit_amounts = np.random.lognormal(mean=3.5, sigma=1.2, size=n_legit)

    for i in range(n_legit):
        legit_data.append(
            {
                "amount": round(float(legit_amounts[i]), 2),
                "hour": int(legit_hours_sampled[i]),
                "day_of_week": fake.random_int(min=0, max=6),
                "merchant_category": legit_cats_sampled[i],
                "is_fraud": 0,
            }
        )

    # 2. Generate Fraudulent Transactions
    fraud_data = []
    fraud_hours_sampled = np.random.choice(hours_pool, size=n_fraud, p=fraud_hour_probs)
    fraud_cats_sampled = np.random.choice(categories, size=n_fraud, p=fraud_cat_probs)
    fraud_amounts = np.random.lognormal(mean=5.2, sigma=1.0, size=n_fraud)

    for i in range(n_fraud):
        fraud_data.append(
            {
                "amount": round(float(fraud_amounts[i]), 2),
                "hour": int(fraud_hours_sampled[i]),
                "day_of_week": fake.random_int(min=0, max=6),
                "merchant_category": fraud_cats_sampled[i],
                "is_fraud": 1,
            }
        )

    # Create dataframe and shuffle all rows
    df = pd.concat(
        [pd.DataFrame(legit_data), pd.DataFrame(fraud_data)], ignore_index=True
    )
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return df


if __name__ == "__main__":
    # Print Synthetic Data statistics
    print("Generating synthetic fraud detection dataset...")
    df = generate_transactions(n_samples=10000, fraud_ratio=0.02)

    # Split data into train (80%) and test (20%)
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)

    # Save to CSV files
    os.makedirs("data", exist_ok=True)
    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)

    print(f"\nDataset generated successfully!")
    print(f"Training Set: {len(train_df):,} transactions")
    print(f"Test Set: {len(test_df):,} transactions")
    print(f"Overall fraud ratio: {df['is_fraud'].mean():.2%}")
    print(
        f"\nAverage Legitimate Transaction Amount: ${df[df['is_fraud'] == 0]['amount'].mean():.2f}"
    )
    print(
        f"\nAverage Fraud Transaction Amount: ${df[df['is_fraud'] == 1]['amount'].mean():.2f}"
    )

    print(f"\n--- Fraud Transaction Distribution by Merchant Category ---")
    print(df[df["is_fraud"] == 1]["merchant_category"].value_counts(normalize=True))

    print(f"\n--- Top Fraud Hours ---")
    print(df[df["is_fraud"] == 1]["hour"].value_counts().head(5))
