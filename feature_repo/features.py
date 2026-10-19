"""
Feast feature definitions for fraud detection

The file defines:
    - Entities: The keys we use to look up features (merchant_category)
    - Data Sources: Where the raw feature data comes from
    - Feature Views: The features themselves and their schemas

The key insight: These definitions are the single source of truth
Both training and serving use these exact definitions
"""

# Import Libraries
from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int64
from sqlalchemy import desc

# ========================================================================
# ENTITIES
# ========================================================================
# An entity is the "key" we use to look up features
# For merchant-level features, the entity is merchant_category

merchant = Entity(
    name="merchant_category",
    description="Merchant category for the transaction (Ex. 'online', 'grocery')",
    value_type=ValueType.STRING,
)
# ========================================================================
# Data Sources
# ========================================================================
# Data Sources tell Feast where to find the raw feature data
# For local development, we use a Parquet file
# For production, this could be BigQuery, SnowFlake, S3 etc.

merchant_stats_source = FileSource(
    name="merchant_stats_source",
    path="../data/merchant_features.parquet",
    timestamp_field="event_timestamp",  # Required in for point-in-time joins
)
# ========================================================================
# Feature Views
# ========================================================================
# A feature view defines a group of related fatures
# It specifies:
#   - Which entity the features are for
#   - The schema (names and types of features)
#   - Where the data comes from
#   - How long features are valid (Time to Live: TTL)

merchant_stats_fv = FeatureView(
    name="merchant_stats",
    description="Aggregated statistics per merchant category",
    entities=[merchant],
    ttl=timedelta(days=7),  # Features are valid for 7 days
    schema=[
        Field(
            name="avg_amount", dtype=Float32, description="Average Transaction Amount"
        ),
        Field(
            name="transaction_count", dtype=Int64, description="Number of Transactions"
        ),
        Field(name="fraud_rate", dtype=Float32, description="Historical Fraud Rate"),
    ],
    source=merchant_stats_source,
    online=True,  # Enable online serving (low-latency retrieval)
)
