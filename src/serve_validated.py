"""
Serve fraud detection model with input validation.

This version adds data validation BEFORE making predictions:
- Invalid inputs are rejected with HTTP 400 and clear error messages
- Valid inputs are processed and predictions returned
"""

# Import Libraries
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from skops.io import get_untrusted_types, load

from src.data_validation import validate_transaction

# Load the saved model, encoder and features at startup
model_path = Path("models") / "mlflow_model.skops"

unknown_types = get_untrusted_types(file=model_path)

loaded_data = load(model_path, trusted=unknown_types)

# Extract Objects
model = loaded_data["model"]
encoder = loaded_data["encoder"]
feature_cols = loaded_data["feature_cols"]

# Temporary Diagnostic
print("Encoder:", type(encoder).__name__)
print("Saved feature names:", feature_cols)
print("Encoder output columns:", encoder.get_feature_names_out())
print(
    "Model feature names:",
    getattr(model, "feature_names_in_", None),
)

# Instantiate FastAPI app
app = FastAPI(
    title="Fraud Detection API (Validated)",
    description=("Fraud Detection API with Pandera Transaction Validation"),
    version="0.1.0",
)


# Define Pydantic Models
class Transaction(BaseModel):
    amount: float = Field(
        description="Transaction Amount",
    )
    hour: int = Field(
        description="Hour of Day (0-23)",
    )
    day_of_week: int = Field(
        description="Day of Week (0=Monday, 6=Sunday)",
    )
    merchant_category: str = Field(
        description="Merchant Category",
    )


class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    validation_passed: bool = True


class ValidationErrorResponse(BaseModel):
    detail: dict[str, Any]


# Define the prediction endpoint
@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ValidationErrorResponse},
    },
)
def predict(tx: Transaction):
    # Convert the Pydantic model to a dictionary
    data = tx.model_dump()

    # Validate the transaction using Pandera
    validation = validate_transaction(data=data)

    if not validation["valid"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "valudation Failed",
                "errors": validation["errors"],
                "input": data,
            },
        )

    # Supply the category as a 2-D DataFrame
    category_df = pd.DataFrame({"merchant_category": [data["merchant_category"]]})

    encoded = encoder.transform(category_df)

    # OneHotEncoder may return a sparse matrix
    if hasattr(encoded, "toarray"):
        encoded = encoded.toarray()

    # Get the column names corresponding to the encoded values
    encoded_columns = encoder.get_feature_names_out()

    encoded_features = dict(zip(encoded_columns, encoded[0].tolist()))

    # Combine the original numeric features with encoded features
    feature_values = {
        "amount": data["amount"],
        "hour": data["hour"],
        "day_of_week": data["day_of_week"],
        **encoded_features,
    }

    # Ensure every feature expected by the saved model is available
    missing_features = [col for col in feature_cols if col not in feature_values]

    if missing_features:
        raise RuntimeError(
            "Encoder / model feature Mismatch"
            f"Missing features: {missing_features}"
            f"Available features: {list(feature_values)}"
        )

    # Preserve the exact feature order used during training
    X = pd.DataFrame(
        [{col: feature_values[col] for col in feature_cols}],
        columns=feature_cols,
    )

    # Predict only after validation succeeds
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]

    return PredictionResponse(
        is_fraud=bool(pred),
        fraud_probability=round(float(prob), 4),
        validation_passed=True,
    )


# Define Health Endpoint
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "validation": "enabled",
    }
