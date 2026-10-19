"""
Serve Fraud detection model as REST API - Naive Version

This is a simple API that:
    1. Loads the trained model at startup
    2. Accepts transaction data via POST request
    3. Returns fraud prediction
"""

# Import Libraries
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from skops.io import get_untrusted_types, load

# Load the trained model, encoder and feature cols on startup
# This is loaded once when the server starts, not on every request

# Inspect types before loading the file
unknown_types = get_untrusted_types(file="models/naive_model.skops")
print(f"Types requiring review: \n{unknown_types}")

# Only trust reviewed types
bundle = load(
    file="models/naive_model.skops",
    trusted=unknown_types,
)

# Unpack the dictionary
model = bundle["model"]
encoder = bundle["encoder"]
feature_cols = bundle["feature_cols"]

print("Model loaded successfully!")

# Create the FastAPI application
app = FastAPI(
    title="Fraud Detection API",
    description="""
    Predict whether a transaction is fradulent

    The API accepts transaction details and returns:
        - Probability whether the transaction is fraudulent

    **NOTE:** This is the naive version without validation or monitoring
    """,
    version="0.0.1",
)

# Define the input schema using Pydantic
# This provides automatic validation and documentation


class Transaction(BaseModel):
    """
    Schema for a transaction to be evaluated for fraud
    """

    amount: float = Field(
        ..., description="Transaction amount in Dollars", examples=[150.00, 30.00]
    )
    hour: int = Field(..., description="Hour of the day (0-23)", examples=[12, 14])
    day_of_week: int = Field(
        ...,
        description="Day of the week starting Monday (0=Monday, 6=Sunday)",
        examples=[0, 3],
    )
    merchant_category: str = Field(
        ..., description="Type of Merchant", examples=["online", "grocery"]
    )


class PredictionResponse(BaseModel):
    """Schema for the prediction response"""

    is_fraud: bool = Field(description="Whether the transaction is predicted as fraud")
    fraud_probability: float = Field(
        description="Probability of the transactin being fraudelent"
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    """
    Predict whether a transaction is fradulent

    Takes transaction details and returns a fraud prediction
    along with probability score
    """
    # Convert the request into a pandas dataframe
    data = pd.DataFrame([transaction.model_dump()])

    # Encode the merchant category using the same encoder from training
    # This ensures consistency between training and serving
    encoded = encoder.transform(data[["merchant_category"]])

    encoded_df = pd.DataFrame(
        data=encoded,
        columns=encoder.get_feature_names_out(["merchant_category"]),
        index=data.index,
    )

    # Combine neumeric features with encoded categorical features
    X_new = pd.concat(
        objs=[data.drop(columns=["merchant_category"]), encoded_df], axis=1
    )

    # Ensure features are in the same order as in the training set
    X_new = X_new.reindex(columns=feature_cols, fill_value=0)

    # Predict fraud probability
    fraud_prediction = model.predict(X_new)[0]
    fraud_probability = model.predict_proba(X_new)[
        0, 1
    ]  # Probability of of class 1 (fraud transaction)

    return PredictionResponse(
        is_fraud=bool(fraud_prediction),
        fraud_probability=round(float(fraud_probability), 4),
    )


@app.get("/health")
def health_check():
    """
    Health Check endpoint

    Returns the status of the API. Useful for:
        - Load balancer health checks
        - Kubernetes liveness probes
        - Monitoring systems
    """
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "Fraud Detection API",
        "version": "0.0.1",
        "docs": "/docs",
        "health": "/health",
    }
