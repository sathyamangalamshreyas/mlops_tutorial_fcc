"""
Data validation for fraud detection.

This module provides functions to validate input data BEFORE making predictions.
Invalid data is rejected with clear error messages.

The key insight: It's better to reject bad input than to make garbage predictions.
"""

# Import Libraries
from typing import Any

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors
from pandera.typing import Series

# Define the valid merchant categories
VALID_CATEGORIES = ["grocery", "restaurant", "retail", "online", "travel"]

# Create empty list
errors = []


# Define pandera data validation class
# The declared columns are required and non-nullable by default
# Extra columns are allowed because we haven't enabled strict column checking
class TransactionBaseSchema(pa.DataFrameModel):
    amount: Series[
        float
    ]  # range constraint not added because the individual and batch validators have different amount policies

    hour: Series[int] = pa.Field(ge=0, le=23)

    day_of_week: Series[int] = pa.Field(ge=0, le=6)

    merchant_category: Series[str] = pa.Field(isin=VALID_CATEGORIES)


# Define rules for amount
# Inherits other rules from TransactionBaseSchema
class TransactionSchema(TransactionBaseSchema):
    amount: Series[float] = pa.Field(
        gt=0,
        le=50_000,
    )


# Create function where at least 99% of amounts must satisfy the range
# This tolerance applies only for batch validation
class BatchTransactionSchema(TransactionBaseSchema):
    @pa.dataframe_check(name="amount_range_99_percent")
    def amount_range_99_percent(
        cls,
        df: pd.DataFrame,
    ) -> bool:
        """
        At least 99% of amounts must be within the allowed range
        """
        if df.empty:
            return False

        in_range = df["amount"].between(
            0.01,
            50_000,
            inclusive="both",
        )

        return bool(in_range.mean() >= 0.99)


# Define function to validate transaction data
def validate_transaction(data: dict[str, Any]) -> dict[str, Any]:
    """
    Validate a single transaction using Pandera.

    Returns:
        {"valid": bool, "errors": list[str]}
    """
    # The original validator accepts both integer and float amounts.
    # Normalize numeric amounts to float without accepting numeric strings.
    row = dict(data)
    amount = row.get("amount")

    if isinstance(amount, (int, float)) and not isinstance(amount, bool):
        row["amount"] = float(amount)

    transaction_df = pd.DataFrame([row])

    try:
        TransactionSchema.validate(
            transaction_df,
            lazy=True,
        )

    except SchemaErrors as exc:
        errors = []

        for _, failure in exc.failure_cases.iterrows():
            column = failure["column"]
            check = str(failure["check"])
            value = failure["failure_case"]

            # Missing columns and null values.
            if check == "column_in_dataframe":
                missing_column = value if isinstance(value, str) else column
                message = f"{missing_column} is required"

            elif check == "not_nullable":
                message = f"{column} is required"

            # Amount constraints.
            elif column == "amount" and check == "greater_than(0)":
                message = f"amount must be positive (got {value})"

            elif column == "amount" and check == "less_than_or_equal_to(50000)":
                message = (
                    f"amount exceeds maximum allowed value of $50,000 (got {value})"
                )

            # Hour constraints.
            elif column == "hour" and check in {
                "greater_than_or_equal_to(0)",
                "less_than_or_equal_to(23)",
            }:
                message = f"hour must be between 0 and 23 (got {value})"

            # Day-of-week constraints.
            elif column == "day_of_week" and check in {
                "greater_than_or_equal_to(0)",
                "less_than_or_equal_to(6)",
            }:
                message = (
                    "day_of_week must be between 0 (Monday) and 6 "
                    f"(Sunday) (got {value})"
                )

            # Merchant-category constraint.
            elif column == "merchant_category" and check.startswith("isin("):
                message = (
                    f"merchant_category must be one of {VALID_CATEGORIES} "
                    f"(got {value!r})"
                )

            # Provide a useful fallback for other schema violations,
            # including unexpected column data types.
            else:
                field = column or "transaction"
                message = f"{field}: validation check {check} failed (got {value!r})"

            if message not in errors:
                errors.append(message)

        return {
            "valid": False,
            "errors": errors,
        }

    return {
        "valid": True,
        "errors": [],
    }


# Define function to validate batch transactions


def validate_batch(df: pd.DataFrame) -> dict[str, Any]:
    """
    Validate a batch of transactions using Pandera

    Returns:
        Dictionary containing success status, number of rows checked,
        and structured validation failure cases.
    """

    try:
        BatchTransactionSchema.validate(df, lazy=True)

    except SchemaErrors as exc:
        return {
            "success": False,
            "rows_checked": len(df),
            "failure_cases": exc.failure_cases.to_dict(orient="records"),
        }

    return {"success": True, "rows_checked": len(df), "failure_cases": []}


if __name__ == "__main__":
    print("-" * 60)
    print("Testing Data Validation")
    print("-" * 60)

    # Test single transaction validation
    print("-" * 40)
    print("\n1. Single Transaction Validation With Pandera")
    print("-" * 40)

    test_cases: list[dict[str, Any]] = [
        {
            "name": "Valid Transaction",
            "data": {
                "amount": 50.0,
                "hour": 14,
                "day_of_week": 3,
                "merchant_category": "grocery",
            },
        },
        {
            "name": "Negative Amount",
            "data": {
                "amount": -100.0,
                "hour": 14,
                "day_of_week": 3,
                "merchant_category": "grocery",
            },
        },
        {
            "name": "Invalid Hour",
            "data": {
                "amount": 50.0,
                "hour": 25,
                "day_of_week": 3,
                "merchant_category": "grocery",
            },
        },
        {
            "name": "Unknown Merchant",
            "data": {
                "amount": 50.0,
                "hour": 14,
                "day_of_week": 3,
                "merchant_category": "unknown",
            },
        },
        {
            "name": "Everything Wrong",
            "data": {
                "amount": -999,
                "hour": 99,
                "day_of_week": 15,
                "merchant_category": "fake",
            },
        },
    ]

    for tc in test_cases:
        result = validate_transaction(tc["data"])
        status = "PASS" if result["valid"] else "FAIL"
        print(f"\n{tc['name']}: {status}")
        if result["errors"]:
            for error in result["errors"]:
                print(f"    - {error}")

    print("\n2. Batch validation with Pandera")
    print("-" * 40)

    train_df = pd.read_csv("data/train.csv")
    results = validate_batch(train_df)

    status = "PASS" if results["success"] else "FAIL"
    print(f"\nTraining data validation: {status}")
    print(f"Rows checked: {results['rows_checked']:,}")

    if results["failure_cases"]:
        print("\nFailure Cases: ")
        for failure in results["failure_cases"]:
            print(
                f"  Column: {failure.get('column')}, "
                f"Check: {failure.get('check')}, "
                f"Value: {failure.get('failure_cases')}, "
                f"Row: {failure.get('index')}"
            )
