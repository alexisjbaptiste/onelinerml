# onelinerml/train.py

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from onelinerml.evaluation import evaluate_model
from onelinerml.models import get_model
from onelinerml.preprocessing import preprocess_data

def train(
    data_source,
    model="auto",
    target_column="target",
    test_size=0.2,
    random_state=42,
    model_save_path="trained_model.joblib",
    preprocessor_save_path="preprocessor.joblib",
    **kwargs
):
    """
    Full training pipeline:
      - Load CSV or DataFrame
      - Preprocess
      - Train/test split
      - Fit model
      - Evaluate
      - Save model
    """
    # Load data
    if isinstance(data_source, str):
        data = pd.read_csv(data_source)
    else:
        data = data_source

    # Preprocess
    X, y, preprocessor = preprocess_data(data, target_column)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Train
    model_instance = get_model(model, y, **kwargs)
    model_instance.fit(X_train, y_train)

    # Evaluate
    metrics = evaluate_model(model_instance, X_test, y_test)

    # Save
    joblib.dump(model_instance, model_save_path)
    joblib.dump(preprocessor, preprocessor_save_path)

    # Report
    print("Evaluation Metrics:", metrics)
    print("Model saved at:", model_save_path)

    return model_instance, metrics


def main():
    """Console entry point for training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train with OneLinerML")
    parser.add_argument("data", help="CSV data path")
    parser.add_argument("--model", default="auto")
    parser.add_argument("--target", dest="target_column", default="target")

    args = parser.parse_args()

    train(
        args.data,
        model=args.model,
        target_column=args.target_column,
    )
