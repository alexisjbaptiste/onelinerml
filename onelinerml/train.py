import pandas as pd
from sklearn.model_selection import train_test_split

from onelinerml.evaluation import evaluate
from onelinerml.model import Model
from onelinerml.models import get_model
from onelinerml.preprocessing import build_preprocessor


def train(data, target="target", model="auto", test_size=0.2, random_state=42,
          save_to=None, **model_params):
    """Train a model in one call. Returns a Model object.

    Args:
        data: CSV file path, URL, or pandas DataFrame.
        target: Name of the target column.
        model: Model name ('auto', 'linear_regression', 'random_forest', etc.).
        test_size: Fraction of data held out for evaluation.
        random_state: Random seed for reproducibility.
        save_to: If set, save the model to this path after training.
        **model_params: Extra keyword arguments passed to the sklearn estimator.

    Returns:
        A Model object with .predict(), .serve(), .save(), and .metrics.
    """
    if isinstance(data, str):
        data = pd.read_csv(data)
    elif not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a file path or pandas DataFrame")

    if target not in data.columns:
        raise ValueError(f"Target column '{target}' not found. "
                         f"Available: {list(data.columns)}")

    y = data[target]
    X = data.drop(columns=[target])
    feature_columns = X.columns.tolist()

    # Split BEFORE preprocessing to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Fit preprocessor on training data only
    preprocessor = build_preprocessor(X_train)
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    # Train
    estimator = get_model(model, y_train, **model_params)
    estimator.fit(X_train_t, y_train)

    # Evaluate
    metrics = evaluate(estimator, X_test_t, y_test)
    print(f"Trained {type(estimator).__name__} | {metrics}")

    result = Model(
        estimator=estimator,
        preprocessor=preprocessor,
        metrics=metrics,
        target_column=target,
        feature_columns=feature_columns,
    )

    if save_to:
        result.save(save_to)

    return result


def main():
    """CLI entry point for training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train with OneLinerML")
    parser.add_argument("data", help="Path to CSV file")
    parser.add_argument("--target", default="target", help="Target column name")
    parser.add_argument("--model", default="auto", help="Model type")
    parser.add_argument("--save-to", default="model.joblib", help="Output path")

    args = parser.parse_args()
    train(args.data, target=args.target, model=args.model, save_to=args.save_to)
