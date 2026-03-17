from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
)

from onelinerml.models import is_regression


def evaluate(estimator, X_test, y_test):
    """Evaluate a trained model and return a metrics dict."""
    y_pred = estimator.predict(X_test)

    if is_regression(y_test):
        return {
            "mse": round(mean_squared_error(y_test, y_pred), 4),
            "r2": round(r2_score(y_test, y_pred), 4),
        }
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
    }
