# onelinerml/evaluation.py
from sklearn.base import is_classifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    if is_classifier(model):
        # For classification tasks, compute additional metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {"mean_squared_error": mse, "r2_score": r2}
