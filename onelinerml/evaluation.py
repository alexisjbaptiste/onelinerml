# onelinerml/evaluation.py
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
import numpy as np

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    # Use regression metrics if target is numeric
    if np.issubdtype(y_test.dtype, np.number):
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return {"mean_squared_error": mse, "r2_score": r2}
    else:
        # For classification tasks, compute additional metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}
