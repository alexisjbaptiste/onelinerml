from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import numpy as np

SUPPORTED_MODELS = {
    "linear_regression": LinearRegression,
    "random_forest": RandomForestRegressor,
    "logistic_regression": LogisticRegression,
    "random_forest_classifier": RandomForestClassifier,
}


def _auto_model(y):
    if np.issubdtype(y.dtype, np.number):
        return LinearRegression()
    return LogisticRegression(max_iter=1000)


def get_model(model_name, y, **kwargs):
    """
    Return an untrained model instance by name.
    Supported models:
      - auto
      - linear_regression
      - random_forest
      - logistic_regression
      - random_forest_classifier
    """
    if model_name == "auto":
        return _auto_model(y)
    model_cls = SUPPORTED_MODELS.get(model_name)
    if model_cls is None:
        raise ValueError(f"Model '{model_name}' is not supported.")
    return model_cls(**kwargs)
