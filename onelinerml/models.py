import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression

MODELS = {
    "linear_regression": LinearRegression,
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
    "logistic_regression": LogisticRegression,
    "random_forest_classifier": RandomForestClassifier,
    "gradient_boosting_classifier": GradientBoostingClassifier,
}


def is_regression(y):
    """Determine if a target array is a regression problem."""
    arr = np.array(y)
    if not np.issubdtype(arr.dtype, np.number):
        return False
    # Float targets are regression; integer targets with many unique values are too
    if np.issubdtype(arr.dtype, np.floating):
        return True
    return len(np.unique(arr)) > 10


def get_model(name, y, **kwargs):
    """Return an untrained model instance. Use 'auto' for automatic selection."""
    if name == "auto":
        if is_regression(y):
            return GradientBoostingRegressor(**kwargs)
        return GradientBoostingClassifier(**kwargs)

    if name not in MODELS:
        supported = ", ".join(sorted(MODELS.keys()))
        raise ValueError(f"Unknown model '{name}'. Choose from: {supported}")
    return MODELS[name](**kwargs)
