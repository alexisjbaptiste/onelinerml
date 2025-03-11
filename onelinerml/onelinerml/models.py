# onelinerml/models.py
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

def get_model(model_name, **kwargs):
    if model_name == "linear_regression":
        return LinearRegression(**kwargs)
    elif model_name == "random_forest":
        return RandomForestRegressor(**kwargs)
    elif model_name == "logistic_regression":
        return LogisticRegression(**kwargs)
    elif model_name == "random_forest_classifier":
        return RandomForestClassifier(**kwargs)
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")
