from importlib import import_module

from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression


def _import_optional(module_name, attr_name):
    try:
        module = import_module(module_name)
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            f"Optional dependency '{module_name}' is required for this model. "
            f"Install it to use '{attr_name}'."
        ) from exc
    return getattr(module, attr_name)

def get_model(model_name, **kwargs):
    """
    Return an untrained model instance by name.
    Supported models:
      - linear_regression
      - random_forest
      - logistic_regression
      - random_forest_classifier
      - isolation_forest
      - xgboost_regressor
      - xgboost_classifier
      - lightgbm_regressor
      - lightgbm_classifier
      - auto_sklearn_regressor
      - auto_sklearn_classifier
    """
    if model_name == "linear_regression":
        return LinearRegression(**kwargs)
    elif model_name == "random_forest":
        return RandomForestRegressor(**kwargs)
    elif model_name == "logistic_regression":
        return LogisticRegression(**kwargs)
    elif model_name == "random_forest_classifier":
        return RandomForestClassifier(**kwargs)
    elif model_name == "isolation_forest":
        return IsolationForest(**kwargs)
    elif model_name == "xgboost_regressor":
        return _import_optional("xgboost", "XGBRegressor")(**kwargs)
    elif model_name == "xgboost_classifier":
        return _import_optional("xgboost", "XGBClassifier")(**kwargs)
    elif model_name == "lightgbm_regressor":
        return _import_optional("lightgbm", "LGBMRegressor")(**kwargs)
    elif model_name == "lightgbm_classifier":
        return _import_optional("lightgbm", "LGBMClassifier")(**kwargs)
    elif model_name == "auto_sklearn_regressor":
        # AutoML regressor with default 1-minute training time by default
        autosklearn_regressor = _import_optional(
            "autosklearn.regression",
            "AutoSklearnRegressor",
        )
        return autosklearn_regressor(time_left_for_this_task=60, **kwargs)
    elif model_name == "auto_sklearn_classifier":
        # AutoML classifier with default 1-minute training time by default
        autosklearn_classifier = _import_optional(
            "autosklearn.classification",
            "AutoSklearnClassifier",
        )
        return autosklearn_classifier(time_left_for_this_task=60, **kwargs)
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")
