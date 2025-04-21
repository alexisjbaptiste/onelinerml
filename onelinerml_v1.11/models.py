from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

# AutoML imports
from autosklearn.regression import AutoSklearnRegressor
from autosklearn.classification import AutoSklearnClassifier

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
        return XGBRegressor(**kwargs)
    elif model_name == "xgboost_classifier":
        return XGBClassifier(**kwargs)
    elif model_name == "lightgbm_regressor":
        return LGBMRegressor(**kwargs)
    elif model_name == "lightgbm_classifier":
        return LGBMClassifier(**kwargs)
    elif model_name == "auto_sklearn_regressor":
        # AutoML regressor with default 1-minute training time by default
        return AutoSklearnRegressor(time_left_for_this_task=60, **kwargs)
    elif model_name == "auto_sklearn_classifier":
        # AutoML classifier with default 1-minute training time by default
        return AutoSklearnClassifier(time_left_for_this_task=60, **kwargs)
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")
