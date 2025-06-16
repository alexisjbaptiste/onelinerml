# onelinerml/preprocessing.py
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(train_data, target_column, test_data=None):
    """Preprocess the dataset.

    If ``test_data`` is provided, the preprocessor is fitted using only
    ``train_data`` and then applied to both train and test splits.
    When ``test_data`` is ``None`` the entire ``train_data`` is used for both
    fitting and transformation (backwards compatibility).
    """
    # Separate target variable from features in the training set
    y_train = train_data[target_column]
    X_train = train_data.drop(columns=[target_column])
    
    # Identify numeric and categorical columns
    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns
    
    # Build pipelines for numeric and categorical data
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean"))
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])
    
    X_train_preprocessed = preprocessor.fit_transform(X_train)

    if test_data is not None:
        y_test = test_data[target_column]
        X_test = test_data.drop(columns=[target_column])
        X_test_preprocessed = preprocessor.transform(X_test)
        return (
            X_train_preprocessed,
            y_train.values,
            X_test_preprocessed,
            y_test.values,
            preprocessor,
        )

    return X_train_preprocessed, y_train.values, preprocessor
