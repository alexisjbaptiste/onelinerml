# onelinerml/preprocessing.py
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(data, target_column):
    # Separate target variable from features
    y = data[target_column]
    X = data.drop(columns=[target_column])
    
    # Identify numeric and categorical columns
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns
    
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
    
    X_preprocessed = preprocessor.fit_transform(X)
    return X_preprocessed, y.values
