import joblib
import numpy as np
import pandas as pd


class Model:
    """A trained ML model with its preprocessor, ready to predict or serve."""

    def __init__(self, estimator, preprocessor, metrics=None, target_column=None,
                 feature_columns=None):
        self.estimator = estimator
        self.preprocessor = preprocessor
        self.metrics = metrics or {}
        self.target_column = target_column
        self.feature_columns = feature_columns

    def predict(self, data):
        """Predict on new data. Accepts a dict, list of dicts, or DataFrame."""
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        elif isinstance(data, list):
            data = pd.DataFrame(data)
        elif not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a dict, list of dicts, or DataFrame")

        if self.feature_columns is not None:
            missing = set(self.feature_columns) - set(data.columns)
            if missing:
                raise ValueError(f"Missing columns: {missing}")

        transformed = self.preprocessor.transform(data)
        return self.estimator.predict(transformed)

    def save(self, path="model.joblib"):
        """Save the entire model (estimator + preprocessor + metadata) to one file."""
        bundle = {
            "estimator": self.estimator,
            "preprocessor": self.preprocessor,
            "metrics": self.metrics,
            "target_column": self.target_column,
            "feature_columns": self.feature_columns,
        }
        joblib.dump(bundle, path)
        print(f"Model saved to {path}")
        return self

    @classmethod
    def load(cls, path="model.joblib"):
        """Load a saved model from a single file."""
        bundle = joblib.load(path)
        return cls(
            estimator=bundle["estimator"],
            preprocessor=bundle["preprocessor"],
            metrics=bundle.get("metrics", {}),
            target_column=bundle.get("target_column"),
            feature_columns=bundle.get("feature_columns"),
        )

    def serve(self, host="0.0.0.0", port=8000):
        """Start a FastAPI prediction server for this model."""
        from onelinerml.api import create_app
        import uvicorn

        app = create_app(self)
        uvicorn.run(app, host=host, port=port)

    def __repr__(self):
        name = type(self.estimator).__name__
        return f"Model({name}, metrics={self.metrics})"
