import joblib
import numpy as np
import pandas as pd


class Model:
    """A trained ML model with its preprocessor, ready to predict, serve, or deploy."""

    def __init__(self, estimator, preprocessor, metrics=None, target_column=None,
                 feature_columns=None, analytics=None):
        self.estimator = estimator
        self.preprocessor = preprocessor
        self.metrics = metrics or {}
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.analytics = analytics

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
            "analytics": self.analytics,
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
            analytics=bundle.get("analytics"),
        )

    def serve(self, host="0.0.0.0", port=8000):
        """Start a FastAPI prediction server (API only, no dashboard)."""
        from onelinerml.api import create_app
        import uvicorn

        app = create_app(self)
        uvicorn.run(app, host=host, port=port)

    def deploy(self, host="0.0.0.0", port=8000, public=False):
        """Deploy API + dashboard. Optionally create a public URL via ngrok.

        Prints the API URL and Dashboard URL immediately.
        If public=True, creates an ngrok tunnel for a public URL.
        """
        from onelinerml.api import create_app
        import uvicorn

        app = create_app(self)
        base = f"http://{host}:{port}"
        if host == "0.0.0.0":
            base = f"http://localhost:{port}"

        print()
        print("=" * 56)
        print("  OneLinerML Deployed!")
        print("=" * 56)
        print(f"  API URL:       {base}/predict")
        print(f"  Dashboard:     {base}/dashboard")
        print(f"  Health check:  {base}/health")

        tunnel_url = None
        if public:
            tunnel_url = _start_tunnel(port)
            if tunnel_url:
                print(f"  Public URL:    {tunnel_url}/predict")
                print(f"  Public Dash:   {tunnel_url}/dashboard")

        print("=" * 56)
        print()

        uvicorn.run(app, host=host, port=port)

    def __repr__(self):
        name = type(self.estimator).__name__
        return f"Model({name}, metrics={self.metrics})"


def _start_tunnel(port):
    """Try to create a public tunnel via pyngrok."""
    try:
        from pyngrok import ngrok
        tunnel = ngrok.connect(port)
        return tunnel.public_url
    except ImportError:
        print("  [!] Install pyngrok for public URLs: pip install pyngrok")
        return None
    except Exception as e:
        print(f"  [!] Could not create tunnel: {e}")
        return None
