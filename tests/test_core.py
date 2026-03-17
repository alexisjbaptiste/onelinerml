import os
import tempfile

import numpy as np
import pandas as pd
import pytest

import onelinerml as ml
from onelinerml.models import get_model, is_regression
from onelinerml.preprocessing import build_preprocessor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def regression_df():
    np.random.seed(42)
    return pd.DataFrame({
        "x1": np.random.randn(100),
        "x2": np.random.randn(100),
        "cat": np.random.choice(["a", "b", "c"], 100),
        "price": np.random.randn(100) * 100,
    })


@pytest.fixture
def classification_df():
    np.random.seed(42)
    return pd.DataFrame({
        "x1": np.random.randn(100),
        "x2": np.random.randn(100),
        "label": np.random.choice(["yes", "no"], 100),
    })


@pytest.fixture
def regression_csv(regression_df, tmp_path):
    path = tmp_path / "data.csv"
    regression_df.to_csv(path, index=False)
    return str(path)


# ---------------------------------------------------------------------------
# Tests: models.py
# ---------------------------------------------------------------------------

class TestModels:
    def test_is_regression_numeric(self):
        assert is_regression(np.arange(100, dtype=float))

    def test_is_regression_categorical(self):
        assert not is_regression(np.array(["a", "b", "c"] * 10))

    def test_is_regression_few_classes(self):
        assert not is_regression(np.array([0, 1] * 50))

    def test_get_model_auto_regression(self):
        m = get_model("auto", np.arange(100, dtype=float))
        assert "Regressor" in type(m).__name__

    def test_get_model_auto_classification(self):
        m = get_model("auto", np.array(["a", "b"] * 50))
        assert "Classifier" in type(m).__name__

    def test_get_model_unknown(self):
        with pytest.raises(ValueError, match="Unknown model"):
            get_model("nonexistent", np.array([1, 2, 3]))

    def test_get_model_with_params(self):
        m = get_model("random_forest", np.arange(100, dtype=float), n_estimators=10)
        assert m.n_estimators == 10


# ---------------------------------------------------------------------------
# Tests: preprocessing.py
# ---------------------------------------------------------------------------

class TestPreprocessing:
    def test_build_preprocessor(self, regression_df):
        X = regression_df.drop(columns=["price"])
        pp = build_preprocessor(X)
        result = pp.fit_transform(X)
        assert result.shape[0] == 100
        # 2 numeric + 3 one-hot categories
        assert result.shape[1] == 5

    def test_handles_missing_values(self):
        df = pd.DataFrame({"a": [1, 2, np.nan], "b": ["x", None, "y"]})
        pp = build_preprocessor(df)
        result = pp.fit_transform(df)
        assert not np.any(np.isnan(result.toarray() if hasattr(result, "toarray") else result))


# ---------------------------------------------------------------------------
# Tests: train + Model
# ---------------------------------------------------------------------------

class TestTrain:
    def test_train_regression_df(self, regression_df):
        model = ml.train(regression_df, target="price")
        assert "r2" in model.metrics
        assert "mse" in model.metrics

    def test_train_classification_df(self, classification_df):
        model = ml.train(classification_df, target="label")
        assert "accuracy" in model.metrics
        assert "f1" in model.metrics

    def test_train_from_csv(self, regression_csv):
        model = ml.train(regression_csv, target="price")
        assert model.metrics

    def test_train_specific_model(self, regression_df):
        model = ml.train(regression_df, target="price", model="linear_regression")
        assert "LinearRegression" in type(model.estimator).__name__

    def test_train_invalid_target(self, regression_df):
        with pytest.raises(ValueError, match="not found"):
            ml.train(regression_df, target="nonexistent")

    def test_train_invalid_data(self):
        with pytest.raises(ValueError, match="file path or pandas"):
            ml.train(12345, target="x")


class TestModel:
    def test_predict_dict(self, regression_df):
        model = ml.train(regression_df, target="price")
        pred = model.predict({"x1": 0.5, "x2": -0.3, "cat": "a"})
        assert len(pred) == 1

    def test_predict_list_of_dicts(self, regression_df):
        model = ml.train(regression_df, target="price")
        pred = model.predict([
            {"x1": 0.5, "x2": -0.3, "cat": "a"},
            {"x1": 1.0, "x2": 0.0, "cat": "b"},
        ])
        assert len(pred) == 2

    def test_predict_dataframe(self, regression_df):
        model = ml.train(regression_df, target="price")
        X = regression_df.drop(columns=["price"]).head(5)
        pred = model.predict(X)
        assert len(pred) == 5

    def test_predict_missing_columns(self, regression_df):
        model = ml.train(regression_df, target="price")
        with pytest.raises(ValueError, match="Missing columns"):
            model.predict({"x1": 0.5})

    def test_save_and_load(self, regression_df, tmp_path):
        model = ml.train(regression_df, target="price")
        path = str(tmp_path / "model.joblib")
        model.save(path)

        loaded = ml.load(path)
        assert loaded.metrics == model.metrics
        assert loaded.target_column == model.target_column
        assert loaded.feature_columns == model.feature_columns

        # Predictions should match
        sample = {"x1": 0.5, "x2": -0.3, "cat": "a"}
        np.testing.assert_array_equal(model.predict(sample), loaded.predict(sample))

    def test_repr(self, regression_df):
        model = ml.train(regression_df, target="price")
        r = repr(model)
        assert "Model(" in r
        assert "metrics=" in r


# ---------------------------------------------------------------------------
# Tests: API
# ---------------------------------------------------------------------------

class TestAPI:
    def test_predict_endpoint(self, regression_df):
        from fastapi.testclient import TestClient
        from onelinerml.api import create_app

        model = ml.train(regression_df, target="price")
        app = create_app(model)
        client = TestClient(app)

        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        resp = client.post("/predict", json={
            "data": [{"x1": 0.5, "x2": -0.3, "cat": "a"}]
        })
        assert resp.status_code == 200
        assert "predictions" in resp.json()

    def test_predict_missing_model(self):
        from fastapi.testclient import TestClient
        from onelinerml.api import create_app

        app = create_app(None)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/predict", json={"data": [{"x": 1}]})
        # Will either be 503 (no model) or 500 (load failed)
        assert resp.status_code in (500, 503)
