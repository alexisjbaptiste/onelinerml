# OneLinerML

Train and deploy ML models in one line.

## Install

```bash
pip install onelinerml
```

## Python API

```python
import onelinerml as ml

# Train on a CSV — returns a Model object
model = ml.train("data.csv", target="price")
# => Trained GradientBoostingRegressor | {'mse': 0.42, 'r2': 0.95}

# Check metrics
print(model.metrics)

# Predict on new data
model.predict({"bedrooms": 3, "sqft": 1500})

# Save and load
model.save("model.joblib")
model = ml.load("model.joblib")

# Deploy as an API server
model.serve(port=8000)

# Or chain it: train and deploy in one line
ml.train("data.csv", target="price").serve()
```

## CLI

```bash
# Train and save
onelinerml-train data.csv --target price --save-to model.joblib

# Serve predictions
onelinerml-serve model.joblib --port 8000
```

## API Endpoints

Once served, the API exposes:

- `GET /` — welcome message
- `GET /health` — health check
- `POST /predict` — send `{"data": [{"col1": val, ...}]}`, get `{"predictions": [...]}`

## Supported Models

| Name | Type |
|------|------|
| `auto` (default) | Auto-detects regression vs classification |
| `linear_regression` | Linear Regression |
| `random_forest` | Random Forest Regressor |
| `gradient_boosting` | Gradient Boosting Regressor |
| `logistic_regression` | Logistic Regression |
| `random_forest_classifier` | Random Forest Classifier |
| `gradient_boosting_classifier` | Gradient Boosting Classifier |

Pass extra parameters to the estimator:

```python
ml.train("data.csv", target="price", model="random_forest", n_estimators=200)
```

## License

MIT
