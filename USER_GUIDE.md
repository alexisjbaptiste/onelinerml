# OneLinerML User Guide

## What OneLinerML does

OneLinerML gives you a minimal workflow for:
- Training an ML model from a CSV or DataFrame in one function call
- Automatic preprocessing (missing values, categorical encoding, scaling)
- Saving/loading everything as a single file
- Deploying a prediction API with one command

## Installation

```bash
pip install onelinerml
```

## 1. Train a model

```python
import onelinerml as ml

model = ml.train("data.csv", target="price")
print(model.metrics)  # {'mse': 0.42, 'r2': 0.95}
```

### Choose a model

By default, `auto` picks the best model type based on your target:
- Numeric target (many unique values) -> GradientBoostingRegressor
- Categorical/few-class target -> GradientBoostingClassifier

You can also specify explicitly:

```python
model = ml.train("data.csv", target="price", model="random_forest")
```

Pass sklearn parameters directly:

```python
model = ml.train("data.csv", target="price", model="random_forest", n_estimators=200)
```

### Use a DataFrame

```python
import pandas as pd

df = pd.read_csv("data.csv")
model = ml.train(df, target="price")
```

## 2. Predict

```python
# Single prediction
model.predict({"bedrooms": 3, "sqft": 1500})

# Batch prediction
model.predict([
    {"bedrooms": 3, "sqft": 1500},
    {"bedrooms": 2, "sqft": 900},
])
```

## 3. Save and load

Everything (model, preprocessor, metadata) is saved as one file:

```python
model.save("model.joblib")
model = ml.load("model.joblib")
```

## 4. Deploy as an API

### From Python

```python
model.serve(port=8000)
# or chain it
ml.train("data.csv", target="price").serve()
```

### From CLI

```bash
onelinerml-train data.csv --target price --save-to model.joblib
onelinerml-serve model.joblib --port 8000
```

### API endpoints

- `GET /health` — returns `{"status": "ok", "model": "GradientBoostingRegressor"}`
- `POST /predict` — send JSON, get predictions:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [{"bedrooms": 3, "sqft": 1500}]}'
# => {"predictions": [350000.0]}
```

## 5. Production tips

- Run behind a process manager (Docker, systemd, Kubernetes).
- Use a reverse proxy (NGINX, Caddy) for TLS and rate limiting.
- Save model files in a persistent volume.
