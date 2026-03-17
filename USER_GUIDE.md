# OneLinerML User Guide

## What OneLinerML does

Train an ML model and deploy it as an API with a metrics dashboard — in one line:

```python
import onelinerml as ml
ml.train("data.csv", target="price").deploy()
```

You immediately get:
- A prediction API at `http://localhost:8000/predict`
- An interactive dashboard at `http://localhost:8000/dashboard`
- Feature correlations, importance, residual plots, and more

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

By default, `auto` picks the best model type:
- Numeric target (many unique values) -> GradientBoostingRegressor
- Categorical/few-class target -> GradientBoostingClassifier

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

# Batch
model.predict([
    {"bedrooms": 3, "sqft": 1500},
    {"bedrooms": 2, "sqft": 900},
])
```

## 3. Save and load

Everything is saved as one file:

```python
model.save("model.joblib")
model = ml.load("model.joblib")
```

## 4. Deploy with dashboard

### From Python

```python
# Deploy locally
model.deploy(port=8000)

# Deploy with public URL (requires: pip install pyngrok)
model.deploy(public=True)

# Or chain it
ml.train("data.csv", target="price").deploy()
```

### From CLI

```bash
onelinerml-train data.csv --target price --save-to model.joblib
onelinerml-serve model.joblib --port 8000
onelinerml-serve model.joblib --public  # public URL via ngrok
```

### Dashboard features

Visit `http://localhost:8000/dashboard` to see:

- **Metrics cards** — R2, MSE, accuracy, F1, etc.
- **Correlation matrix** — heatmap of all numeric feature correlations
- **Feature importance** — ranked bar chart
- **Predictions vs actual** — scatter plot (regression) or confusion matrix (classification)
- **Target distribution** — histogram or pie chart
- **Feature overview** — table with types, statistics, and missing value counts
- **API usage** — ready-to-copy curl command

### API endpoints

- `GET /health` — `{"status": "ok", "model": "GradientBoostingRegressor"}`
- `GET /metrics` — full metrics and analytics as JSON
- `GET /dashboard` — interactive HTML dashboard
- `POST /predict` — predictions:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [{"bedrooms": 3, "sqft": 1500}]}'
# => {"predictions": [350000.0]}
```

## 5. Production tips

- Use `model.save()` to persist, `ml.load()` to reload
- Run behind Docker/systemd/Kubernetes for uptime
- Use a reverse proxy (NGINX, Caddy) for TLS
- Use `--public` with ngrok for quick sharing
