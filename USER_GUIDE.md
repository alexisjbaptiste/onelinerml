# OneLinerML User Guide

## What OneLinerML does
OneLinerML gives you a minimal workflow for:
- Loading a CSV or DataFrame
- Preprocessing numeric + categorical columns
- Training a lightweight scikit-learn model
- Saving model artifacts
- Serving predictions in production via FastAPI

## Installation

```bash
pip install onelinerml
```

## 1. Train in one line

```python
from onelinerml import train

model, metrics = train("data.csv", target_column="price")
print(metrics)
```

### Choose a model (optional)

Supported models:
- `auto` (default): picks Linear Regression for numeric targets, Logistic Regression for categorical targets
- `linear_regression`
- `random_forest`
- `logistic_regression`
- `random_forest_classifier`

```python
model, metrics = train(
    "data.csv",
    target_column="price",
    model="random_forest",
)
```

Artifacts are saved by default:
- `trained_model.joblib`
- `preprocessor.joblib`

## 2. Deploy in production

Start the API server in one command:

```bash
onelinerml-serve --model-path trained_model.joblib --preprocessor-path preprocessor.joblib
```

The API will be available at `http://0.0.0.0:8000`.

### Health check

```bash
curl http://localhost:8000/health
```

### Predict

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [[1.2, "blue", 9], [3.4, "red", 4]]}'
```

### Custom host/port

```bash
onelinerml-serve --host 0.0.0.0 --port 9000
```

## 3. CLI cheatsheet

Train:

```bash
onelinerml-train data.csv --target price --model auto
```

Serve:

```bash
onelinerml-serve --model-path trained_model.joblib --preprocessor-path preprocessor.joblib
```

## 4. Production tips

- Save artifacts in a persistent volume, then point `onelinerml-serve` at them.
- Run behind a process manager (systemd, Docker, Kubernetes) for uptime.
- Use a reverse proxy (NGINX, Caddy) for TLS and rate limiting.
