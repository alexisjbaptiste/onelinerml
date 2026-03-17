# OneLinerML

Train and deploy ML models in one line. Get an API URL and a metrics dashboard instantly.

## Install

```bash
pip install onelinerml
```

## One Line: Train + Deploy + Dashboard

```python
import onelinerml as ml

ml.train("data.csv", target="price").deploy()
```

That's it. You get:
- **API URL** at `http://localhost:8000/predict`
- **Dashboard** at `http://localhost:8000/dashboard` with correlations, feature importance, residuals, and more
- **Health check** at `http://localhost:8000/health`

## Step by Step

```python
import onelinerml as ml

# Train
model = ml.train("data.csv", target="price")
print(model.metrics)  # {'mse': 0.42, 'r2': 0.95}

# Predict
model.predict({"bedrooms": 3, "sqft": 1500})

# Save / load
model.save("model.joblib")
model = ml.load("model.joblib")

# Deploy with dashboard
model.deploy(port=8000)
```

## Public URL (ngrok)

```bash
pip install pyngrok
```

```python
ml.train("data.csv", target="price").deploy(public=True)
# => Public URL: https://abc123.ngrok.io/predict
# => Public Dash: https://abc123.ngrok.io/dashboard
```

## Dashboard

The dashboard shows:
- **Model metrics** (R2, MSE, accuracy, F1, etc.)
- **Correlation matrix** for all numeric features
- **Feature importance** chart
- **Predictions vs actual** scatter plot (regression) or confusion matrix (classification)
- **Target distribution** histogram or pie chart
- **Feature overview** table with types, stats, and missing values
- **API usage** example with curl command

## CLI

```bash
# Train and save
onelinerml-train data.csv --target price --save-to model.joblib

# Deploy with dashboard
onelinerml-serve model.joblib --port 8000

# Deploy with public URL
onelinerml-serve model.joblib --public
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Send `{"data": [{"col": val}]}`, get `{"predictions": [...]}` |
| `/dashboard` | GET | Interactive metrics dashboard |
| `/metrics` | GET | Raw metrics and analytics as JSON |
| `/health` | GET | Health check |

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

```python
ml.train("data.csv", target="price", model="random_forest", n_estimators=200)
```

## License

MIT
