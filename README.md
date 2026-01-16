# OneLinerML 🔥

# For Developers

💡 Build and Deploy ML Models Without Hassle
You're a developer—your time is valuable. You don’t want to spend hours fine-tuning machine learning models when all you need is a working solution. OneLinerML lets you train, evaluate, and deploy ML models with just one line of code.

⚡ Why Developers Love OneLinerML

✅ Minimal Code, Maximum Impact – No complex pipelines. Just pass your dataset, and OneLinerML does the rest.

✅ Prebuilt ML Workflows – Forget about data preprocessing, hyperparameter tuning, or feature scaling—we handle it.

✅ Instant API Deployment – Train a model and deploy it as an API with zero setup.

✅ Scikit-Learn & XGBoost Support – Integrates seamlessly with your favorite ML frameworks.


🚀 Get Started in Seconds
from onelinerml import train
# Train a model in one line
model, report = train("data.csv", target="price", algorithm="random_forest")
print(report)

No debugging nightmares. No messy dependencies. Just results

🔥 Ready to level up your AI projects? Install OneLinerML now:
pip install OneLinerML

## Deployment

Run locally with localtunnel in one line:

```python
from onelinerml import train_and_deploy

result = train_and_deploy("data.csv", target="price", algorithm="random_forest")
print(result["api_url"], result["dashboard_url"])
```

CLI example:

```bash
onelinerml-serve data.csv --target price --model random_forest --deploy-mode local
```

Cloud deployment using a config file:

```bash
onelinerml-serve data.csv --target price --deploy-mode cloud --config-path examples/cloud_config.yaml
```

## Installing Dependencies

Set up a clean Python environment and install the packages required by OneLinerML:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

# For Data Scientists

🚀 Focus on Insights, Not Implementation

You love data science, but you don’t want to waste time on repetitive ML tasks. EasyML automates data preprocessing, feature engineering, and model training—so you can focus on delivering insights.

⚡ What Makes EasyML a Game-Changer?

✅ AutoML Without Complexity – Automatically preprocesses your data, selects features, and optimizes your model.

✅ Interpretability & Explainability – Get model performance metrics out of the box.

✅ Supports Your Favorite ML Tools – Works with Random Forest, XGBoost, Logistic Regression, and more.

✅ Fast Iteration & Prototyping – Test multiple models in seconds, not hours.

📊 Train, Compare, and Deploy Models Effortlessly

🚀 Speed up your workflow and deliver AI solutions faster.

🔥 Start using EasyML today:
pip install OneLinerML


## License

This project is licensed under the [MIT License](LICENSE).
