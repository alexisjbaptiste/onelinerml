# onelinerml/train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from onelinerml.preprocessing import preprocess_data
from onelinerml.models import get_model
from onelinerml.evaluation import evaluate_model

import subprocess
import time
import urllib.request
import os
import joblib
import pickle
from onelinerml.cloud import deploy_api_and_dashboard_cloud

def deploy_api_and_dashboard_localtunnel(
    api_port=8000,
    dashboard_port=8503
):
    """
    Launch the API and Streamlit dashboard, then expose them via localtunnel.
    Returns (api_url, dashboard_url).
    """
    # 1. Ensure localtunnel is installed
    subprocess.run(["npm", "install", "-g", "localtunnel"], check=True)

    # 2. Print external IP (optional)
    try:
        ip = urllib.request.urlopen("https://icanhazip.com").read().decode().strip()
        print(f"Your external IP is: {ip}")
    except Exception as e:
        print(f"Could not fetch external IP: {e}")

    # 3. Start the API server
    subprocess.Popen(
        [
            "python3", "-m", "uvicorn", "onelinerml.api:app",
            "--host", "0.0.0.0", "--port", str(api_port)
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(5)

    # 4. Start the Streamlit dashboard
    subprocess.Popen(
        [
            "streamlit", "run", "onelinerml/dashboard.py",
            "--server.port", str(dashboard_port),
            "--server.enableCORS=false",
            "--server.enableWebsocketCompression=false"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    time.sleep(5)

    # 5. Expose API via localtunnel
    api_lt = subprocess.Popen(
        ["npx", "localtunnel", "--port", str(api_port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )
    api_url = api_lt.stdout.readline().decode().strip()

    # 6. Expose Dashboard via localtunnel
    dash_lt = subprocess.Popen(
        ["npx", "localtunnel", "--port", str(dashboard_port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )
    dash_url = dash_lt.stdout.readline().decode().strip()

    print("API is accessible at:", api_url)
    print("Dashboard is accessible at:", dash_url)

    return api_url, dash_url

def deploy_model_from_path(
    model_save_path="trained_model.joblib",
    preprocessor_save_path="preprocessor.joblib",
    api_port=8000,
    dashboard_port=8503,
    deploy_mode="local",
    config_path=None
):
    """
    Load a pre-trained model (joblib or pickle) and deploy API + dashboard via localtunnel.
    """
    if not os.path.exists(model_save_path):
        raise FileNotFoundError(f"Model file not found: {model_save_path}")

    # Load the model
    try:
        model_instance = joblib.load(model_save_path)
    except Exception:
        with open(model_save_path, 'rb') as f:
            model_instance = pickle.load(f)

    preprocessor_instance = None
    if os.path.exists(preprocessor_save_path):
        try:
            preprocessor_instance = joblib.load(preprocessor_save_path)
        except Exception:
            with open(preprocessor_save_path, 'rb') as f:
                preprocessor_instance = pickle.load(f)

    # Register model for API so predictions use this instance
    import onelinerml.api as api_module
    api_module.model_global = model_instance
    api_module.preprocessor_global = preprocessor_instance

    # Spin up services
    if deploy_mode == "cloud":
        return deploy_api_and_dashboard_cloud(config_path)
    return deploy_api_and_dashboard_localtunnel(api_port, dashboard_port)

def train(
    data_source,
    model="linear_regression",
    target_column="target",
    test_size=0.2,
    random_state=42,
    model_save_path="trained_model.joblib",
    preprocessor_save_path="preprocessor.joblib",
    api_port=8000,
    dashboard_port=8503,
    deploy_mode="local",
    config_path=None,
    deploy: bool = True,
    **kwargs
):
    """
    Full training pipeline:
      - Load CSV or DataFrame
      - Preprocess
      - Train/test split
      - Fit model
      - Evaluate
      - Save model
      - Optionally deploy API + dashboard via local or cloud tunnel
    Parameters
    ----------
    deploy : bool, optional
        If True, deploy the API and dashboard after training. Defaults to True.
    """
    # Load data
    if isinstance(data_source, str):
        data = pd.read_csv(data_source)
    else:
        data = data_source

    # Preprocess
    X, y, preprocessor = preprocess_data(data, target_column)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Train
    model_instance = get_model(model, **kwargs)
    model_instance.fit(X_train, y_train)

    # Evaluate
    metrics = evaluate_model(model_instance, X_test, y_test)

    # Save
    joblib.dump(model_instance, model_save_path)
    joblib.dump(preprocessor, preprocessor_save_path)

    # Deploy if requested
    if deploy:
        if deploy_mode == "cloud":
            api_url, dash_url = deploy_api_and_dashboard_cloud(config_path)
        else:
            api_url, dash_url = deploy_api_and_dashboard_localtunnel(api_port, dashboard_port)

        # Report URLs if deployment happened
        print("API is accessible at:", api_url)
        print("Dashboard is accessible at:", dash_url)

    # Report
    print("Evaluation Metrics:", metrics)
    print("Model saved at:", model_save_path)

    return model_instance, metrics


def main():
    """Console entry point for training and deployment."""
    import argparse

    parser = argparse.ArgumentParser(description="Train and deploy with OneLinerML")
    parser.add_argument("data", help="CSV data path")
    parser.add_argument("--model", default="linear_regression")
    parser.add_argument("--target", dest="target_column", default="target")
    parser.add_argument(
        "--deploy-mode",
        choices=["local", "cloud"],
        default="local",
        help="Deployment mode"
    )
    parser.add_argument("--config-path")

    args = parser.parse_args()

    train(
        args.data,
        model=args.model,
        target_column=args.target_column,
        deploy_mode=args.deploy_mode,
        config_path=args.config_path,
    )

