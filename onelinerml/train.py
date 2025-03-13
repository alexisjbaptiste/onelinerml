# onelinerml/train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from onelinerml.preprocessing import preprocess_data
from onelinerml.models import get_model
from onelinerml.evaluation import evaluate_model
import subprocess
import time
from pyngrok import ngrok
import uvicorn

def deploy_api_and_dashboard():
    API_PORT = 8000
    DASHBOARD_PORT = 8501

    # Start the API server in the background
    api_process = subprocess.Popen(
        ["python3", "-m", "uvicorn", "onelinerml.api:app", "--host", "0.0.0.0", "--port", str(API_PORT)]
    )
    time.sleep(5)

    # Start the Streamlit dashboard in the background
    dashboard_process = subprocess.Popen(
        ["streamlit", "run", "onelinerml/dashboard.py", "--server.port", str(DASHBOARD_PORT)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    time.sleep(5)

    # Open ngrok tunnels for both services
    api_tunnel = ngrok.connect(API_PORT)
    dashboard_tunnel = ngrok.connect(DASHBOARD_PORT)

    return api_tunnel.public_url, dashboard_tunnel.public_url

def train(data_source, model="linear_regression", target_column="target", test_size=0.2, random_state=42, api_key=None, **kwargs):
    # Set ngrok auth token using 'api_key' if provided, otherwise check environment variable
    if api_key is not None:
        ngrok.set_auth_token(api_key)
    else:
        import os
        token = os.environ.get("NGROK_AUTHTOKEN")
        if token:
            ngrok.set_auth_token(token)

    # Load data from a CSV file or DataFrame
    if isinstance(data_source, str):
        data = pd.read_csv(data_source)
    else:
        data = data_source

    # Preprocess the data
    X, y = preprocess_data(data, target_column)
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Train the model
    model_instance = get_model(model, **kwargs)
    model_instance.fit(X_train, y_train)
    
    # Evaluate the model
    metrics = evaluate_model(model_instance, X_test, y_test)
    
    # Deploy the API and dashboard, and retrieve public URLs
    api_url, dashboard_url = deploy_api_and_dashboard()
    
    # Print the evaluation metrics and public URLs
    print("Evaluation Metrics:", metrics)
    print("API is accessible at:", api_url)
    print("Dashboard is accessible at:", dashboard_url)
    
    return model_instance, metrics
