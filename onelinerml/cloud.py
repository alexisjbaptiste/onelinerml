import os
import subprocess
import time
import json
from pyngrok import ngrok

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def _load_config(config_path):
    with open(config_path, "r") as f:
        if config_path.endswith((".yml", ".yaml")):
            if yaml is None:
                raise ImportError("PyYAML is required for YAML config files")
            return yaml.safe_load(f)
        return json.load(f)


def deploy_api_and_dashboard_cloud(config_path):
    """Deploy the API and dashboard using a cloud tunnel (ngrok)."""
    if not config_path or not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = _load_config(config_path)
    api_port = config.get("api_port", 8000)
    dashboard_port = config.get("dashboard_port", 8503)
    region = config.get("region")
    auth_token = config.get("ngrok_auth_token")

    if auth_token:
        ngrok.set_auth_token(auth_token)
    if region:
        ngrok.set_default_region(region)

    subprocess.Popen([
        "python3", "-m", "uvicorn", "onelinerml.api:app",
        "--host", "0.0.0.0", "--port", str(api_port)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(5)

    subprocess.Popen([
        "streamlit", "run", "onelinerml/dashboard.py",
        "--server.port", str(dashboard_port),
        "--server.enableCORS=false",
        "--server.enableWebsocketCompression=false"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(5)

    api_tunnel = ngrok.connect(api_port)
    dash_tunnel = ngrok.connect(dashboard_port)

    api_url = api_tunnel.public_url
    dash_url = dash_tunnel.public_url

    print("API is accessible at:", api_url)
    print("Dashboard is accessible at:", dash_url)

    return api_url, dash_url
