import argparse

from onelinerml.model import Model


def serve(model_path="model.joblib", host="0.0.0.0", port=8000):
    """Load a saved model and start the prediction API server."""
    model = Model.load(model_path)
    model.serve(host=host, port=port)


def main():
    parser = argparse.ArgumentParser(description="Serve OneLinerML predictions")
    parser.add_argument("model_path", nargs="?", default="model.joblib",
                        help="Path to saved model file")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    serve(args.model_path, host=args.host, port=args.port)
