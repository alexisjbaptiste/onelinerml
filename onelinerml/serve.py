import argparse
import os

import uvicorn


def serve(
    model_path="trained_model.joblib",
    preprocessor_path="preprocessor.joblib",
    host="0.0.0.0",
    port=8000,
    reload=False,
):
    os.environ["ONELINERML_MODEL_PATH"] = model_path
    os.environ["ONELINERML_PREPROCESSOR_PATH"] = preprocessor_path
    uvicorn.run("onelinerml.api:app", host=host, port=port, reload=reload)


def main():
    parser = argparse.ArgumentParser(description="Serve OneLinerML predictions.")
    parser.add_argument("--model-path", default="trained_model.joblib")
    parser.add_argument("--preprocessor-path", default="preprocessor.joblib")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    serve(
        model_path=args.model_path,
        preprocessor_path=args.preprocessor_path,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
