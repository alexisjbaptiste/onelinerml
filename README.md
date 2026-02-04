# OneLinerML 🔥

OneLinerML keeps training and deployment dead simple: train once, save artifacts, and serve a prediction API with one command.

## Quickstart

Install:

```bash
pip install onelinerml
```

Train:

```bash
onelinerml-train data.csv --target price
```

Deploy in production (FastAPI + Uvicorn):

```bash
onelinerml-serve --model-path trained_model.joblib --preprocessor-path preprocessor.joblib
```

Need more examples? See the new [User Guide](USER_GUIDE.md).

## License

This project is licensed under the [MIT License](LICENSE).
