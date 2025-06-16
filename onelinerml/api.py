# onelinerml/api.py

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from onelinerml.train import train, deploy_model_from_path
import pandas as pd
from io import StringIO
import joblib, pickle, os

app = FastAPI()

MODEL_PATH = "trained_model.joblib"
model_global = None

@app.on_event("startup")
def load_model_on_startup():
    global model_global
    if os.path.exists(MODEL_PATH):
        try:
            model_global = joblib.load(MODEL_PATH)
        except Exception:
            with open(MODEL_PATH, "rb") as f:
                model_global = pickle.load(f)

class PredictRequest(BaseModel):
    data: list

@app.get("/")
async def root():
    return {"message": "Welcome to the OneLinerML API!"}

@app.post("/train")
async def train_endpoint(
    file: UploadFile = File(...),
    model: str = "linear_regression",
    target_column: str = "target",
    deploy_mode: str = "local",
    config_path: str | None = None
):
    global model_global
    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode("utf-8")))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid CSV format.")
    trained_model, metrics = train(
        df,
        model=model,
        target_column=target_column,
        model_save_path=MODEL_PATH,
        deploy_mode=deploy_mode,
        config_path=config_path,
        deploy=False
    )
    model_global = trained_model
    return {"metrics": metrics}

@app.post("/deploy")
async def deploy_endpoint(
    file: UploadFile = File(...),
    deploy_mode: str = "local",
    config_path: str | None = None
):
    """
    Upload a pre-trained model (joblib or pickle) and deploy it.
    """
    contents = await file.read()
    with open(MODEL_PATH, "wb") as f:
        f.write(contents)
    api_url, dash_url = deploy_model_from_path(
        MODEL_PATH,
        deploy_mode=deploy_mode,
        config_path=config_path
    )
    return {"api_url": api_url, "dashboard_url": dash_url}

@app.post("/predict")
async def predict_endpoint(req: PredictRequest):
    global model_global
    if model_global is None:
        raise HTTPException(status_code=400, detail="Model not available.")
    data = req.data
    if isinstance(data, list) and (not data or not isinstance(data[0], list)):
        data = [data]
    try:
        preds = model_global.predict(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
    return {"prediction": preds.tolist()}
