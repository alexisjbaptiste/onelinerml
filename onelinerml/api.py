# onelinerml/api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import pickle
import pandas as pd

app = FastAPI()

MODEL_PATH = os.getenv("ONELINERML_MODEL_PATH", "trained_model.joblib")
PREPROCESSOR_PATH = os.getenv("ONELINERML_PREPROCESSOR_PATH", "preprocessor.joblib")
model_global = None
preprocessor_global = None

@app.on_event("startup")
def load_model_on_startup():
    global model_global, preprocessor_global
    if os.path.exists(MODEL_PATH):
        try:
            model_global = joblib.load(MODEL_PATH)
        except Exception:
            with open(MODEL_PATH, "rb") as f:
                model_global = pickle.load(f)
    if os.path.exists(PREPROCESSOR_PATH):
        try:
            preprocessor_global = joblib.load(PREPROCESSOR_PATH)
        except Exception:
            with open(PREPROCESSOR_PATH, "rb") as f:
                preprocessor_global = pickle.load(f)

class PredictRequest(BaseModel):
    data: list

@app.get("/")
async def root():
    return {"message": "Welcome to the OneLinerML API!"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict_endpoint(req: PredictRequest):
    global model_global, preprocessor_global
    if model_global is None:
        raise HTTPException(status_code=400, detail="Model not available.")
    data = req.data
    if isinstance(data, list) and (not data or not isinstance(data[0], list)):
        data = [data]
    try:
        if preprocessor_global is not None:
            df = pd.DataFrame(data)
            data = preprocessor_global.transform(df)
        preds = model_global.predict(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
    return {"prediction": preds.tolist()}
