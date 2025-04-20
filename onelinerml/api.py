# onelinerml/api.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from onelinerml.train import train
import pandas as pd
from io import StringIO
import joblib
import os

app = FastAPI()

MODEL_PATH = "trained_model.joblib"

# Load model at startup if exists
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

# Global variable to store the trained model
model_global = load_model()

class TrainRequest(BaseModel):
    csv_data: str
    model: str = "linear_regression"
    target_column: str = "target"

class PredictRequest(BaseModel):
    data: list

@app.get("/")
async def root():
    return {"message": "Welcome to the OneLinerML API!"}

@app.post("/train")
async def train_endpoint(file: UploadFile = File(...), model: str = "linear_regression", target_column: str = "target"):
    global model_global
    try:
        contents = await file.read()
        data = pd.read_csv(StringIO(contents.decode("utf-8")))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid file format or content.")

    trained_model, metrics = train(data, model=model, target_column=target_column, model_save_path=MODEL_PATH)
    model_global = trained_model
    return {"metrics": metrics}

@app.post("/predict")
async def predict_endpoint(req: PredictRequest):
    global model_global
    if model_global is None:
        raise HTTPException(status_code=400, detail="Model not trained yet.")

    input_data = req.data
    if isinstance(input_data, list) and (len(input_data) == 0 or not isinstance(input_data[0], list)):
        input_data = [input_data]

    try:
        prediction = model_global.predict(input_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    return {"prediction": prediction.tolist()}
