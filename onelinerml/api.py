# onelinerml/api.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from onelinerml.train import train  # For retraining via API if needed
import pandas as pd
from io import StringIO

app = FastAPI()

# Global variable to store the trained model
model_global = None

class TrainRequest(BaseModel):
    csv_data: str            # CSV data as a string
    model: str = "linear_regression"
    target_column: str = "target"

# For prediction, we expect the user to supply input data as a list.
class PredictRequest(BaseModel):
    data: list  # e.g., [feature1, feature2, ...] for one instance
                # or [[f11, f12, ...], [f21, f22, ...]] for multiple instances

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
    
    # Train the model and update the global variable
    trained_model, metrics = train(data, model=model, target_column=target_column)
    model_global = trained_model
    return {"metrics": metrics}

@app.post("/predict")
async def predict_endpoint(req: PredictRequest):
    global model_global
    if model_global is None:
        raise HTTPException(status_code=400, detail="Model not trained yet.")
    
    # If the input data is a single instance (i.e., a 1D list), wrap it in a list.
    input_data = req.data
    if isinstance(input_data, list) and (len(input_data) == 0 or not isinstance(input_data[0], list)):
        input_data = [input_data]
    
    try:
        prediction = model_global.predict(input_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
    
    return {"prediction": prediction.tolist()}
