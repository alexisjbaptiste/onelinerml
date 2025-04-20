# onelinerml/api.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from onelinerml.train import train  # For retraining via API if needed
import pandas as pd
from io import StringIO
import traceback
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        logger.info(f"Starting training with model: {model}, target_column: {target_column}")

        # Read and decode the file content
        contents = await file.read()
        logger.info("File read successfully")

        # Convert to DataFrame
        try:
            data = pd.read_csv(StringIO(contents.decode("utf-8")))
            logger.info(f"Data loaded successfully. Shape: {data.shape}")
        except Exception as e:
            logger.error(f"Error loading CSV: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error loading CSV: {str(e)}")

        # Verify target column exists
        if target_column not in data.columns:
            logger.error(f"Target column '{target_column}' not found in data")
            raise HTTPException(
                status_code=400,
                detail=f"Target column '{target_column}' not found in data. Available columns: {list(data.columns)}"
            )

        # Train the model
        try:
            trained_model, metrics = train(data, model=model, target_column=target_column)
            logger.info(f"Model trained successfully. Metrics: {metrics}")
            model_global = trained_model
            return {"metrics": metrics}
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error during training: {str(e)}")

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

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
