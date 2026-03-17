from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


def create_app(model=None):
    """Create a FastAPI app for serving predictions from a Model."""
    state = {"model": model}

    @asynccontextmanager
    async def lifespan(app):
        if state["model"] is None:
            import os
            from onelinerml.model import Model
            path = os.getenv("ONELINERML_MODEL_PATH", "model.joblib")
            state["model"] = Model.load(path)
        yield

    app = FastAPI(lifespan=lifespan)

    class PredictRequest(BaseModel):
        data: list

    @app.get("/")
    async def root():
        return {"message": "OneLinerML API"}

    @app.get("/health")
    async def health():
        m = state["model"]
        return {"status": "ok", "model": type(m.estimator).__name__ if m else None}

    @app.post("/predict")
    async def predict(req: PredictRequest):
        m = state["model"]
        if m is None:
            raise HTTPException(status_code=503, detail="No model loaded")
        try:
            preds = m.predict(req.data)
            return {"predictions": preds.tolist()}
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    return app


# Default app for uvicorn CLI usage: uvicorn onelinerml.api:app
app = create_app()
