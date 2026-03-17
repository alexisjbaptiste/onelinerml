from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel


def create_app(model=None):
    """Create a FastAPI app with prediction API and metrics dashboard."""
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
        return {"message": "OneLinerML API",
                "endpoints": {
                    "predict": "/predict",
                    "dashboard": "/dashboard",
                    "health": "/health",
                    "metrics": "/metrics",
                }}

    @app.get("/health")
    async def health():
        m = state["model"]
        return {"status": "ok", "model": type(m.estimator).__name__ if m else None}

    @app.get("/metrics")
    async def metrics():
        m = state["model"]
        if m is None:
            raise HTTPException(status_code=503, detail="No model loaded")
        return {
            "metrics": m.metrics,
            "analytics": m.analytics,
            "model": type(m.estimator).__name__,
        }

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard():
        m = state["model"]
        if m is None:
            raise HTTPException(status_code=503, detail="No model loaded")
        if m.analytics is None:
            return HTMLResponse(
                "<h1>No analytics available</h1>"
                "<p>Re-train with <code>ml.train()</code> to generate analytics.</p>",
                status_code=200,
            )
        from onelinerml.dashboard import render_dashboard
        return render_dashboard(m.metrics, m.analytics)

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
