# main.py
# This file can be used as an alternative entry point to start the FastAPI server only.
import uvicorn
from onelinerml.api import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
