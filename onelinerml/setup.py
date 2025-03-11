# setup.py
from setuptools import setup, find_packages

setup(
    name="onelinerml",
    version="0.1.4",
    description="A one-line machine learning library with remote API and dashboard support.",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "scikit-learn",
        "fastapi",
        "uvicorn",
        "streamlit",
        "pyngrok",
        "python-multipart",  # Required by FastAPI to handle form data
    ],
    entry_points={
        "console_scripts": [
            "onelinerml-serve=onelinerml.serve:main"
        ]
    },
)
