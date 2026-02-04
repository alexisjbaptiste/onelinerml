from setuptools import setup, find_packages

setup(
    name="onelinerml",
    version="0.1.11",
    description="A one-line machine learning library for training and serving models.",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "scikit-learn",
        "fastapi",
        "uvicorn",
        "joblib",
        "numpy",
        "pydantic",
    ],
    entry_points={
        "console_scripts": [
            "onelinerml-train=onelinerml.train:main",
            "onelinerml-serve=onelinerml.serve:main",
        ]
    },
)
