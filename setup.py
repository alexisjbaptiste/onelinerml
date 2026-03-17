from setuptools import setup, find_packages

setup(
    name="onelinerml",
    version="0.2.0",
    description="Train and deploy ML models in one line.",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas",
        "scikit-learn",
        "fastapi",
        "uvicorn",
        "joblib",
        "numpy",
    ],
    entry_points={
        "console_scripts": [
            "onelinerml-train=onelinerml.train:main",
            "onelinerml-serve=onelinerml.serve:main",
        ]
    },
)
