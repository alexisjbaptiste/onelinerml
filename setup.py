from setuptools import setup, find_packages

setup(
    name="onelinerml",
    version="0.1.6",
    description="A one-line machine learning library with API and dashboard deployment.",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "scikit-learn",
        "fastapi",
        "uvicorn",
        "streamlit",
        "pyngrok",
        "python-multipart"
    ],
    entry_points={
        "console_scripts": [
            "onelinerml-serve=onelinerml.train:train"
        ]
    },
)
