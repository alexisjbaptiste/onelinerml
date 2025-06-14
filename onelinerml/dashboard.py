# onelinerml/dashboard.py
import streamlit as st
import pandas as pd
from onelinerml.train import train

st.title("OneLinerML Dashboard")

data_file = st.file_uploader("Upload CSV Data", type=["csv"])
model_choice = st.selectbox("Select Model", ["linear_regression", "random_forest", "logistic_regression", "random_forest_classifier"])
target_column = st.text_input("Target Column", "target")

if data_file is not None and st.button("Train Model"):
    data = pd.read_csv(data_file)
    model_instance, metrics = train(data, model=model_choice, target_column=target_column)
    st.write("Evaluation Metrics:")
    st.json(metrics)
