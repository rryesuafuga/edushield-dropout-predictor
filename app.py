# app.py

import gradio as gr
import joblib
import numpy as np
import pandas as pd

# Load model & scaler
model = joblib.load('model/dropout_ann_model.joblib')
scaler = joblib.load('model/scaler.joblib')

# Get feature names for the form
feature_names = [
    # add all feature names as per dataset header, e.g.:
    "Marital status", "Application mode", "Application order", "Course",
    "Daytime/evening attendance", "Previous qualification",
    "Previous qualification (grade)", "Nacionality", "Mother's qualification",
    "Father's qualification", "Mother's occupation", "Father's occupation",
    "Admission grade", "Displaced", "Educational special needs", "Debtor",
    "Tuition fees up to date", "Gender", "Scholarship holder", "Age at enrollment",
    "International", "Curricular unit 1st sem. (credited)",
    "Curricular unit 1st sem. (enrolled)", "Curricular unit 1st sem. (evaluations)",
    "Curricular unit 1st sem. (approved)", "Curricular unit 1st sem. (grade)",
    "Curricular unit 1st sem. (without evaluations)", "Curricular unit 2nd sem. (credited)",
    "Curricular unit 2nd sem. (enrolled)", "Curricular unit 2nd sem. (evaluations)",
    "Curricular unit 2nd sem. (approved)", "Curricular unit 2nd sem. (grade)",
    "Curricular unit 2nd sem. (without evaluations)", "Unemployment rate",
    "Inflation rate", "GDP"
]

def predict(*inputs):
    X = np.array(inputs).reshape(1, -1)
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    labels = ['Dropout', 'Enrolled', 'Graduate']
    return labels[pred]

inputs = [gr.Number(label=f) for f in feature_names]
demo = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=gr.Text(label="Predicted Status"),
    title="EduShield UG: Student Dropout Risk Predictor"
)

if __name__ == "__main__":
    demo.launch()
