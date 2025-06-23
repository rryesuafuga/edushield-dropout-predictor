# EduShield UG: Student Dropout Risk Predictor

This app predicts the risk of student dropout using an Artificial Neural Network trained on the [UCI student dropout dataset](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success).

## How it works

- Input student features in the form.
- The trained ANN model predicts if the student is at risk of Dropout, likely to Graduate, or is currently Enrolled.

## Structure

- `data/` - dataset files
- `model/` - saved models
- `train_ann.py` - script for training and saving the ANN model
- `app.py` - Gradio app for Hugging Face Spaces or local use

## How to run locally

```bash
pip install -r requirements.txt
python train_ann.py      # to train the model (run once)
python app.py            # to launch the app
