# EduShield UG: Student Dropout Risk Predictor

EduShield UG is a web application that predicts whether a university student is likely to **drop out**, **stay enrolled**, or **graduate**. You enter information about a student — their background, grades, and enrollment details — and the app returns a prediction along with confidence scores for each outcome.

It is built for educators, academic advisors, and institutional researchers who want an early-warning tool to identify students who may need additional support.

## Key Features

- **Three-outcome prediction** — classifies students as Dropout, Enrolled, or Graduate, with probability percentages for each.
- **Easy-to-use web interface** — a Gradio-powered form with dropdowns and number fields. No coding required to make predictions.
- **Flexible input** — all fields are optional. If you leave something blank, the app fills in a sensible default from the training data.
- **Pre-trained model included** — the repository ships with a trained model, so you can run predictions immediately without retraining.
- **Lightweight and self-contained** — no database needed. Everything runs from a few Python files and two small model files.

## How It Works

Under the hood, EduShield uses an **Artificial Neural Network** (a multi-layer perceptron) trained with scikit-learn. The model has two hidden layers (100 and 50 neurons) and was trained on over 4,400 real student records from a Portuguese university system.

When you submit the form, the app:

1. Converts your inputs into the numeric format the model expects.
2. Scales the features using the same scaler that was used during training.
3. Runs the data through the neural network.
4. Returns the predicted outcome and the probability of each class (e.g., "Dropout: 15%, Enrolled: 22%, Graduate: 63%").

## The Data

The training data comes from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) and contains 4,424 student records with 36 features, including:

| Category | Examples |
|---|---|
| **Demographics** | Age, gender, marital status, nationality |
| **Academic background** | Previous qualifications, admission grade |
| **Family** | Parents' education level and occupation |
| **Financial** | Scholarship holder, tuition fees up to date, debtor status |
| **1st & 2nd semester performance** | Units enrolled, units approved, grades |
| **Economic context** | Unemployment rate, inflation rate, GDP at time of enrollment |

## Project Structure

```
edushield-dropout-predictor/
├── app.py                  # Gradio web app (main entry point)
├── train_ann.py            # Script to train and save the model
├── requirements.txt        # Python dependencies
├── data/
│   └── student-dropout.csv # Training dataset (4,424 records)
└── model/
    ├── dropout_ann_model.joblib   # Trained neural network
    └── scaler.joblib              # Feature scaler
```

## Getting Started

### Prerequisites

- Python 3.8 or later
- pip (Python package manager)

### 1. Clone the repository

```bash
git clone https://github.com/rryesuafuga/edushield-dropout-predictor.git
cd edushield-dropout-predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

This installs Gradio, scikit-learn, pandas, numpy, joblib, and ucimlrepo.

### 3. (Optional) Retrain the model

A pre-trained model is already included in the `model/` folder, so you can skip this step if you just want to run the app. If you want to retrain from scratch:

```bash
python train_ann.py
```

This downloads the dataset from the UCI repository, preprocesses it, trains the neural network, and saves the model and scaler to the `model/` folder. You only need to do this once.

### 4. Launch the app

```bash
python app.py
```

The app will start and print a local URL (typically `http://localhost:7860`). Open that URL in your browser, fill in the student form, and click **Submit** to get a prediction.

## Deploying to Hugging Face Spaces

This app is designed to work on [Hugging Face Spaces](https://huggingface.co/spaces) out of the box:

1. Create a new Space on Hugging Face and select **Gradio** as the SDK.
2. Push this repository to the Space.
3. The app will build and launch automatically — no extra configuration needed.

## License

This project uses data from the UCI Machine Learning Repository. Please see the [dataset page](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) for data usage terms.
