# train_ann.py

import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import joblib
import os

# 1. Fetch dataset from UCI repo
dataset = fetch_ucirepo(id=697)

# 2. Extract features and target as DataFrames
X = dataset.data.features
y = dataset.data.targets

# 3. Check and combine if y is multi-column DataFrame
if isinstance(y, pd.DataFrame):
    y = y.iloc[:, 0]

# 4. Encode categorical columns in X (if any)
cat_cols = X.select_dtypes(include='object').columns
for col in cat_cols:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# 5. Encode target column
y = LabelEncoder().fit_transform(y.astype(str))

# 6. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# 7. Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8. Train ANN model
ann = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', solver='adam', max_iter=200, random_state=42)
ann.fit(X_train_scaled, y_train)

# 9. Evaluate
y_pred = ann.predict(X_test_scaled)
print(classification_report(y_test, y_pred, target_names=['Dropout', 'Enrolled', 'Graduate']))

# 10. Save model & scaler in proper folders
os.makedirs('model', exist_ok=True)
os.makedirs('data', exist_ok=True)
joblib.dump(ann, 'model/dropout_ann_model.joblib')
joblib.dump(scaler, 'model/scaler.joblib')
X.to_csv('data/student-dropout.csv', index=False)

print("Model and scaler saved successfully.")
