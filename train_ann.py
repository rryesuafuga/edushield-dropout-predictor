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
import json

# 1. Fetch dataset from UCI repo
dataset = fetch_ucirepo(id=697)

# 2. Extract features and target as DataFrames
X = dataset.data.features
y = dataset.data.targets

# 3. Check and combine if y is multi-column DataFrame
if isinstance(y, pd.DataFrame):
    y = y.iloc[:, 0]

# 4. Handle missing values
print(f"Missing values per column:")
missing_counts = X.isnull().sum()
if missing_counts.sum() > 0:
    print(missing_counts[missing_counts > 0])
    # Fill missing values with median for numeric, mode for categorical
    for col in X.columns:
        if X[col].isnull().any():
            if X[col].dtype == 'object':
                X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown', inplace=True)
            else:
                X[col].fillna(X[col].median(), inplace=True)
else:
    print("No missing values found.")

# 5. Store original feature names before encoding
feature_names = X.columns.tolist()

# 5. Calculate default values (mean for numeric, mode for categorical)
default_values = {}
for col in X.columns:
    if X[col].dtype == 'object':
        # For categorical, use mode
        mode_val = X[col].mode()
        if len(mode_val) > 0:
            default_values[col] = mode_val[0]
        else:
            default_values[col] = X[col].iloc[0]  # fallback to first value
    else:
        # For numeric, use mean
        default_values[col] = float(X[col].mean())

# 6. Encode categorical columns in X (if any)
cat_cols = X.select_dtypes(include='object').columns
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le
    # Update default value to encoded version
    default_values[col] = le.transform([str(default_values[col])])[0]

# 7. Encode target column
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y.astype(str))

# 8. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# 9. Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 10. Train ANN model with adjusted parameters for better performance
ann = MLPClassifier(
    hidden_layer_sizes=(100, 50),  # Two hidden layers
    activation='relu',  # ReLU activation often works better
    solver='adam',
    max_iter=500,  # More iterations
    early_stopping=True,  # Stop when validation score stops improving
    validation_fraction=0.1,
    random_state=42,
    learning_rate_init=0.001
)
ann.fit(X_train_scaled, y_train)

# 11. Evaluate
y_pred = ann.predict(X_test_scaled)
print("\nModel Performance:")
print("==================")
print(classification_report(y_test, y_pred, target_names=['Dropout', 'Enrolled', 'Graduate']))

# Show confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print("================")
print("Predicted:  Dropout  Enrolled  Graduate")
for i, true_label in enumerate(['Dropout', 'Enrolled', 'Graduate']):
    print(f"{true_label:10} {cm[i,0]:8} {cm[i,1]:9} {cm[i,2]:9}")

# 12. Calculate feature importance (approximation using permutation)
print("\nTraining set class distribution:")
unique, counts = np.unique(y_train, return_counts=True)
for u, c in zip(unique, counts):
    print(f"Class {['Dropout', 'Enrolled', 'Graduate'][u]}: {c} samples ({c/len(y_train)*100:.1f}%)")

# 13. Data validation and cleaning
# Ensure all numeric values are non-negative
numeric_cols = X.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if X[col].min() < 0:
        print(f"Warning: Found negative values in {col}. Setting to 0.")
        X.loc[X[col] < 0, col] = 0

# Calculate min/max ranges for each feature
feature_ranges = {}
for col in X.columns:
    if X[col].dtype in [np.int64, np.float64]:
        feature_ranges[col] = {
            'min': float(X[col].min()),
            'max': float(X[col].max()),
            'mean': float(X[col].mean()),
            'std': float(X[col].std())
        }

# 14. Save model, scaler, and metadata
os.makedirs('model', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Save model and scaler
joblib.dump(ann, 'model/dropout_ann_model.joblib')
joblib.dump(scaler, 'model/scaler.joblib')

# Save comprehensive metadata
metadata = {
    'feature_names': feature_names,
    'default_values': default_values,
    'feature_ranges': feature_ranges,
    'label_encoders': {col: le.classes_.tolist() for col, le in label_encoders.items()},
    'target_classes': target_encoder.classes_.tolist(),
    'model_info': {
        'n_features': len(feature_names),
        'n_classes': len(target_encoder.classes_),
        'model_type': 'MLPClassifier',
        'scaler_type': 'StandardScaler'
    }
}

with open('model/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

# Save the dataset
X.to_csv('data/student-dropout.csv', index=False)

print("\nModel, scaler, and metadata saved successfully.")
print(f"Model accuracy on test set: {ann.score(X_test_scaled, y_test):.3f}")
