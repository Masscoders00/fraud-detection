# main.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib

# Load the dataset
df = pd.read_csv("fraud_data.csv")

# Feature Engineering
df['hour'] = df['step'] % 24
le = LabelEncoder()
df['type'] = le.fit_transform(df['type'])

# Drop irrelevant columns
df = df.drop(['nameOrig', 'nameDest'], axis=1)

# Features and Labels
X = df.drop('isFraud', axis=1)
y = df['isFraud']

# Handle imbalance with SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'fraud_model.pkl')
print("✅ Model trained and saved as fraud_model.pkl")
