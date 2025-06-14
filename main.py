# fraud_detection/main.py

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("fraud_data.csv")  # Make sure the CSV is in the same folder

# Basic EDA
print("Data Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())
print("\nClass Balance:\n", df['isFraud'].value_counts())

# Feature Engineering
df['hour'] = df['step'] % 24  # Add hour of day as a feature
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

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Feature Importance
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.title("Top Feature Importances")
plt.tight_layout()
plt.show()

# Save model
joblib.dump(model, "fraud_model.pkl")
print("✅ Model saved as fraud_model.pkl")
