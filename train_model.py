# train_model.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import os

# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------
print(" Loading dataset from: data/heart.csv")
dataset = pd.read_csv("data/heart.csv")

# ------------------------------------------------------------
# 2. Split predictors and target
# ------------------------------------------------------------
X = dataset.drop("target", axis=1)
y = dataset["target"]

# ------------------------------------------------------------
# 3. Split data into train and test (20% test data)
# Setting random_state ensures reproducible splits
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f" Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# ------------------------------------------------------------
# 4. Create a pipeline with scaler + Random Forest model
# Setting random_state ensures consistent model predictions
# ------------------------------------------------------------
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(
        n_estimators=100,
        random_state=42,       # ensures stable results
        max_depth=6,           # optional tuning for simplicity
        n_jobs=-1
    ))
])

# ------------------------------------------------------------
# 5. Train model
# ------------------------------------------------------------
print(" Training the Random Forest model...")
pipeline.fit(X_train, y_train)

# ------------------------------------------------------------
# 6. Evaluate model
# ------------------------------------------------------------
y_pred = pipeline.predict(X_test)
score = accuracy_score(y_test, y_pred) * 100
print(f" Model Accuracy: {score:.2f}%")

# ------------------------------------------------------------
# 7. Save trained model
# ------------------------------------------------------------
os.makedirs("models", exist_ok=True)

model_bundle = {
    "pipeline": pipeline,
    "accuracy": score
}

MODEL_PATH = "models/heart_disease_model.pkl"
joblib.dump(model_bundle, MODEL_PATH)

print(f" Model saved successfully at: {MODEL_PATH}")
