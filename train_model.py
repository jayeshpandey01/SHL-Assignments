import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load data
print("Loading data...")
df = pd.read_csv("outputs/train_features_full.csv")

# Verify columns
print("Preparing features...")
feature_cols = [col for col in df.columns if col.startswith('emb_')]
if not feature_cols:
    raise ValueError("No embedding columns found in the dataset")

# Select features (embeddings only) and target
X = df[feature_cols]
y = df["grammar_score"]

# Verify data quality
if X.isnull().any().any():
    print("Warning: Found NaN values in features, filling with 0")
    X = X.fillna(0)

if y.isnull().any():
    print("Warning: Found NaN values in target, filling with mean")
    y = y.fillna(y.mean())

# Split data
print("Splitting data...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

# Train model
print("Training model...")
model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
print("Evaluating model...")
train_preds = model.predict(X_train)
val_preds = model.predict(X_val)

# Calculate metrics
train_mse = mean_squared_error(y_train, train_preds)
train_r2 = r2_score(y_train, train_preds)
val_mse = mean_squared_error(y_val, val_preds)
val_r2 = r2_score(y_val, val_preds)

print("\nModel Performance:")
print(f"Train MSE: {train_mse:.4f} | Train R²: {train_r2:.4f}")
print(f"Val MSE: {val_mse:.4f} | Val R²: {val_r2:.4f}")

# Save model
print("\nSaving model...")
joblib.dump(model, "outputs/grammar_scorer.pkl")
print("✅ Model saved successfully")
