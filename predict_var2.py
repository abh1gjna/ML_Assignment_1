import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

TRAIN_FILE = "IMT2024081_train_var2.csv"
TEST_FILE = "IMT2024081_test_var2.csv"
OUTPUT_FILE = "IMT2024081_pred_var2.csv"
FEATURES = ["x1", "x2", "x3"]
TARGET = "y"

BEST_DEGREE = 12
BEST_ALPHA = 1.0

# Loading Data
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

X_train = train_df[FEATURES]
y_train = train_df[TARGET]
X_test = test_df[FEATURES]

print("FINAL VAR2 PREDICTION")
print("Training shape:", train_df.shape)
print("Testing shape:", test_df.shape)

# Pipeline Definition & Training
model = Pipeline([
    ("poly", PolynomialFeatures(degree=BEST_DEGREE, include_bias=False)),
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=BEST_ALPHA)),
])

print("\nTraining final model...")
print(f"Method: Ridge | Degree: {BEST_DEGREE} | Alpha: {BEST_ALPHA}")
model.fit(X_train, y_train)

# Predicting and Saving Results
y_test_pred = model.predict(X_test)
prediction_df = pd.DataFrame({"y": y_test_pred})
prediction_df.to_csv(OUTPUT_FILE, index=False)

print("\nResults saved:", OUTPUT_FILE)
print("Number of predictions:", len(prediction_df))
print("\nPrediction statistics:")
print(prediction_df["y"].describe())