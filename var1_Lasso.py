import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

TRAIN_FILE = "IMT2024081_train_var1.csv"
FEATURES = ["x1", "x2", "x3", "x4", "x5", "x6"]
TARGET = "y"
MIN_DEGREE = 1
MAX_DEGREE = 10
N_FOLDS = 5
RANDOM_STATE = 42

ALPHAS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]

# Loading Data
data = pd.read_csv(TRAIN_FILE)
X = data[FEATURES]
y = data[TARGET]

print("LASSO POLYNOMIAL REGRESSION - VAR1")
print("Training shape:", data.shape)

# Cross Validation
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
results = []

for degree in range(MIN_DEGREE, MAX_DEGREE + 1):
  poly = PolynomialFeatures(degree=degree, include_bias=False)
  poly.fit(X.iloc[:1])
  n_features = poly.transform(X.iloc[:1]).shape[1]

  for alpha in ALPHAS:
    model = Pipeline([
        ("polynomial", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),
        ("regression", Lasso(alpha=alpha, max_iter=1000000, tol=1e-4)),
    ])

    scores = cross_validate(
        model,
        X,
        y,
        cv=kf,
        scoring={"mse": "neg_mean_squared_error", "r2": "r2"},
        n_jobs=-1,
    )

    results.append({
        "Model": "Lasso",
        "Degree": degree,
        "Alpha": alpha,
        "L1_Ratio": 1.0,
        "Polynomial_Features": n_features,
        "CV_MSE": -scores["test_mse"].mean(),
        "CV_MSE_STD": scores["test_mse"].std(),
        "CV_R2": scores["test_r2"].mean(),
    })

  current = [r for r in results if r["Degree"] == degree]
  best_current = min(current, key=lambda x: x["CV_MSE"])
  print(
      f"Degree {degree:2d} | "
      f"Features {n_features:5d} | "
      f"Best Alpha {best_current['Alpha']} | "
      f"CV MSE {best_current['CV_MSE']:.8f} | "
      f"CV R2 {best_current['CV_R2']:.6f}"
  )

# Results
results_df = pd.DataFrame(results).sort_values("CV_MSE").reset_index(drop=True)

print("\nBEST LASSO MODEL - VAR1")
best = results_df.iloc[0]
print(f"Degree:{int(best['Degree'])}")
print(f"Alpha:{best['Alpha']}")
print(f"CV MSE:{best['CV_MSE']:.10f}")
print(f"CV R2:{best['CV_R2']:.10f}")
print(f"Features:{int(best['Polynomial_Features'])}")

# Saving results
OUTPUT = "IMT2024081_var1_Lasso_results.csv"
results_df.to_csv(OUTPUT, index=False)
print("\nResults saved:", OUTPUT)