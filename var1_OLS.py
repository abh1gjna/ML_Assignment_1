# OLS POLYNOMIAL REGRESSION
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

train_file = "IMT2024081_train_var1.csv"
features = ["x1", "x2", "x3", "x4", "x5", "x6"]
target = "y"
min_degree = 1
max_degree = 10
n_folds = 5
random_state = 42

# Loading Data
data = pd.read_csv(train_file)
X = data[features]
y = data[target]

print("OLS POLYNOMIAL REGRESSION")
print("Training shape:", data.shape)

# Cross Validation
kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
results = []

for degree in range(min_degree, max_degree + 1):
  print(f"\nTesting degree {degree}")

  model = Pipeline([
      (
          "polynomial",
          PolynomialFeatures(degree=degree, include_bias=False),
      ),
      ("scaler", StandardScaler()),
      ("regression", LinearRegression()),
  ])

  scores = cross_validate(
      model,
      X,
      y,
      cv=kf,
      scoring={"mse": "neg_mean_squared_error", "r2": "r2"},
      n_jobs=-1,
  )

  mse = -scores["test_mse"].mean()
  mse_std = scores["test_mse"].std()
  r2 = scores["test_r2"].mean()

  poly = PolynomialFeatures(degree=degree, include_bias=False)
  poly.fit(X.iloc[:1])
  n_features = poly.transform(X.iloc[:1]).shape[1]

  results.append({
      "Model": "OLS",
      "Degree": degree,
      "Alpha": 0,
      "L1_Ratio": 0,
      "Polynomial_Features": n_features,
      "CV_MSE": mse,
      "CV_MSE_STD": mse_std,
      "CV_R2": r2,
  })

  print(f"Degree {degree:2d}" f"Features {n_features:5d} "f"CV MSE {mse:.8f}" f"CV R2 {r2:.6f}"
  )

# Results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("CV_MSE").reset_index(drop=True)

print("\nBEST OLS MODEL")
best = results_df.iloc[0]
print(f"Degree:{int(best['Degree'])}")
print(f"CV MSE:{best['CV_MSE']:.10f}")
print(f"CV R2:{best['CV_R2']:.10f}")
print(f"Features:{int(best['Polynomial_Features'])}")

# Saving results
output = "IMT2024081_var1_OLS_results.csv"
results_df.to_csv(output, index=False)
print("\nResults saved:", output)
print("\nComplete results:")
print(results_df.to_string(index=False))