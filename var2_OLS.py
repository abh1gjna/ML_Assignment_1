import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

TRAIN_FILE = "IMT2024081_train_var2.csv"
FEATURES = ["x1", "x2", "x3"]
TARGET = "y"
MIN_DEGREE = 1
MAX_DEGREE = 20
N_FOLDS = 5
RANDOM_STATE = 42

# Loading Data
data = pd.read_csv(TRAIN_FILE)
X = data[FEATURES]
y = data[TARGET]

print("OLS POLYNOMIAL REGRESSION - VAR2")
print("Training shape:", data.shape)

# Cross Validation
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
results = []

for degree in range(MIN_DEGREE, MAX_DEGREE + 1):
  poly = PolynomialFeatures(degree=degree, include_bias=False)
  poly.fit(X.iloc[:1])
  n_features = poly.transform(X.iloc[:1]).shape[1]

  model = Pipeline([
      ("polynomial", PolynomialFeatures(degree=degree, include_bias=False)),
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
  r2 = scores["test_r2"].mean()

  results.append({
      "Model": "OLS",
      "Degree": degree,
      "Alpha": 0,
      "L1_Ratio": 0,
      "Polynomial_Features": n_features,
      "CV_MSE": mse,
      "CV_MSE_STD": scores["test_mse"].std(),
      "CV_R2": r2,
  })

  print(
      f"Degree {degree:2d} | "
      f"Features {n_features:5d} | "
      f"CV MSE {mse:.8f} | "
      f"CV R2 {r2:.6f}"
  )

# Results
results_df = pd.DataFrame(results).sort_values("CV_MSE").reset_index(drop=True)

print("\nBEST OLS MODEL - VAR2")
best = results_df.iloc[0]
print(f"Degree:{int(best['Degree'])}")
print(f"CV MSE:{best['CV_MSE']:.10f}")
print(f"CV R2:{best['CV_R2']:.10f}")
print(f"Features:{int(best['Polynomial_Features'])}")

# Saving results
OUTPUT = "IMT2024081_var2_OLS_results.csv"
results_df.to_csv(OUTPUT, index=False)
print("\nResults saved:", OUTPUT)