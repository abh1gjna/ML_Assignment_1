import os
import matplotlib.pyplot as plt
import pandas as pd

ROLL_NO = "IMT2024081"
METHODS = ["OLS", "Ridge", "Lasso", "ElasticNet"]
OUTPUT_FOLDER = "ML_Plots"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def load_results(var, method):
  filename = f"{ROLL_NO}_var{var}_{method}_results.csv"
  if not os.path.exists(filename):
    print(f"WARNING: {filename} not found.")
    return None
  print(f"Loaded: {filename}")
  return pd.read_csv(filename)


def save_figure(fig, filename):
  path = os.path.join(OUTPUT_FOLDER, filename)
  fig.savefig(path, dpi=300, bbox_inches="tight")
  print(f"Saved: {path}")
  plt.show()
  plt.close(fig)


def format_alpha(alpha):
  return f"{alpha:g}"


def label_lowest_point(ax, df, fontsize=9):
  best_idx = df["CV_MSE"].idxmin()
  best = df.loc[best_idx]
  ax.annotate(
      f"{best['CV_MSE']:.3f}",
      xy=(best["Degree"], best["CV_MSE"]),
      xytext=(0, 10),
      textcoords="offset points",
      ha="center",
      fontsize=fontsize,
      fontweight="bold",
  )


# 1. OLS
def plot_ols(var):
  df = load_results(var, "OLS")
  if df is None:
    return

  df = (
      df.groupby("Degree")["CV_MSE"].min().reset_index().sort_values("Degree")
  )

  fig, ax = plt.subplots(figsize=(11, 7))
  ax.plot(
      df["Degree"],
      df["CV_MSE"],
      marker="o",
      markersize=6,
      linewidth=2,
  )
  label_lowest_point(ax, df, fontsize=9)

  ax.set_yscale("log")
  ax.set_xlabel("Polynomial Degree")
  ax.set_ylabel("5-Fold Cross-Validation MSE")
  ax.set_title(f"Var{var} - OLS\nCV MSE vs Polynomial Degree")
  ax.set_xticks(df["Degree"])
  ax.grid(True, which="both", alpha=0.3)

  fig.tight_layout()
  save_figure(fig, f"Var{var}_OLS_CV_MSE.png")

  best = df.loc[df["CV_MSE"].idxmin()]
  print(
      f"\nOLS - Var{var}\nBest degree = {int(best['Degree'])} | Best CV MSE ="
      f" {best['CV_MSE']:.10f}"
  )


# 2. RIDGE
def plot_ridge(var):
  df = load_results(var, "Ridge")
  if df is None:
    return

  alphas = sorted(df["Alpha"].dropna().unique())
  fig, ax = plt.subplots(figsize=(13, 8))

  for alpha in alphas:
    alpha_df = df[df["Alpha"] == alpha].sort_values("Degree")
    ax.plot(
        alpha_df["Degree"],
        alpha_df["CV_MSE"],
        marker="o",
        markersize=5,
        linewidth=1.8,
        label=f"α = {format_alpha(alpha)}",
    )
    label_lowest_point(ax, alpha_df, fontsize=8)

  ax.set_yscale("log")
  ax.set_xlabel("Polynomial Degree")
  ax.set_ylabel("5-Fold Cross-Validation MSE")
  ax.set_title(
      f"Var{var} - Ridge\nCV MSE vs Polynomial Degree for Different α"
  )
  ax.set_xticks(sorted(df["Degree"].unique()))
  ax.grid(True, which="both", alpha=0.3)
  ax.legend(title="Regularization α", fontsize=8)

  fig.tight_layout()
  save_figure(fig, f"Var{var}_Ridge_All_Alpha.png")

  print(f"\nRIDGE - VAR{var}\n" + "=" * 70)
  for alpha in alphas:
    alpha_df = df[df["Alpha"] == alpha]
    best = alpha_df.loc[alpha_df["CV_MSE"].idxmin()]
    print(
        f"α = {format_alpha(alpha):>8} | Degree = {int(best['Degree']):2d} | MSE"
        f" = {best['CV_MSE']:.10f}"
    )


# 3. LASSO
def plot_lasso(var):
  df = load_results(var, "Lasso")
  if df is None:
    return

  alphas = sorted(df["Alpha"].dropna().unique())
  fig, ax = plt.subplots(figsize=(13, 8))

  for alpha in alphas:
    alpha_df = df[df["Alpha"] == alpha].sort_values("Degree")
    ax.plot(
        alpha_df["Degree"],
        alpha_df["CV_MSE"],
        marker="o",
        markersize=5,
        linewidth=1.8,
        label=f"α = {format_alpha(alpha)}",
    )
    label_lowest_point(ax, alpha_df, fontsize=8)

  ax.set_yscale("log")
  ax.set_xlabel("Polynomial Degree")
  ax.set_ylabel("5-Fold Cross-Validation MSE")
  ax.set_title(
      f"Var{var} - Lasso\nCV MSE vs Polynomial Degree for Different α"
  )
  ax.set_xticks(sorted(df["Degree"].unique()))
  ax.grid(True, which="both", alpha=0.3)
  ax.legend(title="Regularization α", fontsize=8)

  fig.tight_layout()
  save_figure(fig, f"Var{var}_Lasso_All_Alpha.png")

  print(f"\nLASSO - VAR{var}\n" + "=" * 70)
  for alpha in alphas:
    alpha_df = df[df["Alpha"] == alpha]
    best = alpha_df.loc[alpha_df["CV_MSE"].idxmin()]
    print(
        f"α = {format_alpha(alpha):>8} | Degree = {int(best['Degree']):2d} | MSE"
        f" = {best['CV_MSE']:.10f}"
    )


# 4. ELASTIC NET
def plot_elastic_net(var):
  df = load_results(var, "ElasticNet")
  if df is None:
    return

  required_columns = ["Alpha", "L1_Ratio", "Degree", "CV_MSE"]
  missing = [col for col in required_columns if col not in df.columns]
  if missing:
    print("Elastic Net file is missing:", missing)
    return

  l1_ratios = sorted(df["L1_Ratio"].dropna().unique())
  for l1_ratio in l1_ratios:
    ratio_df = df[df["L1_Ratio"] == l1_ratio].copy()
    alphas = sorted(ratio_df["Alpha"].dropna().unique())

    fig, ax = plt.subplots(figsize=(13, 8))
    for alpha in alphas:
      alpha_df = ratio_df[ratio_df["Alpha"] == alpha].sort_values("Degree")
      ax.plot(
          alpha_df["Degree"],
          alpha_df["CV_MSE"],
          marker="o",
          markersize=4,
          linewidth=1.7,
          label=f"α = {format_alpha(alpha)}",
      )
      label_lowest_point(ax, alpha_df, fontsize=7)

    ax.set_yscale("log")
    ax.set_xlabel("Polynomial Degree")
    ax.set_ylabel("5-Fold Cross-Validation MSE")
    ax.set_title(f"Var{var} - Elastic Net\nL1 Ratio = {l1_ratio:g}")
    ax.set_xticks(sorted(ratio_df["Degree"].unique()))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="Regularization α", fontsize=8)

    fig.tight_layout()
    ratio_name = str(l1_ratio).replace(".", "_")
    save_figure(fig, f"Var{var}_ElasticNet_L1Ratio_{ratio_name}.png")


# 5. ALL METHODS TOGETHER
def plot_all_methods(var):
  results = {}
  for method in METHODS:
    df = load_results(var, method)
    if df is None:
      continue
    results[method] = (
        df.groupby("Degree")["CV_MSE"]
        .min()
        .reset_index()
        .sort_values("Degree")
    )

  if not results:
    return

  fig, ax = plt.subplots(figsize=(13, 8))
  for method, method_df in results.items():
    ax.plot(
        method_df["Degree"],
        method_df["CV_MSE"],
        marker="o",
        markersize=5,
        linewidth=2,
        label=method,
    )
    label_lowest_point(ax, method_df, fontsize=8)

  ax.set_yscale("log")
  ax.set_xlabel("Polynomial Degree")
  ax.set_ylabel("Best 5-Fold Cross-Validation MSE")
  ax.set_title(f"Var{var} - Comparison of Regression Methods")

  all_degrees = sorted(
      set().union(*[set(df["Degree"]) for df in results.values()])
  )
  ax.set_xticks(all_degrees)
  ax.grid(True, which="both", alpha=0.3)
  ax.legend()

  fig.tight_layout()
  save_figure(fig, f"Var{var}_All_Methods.png")


# 6. BEST MODEL BAR GRAPH
def plot_best_models(var):
  rows = []
  for method in METHODS:
    df = load_results(var, method)
    if df is None:
      continue
    best = df.loc[df["CV_MSE"].idxmin()]
    row = {
        "Method": method,
        "Degree": int(best["Degree"]),
        "CV_MSE": best["CV_MSE"],
    }
    if "Alpha" in df.columns:
      row["Alpha"] = best["Alpha"]
    if "L1_Ratio" in df.columns:
      row["L1_Ratio"] = best["L1_Ratio"]
    rows.append(row)

  if not rows:
    return

  result = pd.DataFrame(rows)
  print(f"\nBEST MODELS - VAR{var}\n" + "=" * 80)
  print(result.to_string(index=False))

  fig, ax = plt.subplots(figsize=(11, 7))
  bars = ax.bar(result["Method"], result["CV_MSE"])

  ax.set_yscale("log")
  ax.set_xlabel("Regression Method")
  ax.set_ylabel("Best 5-Fold Cross-Validation MSE")
  ax.set_title(f"Var{var} - Best Model Comparison")
  ax.grid(axis="y", which="both", alpha=0.3)

  for bar, (_, row) in zip(bars, result.iterrows()):
    label = f"MSE = {row['CV_MSE']:.4f}\nDegree = {row['Degree']}"
    if "Alpha" in result.columns and pd.notna(row.get("Alpha")):
      label += f"\nα = {format_alpha(row['Alpha'])}"
    if "L1_Ratio" in result.columns and pd.notna(row.get("L1_Ratio")):
      label += f"\nL1 = {row['L1_Ratio']:g}"

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        label,
        ha="center",
        va="bottom",
        fontsize=8,
    )

  fig.tight_layout()
  save_figure(fig, f"Var{var}_Best_Model_Comparison.png")


# 7. OVERALL BEST MODEL
def print_overall_best(var):
  rows = []
  for method in METHODS:
    df = load_results(var, method)
    if df is None:
      continue
    best = df.loc[df["CV_MSE"].idxmin()]
    rows.append({
        "Method": method,
        "Degree": int(best["Degree"]),
        "CV_MSE": best["CV_MSE"],
        "Alpha": best["Alpha"] if "Alpha" in df.columns else None,
        "L1_Ratio": best["L1_Ratio"] if "L1_Ratio" in df.columns else None,
    })

  if not rows:
    return

  result = pd.DataFrame(rows)
  best = result.loc[result["CV_MSE"].idxmin()]

  print(f"\nOVERALL BEST MODEL - VAR{var}\n" + "=" * 80)
  print(f"Method:{best['Method']}")
  print(f"Degree:{int(best['Degree'])}")
  print(f"CV MSE:{best['CV_MSE']:.10f}")
  if pd.notna(best["Alpha"]):
    print(f"Alpha:{format_alpha(best['Alpha'])}")
  if pd.notna(best["L1_Ratio"]):
    print(f"L1 Ratio:{best['L1_Ratio']:g}")



def generate_all_plots(var):
  
  plot_ols(var)
  plot_ridge(var)
  plot_lasso(var)
  plot_elastic_net(var)
  plot_all_methods(var)
  plot_best_models(var)
  print_overall_best(var)


generate_all_plots(1)
generate_all_plots(2)
