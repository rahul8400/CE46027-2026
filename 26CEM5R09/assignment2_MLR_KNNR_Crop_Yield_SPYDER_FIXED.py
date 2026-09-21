# -*- coding: utf-8 -*-
"""
ASSIGNMENT 2
MLR vs KNNR - Crop Yield Dataset
SPYDER FIXED VERSION

This version does NOT depend on the Python script location.
It asks you to select the CSV dataset, so the Spyder temp.py
problem will not occur.
"""

import os
import time
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psutil

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ============================================================
# SETTINGS
# ============================================================

TARGET = "Yield_ton_per_ha"
TEST_SIZE = 0.20
RANDOM_STATE = 42
K_NEIGHBORS = 5


# ============================================================
# SELECT DATASET
# ============================================================

print("\n" + "=" * 70)
print("ASSIGNMENT 2: MLR vs KNNR - CROP YIELD DATASET")
print("=" * 70)

print("\nPlease select your crop yield CSV file...")

try:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    data_file = filedialog.askopenfilename(
        title="SELECT CROP YIELD CSV DATASET",
        filetypes=[
            ("CSV files", "*.csv"),
            ("All files", "*.*")
        ]
    )

    root.destroy()

except Exception as e:
    raise RuntimeError(
        "Could not open the dataset selection window.\n"
        "Please check that Tkinter is installed."
    ) from e


if not data_file:
    raise FileNotFoundError(
        "\nNo CSV file was selected.\n"
        "Run the program again and select crop_yield_dataset.csv.csv."
    )


data_path = Path(data_file)

print("\nSelected dataset:")
print(data_path)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(data_path)

print("\n--- DATASET INFORMATION ---")
print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])

print("\nColumns:")
for col in df.columns:
    print(f"  - {col} ({df[col].dtype})")

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

duplicate_count = int(df.duplicated().sum())

print("\nDuplicate rows:", duplicate_count)

if duplicate_count > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print("Rows after duplicate removal:", len(df))


# ============================================================
# CHECK TARGET
# ============================================================

if TARGET not in df.columns:
    raise ValueError(
        "\nThe target column was not found.\n\n"
        f"Required target column:\n{TARGET}\n\n"
        f"Your columns are:\n{list(df.columns)}"
    )


# ============================================================
# PREPARE X AND Y
# ============================================================

X = df.drop(columns=[TARGET])
y = df[TARGET]

# Convert target to numeric if necessary
y = pd.to_numeric(y, errors="coerce")

target_missing = int(y.isnull().sum())

if target_missing > 0:
    valid = y.notnull()
    X = X.loc[valid].copy()
    y = y.loc[valid].copy()

    print(
        f"\nRemoved {target_missing} rows "
        "with missing/non-numeric target values."
    )


# Feature types
numerical_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nNumerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

print("\n--- TRAIN / TEST SPLIT ---")
print("Training rows :", len(X_train))
print("Testing rows  :", len(X_test))
print("Test size     :", "20%")
print("Random state  :", RANDOM_STATE)


# ============================================================
# PREPROCESSOR
# ============================================================

def make_preprocessor():

    try:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    except TypeError:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )

    transformers = []

    if numerical_features:
        numerical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        transformers.append(
            ("numeric", numerical_pipeline, numerical_features)
        )

    if categorical_features:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", encoder)
        ])

        transformers.append(
            ("categorical", categorical_pipeline, categorical_features)
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )


# ============================================================
# MODELS
# ============================================================

mlr_model = Pipeline([
    ("preprocessor", make_preprocessor()),
    ("model", LinearRegression())
])

knnr_model = Pipeline([
    ("preprocessor", make_preprocessor()),
    ("model", KNeighborsRegressor(
        n_neighbors=K_NEIGHBORS
    ))
])


# ============================================================
# EVALUATION
# ============================================================

process = psutil.Process(os.getpid())


def evaluate_model(name, model):

    print("\n" + "-" * 70)
    print("RUNNING:", name)
    print("-" * 70)

    memory_before = (
        process.memory_info().rss / (1024 ** 2)
    )

    cpu_before = (
        process.cpu_times().user
        + process.cpu_times().system
    )

    # Training
    start = time.perf_counter()

    model.fit(X_train, y_train)

    training_time = time.perf_counter() - start

    # Prediction
    start = time.perf_counter()

    predictions = model.predict(X_test)

    prediction_time = time.perf_counter() - start

    memory_after = (
        process.memory_info().rss / (1024 ** 2)
    )

    cpu_after = (
        process.cpu_times().user
        + process.cpu_times().system
    )

    total_time = training_time + prediction_time
    cpu_time = cpu_after - cpu_before
    memory_change = memory_after - memory_before

    r2 = r2_score(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    result = {
        "Model": name,
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "Training_Time_s": training_time,
        "Prediction_Time_s": prediction_time,
        "Total_Execution_Time_s": total_time,
        "Process_CPU_Time_s": cpu_time,
        "Memory_Change_MB": memory_change
    }

    print("\nR²                       :", f"{r2:.6f}")
    print("RMSE                     :", f"{rmse:.6f}")
    print("MAE                      :", f"{mae:.6f}")
    print("Training time (s)        :", f"{training_time:.6f}")
    print("Prediction time (s)      :", f"{prediction_time:.6f}")
    print("Total execution time (s) :", f"{total_time:.6f}")
    print("Process CPU time (s)     :", f"{cpu_time:.6f}")
    print("Memory change (MB)       :", f"{memory_change:.6f}")

    return result, predictions


# ============================================================
# RUN MODELS
# ============================================================

mlr_result, mlr_predictions = evaluate_model(
    "Multiple Linear Regression (MLR)",
    mlr_model
)

knnr_result, knnr_predictions = evaluate_model(
    f"K-Nearest Neighbors Regression (K={K_NEIGHBORS})",
    knnr_model
)

results_df = pd.DataFrame([
    mlr_result,
    knnr_result
])


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

output_dir = data_path.parent / "assignment2_outputs"
output_dir.mkdir(exist_ok=True)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    output_dir / "model_comparison.csv",
    index=False
)

predictions_df = pd.DataFrame({
    "Actual_Yield": y_test.to_numpy(),
    "MLR_Predicted_Yield": mlr_predictions,
    "KNNR_Predicted_Yield": knnr_predictions
})

predictions_df.to_csv(
    output_dir / "predictions.csv",
    index=False
)


# ============================================================
# SAVE DATASET INFORMATION
# ============================================================

info_df = pd.DataFrame({
    "Item": [
        "Dataset",
        "Rows",
        "Columns",
        "Target",
        "Training rows",
        "Testing rows",
        "Test size",
        "KNN K",
        "Numerical features",
        "Categorical features",
        "Python version",
        "Pandas version",
        "NumPy version",
        "Scikit-learn version"
    ],
    "Value": [
        str(data_path),
        len(df),
        df.shape[1],
        TARGET,
        len(X_train),
        len(X_test),
        "20%",
        K_NEIGHBORS,
        len(numerical_features),
        len(categorical_features),
        platform.python_version(),
        pd.__version__,
        np.__version__,
        __import__("sklearn").__version__
    ]
})

info_df.to_csv(
    output_dir / "dataset_information.csv",
    index=False
)


# ============================================================
# GRAPH FUNCTION
# ============================================================

def save_plot(filename):
    plt.tight_layout()
    plt.savefig(
        output_dir / filename,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


# ============================================================
# GRAPHS
# ============================================================

# 1. MLR Actual vs Predicted
plt.figure()
plt.scatter(y_test, mlr_predictions, alpha=0.6)

min_value = min(y_test.min(), mlr_predictions.min())
max_value = max(y_test.max(), mlr_predictions.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual Yield (ton/ha)")
plt.ylabel("Predicted Yield (ton/ha)")
plt.title("MLR: Actual vs Predicted Yield")
save_plot("01_MLR_actual_vs_predicted.png")


# 2. KNNR Actual vs Predicted
plt.figure()
plt.scatter(y_test, knnr_predictions, alpha=0.6)

min_value = min(y_test.min(), knnr_predictions.min())
max_value = max(y_test.max(), knnr_predictions.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual Yield (ton/ha)")
plt.ylabel("Predicted Yield (ton/ha)")
plt.title(
    f"KNNR (K={K_NEIGHBORS}): Actual vs Predicted Yield"
)
save_plot("02_KNNR_actual_vs_predicted.png")


# 3. MLR Residuals
mlr_residuals = y_test.to_numpy() - mlr_predictions

plt.figure()
plt.scatter(
    mlr_predictions,
    mlr_residuals,
    alpha=0.6
)

plt.axhline(0, linestyle="--")
plt.xlabel("Predicted Yield (ton/ha)")
plt.ylabel("Residual")
plt.title("MLR Residual Plot")
save_plot("03_MLR_residuals.png")


# 4. KNNR Residuals
knnr_residuals = y_test.to_numpy() - knnr_predictions

plt.figure()
plt.scatter(
    knnr_predictions,
    knnr_residuals,
    alpha=0.6
)

plt.axhline(0, linestyle="--")
plt.xlabel("Predicted Yield (ton/ha)")
plt.ylabel("Residual")
plt.title(
    f"KNNR (K={K_NEIGHBORS}) Residual Plot"
)
save_plot("04_KNNR_residuals.png")


# 5. R2
plt.figure()
plt.bar(
    ["MLR", "KNNR"],
    [mlr_result["R2"], knnr_result["R2"]]
)

plt.ylabel("R²")
plt.title("R² Comparison")
save_plot("05_R2_comparison.png")


# 6. RMSE
plt.figure()
plt.bar(
    ["MLR", "KNNR"],
    [mlr_result["RMSE"], knnr_result["RMSE"]]
)

plt.ylabel("RMSE")
plt.title("RMSE Comparison")
save_plot("06_RMSE_comparison.png")


# 7. MAE
plt.figure()
plt.bar(
    ["MLR", "KNNR"],
    [mlr_result["MAE"], knnr_result["MAE"]]
)

plt.ylabel("MAE")
plt.title("MAE Comparison")
save_plot("07_MAE_comparison.png")


# 8. Execution time
plt.figure()
plt.bar(
    ["MLR", "KNNR"],
    [
        mlr_result["Total_Execution_Time_s"],
        knnr_result["Total_Execution_Time_s"]
    ]
)

plt.ylabel("Time (seconds)")
plt.title("Total Execution Time Comparison")
save_plot("08_execution_time_comparison.png")


# 9. Memory
plt.figure()
plt.bar(
    ["MLR", "KNNR"],
    [
        mlr_result["Memory_Change_MB"],
        knnr_result["Memory_Change_MB"]
    ]
)

plt.ylabel("Memory Change (MB)")
plt.title("Memory Change Comparison")
save_plot("09_memory_change_comparison.png")


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(index=False)
)

print("\n" + "=" * 70)
print("FILES GENERATED")
print("=" * 70)

for file in sorted(output_dir.iterdir()):
    print(file.name)

print("\nCompleted successfully!")
print("\nResults folder:")
print(output_dir)

print("=" * 70)
