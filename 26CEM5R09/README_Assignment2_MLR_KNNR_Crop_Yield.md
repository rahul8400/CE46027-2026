# Assignment 2: MLR vs KNNR – Crop Yield Dataset

## 1. Project Title

**Performance Comparison of Multiple Linear Regression (MLR) and K-Nearest Neighbors Regression (KNNR) using a Crop Yield Dataset**

## 2. Objective

The objective of this assignment is to compare the performance of:

1. Multiple Linear Regression (MLR)
2. K-Nearest Neighbors Regression (KNNR)

The models are evaluated using regression performance metrics and computational/resource measures.

## 3. Dataset

The program uses a Crop Yield CSV dataset.

The required target variable is:

`Yield_ton_per_ha`

The program allows the user to select the CSV file manually. Therefore, the CSV file does not need to have a particular filename or be stored in the same folder as the Python script.

## 4. Software and Libraries

The program is designed to run in **Spyder**.

Required Python libraries:

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- psutil
- Tkinter

Main libraries used in the program include:

```text
numpy
pandas
matplotlib
scikit-learn
psutil
tkinter
```

## 5. Model Settings

The program uses the following settings:

- Test size: 20%
- Random state: 42
- Number of neighbors for KNNR: 5
- Target variable: `Yield_ton_per_ha`

The dataset is divided into training and testing data using `train_test_split`.

## 6. Data Preprocessing

The program automatically identifies:

- Numerical features
- Categorical features

For numerical features:

- Missing values are replaced using the median.
- StandardScaler is used for feature scaling.

For categorical features:

- Missing values are replaced using the most frequent value.
- OneHotEncoder is used for categorical encoding.

The same preprocessing approach is applied separately to both models.

## 7. Models

### 7.1 Multiple Linear Regression

Multiple Linear Regression is implemented using:

```python
LinearRegression()
```

It predicts crop yield using multiple input features.

### 7.2 K-Nearest Neighbors Regression

KNN Regression is implemented using:

```python
KNeighborsRegressor(n_neighbors=5)
```

The model predicts crop yield based on the nearest observations in the feature space.

## 8. Performance Metrics

The following metrics are calculated for both models.

### R² Score

R² measures how well the model explains the variation in the target variable.

Higher R² generally indicates better explanatory performance.

### RMSE

Root Mean Squared Error measures the magnitude of prediction errors.

Lower RMSE indicates smaller prediction errors.

### MAE

Mean Absolute Error measures the average absolute difference between actual and predicted values.

Lower MAE indicates smaller prediction errors.

## 9. Computational Performance

The program also calculates:

- Training time
- Prediction time
- Total execution time
- Process CPU time
- Memory change

These values help compare the computational requirements of MLR and KNNR.

## 10. Graphical Analysis

The program automatically creates the following graphs:

1. MLR Actual vs Predicted Yield
2. KNNR Actual vs Predicted Yield
3. MLR Residual Plot
4. KNNR Residual Plot
5. R² Comparison
6. RMSE Comparison
7. MAE Comparison
8. Total Execution Time Comparison
9. Memory Change Comparison

All graphs are saved as PNG files at 300 DPI.

## 11. Output Files

After successful execution, a folder named:

```text
assignment2_outputs
```

is created in the same folder as the selected CSV dataset.

The folder contains:

```text
model_comparison.csv
predictions.csv
dataset_information.csv

01_MLR_actual_vs_predicted.png
02_KNNR_actual_vs_predicted.png
03_MLR_residuals.png
04_KNNR_residuals.png
05_R2_comparison.png
06_RMSE_comparison.png
07_MAE_comparison.png
08_execution_time_comparison.png
09_memory_change_comparison.png
```

## 12. How to Run in Spyder

### Step 1

Open Spyder.

### Step 2

Open:

```text
assignment2_MLR_KNNR_Crop_Yield_SPYDER_FIXED.py
```

### Step 3

Press:

```text
F5
```

or click the green **Run** button.

### Step 4

A file-selection window will appear with the message:

```text
SELECT CROP YIELD CSV DATASET
```

### Step 5

Select your crop yield CSV file.

For example:

```text
crop_yield_dataset.csv.csv
```

### Step 6

Click **Open**.

The program will load the dataset and start the analysis.

### Step 7

Check the Spyder IPython Console for the final model comparison.

## 13. Expected Console Output

The program displays information similar to:

```text
ASSIGNMENT 2: MLR vs KNNR - CROP YIELD DATASET

Selected dataset:
...

--- DATASET INFORMATION ---
Rows    : ...
Columns : ...

RUNNING: Multiple Linear Regression (MLR)

R²                       : ...
RMSE                     : ...
MAE                      : ...
Training time (s)        : ...
Prediction time (s)      : ...
Total execution time (s) : ...
Process CPU time (s)     : ...
Memory change (MB)       : ...

RUNNING: K-Nearest Neighbors Regression (K=5)

R²                       : ...
RMSE                     : ...
MAE                      : ...
Training time (s)        : ...
Prediction time (s)      : ...
Total execution time (s) : ...
Process CPU time (s)     : ...
Memory change (MB)       : ...

FINAL MODEL COMPARISON
```

## 14. Important Notes

- The CSV dataset must contain the column `Yield_ton_per_ha`.
- The program opens a file-selection window, so it is not dependent on Spyder's temporary `temp.py` location.
- If no CSV file is selected, the program stops and asks the user to run it again.
- If the target column is missing, the program displays the available column names.
- Missing feature values are handled automatically.
- Duplicate rows are detected and removed.
- The program is intended for execution in Spyder.

## 15. Assignment Requirements Covered

The program covers the required comparison of MLR and KNNR through:

- R²
- RMSE
- MAE
- Training time
- Prediction time
- Total execution time
- Process CPU time
- Memory change
- Graphical comparison

## 16. Author

**Assignment 2 – AI/ML Course**

**Topic:** Performance Comparison of Multiple Linear Regression and K-Nearest Neighbors Regression on Crop Yield Data
