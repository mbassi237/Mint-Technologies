# Mint-Technologies — Superstore Sales Prediction

An interactive Streamlit application that estimates sales for Superstore orders from a CSV upload.

## Problem Statement

Predict the `Sales` amount for an individual order using information available when the order is placed, including its order date, shipping mode, customer segment, location, region, and product category. Shipping dates and shipping duration are excluded because they are not known at prediction time.

## Dataset

The project uses the Superstore transaction data provided as [`data/train.csv`](data/train.csv). The file contains order dates, shipping and customer details, product categories, and sales values. The original upstream publisher or download URL is not recorded in the supplied project files.

The feature-engineering notebook creates a reproducible 80/20 train/test split (`random_state=42`) and stores the transformed data in `data/engineered_train.csv` and `data/engineered_test.csv`.

## Approach

- **Cleaning:** Parse order and ship dates, convert `Sales` to numeric, remove rows with invalid dates or target values, and fill missing categorical values with `Unknown`.
- **Features:** Derive order year, month, and weekday. One-hot encode categorical fields and standardize the date features using a preprocessor fitted on training data only. Exclude identifiers, names, ship date, and shipping duration from prediction features.
- **Modeling:** Compare Linear Regression, Random Forest, and Gradient Boosting. The advanced models use three-fold cross-validation for hyperparameter selection; the held-out test set is used for final comparison. Linear Regression is retained because it has the best test-set R² among the evaluated models.

## Results

| Model | Test R² | Test MSE |
| --- | ---: | ---: |
| Linear Regression | **0.1638** | **558,897.44** |
| Gradient Boosting | 0.1248 | — |
| Random Forest | 0.0607 | — |

The linear model's test R² is modest, so predictions should be treated as estimates rather than precise sales forecasts. The results are from the saved notebook evaluation on the 20% held-out split.

## How to Run

From the project root, install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

If the model files are not present, run the notebooks in order: `02_feature_engineering.ipynb` creates the preprocessing artifact, and the final cell of `04_advanced_models.ipynb` saves the selected Linear Regression model.

Start the app:

```bash
streamlit run app.py
```

Upload a CSV containing `Order Date`, `Ship Mode`, `Segment`, `Country`, `City`, `State`, `Postal Code`, `Region`, `Category`, and `Sub-Category`. The app adds `Predicted Sales` to the uploaded rows and lets you download the result.
