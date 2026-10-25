from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_PATH = DATA_DIR / "sales_model.joblib"
PREPROCESSOR_PATH = DATA_DIR / "feature_preprocessor.joblib"

CATEGORICAL_COLUMNS = [
    "Ship Mode",
    "Segment",
    "Country",
    "City",
    "State",
    "Postal Code",
    "Region",
    "Category",
    "Sub-Category",
]
REQUIRED_COLUMNS = ["Order Date", *CATEGORICAL_COLUMNS]


@st.cache_resource
def load_artifacts():
    missing_files = [
        path.name for path in (MODEL_PATH, PREPROCESSOR_PATH) if not path.is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            "Missing model artifact(s): "
            f"{', '.join(missing_files)}. Run the model and feature-engineering "
            "notebooks to create them."
        )
    return joblib.load(MODEL_PATH), joblib.load(PREPROCESSOR_PATH)


def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    missing_columns = sorted(set(REQUIRED_COLUMNS).difference(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required CSV columns: {', '.join(missing_columns)}")

    order_dates = pd.to_datetime(data["Order Date"], dayfirst=True, errors="coerce")
    invalid_dates = order_dates.isna()
    if invalid_dates.any():
        rows = ", ".join(str(index + 2) for index in data.index[invalid_dates][:10])
        raise ValueError(
            f"{invalid_dates.sum()} row(s) have an invalid Order Date. "
            f"CSV row(s): {rows}{'...' if invalid_dates.sum() > 10 else ''}."
        )

    features = data[CATEGORICAL_COLUMNS].copy()
    features["Postal Code"] = features["Postal Code"].astype("string")
    features[CATEGORICAL_COLUMNS] = features[CATEGORICAL_COLUMNS].fillna("Unknown")
    features["Order Year"] = order_dates.dt.year
    features["Order Month"] = order_dates.dt.month
    features["Order Weekday"] = order_dates.dt.dayofweek
    return features


st.set_page_config(page_title="Superstore Sales Predictor", page_icon="📊", layout="wide")
st.title("Superstore Sales Predictor")
st.write(
    "Upload a Superstore CSV to estimate sales for each order. "
    "The CSV must include `Order Date` and the order's shipping, customer, "
    "and product category fields."
)
st.caption(
    "Predictions use the saved linear regression model and the preprocessing "
    "transformations fitted during training."
)

try:
    model, preprocessor = load_artifacts()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    try:
        uploaded_data = pd.read_csv(uploaded_file)
    except (pd.errors.ParserError, UnicodeDecodeError) as error:
        st.error(f"Could not read the uploaded CSV: {error}")
        st.stop()

    try:
        features = prepare_features(uploaded_data)
    except ValueError as error:
        st.error(str(error))
        st.stop()

    transformed_features = preprocessor.transform(features)
    predictions = model.predict(transformed_features)

    results = uploaded_data.copy()
    results["Predicted Sales"] = predictions
    st.success(f"Generated predictions for {len(results):,} row(s).")
    st.dataframe(results, use_container_width=True)
    st.download_button(
        "Download predictions",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="superstore_sales_predictions.csv",
        mime="text/csv",
    )
