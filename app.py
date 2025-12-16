import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -------------------------------
# Page config + minimal styling
# -------------------------------
st.set_page_config(page_title="AQI Analytics & Prediction", layout="wide")
# Css start
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.1rem; padding-bottom: 1.1rem; }
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.03);
        padding: 14px;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Analytics & Prediction")
st.caption("CMP7005 – Programming for Data Analysis | Streamlit Deployment")


# -------------------------------
# Helpers
# -------------------------------
def aqi_bucket(value: float) -> str:
    if value <= 50:
        return "Good"
    elif value <= 100:
        return "Satisfactory"
    elif value <= 200:
        return "Moderate"
    elif value <= 300:
        return "Poor"
    elif value <= 400:
        return "Very Poor"
    else:
        return "Severe"


@st.cache_data(show_spinner=False)
def load_data_from_csv(file_or_path) -> pd.DataFrame:
    df = pd.read_csv(file_or_path)

    # Standardise expected columns
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    # Convert numeric columns safely
    non_numeric = {"City", "Date", "AQI_Bucket"}
    for col in df.columns:
        if col not in non_numeric:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean essentials
    if "City" in df.columns:
        df["City"] = df["City"].astype(str).str.strip()

    # Create Year/Month if possible
    if "Date" in df.columns:
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month

    # Drop duplicates
    df = df.drop_duplicates()

    return df


@st.cache_resource(show_spinner=False)
def train_pipeline(df: pd.DataFrame, n_estimators: int, random_state: int):
    target = "AQI"

    # Candidate features (works even if some columns are missing)
    candidate_num = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"]
    numeric_features = [c for c in candidate_num if c in df.columns]
    categorical_features = [c for c in ["City"] if c in df.columns]
    time_features = [c for c in ["Year", "Month"] if c in df.columns]

    features = numeric_features + time_features + categorical_features

    # Basic guards
    if target not in df.columns:
        raise ValueError("AQI column not found.")
    if len(numeric_features) < 3:
        raise ValueError("Not enough pollutant features found to train the model.")

    # Keep rows where target exists (do not drop all missing pollutants here; we will impute)
    df_model = df.dropna(subset=[target]).copy()

    X = df_model[features]
    y = df_model[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # Preprocess
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, numeric_features + time_features),
            ("cat", cat_transformer, categorical_features),
        ],
        remainder="drop"
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )

    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)

    # Evaluate
    preds = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = r2_score(y_test, preds)

    eval_pack = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "y_test": y_test.reset_index(drop=True),
        "preds": pd.Series(preds).reset_index(drop=True),
    }

    meta = {
        "features": features,
        "numeric_features": numeric_features + time_features,
        "categorical_features": categorical_features,
        "target": target
    }

    return pipe, eval_pack, meta


def rf_feature_importance(pipe: Pipeline, meta: dict) -> pd.DataFrame:
    """
    Works with RandomForestRegressor in a Pipeline.
    Handles one-hot expanded features for City.
    """
    model = pipe.named_steps["model"]
    pre = pipe.named_steps["preprocess"]

    # Get output feature names from ColumnTransformer
    try:
        out_names = pre.get_feature_names_out()
        out_names = [n.replace("num__", "").replace("cat__", "") for n in out_names]
    except Exception:
        out_names = [f"f_{i}" for i in range(len(model.feature_importances_))]

    imp = pd.DataFrame({
        "feature": out_names,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    return imp


# -------------------------------
# Sidebar controls
# -------------------------------
st.sidebar.header("App Controls")
uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])

default_path = "all_cities_combined.csv"
data_source_label = "Uploaded file" if uploaded else f"Default file: {default_path}"

n_estimators = st.sidebar.slider("Random Forest trees", 100, 600, 300, 50)
random_state = st.sidebar.number_input("Random seed", value=42, step=1)

st.sidebar.divider()
page = st.sidebar.radio("Navigate", ["Overview", "City Explorer", "Model Evaluation", "AQI Predictor"])


# -------------------------------
# Load dataset
# -------------------------------
try:
    df = load_data_from_csv(uploaded if uploaded else default_path)
except Exception as e:
    st.error(f"Could not load dataset. Check your file name/path and CSV format.\n\nError: {e}")
    st.stop()

if "City" not in df.columns or "AQI" not in df.columns:
    st.error("Your dataset must contain at least: **City** and **AQI** columns.")
    st.stop()

cities = sorted(df["City"].dropna().unique().tolist())


# -------------------------------
# Train model once (cached)
# -------------------------------
try:
    pipe, eval_pack, meta = train_pipeline(df, n_estimators=n_estimators, random_state=random_state)
except Exception as e:
    st.error(f"Model training failed.\n\nReason: {e}")
    st.stop()


# ============================================================
# PAGE: Overview
# ============================================================
if page == "Overview":
    st.subheader("Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Data Source", data_source_label)
    c2.metric("Rows", f"{len(df):,}")
    c3.metric("Columns", f"{df.shape[1]:,}")
    c4.metric("Cities", f"{df['City'].nunique():,}")

    st.divider()

    left, right = st.columns([1.2, 1])

    with left:
        st.markdown("### Quick Peek")
        sample_city = st.selectbox("Sample by city", ["All Cities"] + cities)
        nrows = st.slider("Rows to display", 5, 30, 12)

        if sample_city == "All Cities":
            view = df.sample(n=min(nrows, len(df)), random_state=42)
        else:
            sub = df[df["City"] == sample_city]
            view = sub.sample(n=min(nrows, len(sub)), random_state=42)

        st.dataframe(view, use_container_width=True)

    with right:
        st.markdown("### AQI Distribution")
        fig = px.histogram(df.dropna(subset=["AQI"]), x="AQI", nbins=60)
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    if "AQI_Bucket" in df.columns:
        st.markdown("### AQI Category Breakdown")
        tmp = df.copy()
        tmp["AQI_Bucket"] = tmp["AQI_Bucket"].fillna("Unknown")
        counts = tmp["AQI_Bucket"].value_counts().reset_index()
        counts.columns = ["AQI_Bucket", "Count"]

        fig = px.pie(counts, names="AQI_Bucket", values="Count", hole=0.55)
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("AQI_Bucket column not found — that’s okay. The app can still predict AQI.")


# ============================================================
# PAGE: City Explorer
# ============================================================
elif page == "City Explorer":
    st.subheader("City Explorer")

    city = st.selectbox("Select city", cities)
    city_df = df[df["City"] == city].copy()

    if "Date" not in city_df.columns or city_df["Date"].isna().all():
        st.warning("This dataset does not have usable Date values for time trends.")
        st.stop()

    city_df = city_df.dropna(subset=["Date", "AQI"]).sort_values("Date")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Records", f"{len(city_df):,}")
    k2.metric("Average AQI", f"{city_df['AQI'].mean():.1f}")
    k3.metric("Max AQI", f"{city_df['AQI'].max():.0f}")
    k4.metric("Median AQI", f"{city_df['AQI'].median():.0f}")

    st.divider()

    # Monthly trend + rolling mean
    monthly = city_df.resample("MS", on="Date")["AQI"].mean().reset_index()
    monthly["AQI_rolling3"] = monthly["AQI"].rolling(3, min_periods=1).mean()

    fig = px.line(monthly, x="Date", y=["AQI", "AQI_rolling3"], markers=True,
                  title=f"Monthly AQI Trend – {city}")
    fig.update_layout(height=450, margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Rolling(3) = smoothing to highlight underlying trend while reducing short-term fluctuation.")


# ============================================================
# PAGE: Model Evaluation
# ============================================================
elif page == "Model Evaluation":
    st.subheader("Model Evaluation (Hold-out Test Split)")

    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{eval_pack['mae']:.2f}")
    m2.metric("RMSE", f"{eval_pack['rmse']:.2f}")
    m3.metric("R²", f"{eval_pack['r2']:.3f}")

    st.caption("These metrics are calculated on a 20% test split (unseen data).")

    st.divider()

    # Residual analysis
    y_test = eval_pack["y_test"]
    preds = eval_pack["preds"]
    residuals = y_test - preds

    left, right = st.columns(2)

    with left:
        plot_df = pd.DataFrame({"Actual AQI": y_test, "Predicted AQI": preds})
        fig = px.scatter(plot_df, x="Actual AQI", y="Predicted AQI",
                 title="Actual vs Predicted")

        fig.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        res_df = pd.DataFrame({"Residual": residuals})
        fig = px.histogram(res_df, x="Residual", nbins=50, title="Residual Distribution (Actual - Predicted)")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Feature importance
    st.markdown("### Feature Importance (Random Forest)")
    imp = rf_feature_importance(pipe, meta).head(25)

    fig = px.bar(imp.sort_values("importance"), x="importance", y="feature", orientation="h",
                 title="Top Features Driving AQI Prediction")
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show feature importance table"):
        st.dataframe(imp, use_container_width=True)


# ============================================================
# PAGE: AQI Predictor
# ============================================================
elif page == "AQI Predictor":
    st.subheader("AQI Predictor")

    # Choose city + date context if available
    colA, colB, colC = st.columns(3)
    with colA:
        pred_city = st.selectbox("City", cities)
    with colB:
        pred_year = st.selectbox("Year", sorted(df["Year"].dropna().astype(int).unique()) if "Year" in df.columns else [2024])
    with colC:
        pred_month = st.selectbox("Month", list(range(1, 13)) if "Month" in df.columns else [1])

    st.divider()

    # Build defaults from city medians (more realistic than global means)
    city_slice = df[df["City"] == pred_city]
    num_cols = [c for c in meta["numeric_features"] if c in df.columns and c not in ["Year", "Month"]]
    med = city_slice[num_cols].median(numeric_only=True)

    st.caption("Enter pollutant values. Defaults are city-wise medians (more realistic than zero).")

    inputs = {}
    left, right = st.columns(2)

    for i, col in enumerate(num_cols):
        default_val = float(med.get(col, df[col].median(skipna=True))) if col in df.columns else 0.0
        default_val = 0.0 if np.isnan(default_val) else default_val

        target_col = left if i % 2 == 0 else right
        with target_col:
            inputs[col] = st.number_input(col, value=float(default_val), min_value=0.0, step=1.0)

    if st.button("Predict AQI", type="primary"):
        row = {}

        # Assemble row aligned with features used during training
        for f in meta["features"]:
            if f == "City":
                row[f] = pred_city
            elif f == "Year":
                row[f] = int(pred_year)
            elif f == "Month":
                row[f] = int(pred_month)
            else:
                row[f] = float(inputs.get(f, 0.0))

        X_input = pd.DataFrame([row])
        pred = float(pipe.predict(X_input)[0])

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted AQI", f"{pred:.1f}")
        c2.metric("Predicted Category", aqi_bucket(pred))
        c3.metric("Model", f"RandomForest ({n_estimators} trees)")

        st.caption("Prediction is produced by a trained ML pipeline (imputation + encoding + Random Forest).")
