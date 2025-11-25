import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
import warnings
import logging

from src.data_preprocessing import DataPreprocessor
from src.config import MERGED_DATA_PATH, BEST_PIPELINE, TRAIN_COL_ORDER

warnings.filterwarnings("ignore")

# --------------------------------------------------
# Logger Setup
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.info("Application started")

# --------------------------------------------------
# Streamlit Layout
# --------------------------------------------------
try:
    st.set_page_config(
        page_title="Doctor Patient Risk Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("🩺 Doctor Monitoring Dashboard")
    st.write("Real-time patient monitoring with Explainable AI")
    logger.info("UI initialized")
except Exception as e:
    logger.error(f"Streamlit UI setup failed: {e}")
    st.error(f"UI setup failed: {e}")
    st.stop()

# --------------------------------------------------
# Load Pipeline
# --------------------------------------------------
@st.cache_resource
def load_pipeline():
    try:
        logger.info(f"Loading pipeline from {BEST_PIPELINE}")
        pipeline = joblib.load(BEST_PIPELINE)
        logger.info("Pipeline loaded successfully")
        return pipeline
    except Exception as e:
        logger.error(f"Pipeline loading failed: {e}")
        st.error(f"❌ Could not load ML pipeline: {e}")
        return None

pipeline = load_pipeline()
if pipeline is None:
    st.stop()

# --------------------------------------------------
# Load Live Data
# --------------------------------------------------
@st.cache_data
def load_live_data():
    try:
        logger.info(f"Loading live data from {MERGED_DATA_PATH}")
        df = pd.read_csv(MERGED_DATA_PATH)
        logger.info(f"Live data loaded — shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading live data: {e}")
        st.error(f"❌ Could not load live data: {e}")
        return None

def convert_strlist_columns(df):
    try:
        for col in df.columns:
            if df[col].dtype == object and df[col].str.startswith('[').any():
                df[col] = df[col].apply(lambda x: np.array(eval(x)) if isinstance(x, str) else x)
    except Exception as e:
        logger.warning(f"String list conversion warning: {e}")
    return df

try:
    df_raw = load_live_data()
    if df_raw is None:
        st.stop()
except Exception as e:
    logger.error(f"Data loading failed: {e}")
    st.error(f"❌ Data loading failed: {e}")
    st.stop()

# --------------------------------------------------
# Preprocessing
# --------------------------------------------------
try:
    pre = DataPreprocessor()
    df_clean = pre.clean_input(df_raw)
    logger.info(f"Cleaned data shape: {df_clean.shape}")

    df_features = pre.feature_engineering(df_clean)
    logger.info(f"Feature-engineered shape: {df_features.shape}")

    latest = pre.get_latest_records(df_features)
    logger.info(f"Latest patient records shape: {latest.shape}")
except Exception as e:
    logger.error(f"Preprocessing failed: {e}")
    st.error(f"❌ Preprocessing failed: {e}")
    st.stop()

# --------------------------------------------------
# Prediction
# --------------------------------------------------
try:
    train_feature_order = joblib.load(TRAIN_COL_ORDER)
    latest_for_pred = latest.copy()
    drop_cols = ["Patient_ID", "Date", "Time"]
    latest_for_pred = latest_for_pred.drop(columns=[c for c in drop_cols if c in latest_for_pred.columns])
    latest_for_pred = latest_for_pred.round(2)
    latest_for_pred = latest_for_pred[train_feature_order]

    latest_features_for_expl = latest_for_pred.copy()
    latest_features_for_expl = convert_strlist_columns(latest_features_for_expl)
    latest_for_pred = convert_strlist_columns(latest_for_pred)

    pred = pipeline.predict(latest_for_pred)
    latest["Predicted_Risk_Label"] = pred

    risk_map = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}
    latest["Predicted_Risk"] = latest["Predicted_Risk_Label"].map(risk_map)
    logger.info("Predictions completed")

    st.subheader("📌 Latest Patient Risk Prediction")
    st.dataframe(latest[["Patient_ID", "Predicted_Risk"]])
except Exception as e:
    logger.error(f"Prediction failed: {e}")
    st.error(f"❌ Prediction failed: {e}")
    st.stop()

# --------------------------------------------------
# SHAP Explainability
# --------------------------------------------------
try:
    st.subheader("🧠 Explainability — Why this risk?")
    selector = pipeline.named_steps["select"]
    model = pipeline.named_steps["model"]

    patient_row = latest_features_for_expl.iloc[[0]].copy()
    patient_array = patient_row.to_numpy().astype(float)
    patient_array = np.round(patient_array, 2)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(patient_array)

    pred_class = pred[0]
    shap_for_class = shap_values[pred_class] if isinstance(shap_values, list) else shap_values

    shap_df = pd.DataFrame({
        "Feature": train_feature_order,
        "SHAP_Value": shap_for_class[0]
    }).sort_values(by="SHAP_Value", key=abs, ascending=False)

    st.dataframe(shap_df)
except Exception as e:
    logger.error(f"Explainability failure: {e}")
    st.error(f"Explainability failed: {e}")

# --------------------------------------------------
# Human-readable explanation
# --------------------------------------------------
def explain_human_readable(df):
    try:
        text="Based on the latest vital signs and lab results, the following risk factors were identified:\n\n"
        spo2 = df["Blood Oxygen (SpO₂)"].iloc[0]
        shock = df["ShockIndex"].iloc[0]
        glu = df["Blood Glucose Level (mg/dL)"].iloc[0]
        hr = df["Heart Rate"].iloc[0]

        if spo2 < 94:
            text += "- Low oxygen saturation (SpO₂ < 94%) – major risk\n"
        if shock > 0.9:
            text += "- High Shock Index (>0.9) – major risk\n"
        if glu >= 140:
            text += "- High blood glucose (>140 mg/dL)\n"
        if hr >= 100:
            text += "- Elevated heart rate (>100 bpm)\n"

        return text
    except Exception as e:
        logger.error(f"Human-readable explanation failed: {e}")
        return "❌ Could not generate human-readable explanation."

try:
    st.subheader("📘 Doctor-Friendly Explanation (Rules + Thresholds)")
    st.markdown(explain_human_readable(latest_features_for_expl))
except Exception as e:
    logger.error(f"Explanation display failed: {e}")
    st.error(f"❌ Could not display explanation: {e}")
