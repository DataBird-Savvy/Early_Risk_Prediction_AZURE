import streamlit as st
import pandas as pd
import joblib
import numpy as np
import warnings
import logging
import sys
from src.data_preprocessing import DataPreprocessor
from src.config import MERGED_DATA_PATH, BEST_PIPELINE, TRAIN_COL_ORDER,RISK_COLOR_MAP
from style import explainability_css

# Apply the CSS
st.markdown(explainability_css(), unsafe_allow_html=True)

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
def color_risk(val):
    color = RISK_COLOR_MAP.get(val, "lightgray")
    text_color = "black"  # for light backgrounds
    return f'background-color: {color}; color: {text_color}; font-weight: bold; text-align: center;'


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
    st.dataframe(
    latest[["Patient_ID", "Predicted_Risk"]]
    .style.applymap(color_risk, subset=["Predicted_Risk"])
)
except Exception as e:
    logger.error(f"Prediction failed: {e}")
    st.error(f"❌ Prediction failed: {e}")
    st.stop()

# --------------------------------------------------
# Doctor-Friendly Explainability using Method 1
# --------------------------------------------------
def get_global_feature_importance(pipeline, feature_names):
    """
    Extract top-K feature importances from pipeline (RandomForest/XGBoost + SelectKBest)
    """
    selector = pipeline.named_steps["select"]
    model = pipeline.named_steps["model"]

    # Mask of selected top-K features
    mask = selector.get_support()
    selected_features = np.array(feature_names)[mask]

    # Feature importance
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_[0])
    else:
        raise ValueError("Model does not support basic feature importance.")

    df_imp = (
        pd.DataFrame({
            "feature": selected_features,
            "importance": importance
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return df_imp

def doctor_friendly_importance_styled(importance_df, n_top=5, patient_row=None):
    """
    Convert feature importance to a visually appealing markdown summary.
    """
    top_features = importance_df.head(n_top)
    lines = ["<h5>🔹 Top factors contributing to patient's risk:</h5><ul>"]

    for _, row in top_features.iterrows():
        feature_name = row['feature']
        importance = row['importance']
        if patient_row is not None and feature_name in patient_row.columns:
            value = patient_row[feature_name].iloc[0]
            lines.append(
                f"<li><b>{feature_name}</b>: "
                f"<span style='color:blue'>{value}</span> "
                f"(importance: {importance:.2f})</li>"
            )
        else:
            lines.append(
                f"<li><b>{feature_name}</b> (importance: {importance:.2f})</li>"
            )

    lines.append("</ul>")
    return "".join(lines)




# --------------------------------------------------
# Display Doctor-Friendly Global Feature Importance
# --------------------------------------------------
try:
    st.subheader("🧠 Explainability — Top Risk Factors Per Patient")
    
  

    importance_df = get_global_feature_importance(pipeline, train_feature_order)

    for i in range(len(latest_features_for_expl)):
        patient_row = latest_features_for_expl.iloc[[i]]
        patient_id = latest.iloc[i]["Patient_ID"]
        
        summary_html = doctor_friendly_importance_styled(importance_df, n_top=5, patient_row=patient_row)
   
        
        # Collapsible expander for each patient
        with st.expander(f"Patient ID: {patient_id} — Predicted Risk: {latest.iloc[i]['Predicted_Risk']}"):
            st.markdown(summary_html, unsafe_allow_html=True)
        


    logger.info("Doctor-friendly feature importance displayed for all patients")
except Exception as e:
    logger.error(f"Doctor-friendly explainability failed: {e}")
    st.error(f"❌ Could not generate explainability for all patients: {e}")
