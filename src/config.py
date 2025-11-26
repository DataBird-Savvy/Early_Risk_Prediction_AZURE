MERGED_DATA_PATH="artifacts/merged_patient_kafka_data.csv"
drop_cols = [
    "Patient_ID", "Date", "Time",
    "Electrocardiogram (ECG/EKG)",
    "Blood Pressure",
    "Hydration Levels",
    "Risk_Score",
    'Heart Rate (HR)'

]


BEST_PIPELINE="artifacts/best_pipeline.joblib"

TRAIN_COL_ORDER='artifacts/train_feature_order.joblib'

RISK_COLOR_MAP = {
    "Low": "#d4f7d4",       # very light green
    "Medium": "#ffe5b4",    # very light orange / beige
    "High": "#f7c8c8",      # very light red / pink
    "Critical": "#800000"   # dark red
}
