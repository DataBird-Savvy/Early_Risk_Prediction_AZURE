MERGED_DATA_PATH="artifacts/merged_patient_kafka_data.csv"
drop_cols = [
    "Patient_ID", "Date", "Time",
    "Electrocardiogram (ECG/EKG)",
    "Blood Pressure",
    "Hydration Levels",
    "Risk_Score",
    'Heart Rate (HR)'

]
MODEL_PATH="artifacts/best_model.joblib"
ENCODER_PATH="artifacts/label_encoder.joblib"
TRAIN_ALL_COLS_PATH="artifacts/train_all_cols.joblib"
BEST_PIPELINE="artifacts/best_pipeline.joblib"
TRAIN_NUMERIC_COLS='artifacts/train_numeric_cols.joblib'
TRAIN_COL_ORDER='artifacts/train_feature_order.joblib'