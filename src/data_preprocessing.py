# dataprocessing.py
import pandas as pd
import numpy as np
import sys

class DataPreprocessor:

    def clean_input(self, df):
        """Cleanup + fix missing training columns."""
        drop_extra = ["patient_id_x", "timestamp_x", "patient_id_y", "timestamp_y"]
        df = df.drop(columns=[c for c in drop_extra if c in df.columns])

        # Extract systolic/diastolic
        if "Blood Pressure" in df.columns:
            df["Systolic_BP"] = pd.to_numeric(
                df["Blood Pressure"].astype(str).str.split("/").str[0], errors='coerce'
            )
            df["Diastolic_BP"] = pd.to_numeric(
                df["Blood Pressure"].astype(str).str.split("/").str[1], errors='coerce'
            )

        required_cols = [
            'Patient_ID','Date','Time',
            'Blood Glucose Level (mg/dL)','Blood Oxygen (SpO₂)',
            'Heart Rate','Respiratory Rate (RR)','Body Temperature',
            'Hemoglobin','Glucose','Cholesterol','Platelet Count',
            'WBC Count','RBC Count','Creatinine','Urea','Sodium','Potassium',
            'Calcium','Systolic_BP','Diastolic_BP'
        ]

        for col in required_cols:
            if col not in df.columns:
                df[col] = np.nan

        # Ensure numeric columns are numeric
        numeric_cols = [
            'Systolic_BP', 'Diastolic_BP', 'Heart Rate', 'Blood Glucose Level (mg/dL)',
            'Blood Oxygen (SpO₂)', 'Respiratory Rate (RR)', 'Body Temperature',
            'Hemoglobin', 'Glucose', 'Cholesterol', 'Platelet Count', 'WBC Count',
            'RBC Count', 'Creatinine', 'Urea', 'Sodium', 'Potassium', 'Calcium'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        return df[required_cols]

    def feature_engineering(self, df):
        df = df.copy()

        systolic = "Systolic_BP"
        diastolic = "Diastolic_BP"

        df["PulsePressure"] = df[systolic] - df[diastolic]
        df["MAP"] = (df[systolic] + 2 * df[diastolic]) / 3
        df["ShockIndex"] = df["Heart Rate"] / df[systolic].replace(0, np.nan)

        

        # Flags
        df["is_hypertensive"] = ((df[systolic] >= 130) | (df[diastolic] >= 80)).astype(int)
        df["is_tachycardic"] = (df["Heart Rate"] >= 100).astype(int)
        df["is_hypoxic"] = (df['Blood Oxygen (SpO₂)'] < 94).astype(int)
        df["is_hyperglycemia"] = (df['Blood Glucose Level (mg/dL)'] >= 140).astype(int)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        return df

    def get_latest_records(self, df):
        df["Time"] = df["Time"].astype(str)
        df["DateTime"] = pd.to_datetime(df["Date"] + " " + df["Time"], errors="coerce")

        # Get latest record for each patient
        latest = df.sort_values("DateTime").groupby("Patient_ID").tail(1)

        # ---- ROUND ALL NUMERIC COLUMNS TO 2 DECIMAL PLACES ----
        num_cols = latest.select_dtypes(include=[float, int]).columns
        latest[num_cols] = latest[num_cols].round(2)

        return latest.drop(columns=["DateTime"], errors="ignore")

