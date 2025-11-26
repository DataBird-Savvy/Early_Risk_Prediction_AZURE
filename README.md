
---
## 📸 Screenshots
### 🟥 Real-Time Patient Monitoring Dashboard
![alt text](image-1.png)




# 🏥 Patient Health Monitoring System

## 📌 Overview
An end-to-end patient health monitoring system that:
- Integrates real-time data from wearable devices and lab results
- Uses XGBoost model for risk prediction
- Employs dataset drift detection (via Evidently AI)
- Sends automatic alerts to doctors for abnormal vitals
- Uses Kafka for real-time streaming integration

## 🔑 Key Features
- ✅ Real-time ingestion from wearables: Blood Pressure, Blood Sugar, Heart Rate, SpO₂, ECG, etc.
- ✅ Integration of Lab Records: Hemoglobin, Cholesterol, WBC, etc.
- ✅ Dataset Drift Detection: Uses Evidently AI to monitor data drift
- ✅ Kafka + Zookeeper: Stream architecture using Docker Compose
- ✅ Streamlit Dashboard: Interactive real-time visualization and alerting
- ✅ Alert System: Color-coded alerts for doctors when thresholds are 
- ✅ Collapsible expanders for patient-specific **feature importance*

---

## 🧱 System Architecture
- Data Sources → Kafka Topics
- Kafka Stream → Preprocessing → Model → Predictions
- Drift Monitoring (Evidently AI) → Alerting & Dashboarding

---

## ⚙️ Docker Setup (Kafka + Zookeeper)
📁 docker-compose.yml:
```yaml
version: '3.8'

services:
  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000

  kafka:
    image: confluentinc/cp-kafka:latest
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
```

⏯ Run Kafka & Zookeeper:
```bash
docker-compose up -d
```

---

## 🧪 Dataset Drift Detection with Evidently AI
The system evaluates drift in incoming data using Evidently's DatasetDriftMetric.

📄 drift_monitor.py:
```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import pandas as pd

ref_data = pd.read_csv("data/reference_data.csv")
curr_data = pd.read_csv("data/current_data.csv")

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=ref_data, current_data=curr_data)
report.save_html("drift_report.html")
```

👁 View drift_report.html in browser to see changes in features over time.

---

## 🔍 Model Overview
Architecture used for multi-variate vital forecasting:


---



## 🚨 Alert System: Critical Ranges
If any forecasted vital crosses these ranges, alert is triggered.

| Feature           | Normal Range            |
|------------------|-------------------------|
| Blood Glucose     | 70 - 140 mg/dL         |
| Oxygen (SpO₂)     | 90 - 100%              |
| ECG               | 0.5 - 1.5              |
| Heart Rate        | 50 - 110 bpm           |
| Respiratory Rate  | 12 - 20 breaths/min    |
| Temperature       | 36.1 - 37.8°C          |

---

## 📊 Streamlit Dashboard
📁 main.py includes real-time display:
- Red highlights for abnormal metrics


📦 Run Dashboard:
```bash
streamlit run main.py
```

---



---

## 🛠️ Tech Stack
- Python, Pandas, NumPy
- XGboost
- Scikit-Learn, Joblib
- Evidently AI
- Streamlit
- Kafka + Zookeeper (via Docker)
- Git for version control

---

## 🤝 Contributing
Contributions welcome! Please fork this repo, create a new branch, and submit a PR.

---

## 📜 License
MIT License – see LICENSE file.

---


### 📜 License
🔖 MIT License - Feel free to use & modify! 🚀

