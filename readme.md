# 📈 Sales & Demand Forecasting System
**An End-to-End MLOps Pipeline for Business Intelligence**

This project was developed as part of **Task 1** during my internship at **Future Internships**. It implements a robust machine learning pipeline to forecast retail demand, enabling data-driven inventory management and strategic planning.

## 🚀 Project Overview
Retail environments are highly dynamic. This system automates the extraction of historical sales patterns to predict demand for a 16-day future window. 

The project transitioned from a research prototype to a production-ready modular architecture, favoring a highly optimized **Linear Regression** model for its interpretability and stability in time-series tasks.

## Key Features
- **Automated Feature Engineering:** Generation of lag features (1, 7, 14, 30 days) and rolling mean windows.
- **Temporal Encoding:** Capturing seasonality via Day of Week, Month, and holiday impact.
- **Future Inference Engine:** Custom script for "future suturing" to generate continuous forecasts beyond historical data.
- **Power BI Dashboard:** Interactive visualization of "Actual vs. Predicted" sales with business-ready insights.

##  Tech Stack
- **Language:** Python 
- **Libraries:** Pandas, NumPy, Scikit-learn, Joblib
- **Visualization:** Power BI Desktop
- **Version Control:** Git & GitHub

##  Model Performance
- **Primary Model:** Linear Regression (Baseline Optimized)
- **Error Metric:** RMSE ~483
- **Insight:** The model successfully captures the weekly cycles and correctly identifies the massive sales spikes associated with post-holiday restocks (e.g., August 16th).

## Project Structure
```text
sales-forecasting/
├── data/                   # Raw and processed datasets
├── models/                 # Serialized .pkl models (Joblib)
├── notebooks/              # Research and prototype exploration
├── outputs/                # Generated CSVs for Power BI
├── src/                    # Modular source code
│   ├── preprocessing.py    # Cleaning and Feature Engineering
│   ├── train.py            # Model training script
│   └── predict_future.py   # Future inference script
├── .gitignore              # Files to ignore (data/models)
└── README.md

⚙️ How to Run
Clone the repo:

Bash
git clone https://github.com/donlare188-nanordja/FUTURE_ML_01.git
Install dependencies:

Bash
pip install pandas scikit-learn joblib
Run Inference:

Bash
python src/predict_future.py
🔮 Future Roadmap
[ ] Hybrid Architectures: Integrating BERT/Transformers for sentiment analysis on product metadata.

[ ] Data Enrichment: Adding weather and local economic indicators.

[ ] Deployment: Containerizing microservices using Docker.

🙏 Acknowledgments
Special thanks to Future Internship.