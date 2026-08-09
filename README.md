Agentic AI Dashboard for SME Strategic Planning
MSc Applied AI & Data Science — Dissertation Project
Author: Omotayo Kupoluyi  
Supervisor: Raza Hasan  
Academic Year: 2026–2027

 Overview:
This repository contains the full implementation of the agentic AI dashboard developed for the MSc dissertation “Agentic AI Dashboard for Small Business Strategic Planning.”  
The system integrates multi‑channel operational and financial data, performs automated ETL, generates interpretable forecasts, and produces narrative insights and rule‑based strategic recommendations.

The dashboard is implemented in Streamlit, with supporting analysis in Jupyter Notebook.

 Repository Structure:
Code
MSc-Dissertation-Agentic-AI-Dashboard/
│
├── executive_restaurant_intelligence_dashboard_v2.py   # Main Streamlit dashboard
├── agentic_ai_project.ipynb                            # Jupyter analysis notebook
├── requirements.txt                                     # Python dependencies
│
├── channel_forecasts.pkl                                # Forecast outputs (channels)
├── deliveroo_category_forecasts.pkl                     # Forecast outputs (categories)
│
├── Public_DigitalChannels_Monthly_Revenue_FinancialYear.csv
├── Public_Aggregator_Monthly_Revenue_FinancialYear.csv
├── Public_Deliveroo_Category_Summary_2023_2025.csv
├── Public_EPOS_Annual_Category_Mix_2023_2025.csv
├── Public_PnL_Consolidated.csv                          # P&L dataset
│
└── README.md
 
Running the Dashboard
Install dependencies:

Code
pip install -r requirements.txt
Run the Streamlit dashboard:

Code
streamlit run executive_restaurant_intelligence_dashboard_v2.py
The dashboard will open in your browser.

 Features
Automated ETL pipeline

Multi‑channel performance analysis

Category‑level forecasting

Residual diagnostics

Scenario modelling

Narrative insight generation

Agentic rule‑based strategic recommendations

Consolidated P&L analysis

 Dataset:
All datasets used in this project are publicly archived on Kaggle, anonymised, and contain only aggregated operational and financial information.

 Reproducibility:
This repository contains all files required to:

run the dashboard

reproduce the forecasting outputs

verify the ETL pipeline

inspect the agentic reasoning layer

 Dissertation
The full dissertation is submitted to Southampton Solent University as part of the MSc Applied AI & Data Science programme.
