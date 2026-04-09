

=======================================================================
  AI COMPLIANCE & OPERATIONS ASSISTANT FOR MOBILE FOOD VENDORS
  Suffolk County, NY
  Data Architect Module -- Group Member Setup Guide
=======================================================================

Project:   AI Compliance & Operations Assistant for Mobile Food Vendors
Team Role: Data Architect Lead
Author:    Aisha (Data Architect Lead)
Last Updated: April 2026

-----------------------------------------------------------------------
WHAT THIS MODULE DOES
-----------------------------------------------------------------------

This is the data foundation for the entire AI compliance system.
It sources, cleans, and prepares all datasets that feed into:
  - The NLP regulation extractor (Agent 2)
  - The predictive inspection risk model (Agent 3)
  - The web-based compliance dashboard (Agent 4)

The pipeline runs in 6 steps:

  Step 1 -- Ingest raw data (Suffolk County violations, NOAA weather,
            Census business counts)
  Step 2 -- Clean and normalize all datasets
  Step 3 -- Generate EDA charts and HTML stakeholder report
  Step 4 -- Build ML-ready feature tables
  Step 5 -- Train risk prediction model and score vendors
  Step 6 -- Extract regulation rules and build compliance reference

-----------------------------------------------------------------------
SYSTEM REQUIREMENTS
-----------------------------------------------------------------------

  - Windows 10 or 11 (tested on Windows 11)
  - Python 3.10, 3.11, 3.12, or 3.13
  - Internet connection (for NOAA weather data)
  - ~500 MB disk space

-----------------------------------------------------------------------
STEP 1 -- INSTALL PYTHON (if not already installed)
-----------------------------------------------------------------------

  1. Open: https://www.python.org/downloads/
  2. Download Python 3.12 (recommended)
  3. Run the installer
  4. IMPORTANT: Check "Add Python to PATH" before clicking Install

  Verify Python is installed:
    Open Command Prompt and type:
      python --version
    You should see: Python 3.12.x


-----------------------------------------------------------------------
STEP 2 -- GET THE PROJECT FILES
-----------------------------------------------------------------------

  Option A: Received as a ZIP file
    1. Unzip the file to: C:\Users\YourName\Downloads\files\
    2. Make sure these files are all in the SAME folder:
         run_pipeline.py
         01_data_ingestion.py
         02_data_cleaning.py
         03_eda_report.py
         04_feature_engineering.py
         05_risk_model.py
         06_regulation_extractor.py
         07_dashboard.py
         00_generate_synthetic_data.py
         requirements.txt
         setup_and_run.bat
         GROUP_README.md  (this file)

  Option B: Received from Google Drive / shared folder
    1. Download all .py files and requirements.txt
    2. Put them all in the same folder


-----------------------------------------------------------------------
STEP 3 -- INSTALL DEPENDENCIES
-----------------------------------------------------------------------

  Open Command Prompt (search "cmd" in Windows Start menu):

    cd "C:\Users\YourName\Downloads\files"
    pip install -r requirements.txt

  This installs: pandas, numpy, scikit-learn, matplotlib, seaborn,
  pyarrow, requests, flask

  Takes about 2-3 minutes. You will see a lot of text -- that is normal.

  If you get a "pip not found" error:
    python -m pip install -r requirements.txt


-----------------------------------------------------------------------
STEP 4 -- GET THE SUFFOLK COUNTY VIOLATION DATA
-----------------------------------------------------------------------

  The Suffolk County Open Data portal requires manual download.
  (Their API is restricted -- we cannot automate this part.)

  4a. Open your browser and download each CSV file:

      2024 data:
      https://opendata.suffolkcountyny.gov/datasets/restaurant-violations-2024

      2023 data:
      https://opendata.suffolkcountyny.gov/datasets/restaurant-violations-2023-1

      2022 data:
      https://opendata.suffolkcountyny.gov/datasets/restaurant-violations-2022

      2021 data:
      https://opendata.suffolkcountyny.gov/datasets/restaurant-violations-2021

      2020 data:
      https://opendata.suffolkcountyny.gov/datasets/restaurant-violations-2020

  4b. On each page: click the "Download" button, then choose "Spreadsheet (CSV)"

  4c. Create this folder (if it does not exist):
      C:\Users\YourName\Downloads\files\suffolk_data\raw\inspections\manual\

  4d. Move all 5 downloaded CSV files into that manual\ folder.

  NOTE: If you skip this step, the pipeline will automatically generate
  synthetic (fake) test data so you can still run and test everything.
  Replace with real data when available.


-----------------------------------------------------------------------
STEP 5 -- RUN THE PIPELINE
-----------------------------------------------------------------------

  Option A: Double-click (easiest)
    Double-click "setup_and_run.bat" in your files folder.
    A black window will open and run everything automatically.

  Option B: Command Prompt
    cd "C:\Users\YourName\Downloads\files"
    python run_pipeline.py

  Option C: Single step at a time (for testing)
    python run_pipeline.py --step 1    # just ingestion
    python run_pipeline.py --step 2    # just cleaning
    python run_pipeline.py --step 3    # just EDA charts
    python run_pipeline.py --step 4    # just feature engineering
    python run_pipeline.py --step 5    # just risk model
    python run_pipeline.py --step 6    # just regulation extraction

  Option D: Skip data download (use already-downloaded data)
    python run_pipeline.py --skip-ingest


-----------------------------------------------------------------------
STEP 6 -- VIEW THE OUTPUTS
-----------------------------------------------------------------------

  After the pipeline runs, open these files in your browser:

  Main EDA Report (charts + data summary):
    suffolk_data\reports\data_architect_report.html

  Regulation Reference (searchable compliance rules):
    suffolk_data\regulations\regulation_index.html

  Model Performance Chart:
    suffolk_data\reports\model_performance.png


  To launch the interactive web dashboard:
    pip install flask
    python 07_dashboard.py
    Then open: http://localhost:5000 in your browser
    Press Ctrl+C to stop the dashboard.


-----------------------------------------------------------------------
EXPECTED OUTPUT FILES
-----------------------------------------------------------------------

  suffolk_data\
    raw\
      inspections\suffolk_violations_all.json     (165,000+ rows real data)
      weather\noaa_islip_daily.json               (8,786 weather records)
      census\suffolk_food_business_patterns.json
    clean\
      inspections_clean.parquet                   (ML-ready inspections)
      violations_clean.parquet
      weather_clean.parquet
      master_feature_table.parquet                (joined feature table)
    ml_ready\
      features.parquet                            (model input features)
      labels.parquet                              (pass/fail target)
      risk_scores.parquet                         (vendor risk scores)
    models\
      risk_model.pkl                              (trained Random Forest)
      model_report.txt                            (AUC + accuracy metrics)
    regulations\
      mobile_vendor_rules.json                    (12 compliance rules)
      permit_checklist.json                       (12-step permit process)
      violation_rule_map.json                     (code to rule mapping)
      regulation_index.html                       (searchable HTML)
    reports\
      data_architect_report.html                  (OPEN THIS FIRST)
      inspection_trends.png
      weather_inspection_correlation.png
      model_performance.png
    metadata\
      ingestion_manifest.csv


-----------------------------------------------------------------------
HOW THE DATA FLOWS TO YOUR AGENT
-----------------------------------------------------------------------

  If you are working on Agent 2 (NLP Rule Extractor):
    Read from: suffolk_data\regulations\mobile_vendor_rules.json
               suffolk_data\regulations\violation_rule_map.json
               suffolk_data\raw\regulations\  (for full text files)

  If you are working on Agent 3 (Predictive Model):
    Read from: suffolk_data\ml_ready\features.parquet
               suffolk_data\ml_ready\labels.parquet
               suffolk_data\ml_ready\risk_scores.parquet

  If you are working on Agent 4 (Web Interface):
    Read from: suffolk_data\clean\inspections_clean.parquet
               suffolk_data\ml_ready\risk_scores.parquet
    Use the API: python 07_dashboard.py -> http://localhost:5000/api/


-----------------------------------------------------------------------
API ENDPOINTS (for Agent 4 / Web Interface team)
-----------------------------------------------------------------------

  Start the API server:
    python 07_dashboard.py

  Available endpoints (all return JSON):

  GET http://localhost:5000/api/risk-scores
      All vendor risk scores (0-100) with tier (Low/Medium/High/Critical)
      Add ?mobile_only=true to filter mobile vendors only
      Add ?limit=50 to limit results

  GET http://localhost:5000/api/vendor/FACILITY_NAME
      Full inspection history + risk score for one vendor

  GET http://localhost:5000/api/violations
      Recent 200 violations with severity codes

  GET http://localhost:5000/api/rules
      All 12 mobile vendor compliance rules

  GET http://localhost:5000/api/permit-checklist
      12-step permit application checklist

  GET http://localhost:5000/api/weather
      Current weather risk advisory (heat/rain flags)


-----------------------------------------------------------------------
COMMON ERRORS AND FIXES
-----------------------------------------------------------------------

  ERROR: "python is not recognized"
  FIX:   Reinstall Python and check "Add Python to PATH"

  ERROR: "No module named pandas" (or any other module)
  FIX:   pip install -r requirements.txt

  ERROR: "ModuleNotFoundError: No module named '02_data_cleaning'"
  FIX:   Make sure you are running from the correct folder:
         cd "C:\Users\YourName\Downloads\files"
         python run_pipeline.py

  ERROR: "suffolk_violations_all.json not found"
  FIX:   Either download the CSVs to the manual\ folder (Step 4)
         OR run without --skip-ingest to generate synthetic data:
         python run_pipeline.py

  ERROR: Pipeline stops at Step 5 (risk model)
  FIX:   This is normal if there is not enough data variation.
         Steps 1-4 and 6 still complete successfully.
         The model needs both passing and failing inspections to train.

  ERROR: "Permission denied" when creating folders
  FIX:   Right-click Command Prompt -> Run as Administrator


-----------------------------------------------------------------------
DATA SOURCES (for your report/presentation)
-----------------------------------------------------------------------

  Suffolk County Restaurant Violations (2020-2024):
    Source:  Suffolk County Open Data Portal
    URL:     opendata.suffolkcountyny.gov
    Format:  CSV (manual download required)
    Records: ~165,000 violation records across 5 years

  NOAA Weather Data -- Islip MacArthur Airport:
    Source:  NOAA Climate Data Online (CDO)
    Station: GHCND:USW00094741 (Islip, NY -- closest to Suffolk center)
    URL:     ncdc.noaa.gov/cdo-web
    Records: 8,786 daily records (2020-2025)
    Token:   Pre-configured (ROFECOOtQsJObXMRDAblIUkIvgWOdepd)

  US Census Business Patterns (2022):
    Source:  US Census Bureau API
    URL:     api.census.gov
    Scope:   NAICS 7225 (Restaurants), Suffolk County FIPS 103

  NYS Sanitary Code Part 14:
    Source:  NYS Department of Health
    URL:     health.ny.gov/regulations/nycrr/title_10/part_14/

  Suffolk County Sanitary Code Article 13:
    Source:  Suffolk County Health Services
    URL:     suffolkcountyny.gov/Departments/Health-Services


-----------------------------------------------------------------------
CONTACT / QUESTIONS
-----------------------------------------------------------------------

  Data Architect Lead: Aisha
  Role: Responsible for all data sourcing, cleaning, and pipeline

  If the pipeline fails or you need additional data outputs,
  contact the Data Architect Lead with the full error message
  from the terminal (copy and paste the red text).

  Log file location: files\pipeline.log
  (contains full history of every pipeline run)

=======================================================================

