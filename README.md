# Credit Analysis Dashboard

This repository contains core components of an automated system for analyzing companies' financial statements, aimed at assisting the credit department of a banking institution in making quick and well-founded decisions regarding the granting of loans to organizations.

# How to Run the System

## Project Structure

├── data/                       
│   ├── raw/                    # Original 10-K files downloaded from EDGAR  
│   └── processed/  
│       └── 2025_02_notes/      
│           ├── financial_summary_clean_full/  # Cleaned financial summaries (CSV, JSON, etc.)  
│           └── unique_tags/                   # Extracted unique tags  
├── notebooks/                 
│   └── extracting-financial-10-k-reports-via-sec-edgar-db.ipynb  
├── dashboards/                # Power BI or Tableau dashboards  
├── src/                       
│   ├── etl.py
│   ├── eda.py
│   └── stats.py       
├── requirements.txt           
├── .gitignore  
└── README.md  

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/NataliaStekolnikova/credit-analysis-dashboard.git
cd credit-analysis-dashboard
```

### 2. Create a Virtual Environment

```bash
conda create --n credit_analysis_env python=3.9
conda activate credit_analysis_env
```

### 3. Install Dependencies

```bash
conda install --file requirements.txt
```

## Running the Analysis Pipeline

### Step 1: Download the Latest Data Set

Use the script below to download the latest financial data set for public companies:

```bash
python src/ETL.py
```

This will save the most filings from SEC EDGAR to:

```
data/raw/2025_02_notes.zip
```

Also this script will create the cleaned and tranformed data sets and will store them at: 

```
data/processed/2025_02_notes/financial_summary_clean_full.csv
```

### Step 2: Perform EDA

Use the script below to perform EDA:

```bash
python src/EDA.py
```

### Step 3: Perform Statistical Analysis

Use the script below to perform Statistics:

```bash
python src/stats.py
```

## Optional: Experiments

Use the following notebook to conduct experiments: 

```bash
jupyter notebook notebooks/extracting-financial-10-k-reports-via-sec-edgar-db.ipynb
```

### What Happens Inside the Notebook

- Loads and processes raw financial data from the SEC EDGAR portal  
- Downloads and extracts ZIP files containing XBRL-formatted annual report data (10-K forms)  
- Parses structured data tables such as `sub.tsv`, `num.tsv`, `tag.tsv`, `txt.tsv`, etc.  
- Performs ETL (Extract, Transform, Load) operations:
  - Extracts financial metrics, textual disclosures, and metadata  
  - Loads and organizes tables into pandas DataFrames  
- Prepares a clean dataset for exploratory analysis
- Explores key financial indicators and textual sections relevant to bankruptcy prediction  
- Saves the processed data to local directories for modeling and dashboarding  

## Notes

- Designed for working with public filings from EDGAR database
- Structure of EDGAR filings may vary slightly; parsing logic may require updates over time
