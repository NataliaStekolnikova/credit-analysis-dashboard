# Credit Analysis Dashboard

This repository contains core components of an automated system for analyzing companies' financial statements, aimed at assisting the credit department of a banking institution in making quick and well-founded decisions regarding the granting of loans to organizations.

# How to Run the System

## Project Structure
```
├── data/                                      # 
│   ├── raw/                                   # Original tsv files downloaded and extracted from EDGAR 
│   └── processed/                             # 
│       └── 2025_02_notes/                     # 
│           ├── financial_summary.csv          # Cleaned financial summaries (consumed in Power BI)  
│           └── unique_tags.csv                # Extracted unique tags (financial indicators) 
├── notebooks/                                 # 
│   └── extracting-financial-10-k-reports-via-sec-edgar-db.ipynb  
├── dashboards/                                # 
│   └── Financial state of the companies.pbix  # Power BI dashboard for end-users 
├── src/                                       # 
│   ├── etl.py                                 # 
│   ├── eda.py                                 # 
│   └── stats.py                               # 
├── requirements.txt                           # 
├── .gitignore                                 # 
└── README.md                                  # 
└── LICENSE.txt                                # This repository is licensed under MIT license
```

## Setup Instructions

### Step 1. Clone the Repository

```bash
git clone https://github.com/NataliaStekolnikova/credit-analysis-dashboard.git
cd credit-analysis-dashboard
```

### Step 2. Create a Virtual Environment

```bash
conda create --n credit_analysis_env python=3.9
conda activate credit_analysis_env
```

### Step 3. Install Dependencies

```bash
conda install --file requirements.txt
```

## Running the Analysis Pipeline

### Optional Step 1: Download and Process Financial Data

Use the script below to extract and transform the latest financial dataset from SEC EDGAR:

```bash
python ./src/etl.py 2024
```

This will save the February 2024 filings from the SEC EDGAR database to:

```
./data/raw/2025_02_notes.zip
```

Also this script will create the cleaned and tranformed data sets and will store them at: 

```
data/processed/2025_02_notes/
├── financial_summary_2021.csv   # a transformed and cleaned version of the dataset
├── financial_summary_2022.csv   # a transformed and cleaned version of the dataset
├── financial_summary_2024.csv   # a transformed and cleaned version of the dataset
├── financial_summary_2025.csv   # a transformed and cleaned version of the dataset
├── financial_summary.csv        # the aggregated version of the dataset
└── unique_tags.csv              # extracted list of financial indicators filled by the companies to SEC
```
These files contain the cleaned and structured financial summaries from their financial reports, which are used for:

- Exploratory data analysis (EDA)
- Statistical modeling
- Risk profiling of companies

Dashboard building in Power BI

The unique_tags.csv file includes all standardized XBRL tags extracted from the original reports, useful for tagging, filtering, or tracking metadata.

### Step 2: Perform EDA and Statistical Analysis

Use the script below to perform exploratory and statistical analysis and run the pipeline end-to-end:

```bash
python ./src/stats.py
```

Note, the script ./src/eda.py is used to perform EDA, and is being invoked by stats.py. It requires a dataframe produced by etl.py

## Optional: Experiments

Use the following notebook to conduct experiments: 

```bash
jupyter notebook ./notebooks/extracting-financial-10-k-reports-via-sec-edgar-db.ipynb
```

### What Happens Inside the Notebook

- Loads and processes raw financial data from the SEC EDGAR portal  
- Downloads and extracts ZIP files containing XBRL-formatted financial report data (f.e. 10-K forms)  
- Parses structured data tables such as `sub.tsv`, `num.tsv`, etc.  
- Performs ETL (Extract, Transform, Load) operations:
  - Extracts financial metrics, textual disclosures, and metadata  
  - Pivots dataset and merges them into a single cleaned dataset useful for our purpose
  - Loads and organizes tables into pandas DataFrames useful for further EDA and Statistical analysis  
- Explores key financial indicators and textual sections relevant to bankruptcy prediction  
- Saves the processed data to local directories for further modeling and dashboarding in PowerBI

## Notes

- Designed for working with public filings from EDGAR database
- Structure of EDGAR filings may vary slightly; parsing logic may require updates over time
