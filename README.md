# Credit Analysis Dashboard
# How to Run the System

This system extracts key financial information from 10-K filings submitted to the SEC EDGAR database and saves structured data for further analysis or visualization.

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
│   ├── download_10k.py        # Script to fetch 10-K filings automatically
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
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Analysis Pipeline

### Step 0: Automatically Fetch the Latest 10-K Filing

Use the script below to download the latest 10-K form for any public company by ticker symbol:

```bash
python src/download_10k.py AAPL
```

This will save the most recent 10-K HTML filing from SEC EDGAR to:

```
data/raw/AAPL_10K.html
```

To fetch another company’s report, replace `AAPL` with the desired ticker.


### Step 1: Download 10-K Filing Automatically

Use the script to fetch the latest 10-K filing for a given ticker symbol:

```bash
python src/download_10k.py
```

By default, it will download the latest 10-K for "AAPL" and save it to:

```
data/raw/AAPL_10K.html
```

To change the ticker, edit the script `download_10k.py`.

### Step 2: Open and Run the Notebook

```bash
jupyter notebook notebooks/extracting-financial-10-k-reports-via-sec-edgar-db.ipynb
```

## What Happens Inside the Notebook

- Loads the raw 10-K text or HTML file
- Extracts relevant sections using regular expressions and string parsing:
  - Item 7: Management Discussion & Analysis (MD&A)
  - Item 8: Financial Statements and Supplementary Data
- Cleans and organizes the extracted content
- Saves output to:

```
data/processed/2025_02_notes/
├── financial_summary_clean_full/
└── unique_tags/
```

## Notes

- Designed for working with English-language 10-K filings from EDGAR
- Structure of EDGAR filings may vary slightly; parsing logic may require updates over time
