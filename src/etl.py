import os
import requests
import pandas as pd
import numpy as np
from zipfile import ZipFile, BadZipFile

# --- Step 1: Download the dataset ---
def download_dataset(url, local_filename):
    headers = {
        "User-Agent": "YourName-YourAppName/1.0 (your-email@example.com)",  # Replace with your details
        "Accept-Encoding": "gzip, deflate",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Connection": "keep-alive",
        "Host": "www.sec.gov",
        "Referer": "https://www.sec.gov/",
        "From": "natalia.a.stekolnikova@gmail.com"  # Replace with your email address
    }
    response = requests.get(url, headers=headers, stream=True)
    if response.status_code == 200:
        with open(local_filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded: {local_filename}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")
        response.raise_for_status()

# --- Step 2: Extract the dataset ---
def extract_dataset(local_filename, extract_to):
    try:
        with ZipFile(local_filename, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            print(f"Extracted to {extract_to}")
    except BadZipFile:
        print("Error: The downloaded file is not a valid ZIP file.")

# --- Step 3: Load the dataset ---
def load_data(input_dir):
    sub = pd.read_csv(os.path.join(input_dir, "sub.tsv"), sep='\t', low_memory=False)
    num = pd.read_csv(os.path.join(input_dir, "num.tsv"), sep='\t', low_memory=False)
    print("Files loaded successfully!")
    return sub, num

# --- Step 4: Process the dataset ---
def process_data(sub, num, output_path):
    key_tags = {
        'NetIncomeLoss': 'Net Income',
        'Assets': 'Total Assets',
        'StockholdersEquity': 'Shareholder Equity',
        'NetCashProvidedByUsedInOperatingActivities': 'Cash From Ops',
        'Revenues': 'Revenue',
        'AccountsReceivableNetCurrent': 'Accounts Receivable (Current)',
        'AccountsPayableCurrent': 'Accounts Payable (Current)',
        'AccruedLiabilitiesCurrent': 'Accrued Liabilities (Current)',
        'AccumulatedDepreciationDepletionAndAmortization': 'Accumulated Depreciation/Amortization',
        'AdditionalPaidInCapitalCommonStock': 'Additional Paid-in Capital',
        'AccruedIncomeTaxesNoncurrent': 'Accrued Income Taxes (Noncurrent)',
        'AmortizationOfIntangibleAssets': 'Amortization of Intangibles',
        'AllocatedShareBasedCompensationExpense': 'Share-based Compensation Expense'
    }

    num_filtered = num[num['tag'].isin(key_tags.keys())].copy()
    num_filtered['metric_name'] = num_filtered['tag'].map(key_tags)
    num_filtered = num_filtered.merge(sub[['adsh', 'cik', 'name', 'fy', 'fp', 'form']], on='adsh', how='left')

    edf = num_filtered.pivot_table(
        index=['cik', 'name', 'adsh', 'fy', 'fp', 'form'],
        columns='metric_name',
        values='value',
        aggfunc='first'
    ).reset_index()

    edf['ROE'] = edf['Net Income'] / edf['Shareholder Equity'].replace(0, np.nan)
    edf['ROA'] = edf['Net Income'] / edf['Total Assets'].replace(0, np.nan)
    edf['ROI'] = edf['Net Income'] / (edf['Total Assets'] - edf['Accounts Payable (Current)'])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    edf.to_csv(output_path, index=False)
    print(f"Clean dataset saved at: {output_path}")

# --- Main Execution ---
if __name__ == "__main__":
    DATA_URL = "https://www.sec.gov/files/dera/data/financial-statement-notes-data-sets/2025_02_notes.zip"
    LOCAL_ZIP_PATH = "C:/Users/natal/credit-analysis-dashboard-v1/data/raw/2025_02_notes.zip"
    EXTRACT_DIR = "C:/Users/natal/credit-analysis-dashboard-v1/data/raw/2025_02_notes"
    OUTPUT_PATH = "C:/Users/natal/credit-analysis-dashboard/data/processed/2025_02_notes/financial_summary_clean_full.csv"

    # Step 1: Download
    download_dataset(DATA_URL, LOCAL_ZIP_PATH)

    # Step 2: Extract
    extract_dataset(LOCAL_ZIP_PATH, EXTRACT_DIR)

    # Step 3: Load
    sub_df, num_df = load_data(EXTRACT_DIR)

    # Step 4: Transform
    process_data(sub_df, num_df, OUTPUT_PATH)