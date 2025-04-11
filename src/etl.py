"""
The MIT License (MIT)

Copyright 2025 Natalia Stekolnikova

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import requests
import pandas as pd
import numpy as np
from zipfile import ZipFile, BadZipFile
from datetime import datetime

def download_dataset(url, local_filename):
    headers = {
        "User-Agent": "NataliaStekolnikova-FinRiskApp/1.0 (example@gmail.com)",  # Replace with your details
        "Accept-Encoding": "gzip, deflate",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Connection": "keep-alive",
        "Host": "www.sec.gov",
        "Referer": "https://www.sec.gov/",
        "From": "example@gmail.com"  # Replace with your email address
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

def extract_dataset(local_filename, extract_to):
    try:
        with ZipFile(local_filename, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            print(f"Extracted to {extract_to}")
    except BadZipFile:
        print("Error: The downloaded file is not a valid ZIP file.")

def load_data(input_dir):
    # file_names = ["sub", "num", "tag", "cal", "dim", "pre", "ren", "txt"]
    file_names = ["sub", "num"]
    dataframes = {}
    for file_name in file_names:
        file_path = os.path.join(input_dir, f"{file_name}.tsv")
        if os.path.exists(file_path):
            dataframes[file_name] = pd.read_csv(file_path, sep='\t', low_memory=False)
            print(f"Loaded {input_dir}/{file_name}.tsv")
        else:
            print(f"File {file_name}.tsv not found in {input_dir}")
    return dataframes

def process_data(dataframes, output_path):
    sub = dataframes["sub"]
    num = dataframes["num"]

    # Drop unnecessary columns from num
    num = num.drop(columns=['footnote', 'coreg', 'iprx', 'footlen', 'durp', 'datp'], errors='ignore')

    # Convert 'value' to numeric and handle nulls
    num['value'] = num['value'].apply(lambda x: round(x, 2) if pd.notnull(x) else x)

    # Filter num for relevant quarters
    num = num[num['qtrs'] <= 4]

    # Define key tags for financial metrics
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
        'AmortizationOfIntangibleAssets': 'Amortization of Intangibles',
        'AllocatedShareBasedCompensationExpense': 'Share-based Compensation Expense'
    }

    # Filter num for key tags
    num_filtered = num[num['tag'].isin(key_tags.keys())].copy()
    num_filtered['metric_name'] = num_filtered['tag'].map(key_tags)

    # Merge with sub to add metadata
    num_filtered = num_filtered.merge(
        sub[['adsh', 'cik', 'name', 'countryba', 'stprba', 'cityba', 'accepted', 'form', 'period', 'fy', 'fp']],
        on='adsh', how='left'
    )

    # Drop duplicates and handle missing values
    num_filtered = num_filtered.dropna(subset=['value', 'fy', 'fp']).drop_duplicates()

    # Pivot the data to create a clean dataset
    edf = num_filtered.pivot_table(
        index=['adsh', 'cik', 'name', 'countryba', 'stprba', 'cityba', 'accepted', 'form', 'period', 'fy', 'fp'],
        columns='metric_name',
        values='value',
        aggfunc='first'
    ).reset_index()

    # Add calculated financial ratios
    edf['ROE'] = edf['Net Income'] / edf['Shareholder Equity'].replace(0, np.nan)
    edf['ROA'] = edf['Net Income'] / edf['Total Assets'].replace(0, np.nan)
    edf['ROI'] = edf['Net Income'] / (edf['Total Assets'] - edf['Accounts Payable (Current)']).replace(0, np.nan)
    edf['Current Ratio'] = edf['Accounts Receivable (Current)'] / edf['Accounts Payable (Current)']
    edf['Debt to Equity'] = (edf['Total Assets'] - edf['Shareholder Equity']) / edf['Shareholder Equity']
    edf['Profit Margin'] = edf['Net Income'] / edf['Revenue']
    edf['Asset Turnover'] = edf['Revenue'] / edf['Total Assets']

    # Replace infinite values with NaN
    edf.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Add financial risk flag
    edf['Risk Flag'] = (
        (edf['Current Ratio'] < 1) |
        (edf['Debt to Equity'] > 2) |
        (edf['ROE'] < 0) |
        (edf['ROI'] < 0) |
        (edf['Profit Margin'] < 0) |
        edf[['Current Ratio', 'Debt to Equity', 'ROE', 'ROI', 'Profit Margin']].isnull().any(axis=1)
    )
    edf['Risk Category'] = edf['Risk Flag'].map({True: 'High Risk', False: 'Low Risk'})

    # Save the processed dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    edf.to_csv(output_path, index=False)
    print(f"Clean dataset saved at: {output_path}")

    return edf

def main(years_of_interest):
    print("Running ETL pipeline...")
    BASE_URL = "https://www.sec.gov/files/dera/data/financial-statement-notes-data-sets/{}_02_notes.zip"
    # LOCAL_DIR = "C:/Users/natal/credit-analysis-dashboard-v1/data/raw/"
    LOCAL_DIR = "../credit-analysis-dashboard-v1/data/raw/"
    OUTPUT_PATH = "./data/processed/2025_02_notes/financial_summary_{}.csv"

    # Download and process data for the last 10 years
    for year in years_of_interest:
        url = BASE_URL.format(year)
        local_zip_path = os.path.join(LOCAL_DIR, f"{year}_02_notes.zip")
        extract_dir = os.path.join(LOCAL_DIR, f"{year}_02_notes")

        # Step 1: Download
        download_dataset(url, local_zip_path)

        # Step 2: Extract
        extract_dataset(local_zip_path, extract_dir)

        # Step 3: Load
        dataframes = load_data(extract_dir)

        # Step 4: Process
        export_path = OUTPUT_PATH.format(year)
        processed_dataframes = process_data(dataframes, export_path)

        return processed_dataframes

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Convert all arguments after the script name to integers
        years_of_interest = [int(year) for year in sys.argv[1:]]
        main(years_of_interest)
    else:
        raise ValueError("You must pass one or more years (e.g., python etl.py 2023 2024)")
