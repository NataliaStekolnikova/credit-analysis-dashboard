"""
The MIT License (MIT)

Copyright 2025 Natalia Stekolnikova

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

st.set_page_config(page_title="Credit Risk & Investment Analysis", layout="wide")

# --- Top menu for selecting mode ---
menu = st.selectbox(
    "Select Analysis Mode",
    ("🏦 Credit Risk Analysis", "💰 Investment Project Analysis")
)

# --- Tab 1: Company Credit Risk Analysis ---
if menu == "🏦 Credit Risk Analysis":
    st.title("🏦 Bankruptcy Risk Scoring System")

    st.markdown("Upload financial data to calculate the bankruptcy risk of companies based on key financial metrics.")

    # --- Data upload ---
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # --- Function to calculate risk score ---
        def calculate_risk_score(row):
            score = 0
            if pd.notna(row['Current Ratio']) and row['Current Ratio'] < 1:
                score += 1
            if pd.notna(row['Debt to Equity']) and row['Debt to Equity'] > 2:
                score += 1
            if pd.notna(row['ROE']) and row['ROE'] < 0:
                score += 1
            if pd.notna(row['ROI']) and row['ROI'] < 0:
                score += 1
            if pd.notna(row['Profit Margin']) and row['Profit Margin'] < 0:
                score += 1
            if row[['Current Ratio', 'Debt to Equity', 'ROE', 'ROI', 'Profit Margin']].isnull().any():
                score += 1
            return score

        # --- Applying the function ---
        df['Risk Score'] = df.apply(calculate_risk_score, axis=1)
        df['Risk Level'] = df['Risk Score'].apply(lambda x: 'High Risk' if x >= 3 else ('Medium Risk' if x >= 1 else 'Low Risk'))
        df['Risk Level'] = pd.Categorical(df['Risk Level'], categories=['Low Risk', 'Medium Risk', 'High Risk'], ordered=True)

        st.success("✅ Bankruptcy risk scoring completed!")

        # --- Statistics ---
        st.subheader("Summary Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Companies", len(df))
        with col2:
            risk_counts = df['Risk Level'].value_counts()
            st.metric("Low Risk Companies", risk_counts.get('Low Risk', 0))

        # --- Select risk level to display ---
        st.subheader("Top 10 Companies by Risk Level")
        risk_option = st.selectbox(
            "Select Risk Level to Display",
            options=['Low Risk', 'Medium Risk', 'High Risk'],
            index=0
        )
        filtered_df = df[df['Risk Level'] == risk_option]
        display_columns = (
            ['name', 'Risk Level', 'Current Ratio', 'Debt to Equity', 'ROE', 'ROI', 'Profit Margin']
            if 'name' in df.columns
            else ['Risk Level', 'Current Ratio', 'Debt to Equity', 'ROE', 'ROI', 'Profit Margin']
        )
        top10 = filtered_df[display_columns].head(10)
        st.dataframe(top10)

        # --- Show charts side by side ---
        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            st.subheader("Distribution of Risk Categories")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            sns.countplot(x='Risk Level', data=df, palette="Pastel1", order=['Low Risk', 'Medium Risk', 'High Risk'])
            plt.title('Risk Level Distribution')
            plt.xlabel('Risk Category')
            plt.ylabel('Number of Companies')
            st.pyplot(fig1)

        with col_chart2:
            st.subheader("Feature Importance Analysis")

            # Prepare data for feature importance
            df_corr = df.copy()
            # Create a binary risk flag: 1 for High Risk, 0 otherwise
            df_corr['Risk Flag'] = (df_corr['Risk Level'] == 'High Risk').astype(int)

            # Define features (only those present in df)
            features = [
            "ROE",
            "ROI",
            "Current Ratio",
            "Debt to Equity",
            "Profit Margin"
            ]
            available_features = [f for f in features if f in df_corr.columns]
            df_clean = df_corr.dropna(subset=available_features + ['Risk Flag']).copy()
            if len(df_clean) > 5 and len(available_features) > 1 and df_clean['Risk Flag'].nunique() > 1:
                X = df_clean[available_features]
                y = df_clean['Risk Flag']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
                rf.fit(X_train, y_train)
                importances = pd.Series(rf.feature_importances_, index=available_features).sort_values(ascending=False)
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                sns.barplot(x=importances.values, y=importances.index, ax=ax2)
                plt.title("Feature Importance")
                plt.xlabel("Importance")
                plt.ylabel("Features")
                plt.tight_layout()
                st.pyplot(fig2)
            else:
                st.info("Not enough data or features to compute feature importance.")

    else:
        st.warning("⚠️ Please upload a CSV file to begin.")

# --- Tab 2: Financial Analysis of Investment Project ---
elif menu == "💰 Investment Project Analysis":
    st.title("💰 Investment Project Financial Analysis")

    st.markdown("Analyze the financial viability of an investment project based on cash flows and key metrics like NPV, IRR, and MIRR.")

    # --- Project data (user-editable) ---
    st.subheader("💵 Project Cash Flows")
    st.markdown("Edit the cash flows for each year below:")

    default_cash_flows = [-40000, 8000, 15000, 22000, 30000, 38000]
    years = list(range(len(default_cash_flows)))
    cash_flow_df = pd.DataFrame({
        "Year": years,
        "Cash Flow": default_cash_flows
    })

    edited_df = st.data_editor(
        cash_flow_df,
        num_rows="dynamic",
        use_container_width=True,
        key="cash_flow_editor"
    )

    cash_flows = edited_df["Cash Flow"].tolist()
    discount_rate = st.number_input(
        "Discount Rate (%)", min_value=0.0, max_value=1.0, value=0.15, step=0.01, format="%.2f"
    )
    finance_rate = st.number_input(
        "Finance Rate (%)", min_value=0.0, max_value=1.0, value=0.10, step=0.01, format="%.2f"
    )
    reinvestment_rate = st.number_input(
        "Reinvestment Rate (%)", min_value=0.0, max_value=1.0, value=0.12, step=0.01, format="%.2f"
    )

    # --- Calculations ---
    npv = npf.npv(discount_rate, cash_flows)
    pi = npv / abs(cash_flows[0])

    discounted_cash_flows = [cf / (1 + discount_rate) ** i for i, cf in enumerate(cash_flows)]
    cumulative_dcf = np.cumsum(discounted_cash_flows)
    dpp = None
    for i, total in enumerate(cumulative_dcf):
        if total >= 0:
            dpp = i
            break

    irr = npf.irr(cash_flows)

    positive_flows = [cf if cf > 0 else 0 for cf in cash_flows]
    negative_flows = [cf if cf < 0 else 0 for cf in cash_flows]

    fv_positive = sum([cf * (1 + reinvestment_rate) ** (len(cash_flows) - i - 1) for i, cf in enumerate(positive_flows)])
    pv_negative = sum([cf / (1 + finance_rate) ** i for i, cf in enumerate(negative_flows)])

    mirr = (fv_positive / abs(pv_negative)) ** (1 / (len(cash_flows) - 1)) - 1

    std_dev = np.std(cash_flows[1:])
    mean_cash_flow = np.mean(cash_flows[1:])
    cv = std_dev / mean_cash_flow

    # --- Output ---
    st.subheader("📊 Investment Metrics Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("NPV", f"${npv:,.2f}")
        st.metric("DPP", f"{dpp} years")
        st.metric("σ (Standard Deviation)", f"${std_dev:.2f}")

    with col2:
        st.metric("PI (Profitability Index)", f"{pi:.2f}")
        st.metric("IRR", f"{irr:.2%}")

    with col3:
        st.metric("MIRR", f"{mirr:.2%}")
        st.metric("CV (Coefficient of Variation)", f"{cv:.2f}")