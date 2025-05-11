# app.py

import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Credit Risk & Investment Analysis")	

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
    uploaded_file = st.file_uploader("c", type=["csv"])

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
            st.metric("High Risk Companies", risk_counts.get('High Risk', 0))

        # --- Top 10 companies with the highest risk ---
        st.subheader("Top 10 Companies by Risk Score")
        if 'name' in df.columns:
            top10 = df[['name', 'Risk Score', 'Risk Level', 'Current Ratio', 'Debt to Equity', 'Profit Margin']].sort_values('Risk Score', ascending=False).head(10)
        else:
            top10 = df[['Risk Score', 'Risk Level', 'Current Ratio', 'Debt to Equity', 'Profit Margin']].sort_values('Risk Score', ascending=False).head(10)
        st.dataframe(top10)

        # --- Risk level distribution chart ---
        st.subheader("Distribution of Risk Categories")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.countplot(x='Risk Level', data=df, palette="Pastel1", order=['Low Risk', 'Medium Risk', 'High Risk'])
        plt.title('Risk Level Distribution')
        plt.xlabel('Risk Category')
        plt.ylabel('Number of Companies')
        st.pyplot(fig1)

        # --- Correlation heatmap ---
        st.subheader("Correlation Heatmap (Financial Metrics vs Risk Score)")
        metrics = ['Current Ratio', 'Debt to Equity', 'ROE', 'ROI', 'Profit Margin', 'Risk Score']
        corr_df = df[metrics].dropna()
        corr_matrix = corr_df.corr()

        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0)
        plt.title('Correlation Matrix')
        st.pyplot(fig2)

    else:
        st.warning("⚠️ Please upload a CSV file to begin.")

# --- Tab 2: Financial Analysis of Investment Project ---
elif menu == "💰 Investment Project Analysis":
    st.title("💰 Investment Project Financial Analysis")

    st.markdown("Analyze the financial viability of an investment project based on cash flows and key metrics like NPV, IRR, and MIRR.")

    # --- Project data ---
    cash_flows = [-40000, 8000, 15000, 22000, 30000, 38000]
    discount_rate = 0.15  # discount rate
    finance_rate = 0.10   # project financing rate
    reinvestment_rate = 0.12  # reinvestment rate of income

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
