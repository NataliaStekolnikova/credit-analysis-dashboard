import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

# --- Load the processed dataset ---
DATA_PATH = "C:/Users/natal/credit-analysis-dashboard/data/processed/2025_02_notes/financial_summary_clean_full.csv"
edf_full_clean = pd.read_csv(DATA_PATH)

# --- Descriptive Statistics ---
def descriptive_statistics(df):
    print("\nBasic descriptive statistics:")
    desc_stats = df[['Net Income', 'Revenue', 'ROE', 'ROA']].describe().T.copy()
    desc_stats['median'] = df[['Net Income', 'Revenue', 'ROE', 'ROA']].median()
    desc_stats['mode'] = df[['Net Income', 'Revenue', 'ROE', 'ROA']].mode().iloc[0]
    desc_stats['range'] = desc_stats['max'] - desc_stats['min']
    desc_stats['variance'] = df[['Net Income', 'Revenue', 'ROE', 'ROA']].var()
    print(desc_stats)

# --- Visualizations: Histograms and Boxplots ---
def plot_distributions(df):
    for metric in ['Net Income', 'Revenue', 'ROE', 'ROA']:
        plt.figure(figsize=(10, 4))
        sns.histplot(df[metric].dropna(), kde=True, bins=30)
        plt.title(f'Histogram + KDE of {metric}')
        plt.show()

        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[metric].dropna())
        plt.title(f'Boxplot of {metric}')
        plt.show()

# --- Correlation Analysis ---
def correlation_analysis(df):
    numeric_df = df.select_dtypes(include=['number', 'category'])
    correlation_matrix = numeric_df.corr()
    plt.figure(figsize=(14, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar_kws={"shrink": 0.8})
    plt.title("Correlation Matrix of Financial Metrics", fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# --- Hypothesis Testing ---
def hypothesis_test(df):
    filtered_df = df[df['form'].isin(['10-K', '10-Q'])].dropna(subset=['ROA'])
    roa_10k = filtered_df[filtered_df['form'] == '10-K']['ROA']
    roa_10q = filtered_df[filtered_df['form'] == '10-Q']['ROA']
    t_stat, p_value = ttest_ind(roa_10k, roa_10q, equal_var=False)
    print(f"T-statistic: {t_stat:.2f}, P-value: {p_value:.2e}")

# --- Linear Regression Model ---
def linear_regression_model(df):
    df_model = df[['ROI', 'Net Income', 'Revenue', 'Cash From Ops', 'Total Assets', 'Shareholder Equity', 'form', 'fp']].dropna()
    df_model = df_model[(df_model['ROI'] > -2) & (df_model['ROI'] < 2)]
    for col in ['Net Income', 'Revenue', 'Cash From Ops', 'Total Assets', 'Shareholder Equity']:
        df_model[f'log_{col}'] = np.log(df_model[col].abs() + 1)
    df_model = pd.get_dummies(df_model, columns=['form', 'fp'], drop_first=True)
    X_cols = [col for col in df_model.columns if col.startswith('log_') or col.startswith('form_') or col.startswith('fp_')]
    X = df_model[X_cols]
    y = df_model['ROI']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = sm.add_constant(X_scaled)
    model = sm.OLS(y, X_scaled).fit()
    print(model.summary())

# --- Main Execution ---
if __name__ == "__main__":
    descriptive_statistics(edf_full_clean)
    plot_distributions(edf_full_clean)
    correlation_analysis(edf_full_clean)
    hypothesis_test(edf_full_clean)
    linear_regression_model(edf_full_clean)