from etl import main as run_etl
from eda import main as run_eda

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, mannwhitneyu, pearsonr
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy import stats


def correlation_analysis(df):
    """
    Perform correlation analysis between financial metrics and Risk Flag.
    """
    print("\n--- Correlation Analysis ---")
    features = [
        'Net Income', 'Total Assets', 'Shareholder Equity', 'Cash From Ops', 'Revenue',
        'Accounts Receivable (Current)', 'Accounts Payable (Current)', 'Accrued Liabilities (Current)',
        'Additional Paid-in Capital', 'ROI', 'ROE', 'ROA',
        'Amortization of Intangibles', 'Share-based Compensation Expense', 'Risk Flag'
    ]

    # Remove outliers (1% and 99% quantiles)
    trimmed_df = df.copy()
    for feature in features:
        lower_bound = trimmed_df[feature].quantile(0.01)
        upper_bound = trimmed_df[feature].quantile(0.99)
        trimmed_df = trimmed_df[(trimmed_df[feature] >= lower_bound) & (trimmed_df[feature] <= upper_bound)]

    # Calculate correlation matrix
    correlation_matrix = trimmed_df[features].corr()

    # Plot correlation heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(
        correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm",
        square=True, cbar_kws={"shrink": 0.8}, linewidths=0.5
    )
    plt.title("Correlation Matrix (After Outlier Removal)", fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def hypothesis_test(df):
    """
    Perform hypothesis testing on financial metrics.
    """
    print("\n--- Hypothesis Testing ---")

    # Hypothesis Test 1: Pearson Correlation between Accrued Liabilities and Risk Flag
    corr_df = df.dropna(subset=['Accrued Liabilities (Current)', 'Risk Flag'])
    x = corr_df['Accrued Liabilities (Current)']
    y = corr_df['Risk Flag']

    # Remove 1% outliers
    lower_bound = x.quantile(0.01)
    upper_bound = x.quantile(0.99)
    filtered_corr_df = corr_df[(x >= lower_bound) & (x <= upper_bound)]
    x_filtered = filtered_corr_df['Accrued Liabilities (Current)']
    y_filtered = filtered_corr_df['Risk Flag']

    # Pearson correlation test
    r, p_value = pearsonr(x_filtered, y_filtered)
    print(f"Pearson Correlation (Accrued Liabilities vs Risk Flag): r = {r:.4f}, p-value = {p_value:.4e}")

    # Hypothesis Test 2: Difference in Profit Margin between Risk Groups
    pm_series = df['Profit Margin'].dropna()
    lower_bound = pm_series.quantile(0.01)
    upper_bound = pm_series.quantile(0.99)
    filtered_df = df[
        (df['Profit Margin'] >= lower_bound) &
        (df['Profit Margin'] <= upper_bound)
    ].dropna(subset=['Risk Category'])

    low_risk = filtered_df[filtered_df['Risk Category'] == 'Low Risk']['Profit Margin']
    high_risk = filtered_df[filtered_df['Risk Category'] == 'High Risk']['Profit Margin']

    # Welch's t-test
    t_stat, p_ttest = ttest_ind(low_risk, high_risk, equal_var=False)
    print(f"Welch's t-test (Profit Margin): t = {t_stat:.2f}, p-value = {p_ttest:.4e}")

    # Mann-Whitney U Test
    u_stat, p_mannwhitney = mannwhitneyu(low_risk, high_risk, alternative='two-sided')
    print(f"Mann-Whitney U Test (Profit Margin): U = {u_stat:.2f}, p-value = {p_mannwhitney:.4e}")

    # Hypothesis Test 3: Logistic Regression (Debt to Equity vs Risk Flag)
    logit_df = df.dropna(subset=['Debt to Equity', 'Risk Flag'])
    lower_bound = logit_df['Debt to Equity'].quantile(0.01)
    upper_bound = logit_df['Debt to Equity'].quantile(0.99)
    filtered_logit_df = logit_df[(logit_df['Debt to Equity'] >= lower_bound) & (logit_df['Debt to Equity'] <= upper_bound)]

    X = sm.add_constant(filtered_logit_df[['Debt to Equity']])
    y = filtered_logit_df['Risk Flag']
    model = sm.Logit(y, X).fit(disp=False)
    coef = model.params['Debt to Equity']
    p_value = model.pvalues['Debt to Equity']
    odds_ratio = np.exp(coef)
    print(f"Logistic Regression (Debt to Equity vs Risk Flag): Coef = {coef:.4f}, p-value = {p_value:.4e}, Odds Ratio = {odds_ratio:.4f}")


def linear_regression_model(df):
    """
    Perform linear regression analysis on Revenue and Net Income.
    """
    print("\n--- Linear Regression Analysis ---")

    # Data preparation with 1% outlier trimming
    df_lr = df.dropna(subset=['Revenue', 'Net Income'])
    lower_rev, upper_rev = df_lr['Revenue'].quantile([0.01, 0.99])
    lower_ni, upper_ni = df_lr['Net Income'].quantile([0.01, 0.99])

    df_trimmed = df_lr[
        (df_lr['Revenue'].between(lower_rev, upper_rev)) &
        (df_lr['Net Income'].between(lower_ni, upper_ni))
    ]

    # Define X and y
    X = df_trimmed[['Revenue']]
    y = df_trimmed['Net Income']

    # Fit LinearRegression model
    lin_model = LinearRegression()
    lin_model.fit(X, y)
    y_pred = lin_model.predict(X)

    # Supplement with statsmodels for p-value and CI
    X_sm = sm.add_constant(X)
    ols_model = sm.OLS(y, X_sm).fit()
    p_value = ols_model.pvalues['Revenue']
    conf_int = ols_model.conf_int().loc['Revenue']
    lower_bound, upper_bound = conf_int

    # Extract metrics
    intercept = lin_model.intercept_
    coef = lin_model.coef_[0]
    r2_sklearn = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)

    print(f"Linear Regression (Revenue → Net Income): Coef = {coef:.4f}, p-value = {p_value:.4e}, R² = {r2_sklearn:.4f}, RMSE = {rmse:.2e}")

    # Plot regression line with confidence interval
    plot_df = df_trimmed[['Revenue', 'Net Income']].copy()
    plot_df['y_pred'] = y_pred
    plot_df = plot_df.sort_values(by='Revenue')

    X_sorted = plot_df['Revenue'].values
    y_sorted = plot_df['Net Income'].values
    y_pred_sorted = plot_df['y_pred'].values

    n = len(X_sorted)
    t_value = stats.t.ppf(0.975, df=n - 2)
    mean_x = np.mean(X_sorted)
    s_err = np.sqrt(np.sum((y_sorted - y_pred_sorted)**2) / (n - 2))
    conf = t_value * s_err * np.sqrt(
        1/n + (X_sorted - mean_x)**2 / np.sum((X_sorted - mean_x)**2)
    )

    lower = y_pred_sorted - conf
    upper = y_pred_sorted + conf

    plt.figure(figsize=(10, 6))
    plt.scatter(X_sorted, y_sorted, color='lightblue', alpha=0.6, label='Actual Data')
    plt.plot(X_sorted, y_pred_sorted, color='darkblue', linewidth=2, label='Regression Line')
    plt.fill_between(X_sorted, lower, upper, color='skyblue', alpha=0.3, label='95% Confidence Interval')
    plt.title('Linear Regression with 95% Confidence Interval\nRevenue → Net Income', fontsize=14)
    plt.xlabel('Revenue')
    plt.ylabel('Net Income')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = run_etl()
    run_eda(df)

    correlation_analysis(df)
    hypothesis_test(df)
    linear_regression_model(df)