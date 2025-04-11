"""
Copyright 2025 Natalia Stekolnikova

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from etl import main as run_etl
from eda import main as run_eda

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from scipy.stats import ttest_ind, mannwhitneyu, pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def hypothesis_test_pearson_corr_of_accrued_liabiilties_and_risk_flag(edf):
    corr_df = edf.dropna(subset=['Accrued Liabilities (Current)', 'Risk Flag'])

    x = corr_df['Accrued Liabilities (Current)']
    y = corr_df['Risk Flag']  # Assumes binary (0 = low risk, 1 = high risk)

    # Perform Pearson correlation test
    r, p_value = pearsonr(x, y)

    # Format results in markdown
    markdown_result = f"""
    ### Pearson Correlation Test 1a: Accrued Liabilities vs Risk Flag

    | Metric           | Value        |
    |------------------|--------------|
    | Correlation (r)  | `{r:.4f}`    |
    | p-value          | `{p_value:.4e}` |

    **Conclusion**:
    {"✅ Reject H₀ — statistically significant linear correlation." if p_value < 0.05 else "— Fail to reject H₀ — no significant linear correlation."}
    """

    print(markdown_result)

def hypothesis_test_pearson_corr_of_accrued_liabiilties_and_risk_flag_out(edf):
    corr_df = edf.dropna(subset=['Accrued Liabilities (Current)', 'Risk Flag'])

    x = corr_df['Accrued Liabilities (Current)']
    y = corr_df['Risk Flag']  # Assumes binary (0 = low risk, 1 = high risk)

    # Perform Pearson correlation test
    r, p_value = pearsonr(x, y)

    # Remove 1% of outliers from 'Accrued Liabilities (Current)'
    lower_bound = x.quantile(0.01)
    upper_bound = x.quantile(0.99)
    filtered_corr_df = corr_df[(x >= lower_bound) & (x <= upper_bound)]

    # Update x and y after filtering
    x_filtered = filtered_corr_df['Accrued Liabilities (Current)']
    y_filtered = filtered_corr_df['Risk Flag']

    # Perform Pearson correlation test on filtered data
    r_filtered, p_value_filtered = pearsonr(x_filtered, y_filtered)

    # Format results in markdown
    markdown_result = f"""
    ### Pearson Correlation Test 1b: Accrued Liabilities vs Risk Flag (After Removing 1% Outliers)

    | Metric           | Value        |
    |------------------|--------------|
    | Correlation (r)  | `{r_filtered:.4f}`    |
    | p-value          | `{p_value_filtered:.4e}` |

    **Conclusion**:
    {"✅ Reject H₀ — statistically significant linear correlation." if p_value_filtered < 0.05 else "— Fail to reject H₀ — no significant linear correlation."}
    """

    print(markdown_result)

def hypothesis_test_t_test_or_mann_whitney_u_of_profit_margin_between_risk_groups(edf):
    # Drop rows with missing Profit Margin or Risk Category
    df_valid = edf.dropna(subset=['Profit Margin', 'Risk Category'])

    # Split the data into Low Risk and High Risk groups
    low_risk = df_valid[df_valid['Risk Category'] == 'Low Risk']['Profit Margin']
    high_risk = df_valid[df_valid['Risk Category'] == 'High Risk']['Profit Margin']

    # Welch’s t-test (does not assume equal variances)
    t_stat, p_ttest = ttest_ind(low_risk, high_risk, equal_var=False)

    # Mann-Whitney U Test (non-parametric)
    u_stat, p_mannwhitney = mannwhitneyu(low_risk, high_risk, alternative='two-sided')

    # Calculate mean Profit Margin for both groups
    mean_low = low_risk.mean()
    mean_high = high_risk.mean()

    # Markdown table with results
    markdown_result = f"""
    ### Hypothesis Test 2a: Difference in Profit Margin between Risk Groups

    | Test                   | Statistic         | p-value           | Conclusion                   |
    |------------------------|-------------------|-------------------|------------------------------|
    | **t-test (Welch)**     | t = `{t_stat:.2f}` | `{p_ttest:.2e}`   | {"✅ Significant" if p_ttest < 0.05 else "— Not significant"} |
    | **Mann-Whitney U test**| U = `{u_stat:.2f}` | `{p_mannwhitney:.2e}` | {"✅ Significant" if p_mannwhitney < 0.05 else "— Not significant"} |

    **Mean Profit Margin**:
    - Low Risk: `{mean_low:.4f}`
    - High Risk: `{mean_high:.4f}`
    """

    print(markdown_result)

def hypothesis_test_t_test_or_mann_whitney_u_of_profit_margin_between_risk_groups_out(edf):
    pm_series = edf['Profit Margin'].dropna()

    # Remove outliers based on the 1st and 99th percentiles
    lower_bound = pm_series.quantile(0.01)
    upper_bound = pm_series.quantile(0.99)

    # Filter the dataframe
    filtered_df = edf[
        (edf['Profit Margin'] >= lower_bound) &
        (edf['Profit Margin'] <= upper_bound)
    ].dropna(subset=['Risk Category'])

    # Step 2: Split by risk category
    low_risk = filtered_df[filtered_df['Risk Category'] == 'Low Risk']['Profit Margin']
    high_risk = filtered_df[filtered_df['Risk Category'] == 'High Risk']['Profit Margin']

    # Step 3: Perform statistical tests
    t_stat, p_ttest = ttest_ind(low_risk, high_risk, equal_var=False)
    u_stat, p_mannwhitney = mannwhitneyu(low_risk, high_risk, alternative='two-sided')

    # Step 4: Calculate group means
    mean_low = low_risk.mean()
    mean_high = high_risk.mean()

    # Step 5: Present results
    markdown_result = f"""
    ### Hypothesis Test 2b (after removing 1% outliers): Profit Margin vs Risk

    | Test                   | Statistic         | p-value           | Conclusion                   |
    |------------------------|-------------------|-------------------|------------------------------|
    | **t-test (Welch)**     | t = `{t_stat:.2f}` | `{p_ttest:.2e}`   | {"✅ Significant" if p_ttest < 0.05 else "— Not significant"} |
    | **Mann-Whitney U test**| U = `{u_stat:.2f}` | `{p_mannwhitney:.2e}` | {"✅ Significant" if p_mannwhitney < 0.05 else "— Not significant"} |

    **Mean Profit Margin (after trimming outliers)**:
    - Low Risk: `{mean_low:.4f}`
    - High Risk: `{mean_high:.4f}`
    """

    print(markdown_result)

def hypothesis_test_log_regresion_t_test_of_debt_to_equity_higher_risk(edf):
    logit_df = edf.dropna(subset=['Debt to Equity', 'Risk Flag'])

    # Define the independent variable and the target variable
    X = logit_df[['Debt to Equity']]
    X = sm.add_constant(X)  # Adds the intercept term
    y = logit_df['Risk Flag']

    # Fit logistic regression model
    model = sm.Logit(y, X).fit(disp=False)

    # Extract model parameters
    coef = model.params['Debt to Equity']
    p_value = model.pvalues['Debt to Equity']
    odds_ratio = round(np.exp(coef), 4)

    # Calculate 95% confidence interval for odds ratio
    conf = model.conf_int()
    lower_bound = np.exp(conf.loc['Debt to Equity'][0])
    upper_bound = np.exp(conf.loc['Debt to Equity'][1])

    # Display results in Markdown format
    markdown_output = f"""
    ### Hypothesis Test 3a: Debt to Equity vs Risk Flag

    | Metric                          | Value                          |
    |---------------------------------|--------------------------------|
    | **Coefficient (β)**             | `{coef:.4f}`                  |
    | **Odds Ratio**                  | `{odds_ratio:.4f}`            |
    | **95% Confidence Interval**     | [`{lower_bound:.4f}`, `{upper_bound:.4f}`] |
    | **p-value**                     | `{p_value:.4f}` {"✅ Significant" if p_value < 0.05 else "— Not significant"} |

    **Interpretation**:
    - {"The effect of Debt to Equity on the probability of being high risk is statistically significant." if p_value < 0.05 else "There is no statistically significant effect of Debt to Equity on the probability of being high risk."}
    - An increase of one unit in the Debt to Equity ratio multiplies the odds of being flagged as high risk by approximately `{odds_ratio:.4f}`.
    """

    print(markdown_output)


def hypothesis_test_log_regresion_t_test_of_debt_to_equity_higher_risk_out(edf):
    logit_df = edf.dropna(subset=['Debt to Equity', 'Risk Flag'])

    # Define the independent variable and the target variable
    X = logit_df[['Debt to Equity']]
    X = sm.add_constant(X)  # Adds the intercept term
    y = logit_df['Risk Flag']

    # Remove outliers based on the 1st and 99th percentiles
    lower_bound = logit_df['Debt to Equity'].quantile(0.01)
    upper_bound = logit_df['Debt to Equity'].quantile(0.99)

    # Filter the DataFrame to exclude outliers
    filtered_logit_df = logit_df[(logit_df['Debt to Equity'] >= lower_bound) & (logit_df['Debt to Equity'] <= upper_bound)]

    # Define the independent variable and the target variable
    X_filtered = filtered_logit_df[['Debt to Equity']]
    X_filtered = sm.add_constant(X_filtered)  # Adds the intercept term
    y_filtered = filtered_logit_df['Risk Flag']

    # Fit logistic regression model
    filtered_model = sm.Logit(y_filtered, X_filtered).fit(disp=False)

    # Extract model parameters
    coef_filtered = filtered_model.params['Debt to Equity']
    p_value_filtered = filtered_model.pvalues['Debt to Equity']
    odds_ratio_filtered = round(np.exp(coef_filtered), 4)

    # Calculate 95% confidence interval for odds ratio
    conf_filtered = filtered_model.conf_int()
    lower_bound_filtered = np.exp(conf_filtered.loc['Debt to Equity'][0])
    upper_bound_filtered = np.exp(conf_filtered.loc['Debt to Equity'][1])

    # Display results in Markdown format
    markdown_output_filtered = f"""
    ### Hypothesis Test 3b (After Removing 1% Outliers): Debt to Equity vs Risk Flag

    | Metric                          | Value                          |
    |---------------------------------|--------------------------------|
    | **Coefficient (β)**             | `{coef_filtered:.4f}`         |
    | **Odds Ratio**                  | `{odds_ratio_filtered:.4f}`   |
    | **95% Confidence Interval**     | [`{lower_bound_filtered:.4f}`, `{upper_bound_filtered:.4f}`] |
    | **p-value**                     | `{p_value_filtered:.4f}` {"✅ Significant" if p_value_filtered < 0.05 else "— Not significant"} |

    **Interpretation**:
    - {"The effect of Debt to Equity on the probability of being high risk is statistically significant." if p_value_filtered < 0.05 else "There is no statistically significant effect of Debt to Equity on the probability of being high risk."}
    - An increase of one unit in the Debt to Equity ratio multiplies the odds of being flagged as high risk by approximately `{odds_ratio_filtered:.4f}`.
    """

    print(markdown_output_filtered)

def hypothesis_test_t_test_mann_whitney_u_of_roi_higher_risk(edf):
    # Filter to valid ROI and Risk Category values (no NaNs)
    roi_df = edf.dropna(subset=['ROI', 'Risk Category'])

    # Split into Low Risk and High Risk groups
    low_risk = roi_df[roi_df['Risk Category'] == 'Low Risk']['ROI']
    high_risk = roi_df[roi_df['Risk Category'] == 'High Risk']['ROI']

    # Perform hypothesis tests
    t_stat, p_ttest = ttest_ind(low_risk, high_risk, equal_var=False)
    u_stat, p_mannwhitney = mannwhitneyu(low_risk, high_risk, alternative='two-sided')

    # Prepare Markdown output
    markdown_output = f"""
    ### Hypothesis Test 4a: ROI vs Risk Category (No Outlier Removal)

    | Test                        | Statistic         | p-value           | Conclusion                   |
    |-----------------------------|-------------------|-------------------|------------------------------|
    | **T-test (Welch's)**        | t = `{t_stat:.2f}` | `{p_ttest:.4e}`   | {"✅ Significant" if p_ttest < 0.05 else "— Not significant"} |
    | **Mann–Whitney U test**     | U = `{u_stat:.2f}` | `{p_mannwhitney:.4e}` | {"✅ Significant" if p_mannwhitney < 0.05 else "— Not significant"} |

    **Group Statistics**:
    - **Low Risk** — Mean ROI: `{low_risk.mean():.4f}`, Median: `{low_risk.median():.4f}`
    - **High Risk** — Mean ROI: `{high_risk.mean():.4f}`, Median: `{high_risk.median():.4f}`

    **Interpretation**:
    - {"High-risk firms tend to have significantly lower ROI, based on both tests." if p_ttest < 0.05 and p_mannwhitney < 0.05 else "There is no strong statistical evidence that ROI differs significantly between risk categories."}
    """

    # Display result
    print(markdown_output)

def hypothesis_test_t_test_mann_whitney_u_of_roi_higher_risk_out(edf):
    roi_df = edf.dropna(subset=['ROI', 'Risk Category'])

    # Optional: Trim 5% outliers (to reduce skew, improve test robustness)
    lower = roi_df['ROI'].quantile(0.01)
    upper = roi_df['ROI'].quantile(0.99)
    roi_df = roi_df[(roi_df['ROI'] >= lower) & (roi_df['ROI'] <= upper)]

    # Split into two groups
    low_risk = roi_df[roi_df['Risk Category'] == 'Low Risk']['ROI']
    high_risk = roi_df[roi_df['Risk Category'] == 'High Risk']['ROI']

    # Perform tests
    t_stat, p_ttest = ttest_ind(low_risk, high_risk, equal_var=False)
    u_stat, p_mannwhitney = mannwhitneyu(low_risk, high_risk, alternative='two-sided')

    # Prepare Markdown output
    markdown_output = f"""
    ### Hypothesis Test 4b: ROI vs Risk Category (After 1% of Outlier Removal)

    | Test                        | Statistic         | p-value           | Conclusion                   |
    |-----------------------------|-------------------|-------------------|------------------------------|
    | **T-test (Welch's)**        | t = `{t_stat:.2f}` | `{p_ttest:.4e}`   | {"✅ Significant" if p_ttest < 0.05 else "— Not significant"} |
    | **Mann–Whitney U test**     | U = `{u_stat:.2f}` | `{p_mannwhitney:.4e}` | {"✅ Significant" if p_mannwhitney < 0.05 else "— Not significant"} |

    **Group Statistics**:
    - **Low Risk** — Mean ROI: `{low_risk.mean():.4f}`, Median: `{low_risk.median():.4f}`
    - **High Risk** — Mean ROI: `{high_risk.mean():.4f}`, Median: `{high_risk.median():.4f}`

    **Interpretation**:
    - {"High-risk firms tend to have significantly lower ROI, based on both tests." if p_ttest < 0.05 and p_mannwhitney < 0.05 else "There is no strong statistical evidence that ROI differs significantly between risk categories."}
    """

    print(markdown_output)

def interpret_r2(value):
    if value >= 0.75:
        return "✅ Strong explanatory power"
    elif value >= 0.50:
        return "✅ Moderate explanatory power"
    elif value >= 0.25:
        return "⚠️ Weak explanatory power"
    else:
        return "⚠️ Very weak explanatory power"
    
def interpret_mse(value):
    return "Lower MSE indicates better model fit; scale depends on target variable (Net Income)."

def interpret_rmse(value):
    return f"Average prediction error is approximately ±{value:,.0f} currency units (e.g., dollars)."

def hypothesis_test_linear_regression_revenue_and_net_income(edf):
    # Data preparation with 1% outlier trimming
    df_lr = edf.dropna(subset=['Revenue', 'Net Income'])

    # Define X and y
    X = df_lr[['Revenue']]
    y = df_lr['Net Income']

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
    r_squared = ols_model.rsquared
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)

    markdown_output = f"""
    ### Hypothesis Test 5a: Revenue → Net Income (Linear Regression)

    | Metric                          | Value                          | Interpretation                              |
    |---------------------------------|--------------------------------|----------------------------------------------|
    | **Intercept (α)**               | `{intercept:.4f}`              | —                                            |
    | **Coefficient (β for Revenue)** | `{coef:.4f}`                   | Positive slope implies positive effect       |
    | **95% Confidence Interval (β)** | [`{lower_bound:.4f}`, `{upper_bound:.4f}`] | —                                |
    | **p-value (Revenue)**           | `{p_value:.4e}` {"✅ Significant" if p_value < 0.05 else "— Not significant"} | {"Reject H₀" if p_value < 0.05 else "Fail to reject H₀"} |
    | **R²**                          | `{r2_sklearn:.4f}`             | {interpret_r2(r2_sklearn)}                  |
    | **Mean Squared Error (MSE)**    | `{mse:.2e}`                    | {interpret_mse(mse)}                        |
    | **Root Mean Squared Error (RMSE)** | `{rmse:.2e}`                | {interpret_rmse(rmse)}                      |

    **Interpretation**:
    - {"There is a statistically significant positive relationship between Revenue and Net Income." if p_value < 0.05 else "There is no statistically significant linear effect of Revenue on Net Income."}
    - Each unit increase in Revenue predicts an average Net Income increase of `{coef:.4f}`.
    - The model explains about `{r_squared:.1%}` of the variance in Net Income.
    """

    print(markdown_output)

    # Plot regression line with confidence interval
    # Sort X for plotting
    # Create a new DataFrame for sorting and alignment
    plot_df = df_lr[['Revenue', 'Net Income']].copy()
    plot_df['y_pred'] = y_pred
    plot_df = plot_df.sort_values(by='Revenue')

    X_sorted = plot_df['Revenue'].values
    y_sorted = plot_df['Net Income'].values
    y_pred_sorted = plot_df['y_pred'].values

    # Compute confidence interval
    n = len(X_sorted)
    t_value = stats.t.ppf(0.975, df=n - 2)
    mean_x = np.mean(X_sorted)
    s_err = np.sqrt(np.sum((y_sorted - y_pred_sorted)**2) / (n - 2))
    conf = t_value * s_err * np.sqrt(
        1/n + (X_sorted - mean_x)**2 / np.sum((X_sorted - mean_x)**2)
    )

    lower = y_pred_sorted - conf
    upper = y_pred_sorted + conf

    # Plot
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

def hypothesis_test_linear_regression_revenue_and_net_income_out(edf):
    # Data preparation with 1% outlier trimming
    df_lr = edf.dropna(subset=['Revenue', 'Net Income'])
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
    r_squared = ols_model.rsquared
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)

    markdown_output = f"""
    ### Hypothesis Test 5b: Revenue → Net Income (Linear Regression, 1% Outlier Removal)

    | Metric                          | Value                          | Interpretation                              |
    |---------------------------------|--------------------------------|----------------------------------------------|
    | **Intercept (α)**               | `{intercept:.4f}`              | —                                            |
    | **Coefficient (β for Revenue)** | `{coef:.4f}`                   | Positive slope implies positive effect       |
    | **95% Confidence Interval (β)** | [`{lower_bound:.4f}`, `{upper_bound:.4f}`] | —                                |
    | **p-value (Revenue)**           | `{p_value:.4e}` {"✅ Significant" if p_value < 0.05 else "— Not significant"} | {"Reject H₀" if p_value < 0.05 else "Fail to reject H₀"} |
    | **R²**                          | `{r2_sklearn:.4f}`             | {interpret_r2(r2_sklearn)}                  |
    | **Mean Squared Error (MSE)**    | `{mse:.2e}`                    | {interpret_mse(mse)}                        |
    | **Root Mean Squared Error (RMSE)** | `{rmse:.2e}`               | {interpret_rmse(rmse)}                      |

    **Interpretation**:
    - {"There is a statistically significant positive relationship between Revenue and Net Income." if p_value < 0.05 else "There is no statistically significant linear effect of Revenue on Net Income."}
    - Each unit increase in Revenue predicts an average Net Income increase of `{coef:.4f}`.
    - The model explains about `{r_squared:.1%}` of the variance in Net Income.
    """

    print(markdown_output)

    # Plot regression line with confidence interval
    # Sort X for plotting
    # Create a new DataFrame for sorting and alignment
    plot_df = df_trimmed[['Revenue', 'Net Income']].copy()
    plot_df['y_pred'] = y_pred
    plot_df = plot_df.sort_values(by='Revenue')

    X_sorted = plot_df['Revenue'].values
    y_sorted = plot_df['Net Income'].values
    y_pred_sorted = plot_df['y_pred'].values

    # Compute confidence interval
    n = len(X_sorted)
    t_value = stats.t.ppf(0.975, df=n - 2)
    mean_x = np.mean(X_sorted)
    s_err = np.sqrt(np.sum((y_sorted - y_pred_sorted)**2) / (n - 2))
    conf = t_value * s_err * np.sqrt(
        1/n + (X_sorted - mean_x)**2 / np.sum((X_sorted - mean_x)**2)
    )

    lower = y_pred_sorted - conf
    upper = y_pred_sorted + conf

    # Plot
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
    years_of_interest = [2024] # Currently supports only 1 year at a time

    edf = run_etl(years_of_interest)

    run_eda(edf)

    print("Running Statistics pipeline...")
    hypothesis_test_pearson_corr_of_accrued_liabiilties_and_risk_flag(edf)
    hypothesis_test_pearson_corr_of_accrued_liabiilties_and_risk_flag_out(edf)

    hypothesis_test_t_test_or_mann_whitney_u_of_profit_margin_between_risk_groups(edf)
    hypothesis_test_t_test_or_mann_whitney_u_of_profit_margin_between_risk_groups_out(edf)

    hypothesis_test_log_regresion_t_test_of_debt_to_equity_higher_risk(edf)
    hypothesis_test_log_regresion_t_test_of_debt_to_equity_higher_risk_out(edf)

    hypothesis_test_t_test_mann_whitney_u_of_roi_higher_risk(edf)
    hypothesis_test_t_test_mann_whitney_u_of_roi_higher_risk_out(edf)

    hypothesis_test_linear_regression_revenue_and_net_income(edf)
    hypothesis_test_linear_regression_revenue_and_net_income_out(edf)