import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- Visualizations: Histograms and Boxplots ---
def plot_distributions(edf):
    features = [
        'Accounts Payable (Current)', 'Accounts Receivable (Current)', 'Accrued Liabilities (Current)',
        'Additional Paid-in Capital', 'Amortization of Intangibles', 'Cash From Ops', 'Net Income', 'Revenue',
        'Share-based Compensation Expense', 'Shareholder Equity', 'Total Assets', 'ROE', 'ROA', 'ROI',
        'Current Ratio', 'Debt to Equity', 'Profit Margin', 'Asset Turnover', 'Financial Risk Flag'
    ]
    for metric in features:
        # Remove 1% of outliers
        lower_bound = edf[metric].quantile(0.01)
        upper_bound = edf[metric].quantile(0.99)
        trimmed_data = edf[(edf[metric] >= lower_bound) & (edf[metric] <= upper_bound)][metric].dropna()

        # Plot histogram with KDE
        plt.figure(figsize=(10, 4))
        sns.histplot(trimmed_data, kde=True, bins=30)
        plt.title(f'Histogram + KDE of {metric} (1% Outliers Removed)')
        plt.show()

        # Plot boxplot
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=trimmed_data)
        plt.title(f'Boxplot of {metric} (1% Outliers Removed)')
        plt.show()

# --- Correlation Analysis ---
def correlation_analysis(edf):
    # Select the specified columns
    selected_columns = [
        'Accounts Payable (Current)', 'Accounts Receivable (Current)', 'Accrued Liabilities (Current)',
        'Additional Paid-in Capital', 'Amortization of Intangibles', 'Cash From Ops', 'Net Income', 'Revenue',
        'Share-based Compensation Expense', 'Shareholder Equity', 'Total Assets', 'ROE', 'ROA', 'ROI',
        'Current Ratio', 'Debt to Equity', 'Profit Margin', 'Asset Turnover', 'Financial Risk Flag'
    ]

    # Filter the DataFrame to include only the selected columns
    filtered_edf = edf[selected_columns].dropna()

    # Calculate correlation matrix
    correlation_matrix = filtered_edf.corr()

    # Apply a mask to highlight values greater than 0.8
    mask = correlation_matrix < 0.8

    # Plot correlation heatmap with grid
    plt.figure(figsize=(14, 10))
    sns.heatmap(
        correlation_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", 
        square=True, cbar_kws={"shrink": 0.8}, linewidths=0.5
    )
    plt.title("Correlation Matrix (Values > 0.7 Highlighted)", fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# --- Principal Component Analysis (PCA) ---
def pca_analysis(edf):
    features = [
        'Net Income', 'Total Assets', 'Shareholder Equity', 'Cash From Ops', 'Revenue',
        'Accounts Receivable (Current)', 'Accounts Payable (Current)', 'Accrued Liabilities (Current)',
        'Additional Paid-in Capital', 'ROI', 'ROE', 'ROA',
        'Amortization of Intangibles', 'Share-based Compensation Expense'
    ]
    target = 'Financial Risk Flag'

    # Drop rows with missing values in the selected features and target
    df_valid = edf[features + [target]].dropna()

    # Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_valid[features])

    # Apply PCA
    pca = PCA()
    pca_data = pca.fit_transform(scaled_features)

    # Create a DataFrame for the PCA results
    explained_variance = pd.DataFrame({
        'Principal Component': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
        'Explained Variance Ratio': pca.explained_variance_ratio_,
        'Cumulative Explained Variance': pca.explained_variance_ratio_.cumsum()
    })
    print("\n--- PCA Explained Variance Table ---")
    print(explained_variance)

    # Highlight the number of components needed to explain at least 95% of the variance
    components_needed = (explained_variance['Cumulative Explained Variance'] >= 0.95).idxmax() + 1
    print(f"\nNumber of components needed to explain at least 95% of the variance: {components_needed}")

    # Visualize the explained variance ratio
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_, marker='o', label='Explained Variance')
    plt.axhline(y=1/len(features), color='r', linestyle='--', label='Ideal Line (1/number of features)')
    plt.title('Explained Variance by Principal Component')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Visualize the first two principal components with respect to the Risk Flag
    pca_df = pd.DataFrame(pca_data[:, :2], columns=['PC1', 'PC2'])
    pca_df[target] = df_valid[target].values

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue=target, palette='coolwarm', alpha=0.7)
    plt.title('PCA: First Two Principal Components')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Risk Flag')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Display PCA loadings
    loadings = pd.DataFrame(
        pca.components_.T[:, :3],
        columns=['PC1', 'PC2', 'PC3'],
        index=features
    ).sort_values(by='PC1', ascending=False)
    print("\n--- PCA Loadings ---")
    print(loadings)

# --- Boxplot Visualization ---
def boxplot_visualization(edf):
    features = ['Current Ratio', 'Debt to Equity', 'ROE', 'ROA', 'ROI', 'Profit Margin']
    df_trimmed = edf.copy()
    for feature in features:
        lower_bound = df_trimmed[feature].quantile(0.05)
        upper_bound = df_trimmed[feature].quantile(0.95)
        df_trimmed = df_trimmed[(df_trimmed[feature] >= lower_bound) & (df_trimmed[feature] <= upper_bound)]

    df_melted = df_trimmed.melt(id_vars=['Risk Category'], value_vars=features, var_name='Feature', value_name='Value')

    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df_melted, x='Feature', y='Value', hue='Risk Category', palette='pastel')
    plt.title('Boxplot of Features by Risk Category (Outliers Removed)', fontsize=16)
    plt.ylabel('Value')
    plt.xlabel('Feature')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Risk Category')
    plt.tight_layout()
    plt.show()


def main(edf):
    print("Running EDA pipeline...")

    correlation_analysis(edf)
    plot_distributions(edf)
    pca_analysis(edf)
    boxplot_visualization(edf)

if __name__ == "__main__":
    main()