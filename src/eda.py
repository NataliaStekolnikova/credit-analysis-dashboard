import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def barchart_visualization(edf):
    print("\nEDA Bar Chart Visualizations:")

    # Plot: Top 10 companies by ROE (excluding extreme outliers)
    filtered_roe = edf.dropna(subset=['ROE'])
    filtered_roe = filtered_roe[filtered_roe['ROE'] < filtered_roe['ROE'].quantile(0.99)]
    top_10_roe = filtered_roe.sort_values(by='ROE', ascending=False).head(10)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=top_10_roe, x='name', y='ROE', palette='viridis')
    plt.title('Top 10 Companies by Return on Equity (ROE)', fontsize=14)
    plt.ylabel('ROE')
    plt.xlabel('Company')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def boxplot_visualization(edf):
    print("\nEDA Bar Plot Visualizations:")

    features = ['Current Ratio', 'Debt to Equity', 'ROE', 'ROA', 'ROI', 'Profit Margin']

    edf_trimmed = edf.copy()
    for feature in features:
        lower_bound = edf_trimmed[feature].quantile(0.10)
        upper_bound = edf_trimmed[feature].quantile(0.90)
        edf_trimmed = edf_trimmed[(edf_trimmed[feature] >= lower_bound) & (edf_trimmed[feature] <= upper_bound)]

    # Normalize features for better visual comparison
    scaler = StandardScaler()
    edf_trimmed[features] = scaler.fit_transform(edf_trimmed[features])

    # Prepare data for visualization
    edf_melted = edf_trimmed.melt(id_vars=['Risk Category'], value_vars=features, var_name='Feature', value_name='Value')

    # Plot grouped boxplot
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=edf_melted, x='Feature', y='Value', hue='Risk Category', palette='pastel')
    plt.title('Boxplot of Features by Risk Category (Outliers Removed)', fontsize=16)
    plt.ylabel('Normalized Value')
    plt.xlabel('Feature')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Risk Category')
    plt.tight_layout()
    plt.show()

# Correlation Analysis
def eda_correlation_analysis(edf):
    print("\nEDA Correlation Analysis:")

    features = [
        'Net Income', 'Total Assets', 'Shareholder Equity', 'Cash From Ops', 'Revenue',
        'Accounts Receivable (Current)', 'Accounts Payable (Current)', 'Accrued Liabilities (Current)',
        'Additional Paid-in Capital', 'ROI', 'ROE', 'ROA',
        'Amortization of Intangibles', 'Share-based Compensation Expense'
    ]

    trimmed_edf = edf.copy()

    # Histogram and Boxplot with Outlier Removal
    for metric in features:
        print(f"\nHistogram and Boxplot for {metric}:")
        
        # Remove 5% of outliers
        lower_bound = edf[metric].quantile(0.05)
        upper_bound = edf[metric].quantile(0.95)
        trimmed_edf = edf[(edf[metric] >= lower_bound) & (edf[metric] <= upper_bound)][metric].dropna()

        # Plot histogram with KDE
        plt.figure(figsize=(10, 4))
        sns.histplot(trimmed_edf, kde=True, bins=30)
        plt.title(f'Histogram + KDE of {metric} (5% Outliers Removed)')
        plt.show()

        # Plot boxplot
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=trimmed_edf)
        plt.title(f'Boxplot of {metric} (5% Outliers Removed)')
        plt.show()

def pca_analysis(edf):
    print("\nEDA Principal Component Analysis (PCA):")

    features = [
        'Net Income', 'Total Assets', 'Shareholder Equity', 'Cash From Ops', 'Revenue',
        'Accounts Receivable (Current)', 'Accounts Payable (Current)', 'Accrued Liabilities (Current)',
        'Additional Paid-in Capital', 'ROI', 'ROE', 'ROA',
        'Amortization of Intangibles', 'Share-based Compensation Expense'
    ]
    target = 'Risk Flag'

    # Drop rows with missing values in the selected features and target
    df_valid = edf[features + [target]].dropna()

    # Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_valid[features])

    # Apply PCA
    pca = PCA()
    pca_data = pca.fit_transform(scaled_features)

    # Explained variance table
    explained_variance = pd.DataFrame({
        'Principal Component': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
        'Explained Variance Ratio': pca.explained_variance_ratio_,
        'Cumulative Explained Variance': pca.explained_variance_ratio_.cumsum()
    })

    print("\nPCA Explained Variance Table:")
    print(explained_variance)

    # Number of components needed to explain 95% variance
    components_needed = (explained_variance['Cumulative Explained Variance'] >= 0.95).idxmax() + 1
    print(f"\nNumber of components needed to explain at least 95% of the variance: {components_needed}")

    # Focused Explained Variance Plot
    explained = pca.explained_variance_ratio_ * 100  # Convert to %
    cumulative = explained.cumsum()

    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained)+1), explained, alpha=0.7, label='Individual Explained Variance')
    plt.plot(range(1, len(cumulative)+1), cumulative, color='red', marker='o', linewidth=2, label='Cumulative Explained Variance')
    plt.axhline(y=95, color='green', linestyle='--', linewidth=1, label='95% Variance Threshold')
    plt.axvline(x=components_needed, color='purple', linestyle='--', linewidth=1, label=f'{components_needed} Components')

    plt.title('Explained Variance by Principal Components', fontsize=14)
    plt.xlabel('Principal Component Number')
    plt.ylabel('Explained Variance (%)')
    plt.xticks(range(1, len(explained)+1))
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # Focused PC1 vs PC2 Scatter Plot
    pca_df = pd.DataFrame(pca_data[:, :2], columns=['PC1', 'PC2'])
    pca_df[target] = df_valid[target].values

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=pca_df,
        x='PC1', y='PC2',
        hue=target,
        palette='Set1',
        alpha=0.8,
        s=60,
        edgecolor='black'
    )
    plt.title('PCA Projection: PC1 vs PC2 by Risk Category', fontsize=14)
    plt.xlabel(f'PC1 ({explained[0]:.1f}% Variance)')
    plt.ylabel(f'PC2 ({explained[1]:.1f}% Variance)')
    plt.legend(title='Risk Flag', loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Optional: Print PCA Loadings
    print("\nPCA Loadings (top 3 components):")
    loadings = pd.DataFrame(
        pca.components_.T[:, :3],
        columns=['PC1', 'PC2', 'PC3'],
        index=features
    ).sort_values(by='PC1', ascending=False)
    print(loadings)

    pca_df = pd.DataFrame(pca_data[:, :3], columns=['PC1', 'PC2', 'PC3'])
    pca_df['Risk Flag'] = edf.loc[df_valid.index, 'Risk Flag'].values  # if using flag

    sns.pairplot(pca_df, hue='Risk Flag', diag_kind='kde')
    plt.suptitle('PCA Component Clusters by Risk Flag', y=1.02)
    plt.show()

def pca_analysis2(edf):
    print("\nEDA Principal Component Analysis (PCA):")
    features = [
        'Net Income', 'Total Assets', 'Shareholder Equity', 'Cash From Ops', 'Revenue',
        'Accounts Receivable (Current)', 'Accounts Payable (Current)', 'Accrued Liabilities (Current)',
        'Additional Paid-in Capital', 'ROI', 'ROE', 'ROA',
        'Amortization of Intangibles', 'Share-based Compensation Expense'
    ]
    target = 'Risk Flag'

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
    print("\nPCA Explained Variance Table:")
    print(explained_variance)

    # Highlight the number of components needed to explain at least 95% of the variance
    components_needed = (explained_variance['Cumulative Explained Variance'] >= 0.95).idxmax() + 1
    print(f"\nNumber of components needed to explain at least 95% of the variance: {components_needed}")

    # Visualize the explained variance ratio
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_, marker='o', label='Explained Variance')
    plt.axhline(y=1/len(features), color='r', linestyle='--', label='Ideal Line (1/number of features)')
    plt.title('Percentage of Variance Explained by Each Principal Component')
    plt.xlabel('Principal Component Number')
    plt.ylabel('Explained Variance Ratio (%)')
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

    print("\nPCA Loadings:")
    loadings = pd.DataFrame(
        pca.components_.T[:, :3],
        columns=['PC1', 'PC2', 'PC3'],
        index=features
    ).sort_values(by='PC1', ascending=False)
    print(loadings)

def main(edf):
    print("Running EDA pipeline...")
    barchart_visualization(edf)
    boxplot_visualization(edf)
    eda_correlation_analysis(edf)
    pca_analysis(edf)

if __name__ == "__main__":
    main()