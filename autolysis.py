import os
import sys
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore, chi2_contingency
import numpy as np
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Get API token from environment variable
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN", "")  

def read_csv(filename):
    """Load CSV file with fallback encoding."""
    try:
        df = pd.read_csv(filename, encoding="utf-8")
        logging.info(f"Dataset loaded: {filename}")
        return df
    except UnicodeDecodeError:
        logging.warning(f"Encoding issue detected with {filename}. Trying 'latin1'.")
        return pd.read_csv(filename, encoding="latin1")
    except Exception as e:
        logging.error(f"Error loading {filename}: {e}")
        sys.exit(1)

def analyze_data(df):
    """Perform both traditional EDA and use AI Proxy for additional insights."""
    # Step 1: Traditional EDA

    # Data Overview
    data_overview = {
        "shape": df.shape,
        "data_types": df.dtypes.to_dict(),
        "summary_statistics": df.describe(include='all').to_dict(),
    }

    # Missing Data Analysis
    missing_data = df.isnull().sum()
    missing_percentage = (missing_data / len(df)) * 100
    missing_overview = missing_data[missing_data > 0]

    # Univariate Analysis (Visualize distributions of numerical features)
    numeric_columns = df.select_dtypes(include=["number"]).columns
    categorical_columns = df.select_dtypes(include=["object"]).columns

    # Visualize numeric distributions
    for col in numeric_columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(f"{col}_distribution.png")
        plt.close()

    # Visualize categorical distributions
    for col in categorical_columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x=col)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(f"{col}_countplot.png")
        plt.close()

    # Bivariate Analysis (Correlations for numeric and chi-square for categorical)
    correlation_matrix = df[numeric_columns].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png")
    plt.close()

    # Chi-Square Test for Categorical Variables
    chi_square_results = {}
    for col1 in categorical_columns:
        for col2 in categorical_columns:
            if col1 != col2:
                contingency_table = pd.crosstab(df[col1], df[col2])
                chi2, p, _, _ = chi2_contingency(contingency_table)
                chi_square_results[(col1, col2)] = (chi2, p)

    # Outlier Detection (Z-Score method)
    outliers = {}
    for col in numeric_columns:
        z_scores = zscore(df[col].dropna())
        outlier_indices = np.where(np.abs(z_scores) > 3)[0]  # Z-score > 3 indicates outliers
        if len(outlier_indices) > 0:
            outliers[col] = outlier_indices.tolist()

    # Step 2: AI Proxy Analysis (summarizing findings)
    ai_analysis = ""
    if AIPROXY_TOKEN:
        ai_analysis = analyze_using_api(df)

    # Combine results
    analysis_summary = f"""
    ### Data Overview:
    - Shape: {data_overview['shape']}
    - Data Types: {data_overview['data_types']}
    
    ### Summary Statistics:
    {data_overview['summary_statistics']}

    ### Missing Data:
    {missing_overview}

    ### Bivariate Analysis:
    - Correlation Matrix saved as 'correlation_heatmap.png'
    """
    for (col1, col2), (chi2, p) in chi_square_results.items():
        analysis_summary += f"\n{col1} vs {col2}: Chi2 = {chi2:.2f}, p-value = {p:.4f}"

    # Outlier Detection:
    if outliers:
        analysis_summary += "\nOutliers detected in the following columns (Z-Score > 3):"
        for col, indices in outliers.items():
            analysis_summary += f"\n{col}: {indices}"

    # Add AI analysis at the end
    if ai_analysis:
        analysis_summary += f"\n\n### AI Insights:\n{ai_analysis}"

    return analysis_summary

def analyze_using_api(df):
    """Send curated data to AI Proxy for additional insights."""
    if not AIPROXY_TOKEN:
        logging.error("API token is not set in environment variables.")
        return "Error: API token is missing."

    # Data curation: Select only relevant columns and a sample of rows
    df_cleaned = df.select_dtypes(include=["number", "object"]).dropna(how="all")
    
    # Sample first 5 rows of the cleaned dataframe
    df_sample = df_cleaned.head(5)

    # Only send columns with significant data
    df_string = df_sample.to_string()

    # Create the prompt dynamically based on the cleaned data
    prompt = f"""
    You are a data analysis assistant. Given the following dataset:

    {df_string}

    Please provide:
    1. A summary of the dataset, including key trends and observations.
    2. Identify any missing or unusual values.
    3. Analyze relationships between any categorical variables.
    4. Provide any recommendations or insights based on the data.
    """

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful data analysis assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 1000,
        "temperature": 0.7,
    }

    # Send the request to AI Proxy
    response = send_api_request(data)

    return response

def send_api_request(data):
    """Helper function to send the API request and handle responses."""
    api_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(api_url, json=data, headers=headers)

        if response.status_code == 200:
            response_data = response.json()
            if "choices" in response_data:
                return response_data["choices"][0]["message"]["content"].strip()
            else:
                logging.error(f"Unexpected response format: {response_data}")
                return "Error: No analysis available in the response."
        else:
            logging.error(f"API request failed: {response.status_code} - {response.text}")
            return f"Error: API request failed with status {response.status_code}"

    except Exception as e:
        logging.error(f"An error occurred while making the API request: {e}")
        return f"Error: {str(e)}"

def visualize_data(df):
    """Generate visualizations for the dataset."""
    charts = []
    numeric_columns = df.select_dtypes(include=["number"]).columns

    for col in numeric_columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(f"{col}_distribution.png")
        charts.append(f"{col}_distribution.png")
        plt.close()

    return charts

def save_markdown(df, analysis, charts):
    """Generate a README.md file summarizing the analysis."""
    df_info = {
        "shape": df.shape,
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
    }

    narration = f"""
    This dataset contains {df_info['shape'][0]} rows and {df_info['shape'][1]} columns.
    It has attributes such as {', '.join(df_info['columns'][:5])} (and more).
    Missing values are present in {', '.join([col for col, val in df_info['missing_values'].items() if val > 0])}.
    """

    readme_content = f"""# Analysis Report

## Narration

{narration}

## Dataset Analysis

- **Shape**: {df_info['shape']}
- **Columns**: {', '.join(df_info['columns'])}
- **Missing Values**: {df_info['missing_values']}

## Summary Statistics

{analysis}

## Visualizations

"""
    for chart in charts:
        readme_content += f"![Chart](./{chart})\n"

    readme_file = "README.md"
    with open(readme_file, "w") as f:
        f.write(readme_content)

    logging.info("README.md generated successfully.")

def main():
    if len(sys.argv) != 2:
        logging.error("Usage: python autolysis.py <path_to_csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    df = read_csv(file_path)
    analysis = analyze_data(df)
    charts = visualize_data(df)
    save_markdown(df, analysis, charts)
    logging.info("Analysis completed successfully.")

if __name__ == "__main__":
    main()
