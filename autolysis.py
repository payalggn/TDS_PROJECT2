# ///
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "seaborn",
#   "matplotlib",
#   "scipy",
#   "tabulate",
#   "requests",
# ]
# ///

import os
import sys
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN", "")  # Get token from environment variable

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

def prepare_prompt(df):
    """Prepare dynamic prompt based on the dataset."""
    df_info = {
        "shape": df.shape,
        "columns": list(df.columns),
        "numeric_columns": df.select_dtypes(include=["number"]).columns.tolist(),
        "categorical_columns": df.select_dtypes(exclude=["number"]).columns.tolist(),
    }

    prompt = f"""
    You are a data analysis assistant. Here is the dataset:
    
    - Shape: {df_info['shape']}
    - Columns: {', '.join(df_info['columns'])}
    - Numeric Columns: {', '.join(df_info['numeric_columns'])}
    - Categorical Columns: {', '.join(df_info['categorical_columns'])}

    Please analyze the dataset, identify trends, and offer insights on:
    1. General observations about the data.
    2. Statistical relationships between variables.
    3. Missing or unusual values.
    4. Any potential issues or suggestions for further analysis.
    """
    
    return prompt

def send_api_request(prompt):
    """Send a request to the AI Proxy API."""
    if not AIPROXY_TOKEN:
        logging.error("API token is not set in environment variables.")
        return "Error: API token is missing."

    api_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json",
    }

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful data analysis assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 1000,
        "temperature": 0.7,
    }

    try:
        response = requests.post(api_url, json=data, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            logging.error(f"API request failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        logging.error(f"An error occurred while making the API request: {e}")
        return None

def process_api_response(response_data):
    """Process the response from the API and return the analysis."""
    if "choices" in response_data:
        analysis = response_data["choices"][0]["message"]["content"].strip()
        return analysis
    else:
        logging.error(f"Unexpected response format: {response_data}")
        return "Error: No analysis available in the response."

def analyze_data(df):
    """Analyze data using the AI Proxy and apply statistical methods."""
    if not AIPROXY_TOKEN:
        logging.error("API token is not set in environment variables.")
        return "Error: API token is missing."

    # Basic Descriptive Statistics
    description = df.describe(include='all')

    # Correlation Analysis (for numeric columns)
    numeric_columns = df.select_dtypes(include=["number"]).columns
    correlation_matrix = df[numeric_columns].corr()

    # Chi-Square Test for Categorical Variables
    categorical_columns = df.select_dtypes(include=["object"]).columns
    chi_square_results = {}
    for col1 in categorical_columns:
        for col2 in categorical_columns:
            if col1 != col2:
                # Creating a contingency table for chi-square test
                contingency_table = pd.crosstab(df[col1], df[col2])
                chi2, p, _, _ = chi2_contingency(contingency_table)
                chi_square_results[(col1, col2)] = (chi2, p)

    # Summarize the results
    analysis_summary = f"""
    ### Descriptive Statistics:
    {description.to_string()}

    ### Correlation Analysis:
    {correlation_matrix.to_string()}

    ### Chi-Square Test Results:
    """
    for (col1, col2), (chi2, p) in chi_square_results.items():
        analysis_summary += f"\n{col1} vs {col2}: Chi2 = {chi2:.2f}, p-value = {p:.4f}"

    # Send API request for advanced analysis
    df_string = df.head(10).to_string()  # Use top 10 rows for AI model analysis
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

    try:
        response = send_api_request(data)
        if response:
            analysis_summary += f"\n\nAPI Analysis:\n{response}"
        else:
            logging.warning("API response missing or failed.")
    except Exception as e:
        logging.error(f"Error with API request: {str(e)}")

    return analysis_summary

def visualize_data(df):
    """Generate visualizations for the dataset and save to the current working directory."""
    charts = []

    # Identify numeric columns
    numeric_columns = df.select_dtypes(include=["number"]).columns

    # Correlation Heatmap with low detail
    if len(numeric_columns) > 1:  # Correlation requires at least two numeric columns
        plt.figure(figsize=(9.6, 5.4))  # Set the figsize to 960x540 pixels (9.6 inches at 100 dpi)
        heatmap = sns.heatmap(
            df[numeric_columns].corr(),
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            cbar_kws={'shrink': 0.8},
            annot_kws={"size": 8},  # Reduce annotation size for low detail
        )
        heatmap.set_title("Correlation Heatmap", fontsize=12, pad=20)  # Reduce font size for low detail
        plt.tight_layout(pad=3.0)
        heatmap_file = "heatmap.png"  # Save in the current directory
        plt.savefig(heatmap_file, dpi=100)  # Save as 960x540 pixels at 100 dpi
        charts.append(heatmap_file)
        plt.close()

    # Line Plot of Numeric Columns with low detail
    if len(numeric_columns) >= 2:
        plt.figure(figsize=(9.6, 5.4))  # Set the figsize to 960x540 pixels (9.6 inches at 100 dpi)
        for col in numeric_columns:
            df[col].dropna().reset_index(drop=True).plot(label=col, linewidth=1)  # Reduce line width for low detail
        plt.title("Line Plot of Numeric Columns", fontsize=12, pad=20)  # Reduce font size for low detail
        plt.xlabel("Index", fontsize=10)  # Reduce font size for low detail
        plt.ylabel("Values", fontsize=10)  # Reduce font size for low detail
        plt.legend(loc="best", fontsize=8)  # Reduce legend font size for low detail
        plt.tight_layout(pad=3.0)
        lineplot_file = "lineplot.png"  # Save in the current directory
        plt.savefig(lineplot_file, dpi=100)  # Save as 960x540 pixels at 100 dpi
        charts.append(lineplot_file)
        plt.close()

    # Histogram of the Second Column with low detail
    if len(df.columns) > 1:  # Check if the dataset has at least two columns
        second_column = df.columns[1]
        if df[second_column].dtype in ["int64", "float64"]:  # Check if second column is numeric
            plt.figure(figsize=(9.6, 5.4))  # Set the figsize to 960x540 pixels (9.6 inches at 100 dpi)
            df[second_column].dropna().plot(kind="hist", bins=20, color="skyblue", edgecolor="black")  # Reduce bins for low detail
            plt.title(f"Histogram of {second_column}", fontsize=12, pad=20)  # Reduce font size for low detail
            plt.xlabel(second_column, fontsize=10)  # Reduce font size for low detail
            plt.ylabel("Frequency", fontsize=10)  # Reduce font size for low detail
            plt.tight_layout(pad=3.0)
            histogram_file = "histogram.png"  # Save in the current directory
            plt.savefig(histogram_file, dpi=100)  # Save as 960x540 pixels at 100 dpi
            charts.append(histogram_file)
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

    readme_file = "README.md"  # Save in the current directory
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
