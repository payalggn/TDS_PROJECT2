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

def analyze_data(df):
    """Analyze data using the AI Proxy."""
    if not AIPROXY_TOKEN:
        logging.error("API token is not set in environment variables.")
        return "Error: API token is missing."

    # Define the AI Proxy endpoint
    api_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

    # Set up the headers for the API request
    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json",
    }

    # Convert the first few rows of the dataframe to a string for the prompt
    df_string = df.head(10).to_string()

    # Create the prompt for analysis
    prompt = f"""
    You are a data analysis assistant. Given the following dataset:

    {df_string}

    Please provide:
    1. A summary of the dataset, including key trends and observations.
    2. Identify any missing or unusual values.
    3. Analyze relationships between any categorical variables.
    4. Provide any recommendations or insights based on the data.

    Respond with the analysis and insights.
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
        # Send the request to the AI Proxy
        response = requests.post(api_url, json=data, headers=headers)

        if response.status_code == 200:
            response_data = response.json()
            if "choices" in response_data:
                analysis = response_data["choices"][0]["message"]["content"].strip()
                return analysis
            else:
                logging.error(f"Unexpected response format: {response_data}")
                return "Error: No analysis available in the response."
        else:
            logging.error(f"API request failed: {response.status_code} - {response.text}")
            return f"Error: API request failed with status {response.status_code}"
    except Exception as e:
        logging.error(f"An error occurred while making the API request: {e}")
        return f"Error: {str(e)}"

def visualize_data(df, output_folder):
    """Generate visualizations for the dataset and save to the specified folder."""
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
        heatmap_file = os.path.join(output_folder, "heatmap.png")
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
        lineplot_file = os.path.join(output_folder, "lineplot.png")
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
            histogram_file = os.path.join(output_folder, "histogram.png")
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
