import os
import sys
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import chi2_contingency
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN", "")  # Get token from environment variable
UPLOAD_ENDPOINT = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"  # Set this to your intended endpoint

def read_csv(filename):
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

import requests
import logging
import sys

def analyze_data(df):
    # Check if the API token is available
    if not AIPROXY_TOKEN:
        logging.error("API token is not set in environment variables.")
        return "Error: API token is missing."

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
    3. Perform an analysis of relationships between any categorical variables.
    4. Provide any recommendations or insights based on the data.

    Respond with the analysis and insights.
    """

    # Create the data for the API request
    data = {
        "model": "gpt-4o-mini",  # Use the model supported by the proxy
        "messages": [
            {"role": "system", "content": "You are a helpful data analysis assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1000,
        "temperature": 0.7,
    }

    try:
        # Send the request to the AI Proxy
        response = requests.post(api_url, json=data, headers=headers)

        # Check if the response is valid
        if response.status_code == 200:
            response_data = response.json()
            if 'choices' in response_data:
                # Extract the content of the analysis if 'choices' is present
                analysis = response_data['choices'][0]['message']['content'].strip()
                return analysis
            else:
                # Log and handle the case when 'choices' is not present
                logging.error("No 'choices' found in the response: %s", response_data)
                return "Error: No analysis available."
        else:
            # Handle failed API response
            logging.error(f"Error with AIPROXY API: {response.status_code} - {response.text}")
            return f"Request failed with status code {response.status_code}"
    
    except Exception as e:
        # Handle general errors
        logging.error(f"An error occurred: {str(e)}")
        return f"An error occurred: {str(e)}"

def visualize_data(df):
    charts = []

    numeric_columns = df.select_dtypes(include=["number"]).columns
    if len(numeric_columns) > 0:
        plt.figure(figsize=(18, 10))
        heatmap = sns.heatmap(
            df[numeric_columns].corr(),
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            cbar_kws={'shrink': 0.8}
        )
        heatmap.set_title("Correlation Heatmap", fontsize=16, pad=20)
        plt.tight_layout(pad=3.0)
        heatmap_file = "heatmap.png"  # Save in the current directory
        plt.savefig(heatmap_file, dpi=300)
        charts.append(heatmap_file)
        plt.close()

    if len(numeric_columns) >= 2:
        plt.figure(figsize=(18, 10))
        for col in numeric_columns:
            df[col].dropna().reset_index(drop=True).plot(label=col, linewidth=2)
        plt.title("Line Plot of Numeric Columns", fontsize=16, pad=20)
        plt.xlabel("Index", fontsize=14)
        plt.ylabel("Values", fontsize=14)
        plt.legend(loc="best")
        plt.tight_layout(pad=3.0)
        lineplot_file = "lineplot.png"  # Save in the current directory
        plt.savefig(lineplot_file, dpi=300)
        charts.append(lineplot_file)
        plt.close()

    if len(df.columns) > 1:
        second_column = df.columns[1]
        plt.figure(figsize=(18, 10))
        df[second_column].dropna().plot(kind="hist", bins=30, color="skyblue", edgecolor="black")
        plt.title(f"Histogram of {second_column}", fontsize=16, pad=20)
        plt.xlabel(second_column, fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.tight_layout(pad=3.0)
        histogram_file = "histogram.png"  # Save in the current directory
        plt.savefig(histogram_file, dpi=300)
        charts.append(histogram_file)
        plt.close()

    return charts

def save_markdown(df, analysis, charts):
    df_info = {
        "shape": df.shape,
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict()
    }

    narration = f"""This dataset contains {df_info['shape'][0]} rows and {df_info['shape'][1]} columns, providing valuable insights into various attributes such as {', '.join(df_info['columns'][:5])}, and more. The dataset reveals interesting trends and patterns. For instance, the presence of missing values in some columns such as {', '.join([col for col, val in df_info['missing_values'].items() if val > 0])} highlights areas that may require further attention. Analyzing the relationships between these variables sheds light on significant correlations and distributions, offering a deeper understanding of the data."""

    readme_content = f"""# Analysis Report

## Narration

{narration}

## Dataset Analysis

- *Shape*: {df_info['shape']}
- *Columns*: {', '.join(df_info['columns'])}
- *Missing Values*: {df_info['missing_values']}

## Summary Statistics

{analysis}

## Visualizations

"""
    for chart in charts:
        readme_content += f'<img src="./{chart}" alt="Chart" width="540" />\n'

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
