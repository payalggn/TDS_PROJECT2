# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx",
#   "pandas",
#   "seaborn",
#   "matplotlib",
#   "scipy",
#   "requests",
#   "openai==0.27.6",
# ]
# ///

import os
import sys
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import chi2_contingency
import requests
import openai

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

AIPROXY_TOKEN = ""
UPLOAD_ENDPOINT = "https://aiproxy.sanand.workers.dev/"

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

def analyze_data(df):
    openai.api_key = AIPROXY_TOKEN

    df_string = df.head(10).to_string()

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

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a helpful data analysis assistant."},
                  {"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0.7,
    )

    analysis = response['choices'][0]['message']['content'].strip()
    return analysis

def visualize_data(df, output_folder):
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
        heatmap_file = os.path.join(output_folder, "heatmap.png")
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
        lineplot_file = os.path.join(output_folder, "lineplot.png")
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
        histogram_file = os.path.join(output_folder, "histogram.png")
        plt.savefig(histogram_file, dpi=300)
        charts.append(histogram_file)
        plt.close()

    return charts

def save_markdown(df, analysis, charts, output_folder):
    df_info = {
        "shape": df.shape,
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict()
    }

    narration = f"""This dataset contains {df_info['shape'][0]} rows and {df_info['shape'][1]} columns, providing valuable insights into various attributes such as {', '.join(df_info['columns'][:5])}, and more. The dataset reveals interesting trends and patterns. For instance, the presence of missing values in some columns such as {', '.join([col for col, val in df_info['missing_values'].items() if val > 0])} highlights areas that may require further attention. Analyzing the relationships between these variables sheds light on significant correlations and distributions, offering a deeper understanding of the data."""

    payload = {
        "token": AIPROXY_TOKEN,
        "action": "generate_readme",
        "analysis": {
            "df_info": df_info,
            "insights": analysis,
        },
        "charts": charts
    }

    try:
        response = requests.post(UPLOAD_ENDPOINT, json=payload)
        response.raise_for_status()
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
            chart_name = Path(chart).name
            readme_content += f'<img src="./{chart_name}" alt="Chart" width="540" />\n'

        readme_file = os.path.join(output_folder, "README.md")
        with open(readme_file, "w") as f:
            f.write(readme_content)

        logging.info("README.md generated successfully.")
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to generate README.md: {e}")

def main():
    if len(sys.argv) != 2:
        logging.error("Usage: python autolysis.py <path_to_csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    df = read_csv(file_path)
    analysis = analyze_data(df)
    charts = visualize_data(df, ".")
    save_markdown(df, analysis, charts, ".")
    logging.info("Analysis completed successfully.")

if __name__ == "__main__":
    main()
