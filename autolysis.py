# ///
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "seaborn",
#   "matplotlib",
#   "scipy",
#   "requests",
#   "scikit-learn",
# ]
# ///

import os
import sys
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, ttest_ind
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
import numpy as np
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

def summarize_data(df):
    """Generate a summarized version of the dataset."""
    summary = {
        'Shape': df.shape,
        'Columns': list(df.columns),
        'Missing Values': df.isnull().sum().to_dict(),
        'Basic Stats': df.describe(include='all').to_dict(),
        'Unique Values': {col: df[col].nunique() for col in df.columns}
    }
    return summary

def generate_prompt(summary):
    """Generate a more concise prompt for the AI model."""
    prompt = f"""
    You are a data analysis assistant. Below is a summary of the dataset:

    - **Shape**: {summary['Shape']}
    - **Columns**: {', '.join(summary['Columns'])}
    - **Missing Values**: {summary['Missing Values']}
    - **Basic Stats**: {summary['Basic Stats']}
    - **Unique Values**: {summary['Unique Values']}

    Based on this, please provide:
    1. A summary of the dataset, including key trends and observations.
    2. Identify any missing or unusual values.
    3. Provide recommendations or insights based on the data.
    """
    return prompt

def analyze_data(df):
    """Analyze data using the AI Proxy API and summarize the data."""
    if not AIPROXY_TOKEN:
        logging.error("API token is not set in environment variables.")
        return "Error: API token is missing."

    # Summarize data to send a compact version
    summary = summarize_data(df)
    
    # Generate a concise prompt for the AI model
    prompt = generate_prompt(summary)

    # AI analysis via API
    api_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {AIPROXY_TOKEN}", "Content-Type": "application/json"}
    
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
                ai_analysis = response_data["choices"][0]["message"]["content"].strip()
                return ai_analysis, summary
            else:
                logging.error(f"Unexpected response format: {response_data}")
                return "Error: No analysis available in the response.", summary
        else:
            logging.error(f"API request failed: {response.status_code} - {response.text}")
            return f"Error: API request failed with status {response.status_code}", summary
    except Exception as e:
        logging.error(f"An error occurred while making the API request: {e}")
        return f"Error: {str(e)}", summary

def hypothesis_testing(df):
    """Perform hypothesis testing (t-test, chi-square) for statistical analysis."""
    logging.info("Performing hypothesis testing...")

    # Example t-test: Compare means between two groups (e.g., ratings of books in two languages)
    if 'language_code' in df.columns and 'average_rating' in df.columns:
        lang_english = df[df['language_code'] == 'eng']['average_rating']
        lang_other = df[df['language_code'] != 'eng']['average_rating']

        t_stat, p_value = ttest_ind(lang_english.dropna(), lang_other.dropna())
        logging.info(f"T-test result between English and other languages: t-stat={t_stat}, p-value={p_value}")

        if p_value < 0.05:
            logging.info("There is a significant difference in ratings between English and other languages.")
        else:
            logging.info("No significant difference in ratings between English and other languages.")

    # Chi-square test for categorical variables (e.g., publication year vs. language code)
    if 'original_publication_year' in df.columns and 'language_code' in df.columns:
        contingency_table = pd.crosstab(df['original_publication_year'], df['language_code'])
        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

        logging.info(f"Chi-square test result: chi2_stat={chi2_stat}, p-value={p_value}")
        if p_value < 0.05:
            logging.info("There is a significant association between publication year and language code.")
        else:
            logging.info("No significant association between publication year and language code.")

def cross_validation(df):
    """Perform cross-validation on a simple logistic regression model."""
    logging.info("Performing cross-validation...")

    # Use numeric columns for model
    numeric_columns = df.select_dtypes(include=["number"]).columns
    if len(numeric_columns) < 2:
        logging.warning("Not enough numeric columns for cross-validation.")
        return None

    X = df[numeric_columns].dropna()
    y = df['ratings_count']  # Assuming 'ratings_count' is a target variable for prediction

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Logistic Regression model
    model = LogisticRegression(max_iter=1000)

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    logging.info(f"Cross-validation scores: {cv_scores}")
    logging.info(f"Mean cross-validation score: {cv_scores.mean()}")

def dynamic_analysis_workflow(df, analysis_functions):
    """Execute dynamic analysis workflow by calling functions based on user input or data attributes."""
    results = {}
    
    for func_name, func in analysis_functions.items():
        logging.info(f"Executing {func_name}...")
        try:
            result = func(df)
            results[func_name] = result
        except Exception as e:
            logging.error(f"Error executing {func_name}: {e}")
            results[func_name] = f"Error: {e}"

    return results

def visualize_data(df):
    """Generate visualizations for the dataset and save to the current working directory."""
    charts = []

    # Identify numeric columns
    numeric_columns = df.select_dtypes(include=["number"]).columns

    # Correlation Heatmap
    if len(numeric_columns) > 1:  # Correlation requires at least two numeric columns
        plt.figure(figsize=(9.6, 5.4))
        heatmap = sns.heatmap(
            df[numeric_columns].corr(),
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            cbar_kws={'shrink': 0.8},
            annot_kws={"size": 8},
        )
        heatmap.set_title("Correlation Heatmap", fontsize=12, pad=20)
        plt.tight_layout(pad=3.0)
        heatmap_file = "heatmap.png"
        plt.savefig(heatmap_file, dpi=100)
        charts.append(heatmap_file)
        plt.close()

    # Line Plot
    if len(numeric_columns) >= 2:
        plt.figure(figsize=(9.6, 5.4))
        for col in numeric_columns:
            df[col].dropna().plot(kind="line", label=col)
        plt.legend()
        plt.title("Line Plot of Numeric Columns")
        lineplot_file = "lineplot.png"
        plt.savefig(lineplot_file, dpi=100)
        charts.append(lineplot_file)
        plt.close()

    return charts

def save_markdown(df, analysis, charts):
    """Generate a markdown report of the analysis."""
    readme_content = f"""
# Analysis Report

## Dataset Overview

- **Shape**: {df.shape}
- **Columns**: {', '.join(df.columns)}
- **Missing Values**: {df.isnull().sum().to_dict()}

## Summary Statistics

{analysis}

## Visualizations
"""
    for chart in charts:
        readme_content += f"![Chart]({chart})\n"

    readme_filename = "README.md"
    with open(readme_filename, "w") as file:
        file.write(readme_content)

    logging.info(f"Markdown report saved as {readme_filename}")
    return readme_filename

def main(filename):
    """Main function to load dataset, analyze, visualize, and generate report."""
    df = read_csv(filename)

    # Define dynamic analysis functions to be used in the workflow
    analysis_functions = {
        'Data Analysis': analyze_data,
        'Hypothesis Testing': hypothesis_testing,
        'Cross-Validation': cross_validation,
    }

    # Perform dynamic analysis workflow
    results = dynamic_analysis_workflow(df, analysis_functions)

    # Generate visualizations
    charts = visualize_data(df)
    logging.info("Visualizations generated.")

    # Save the report
    readme_file = save_markdown(df, results, charts)
    logging.info(f"Markdown report saved: {readme_file}")

if __name__ == "__main__":
    main("goodreads.csv")  # Provide your CSV file path
