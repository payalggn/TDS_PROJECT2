# TDS Project 2

## Overview
`autolysis.py` is a Python script designed to perform comprehensive analysis and generate insights from a CSV dataset. The script reads a CSV file, analyzes the data (statistical analysis, chi-square tests, and visualizations), and produces a detailed report in markdown format.

## Features
- **Data Loading**: Reads CSV files with error handling for encoding issues.
- **API Integration**: Sends dataset summary to an AI API for high-level insights.
- **Statistical Analysis**: Generates descriptive statistics, correlation matrix, and chi-square tests for categorical data.
- **Visualization**: Produces heatmaps, line plots, and histograms.
- **Markdown Report**: Saves the analysis and visualizations in a markdown file.

## Requirements
- Python 3.11 or higher
- Required packages:
  - `pandas`
  - `seaborn`
  - `matplotlib`
  - `scipy`
  - `requests`
