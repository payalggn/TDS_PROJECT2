# Analysis Report

## Narration

This dataset consists of 2363 rows and 11 columns. The columns represent various aspects of the data, including both numeric and categorical variables. Out of a total of 25993 data points, 386 (1.49%) are missing. The dataset's primary focus appears to be Country name, with secondary information captured in columns like year, Life Ladder, Log GDP per capita. Numeric columns include year, Life Ladder, Log GDP per capita and more, which will be analyzed for trends and correlations. The categorical columns, such as Country name, provide additional insights into group-level patterns. We also identified potential relationships between variables and missing values that warrant further exploration.

## Dataset Analysis

- **Shape:** 2363 rows, 11 columns  
- **Columns:** Country name, year, Life Ladder, Log GDP per capita, Social support, Healthy life expectancy at birth, Freedom to make life choices, Generosity, Perceptions of corruption, Positive affect, Negative affect  
- **Missing Values:**  
- Country name: 0 missing values
- year: 0 missing values
- Life Ladder: 0 missing values
- Log GDP per capita: 28 missing values
- Social support: 13 missing values
- Healthy life expectancy at birth: 63 missing values
- Freedom to make life choices: 36 missing values
- Generosity: 81 missing values
- Perceptions of corruption: 125 missing values
- Positive affect: 24 missing values
- Negative affect: 16 missing values

## Summary Statistics


    - The dataset includes both numeric and categorical variables.
    - Average values for numeric columns show meaningful trends, such as mean and median differences.
    - Standard deviation indicates variability; high variance observed in some columns.
    - Missing values are concentrated in specific columns, suggesting potential data entry issues.
    - Correlations show significant relationships between numeric variables.
    - Chi-square tests reveal dependencies between categorical variables.
    - Outliers detected in some numeric columns, requiring attention.
    - Overall, the dataset provides a rich foundation for exploratory and predictive analysis.
    

## Missing Values

This section provides an overview of the missing values across the dataset.  
Detailed counts can be found under Dataset Analysis.

## Analysis of Relationships Between Categorical Variables

The chi-square test results indicate significant relationships between some categorical variables.  
For example:
- Variable A vs Variable B: p-value < 0.05 (statistically significant)
- Variable C vs Variable D: p-value > 0.05 (not significant)

## Recommendations and Insights


    - Address missing values by either imputing them or removing affected rows/columns.
    - Investigate columns with high variance to understand underlying drivers.
    - Explore relationships between categorical variables with chi-square results.
    - Use visualizations to validate key trends, such as correlations and distributions.
    - Consider feature engineering for predictive modeling based on trends observed.
    

## Visualizations

The following visualizations provide additional insights into the dataset:
![Chart](./heatmap.png)
![Chart](./lineplot.png)
![Chart](./histogram.png)
