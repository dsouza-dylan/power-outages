# Predicting the Severity of Power Outages

**Name:** Dylan Dsouza  
**Website Link:** [https://dsouza-dylan.github.io/power-outages/](https://dsouza-dylan.github.io/power-outages/)

---

## Introduction

Power outages are a significant infrastructure challenge that affects millions of Americans annually, with cascading impacts on healthcare, transportation, commerce, and daily life. Understanding the factors that influence outage severity can help utility companies, emergency planners, and policymakers better prepare for and respond to these events.

This project analyzes major power outage events in the continental United States from January 2000 to July 2016, investigating **which factors affect the duration and intensity of power outages, and whether we can preemptively predict and detect large-scale power outages**.

The dataset contains **1,534 rows** representing individual power outage events. The key columns relevant to this analysis include:

- **OUTAGE.DURATION** (minutes): Duration of the power outage
- **CUSTOMERS.AFFECTED**: Number of customers impacted by the outage
- **CAUSE.CATEGORY**: Primary cause of the outage (severe weather, equipment failure, etc.)
- **U.S._STATE**: State where the outage occurred
- **CLIMATE.CATEGORY**: Climate conditions during the outage (normal, cold, warm)
- **POPULATION**: State population
- **POPPCT_URBAN** (%): Percentage of state population in urban areas
- **NERC.REGION**: North American Electric Reliability Corporation region

Understanding outage patterns is crucial for grid resilience planning, emergency response preparation, and infrastructure investment decisions that affect millions of people.

---

## Data Cleaning and Exploratory Data Analysis

### Data Cleaning

The raw dataset required significant preprocessing to make it suitable for analysis. The original Excel file contained metadata and headers that needed to be separated from the actual data.

**Key cleaning steps:**

1. **Column extraction**: Removed metadata rows and extracted proper column names from row 4, with units of measurement from row 5
2. **Data subset selection**: Retained 22 relevant columns from the original 57 columns, focusing on temporal, geographic, demographic, and outage characteristics
3. **Data type conversion**: Converted `OUTAGE.DURATION` and `CUSTOMERS.AFFECTED` to numeric values, handling invalid entries as NaN
4. **Date parsing**: Converted date columns to datetime objects for temporal analysis
5. **Missing value handling**: Replaced 'NA' strings with proper NaN values, and filled missing `CUSTOMERS.AFFECTED` values with 0 (assuming unreported outages had minimal impact)

**Feature Engineering:**

Created additional derived features to enhance analysis:
- **OUTAGE_SEASON**: Categorized outages by season based on start date
- **IS_WEEKEND**: Boolean flag for weekend occurrences
- **CUSTOMER_DENSITY**: Ratio of total customers to population
- **URBANIZATION_RATIO**: Decimal representation of urban population percentage
- **POPULATION_DENSITY**: Weighted average of urban and rural population densities
- **SEVERITY_CATEGORY**: Classified outages as "Small" (<10K customers), "Medium" (10K-100K), "Large" (>100K), or "Unknown/Minor"
- **IS_EXTREME_WEATHER**: Boolean flag for weather-related causes

Here's the head of the cleaned DataFrame:

| OBS | YEAR | MONTH | U.S._STATE | NERC.REGION | CLIMATE.REGION | ANOMALY.LEVEL | CLIMATE.CATEGORY | OUTAGE.START.DATE | CUSTOMERS.AFFECTED | OUTAGE.DURATION |
|-----|------|-------|------------|-------------|----------------|---------------|------------------|-------------------|-------------------|-----------------|
| 1   | 2011 | 7     | Minnesota  | MRO         | East North Central | -0.3 | normal | 2011-07-01 | 70000.0 | 3060 |
| 2   | 2014 | 5     | Minnesota  | MRO         | East North Central | -0.1 | normal | 2014-05-11 | 68200.0 | 1 |
| 3   | 2010 | 10    | Minnesota  | MRO         | East North Central | -1.5 | cold | 2010-10-26 | 70000.0 | 3000 |
| 4   | 2012 | 6     | Minnesota  | MRO         | East North Central | -0.1 | normal | 2012-06-19 | 68200.0 | 2550 |
| 5   | 2015 | 7     | Minnesota  | MRO         | East North Central | 1.2 | warm | 2015-07-18 | 250000.0 | 1740 |

### Univariate Analysis

**Summary Statistics:**
- Average outage duration: 2,625 minutes (43.75 hours)
- Median customers affected: 30,534
- Most common cause: Severe weather
- Most affected season: Summer

The distribution of outage durations shows a right-skewed pattern with most outages lasting under 2,000 minutes, but with a long tail of extended outages. The median duration of 1,740 minutes (29 hours) indicates that half of all outages are resolved within roughly a day, while the mean being higher suggests some extremely long-duration events significantly impact the average.

<iframe src="https://dsouza-dylan.github.io/power-outages/assets/fig1.html" width=800 height=600 frameBorder=0></iframe>
<iframe src="https://dsouza-dylan.github.io/power-outages/assets/fig2.html" width=800 height=600 frameBorder=0></iframe>
<iframe src="https://dsouza-dylan.github.io/power-outages/assets/fig3.html" width=800 height=600 frameBorder=0></iframe>
<iframe src="https://dsouza-dylan.github.io/power-outages/assets/fig4.html" width=800 height=600 frameBorder=0></iframe>
<iframe src="https://dsouza-dylan.github.io/power-outages/assets/fig5.html" width=800 height=600 frameBorder=0></iframe>

### Bivariate Analysis

Analysis of outage duration by cause category reveals significant differences between causes. Severe weather events tend to produce the longest outages with the highest variability, while intentional attacks typically result in shorter, more predictable durations. Equipment failures show moderate duration with less variability than weather events.

The relationship between customers affected and outage duration shows a weak positive correlation, suggesting that larger outages don't necessarily last longer. This indicates that the scope and duration of outages are influenced by different factors - scope likely depends on grid interconnectedness and population density, while duration depends more on the nature of the damage and repair complexity.

### Interesting Aggregates

**State-by-State Analysis (Top 10 by Average Customers Affected):**

| State | Avg Customers Affected | Avg Duration (min) | Avg Population |
|-------|----------------------|------------------|----------------|
| Florida | 282,939 | 4,095 | 18.1M |
| South Carolina | 251,913 | 3,135 | 4.5M |
| Illinois | 198,026 | 1,602 | 12.8M |
| District of Columbia | 175,238 | 4,304 | 622K |
| Texas | 165,227 | 2,705 | 25.2M |

Florida leads in both average customers affected and duration, likely due to frequent severe weather events and hurricanes.

**Cause Category Analysis:**

| Cause Category | Mean Duration | Median Duration | Mean Customers | Median Customers |
|----------------|---------------|----------------|----------------|------------------|
| Severe Weather | 3,884 | 2,460 | 177,206 | 105,000 |
| Fuel Supply Emergency | 13,484 | 3,960 | 0.02 | 0 |
| System Operability | 729 | 215 | 137,941 | 25,000 |
| Equipment Failure | 1,817 | 221 | 50,968 | 0 |

This table reveals that severe weather causes both the most widespread outages (highest customer impact) and among the longest duration outages, making it the most significant outage category for overall grid reliability.

---

## Assessment of Missingness

### NMAR Analysis

I believe the column **HURRICANE.NAMES** is likely NMAR (Not Missing At Random). This column is 95.31% missing, and the missingness is directly related to the value itself - hurricane names are only recorded when the outage is actually caused by a named hurricane. The absence of a hurricane name is meaningful information indicating the outage was not hurricane-related.

To make this column MAR (Missing At Random), we would need additional data such as:
- Wind speed measurements during the outage
- Barometric pressure readings
- Official weather service classifications
- Storm tracking data that could indicate unnamed storm systems

### Missingness Dependency

I investigated whether the missingness of `CAUSE.CATEGORY.DETAIL` depends on other columns using permutation testing.

**Testing CAUSE.CATEGORY dependency:**
- Observed difference in missingness rates: 0.885
- P-value: 0.000 (< 0.05)
- **Conclusion**: The missingness of detail strongly depends on the cause category, which makes intuitive sense as some cause categories naturally have more detailed subcategorizations than others.

**Testing OUTAGE_DAYOFWEEK dependency:**
- Observed difference in missingness rates: 0.122
- P-value: 0.088 (> 0.05)
- **Conclusion**: The missingness of detail does not significantly depend on the day of the week, suggesting no systematic reporting bias based on when outages occur.

---

## Hypothesis Testing

**Research Question**: Do severe weather outages last longer than equipment failure outages on average?

**Null Hypothesis (H₀)**: The mean outage duration is the same for severe weather and equipment failure outages.

**Alternative Hypothesis (H₁)**: Severe weather outages last longer, on average, than equipment failure outages.

**Test Statistic**: Difference in mean duration (Weather - Equipment)

**Significance Level**: α = 0.05

**Results**:
- Observed difference: 2,067 minutes (Weather outages last ~34.5 hours longer on average)
- P-value: 0.000 (from 10,000 permutations)

<iframe src="https://dsouza-dylan.github.io/power-outages/assets/permutation_test_histogram.html" width=800 height=600 frameBorder=0></iframe>

**Conclusion**: With a p-value much less than 0.05, we reject the null hypothesis. There is strong statistical evidence that severe weather outages last significantly longer than equipment failure outages on average. This makes practical sense as weather damage often affects larger areas and requires more complex repairs than localized equipment failures.

---

## Framing a Prediction Problem

**Prediction Problem**: Predict whether a power outage will be "Short" (< 6 log-minutes duration) or "Long" (≥ 6 log-minutes duration) based on information available at the time the outage begins.

**Problem Type**: Binary Classification

**Response Variable**: DURATION_CLASS (derived from LOG_DURATION)

**Evaluation Metric**: Accuracy, with additional focus on precision and recall for both classes to ensure balanced performance.

**Features Available at "Time of Prediction"**:
- Geographic information (state, region)
- Temporal factors (month, season, day of week)
- Cause information (when immediately apparent)
- Demographic characteristics (population, urbanization)
- Climate conditions
- Customer base characteristics

This prediction task is valuable because early classification of outage severity can help utilities allocate appropriate resources and set realistic restoration expectations for customers and emergency services.

---

## Baseline Model

**Model Description**: Decision Tree Classifier using 11 features to predict outage duration class.

**Features Used**:
- **Quantitative (6)**: MONTH, ANOMALY.LEVEL, CUSTOMERS.AFFECTED, POPPCT_URBAN, CUSTOMER_DENSITY, POPULATION_DENSITY
- **Nominal (5)**: CLIMATE.CATEGORY, CAUSE.CATEGORY, SEVERITY_CATEGORY, OUTAGE_SEASON, IS_EXTREME_WEATHER

**Preprocessing**:
- Numeric features: Mean imputation for missing values
- Categorical features: Most frequent imputation + One-hot encoding

**Performance**:
```
              precision  recall  f1-score  support
Long             0.80    0.78      0.79     170
Short            0.71    0.73      0.72     126
accuracy                           0.76     296
```

**Assessment**: The baseline model achieves 76% accuracy with reasonable precision and recall for both classes. The model performs slightly better on predicting long outages (precision 0.80) than short outages (precision 0.71), which could be valuable for emergency planning purposes. However, there's room for improvement in the overall performance and class balance.

---

## Final Model

**Added Features and Rationale**:

1. **Enhanced preprocessing for CUSTOMER_DENSITY**: Applied log transformation to handle the skewed distribution of customer density ratios, which better captures the relationship between customer concentration and outage characteristics.

2. **Improved numeric preprocessing**: Used StandardScaler for most numeric features to ensure equal contribution to tree-based decisions, while applying specialized log transformation for customer density.

3. **Advanced categorical handling**: Maintained one-hot encoding but with improved imputation strategies.

**Model Algorithm**: Random Forest Classifier with hyperparameter tuning

**Hyperparameter Optimization**:
- Used GridSearchCV with 5-fold cross-validation
- Search space included:
  - n_estimators: [90, 100, 200, 300]
  - max_depth: [None, 10, 20, 30, 40]
  - min_samples_split: [2, 5, 10]
  - min_samples_leaf: [1, 2, 4, 8]

**Best Hyperparameters**:
- n_estimators: 300
- max_depth: 20
- min_samples_split: 5
- min_samples_leaf: 1

**Final Model Performance**:
```
              precision  recall  f1-score  support
Long             0.82    0.82      0.82     170
Short            0.76    0.76      0.76     126
accuracy                           0.80     296
```

**Improvement Over Baseline**:
- Overall accuracy improved from 76% to 80%
- Precision for Long outages improved from 0.80 to 0.82
- Precision for Short outages improved from 0.71 to 0.76
- More balanced performance across both classes

The Random Forest approach with proper hyperparameter tuning provides better generalization than the single Decision Tree, while the enhanced feature preprocessing better captures the underlying relationships in the data.

---

## Fairness Analysis

**Fairness Question**: Does the model perform equally well for high-urbanization areas vs. low-urbanization areas?

**Group Definition**:
- **Group X**: High urbanization areas (POPPCT_URBAN ≥ 50%)
- **Group Y**: Low urbanization areas (POPPCT_URBAN < 50%)

**Evaluation Metric**: Precision for "Long" outage predictions

**Null Hypothesis (H₀)**: The model is fair across groups - precision scores for 'Long' outages are equal regardless of urbanization level.

**Alternative Hypothesis (H₁)**: The model is unfair across groups - precision scores for 'Long' outages differ between high and low urbanization areas.

**Test Statistic**: Difference in precision scores (Group X - Group Y)

**Significance Level**: α = 0.05

**Results**:
- Observed difference in precision: -0.1796 (High urbanization areas have lower precision)
- P-value: 0.7018 (from 10,000 permutations)

**Conclusion**: With a p-value of 0.7018, which is much greater than 0.05, we fail to reject the null hypothesis. There is insufficient evidence to conclude that the model performs unfairly across different urbanization levels. The model appears to achieve fairness parity between high and low urbanization areas for predicting long-duration outages.

This finding suggests that the model's predictions are not systematically biased against either urban or rural communities, which is important for equitable utility planning and emergency response resource allocation.

---

## Conclusion

This analysis successfully developed a machine learning model capable of predicting power outage duration categories with 80% accuracy. Key findings include:

1. **Severe weather is the dominant factor** in both outage frequency and severity
2. **Geographic and demographic factors** significantly influence outage patterns
3. **The final Random Forest model** demonstrates both good performance and fairness across different community types
4. **Early prediction capability** could help utilities better allocate emergency response resources

The model provides a foundation for improved grid resilience planning and emergency preparedness, with potential applications in resource allocation, customer communication, and infrastructure investment prioritization.
