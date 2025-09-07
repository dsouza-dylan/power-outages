# Project Report

## Introduction

### Introduction and Question Identification
# Predicting the Severity of Power Outages

**Name**: Dylan Dsouza

**Website Link**: https://dsouza-dylan.github.io/power-outages/

## Step 1: Introduction

First, I begin by understanding the data given within the Excel sheet.

Next, I maintain a dictionary of each variable and its corresponding unit of measurement.

Finally, I pick out the exact columns within the Excel sheet to create the raw DataFrame to be used for this project.

Looking at the data, I want to investigate **which factors affect the duration and intensity of a power outage, and if we can preemptively predict and detect large scale power outages in the United States**.

## Step 2: Data Cleaning and Exploratory Data Analysis

Since the raw DataFrame has multiple irrelevant columns, I eliminate most and retain the ones relevant for the purpose of this project, replacing unavailable values whenever encountered.

Next, I convert certain relevant columns to numeric data, and the date columns to datetime objects.

### Feature Engineering

Based on the existing columns of the given data, I engineered additional relevant and related columns to work with in subsequent steps of the project.

I also applied a logarithmic transformation using 1 plus the input value, to account for the fact that the logarithm of 0 is undefined.

Finally, as my investigation primarily involves the number of customers affected by a power outage, I chose to fill missing values for this column with 0. Here, I use the underlying assumption that if an outage was not overly significant, the number of customers affected by it was also insignificant/not reported due to its small-scale impact.

### Univariate Analysis

Before delving into univariate analysis, here are some summary statistics to better gauge the data I am working with:

With this understanding, I plot the distribution of outage durations. To ensure better bin values, I excluded those power outages exceeding or equal to 10000 minutes as they will be outliers for this plot. Note that I only took this step of filtering outliers for this plot, and the remainder of the project has data unaltered to retain its quality.

Next, I plotted a pie chart depicting the severity of each power outage based on the previous feature engineering I conducted.

Finally, I attempted to uncover seasonal patterns by looking at the average number of customers affected in each seasons, and the average duration of a power outage in each season.

### Bivariate Analysis

First, I looked at a box plot representing the outage duration distibution based on the cause category, again filtering outliers exceeding 10000 minutes.

Next, I looked at the spread of outage durations considering the number of customers affected. This time, I only considered data when the number of customers affected exceeded 10000, again to account for skew/outliers.

### Interesting Aggregations

The first aggregation I attempted was to look at the data state-by-state, considering the mean of 3 columns as given below:

Next, I attemped a similar aggregation for the customers affected by each cause category of power outage, considering mean and median as aggregation functions.

Finally, an aggregation for the average duration of power outages for each season given a specific cause category:

## Step 3: Assessment of Missingness

I first looked at how much percentage of values the data is missing in each column, which also serves as a way to cross-validate any null replacement steps I have taken above.

The missing column I chose to examine was 'CAUSE.CATEGORY.DETAIL', which I essentially one-hot encoded, using an additional column.

Subsequently, I wrote a helper function for permutation testing, which can be used to test different columns for whether or not they depend on missingness of 'CAUSE.CATEGORY.DETAIL'.

'CAUSE.CATEGORY' has a large observed difference in missingness of close to 0.89. The negligible p-value validates this, and we can safely conclude that the missingness of 'CAUSE.CATEGORY.DETAIL' **does depend** on 'CAUSE.CATEGORY', which also makes intuitive sense.

In contrast, 'OUTAGE_DAYOFWEEK' seems to show little to no difference (approximately 0.12) in missingness when compared with 'CAUSE.CATEGORY.DETAIL', and the p-value exceeds 0.05, suggesting that missingness **does not depend** on day of the week.

## Step 4: Hypothesis Testing

For this hypothesis test, I chose to investigate whether mean outage duration is the same for severe weather and equipment failure, or if severe weather outages last longer than equipment failure outages:

**Null Hypothesis (H₀):** The mean outage duration is the same for severe weather and equipment failure outages.

**Alternative Hypothesis (H₁)**: Severe weather outages last longer, on average, than equipment failure outages.

Once I computed the observed difference, I conducted a permutation test to assess whether the said difference could be attributed to randomness, or whether there was an actual association present.

Since our p-value is much lesser than 0.05, we can safely **reject the null hypothesis**, and conclude that **outages due to severe weather conditions do last longer than equipment failure outages**, on average. I also added a plot for better visualization of the result of this test.

## Step 5: Framing a Prediction Problem

The goal of this project is to **predict the category of a major power outage's duration (DURATION_CLASS), classified as Short or Long**, based on factors such as location, time, cause, and demographic characteristics.

**Target Variable:** DURATION_CLASS (derived from LOG_DURATION)

## Step 6: Baseline Model

First, I dropped the column I had previously added from the missingness analysis section.

Next, I used the data and classified each outage as either 'Short' or 'Long', based on the logarithmic duration. After excluding the duration columns from the training data (to prevent any biases and possible redundancies), I preprocessed the features, trained a **Decision Tree** classifier, and finally evaluated how well it predicts outage duration categories using a Classification Report and a Confusion Matrix.

## Step 7: Final Model

To improve my baseline model, I pivoted to a **Random Forest** classifier, introducing Grid Search for more effective hyperparameter tuning, specifically:

**Number of Trees:** While more trees generally reduce variance, they do increase computation time about which I am vary.

**Maximum Depth of Each Tree:** To reduce overfitting but ensure enough depth to capture salient features.

**Minimum Samples Needed to Split Node:** To prevent splitting nodes on noisy partitions, which can contribute to overfitting.

**Minimum Samples Required at Leaf Node:** To ensure leaf nodes are not too small, i.e. again to prevent overfitting and decrease sensitivity to noise.

Finally, I evaluated model performance using the same Classification Report and Confusion Matrix metrics, and fortunately, the final model showed a slight but significant increase in almost every metric!

## Step 8: Fairness Analysis

Assuming we partition the data into 2 groups, specifically:

**GROUP X:** High percentage urban areas (POPPCT_URBAN >= 50%)

**GROUP Y:** Low percentage urban areas (POPPCT_URBAN < 50%)

Using the metric of **precision scores for 'Long' outages**, we have the following hypotheses:

**Null Hypothesis (H₀):** The model is fair across groups (GROUP X = GROUP Y), as the precision scores are the same for 'Long' outages regardless of group chosen.

**Alternative Hypothesis (H₁):** The model is unfair across groups (GROUP X != GROUP Y), as the precision scores are not the same for 'Long' outages, and probably depend on group chosen.

Once I computed the observed difference, I conducted a permutation test to assess whether the said difference could be attributed to randomness, or whether there was an actual association present.

Since our p-value is much greater than 0.05, we can safely **fail to reject the null hypothesis**, and conclude that the model is fair across groups.

---

## Data Cleaning and Exploratory Data Analysis

### Data Cleaning
(Describe cleaning steps here, show cleaned DataFrame head.)

---

### Univariate Analysis
(Embed at least one plotly visualization of a single column with interpretation.)

---

### Bivariate Analysis
(Embed at least one plotly visualization of two columns with interpretation.)

---

### Interesting Aggregates
(Embed grouped table or pivot table and explain significance.)

---

## Assessment of Missingness

### NMAR Analysis
(State whether a column is NMAR and reasoning.)

---

### Missingness Dependency
(Present permutation test results, embed plotly plot, interpret.)

---

## Hypothesis Testing

### Hypothesis Testing
(State null & alternative hypotheses, test statistic, significance level, p-value, conclusion.)

---

## Framing a Prediction Problem

### Problem Identification
(Clearly define prediction problem and justify features available at prediction time.)

---

### Baseline Model
(Describe baseline model, features, encodings, performance, assessment.)

---

### Final Model
(State added features and rationale, algorithm, hyperparameters, performance comparison.)

---

## Fairness Analysis

### Fairness Analysis
(Define groups, evaluation metric, hypotheses, test statistic, significance level, p-value, conclusion.)
