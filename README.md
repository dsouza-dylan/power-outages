# Predicting the Severity of Power Outages

**Name**: Dylan Dsouza

**Website Link**: https://dsouza-dylan.github.io/power-outages/


```python
import pandas as pd
import numpy as np
from pathlib import Path
import openpyxl
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
pd.options.plotting.backend = 'plotly'
import plotly.io as pio
pio.renderers.default = "iframe"
```


```python
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, QuantileTransformer, FunctionTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, precision_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

## Step 1: Introduction

First, I begin by understanding the data given within the Excel sheet.


```python
df = pd.read_excel('outage.xlsx')
raw_df = df.copy()
raw_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Major power outage events in the continental U.S.</th>
      <th>Unnamed: 1</th>
      <th>Unnamed: 2</th>
      <th>Unnamed: 3</th>
      <th>Unnamed: 4</th>
      <th>Unnamed: 5</th>
      <th>Unnamed: 6</th>
      <th>Unnamed: 7</th>
      <th>Unnamed: 8</th>
      <th>Unnamed: 9</th>
      <th>...</th>
      <th>Unnamed: 47</th>
      <th>Unnamed: 48</th>
      <th>Unnamed: 49</th>
      <th>Unnamed: 50</th>
      <th>Unnamed: 51</th>
      <th>Unnamed: 52</th>
      <th>Unnamed: 53</th>
      <th>Unnamed: 54</th>
      <th>Unnamed: 55</th>
      <th>Unnamed: 56</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Time period: January 2000 - July 2016</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Regions affected: Outages reported in this dat...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>variables</td>
      <td>OBS</td>
      <td>YEAR</td>
      <td>MONTH</td>
      <td>U.S._STATE</td>
      <td>POSTAL.CODE</td>
      <td>NERC.REGION</td>
      <td>CLIMATE.REGION</td>
      <td>ANOMALY.LEVEL</td>
      <td>CLIMATE.CATEGORY</td>
      <td>...</td>
      <td>POPPCT_URBAN</td>
      <td>POPPCT_UC</td>
      <td>POPDEN_URBAN</td>
      <td>POPDEN_UC</td>
      <td>POPDEN_RURAL</td>
      <td>AREAPCT_URBAN</td>
      <td>AREAPCT_UC</td>
      <td>PCT_LAND</td>
      <td>PCT_WATER_TOT</td>
      <td>PCT_WATER_INLAND</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1535</th>
      <td>NaN</td>
      <td>1530</td>
      <td>2011</td>
      <td>12</td>
      <td>North Dakota</td>
      <td>ND</td>
      <td>MRO</td>
      <td>West North Central</td>
      <td>-0.9</td>
      <td>cold</td>
      <td>...</td>
      <td>59.9</td>
      <td>19.9</td>
      <td>2192.2</td>
      <td>1868.2</td>
      <td>3.9</td>
      <td>0.27</td>
      <td>0.1</td>
      <td>97.599649</td>
      <td>2.401765</td>
      <td>2.401765</td>
    </tr>
    <tr>
      <th>1536</th>
      <td>NaN</td>
      <td>1531</td>
      <td>2006</td>
      <td>NaN</td>
      <td>North Dakota</td>
      <td>ND</td>
      <td>MRO</td>
      <td>West North Central</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>59.9</td>
      <td>19.9</td>
      <td>2192.2</td>
      <td>1868.2</td>
      <td>3.9</td>
      <td>0.27</td>
      <td>0.1</td>
      <td>97.599649</td>
      <td>2.401765</td>
      <td>2.401765</td>
    </tr>
    <tr>
      <th>1537</th>
      <td>NaN</td>
      <td>1532</td>
      <td>2009</td>
      <td>8</td>
      <td>South Dakota</td>
      <td>SD</td>
      <td>RFC</td>
      <td>West North Central</td>
      <td>0.5</td>
      <td>warm</td>
      <td>...</td>
      <td>56.65</td>
      <td>26.73</td>
      <td>2038.3</td>
      <td>1905.4</td>
      <td>4.7</td>
      <td>0.3</td>
      <td>0.15</td>
      <td>98.307744</td>
      <td>1.692256</td>
      <td>1.692256</td>
    </tr>
    <tr>
      <th>1538</th>
      <td>NaN</td>
      <td>1533</td>
      <td>2009</td>
      <td>8</td>
      <td>South Dakota</td>
      <td>SD</td>
      <td>MRO</td>
      <td>West North Central</td>
      <td>0.5</td>
      <td>warm</td>
      <td>...</td>
      <td>56.65</td>
      <td>26.73</td>
      <td>2038.3</td>
      <td>1905.4</td>
      <td>4.7</td>
      <td>0.3</td>
      <td>0.15</td>
      <td>98.307744</td>
      <td>1.692256</td>
      <td>1.692256</td>
    </tr>
    <tr>
      <th>1539</th>
      <td>NaN</td>
      <td>1534</td>
      <td>2000</td>
      <td>NaN</td>
      <td>Alaska</td>
      <td>AK</td>
      <td>ASCC</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>66.02</td>
      <td>21.56</td>
      <td>1802.6</td>
      <td>1276</td>
      <td>0.4</td>
      <td>0.05</td>
      <td>0.02</td>
      <td>85.761154</td>
      <td>14.238846</td>
      <td>2.901182</td>
    </tr>
  </tbody>
</table>
<p>1540 rows × 57 columns</p>
</div>



Next, I maintain a dictionary of each variable and its corresponding unit of measurement.


```python
unit_lst = raw_df.iloc[5].values
var_lst = raw_df.iloc[4].values
var_unit_dct = dict(zip(var_lst[1:], unit_lst[1:]))
var_unit_dct
```




    {'OBS': nan,
     'YEAR': nan,
     'MONTH': nan,
     'U.S._STATE': nan,
     'POSTAL.CODE': nan,
     'NERC.REGION': nan,
     'CLIMATE.REGION': nan,
     'ANOMALY.LEVEL': 'numeric',
     'CLIMATE.CATEGORY': nan,
     'OUTAGE.START.DATE': 'Day of the week, Month Day, Year',
     'OUTAGE.START.TIME': 'Hour:Minute:Second (AM / PM)',
     'OUTAGE.RESTORATION.DATE': 'Day of the week, Month Day, Year',
     'OUTAGE.RESTORATION.TIME': 'Hour:Minute:Second (AM / PM)',
     'CAUSE.CATEGORY': nan,
     'CAUSE.CATEGORY.DETAIL': nan,
     'HURRICANE.NAMES': nan,
     'OUTAGE.DURATION': 'mins',
     'DEMAND.LOSS.MW': 'Megawatt',
     'CUSTOMERS.AFFECTED': nan,
     'RES.PRICE': 'cents / kilowatt-hour',
     'COM.PRICE': 'cents / kilowatt-hour',
     'IND.PRICE': 'cents / kilowatt-hour',
     'TOTAL.PRICE': 'cents / kilowatt-hour',
     'RES.SALES': 'Megawatt-hour',
     'COM.SALES': 'Megawatt-hour',
     'IND.SALES': 'Megawatt-hour',
     'TOTAL.SALES': 'Megawatt-hour',
     'RES.PERCEN': '%',
     'COM.PERCEN': '%',
     'IND.PERCEN': '%',
     'RES.CUSTOMERS': nan,
     'COM.CUSTOMERS': nan,
     'IND.CUSTOMERS': nan,
     'TOTAL.CUSTOMERS': nan,
     'RES.CUST.PCT': '%',
     'COM.CUST.PCT': '%',
     'IND.CUST.PCT': '%',
     'PC.REALGSP.STATE': 'USD',
     'PC.REALGSP.USA': 'USD',
     'PC.REALGSP.REL': 'fraction',
     'PC.REALGSP.CHANGE': '%',
     'UTIL.REALGSP': 'USD',
     'TOTAL.REALGSP': 'USD',
     'UTIL.CONTRI': '%',
     'PI.UTIL.OFUSA': '%',
     'POPULATION': nan,
     'POPPCT_URBAN': '%',
     'POPPCT_UC': '%',
     'POPDEN_URBAN': 'persons per square mile',
     'POPDEN_UC': 'persons per square mile',
     'POPDEN_RURAL': 'persons per square mile',
     'AREAPCT_URBAN': '%',
     'AREAPCT_UC': '%',
     'PCT_LAND': '%',
     'PCT_WATER_TOT': '%',
     'PCT_WATER_INLAND': '%'}



Finally, I pick out the exact columns within the Excel sheet to create the raw DataFrame to be used for this project.


```python
raw_df.columns = var_lst
raw_df = raw_df.iloc[6:]
raw_df = raw_df.drop(columns='variables').set_index('OBS')
raw_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>YEAR</th>
      <th>MONTH</th>
      <th>U.S._STATE</th>
      <th>POSTAL.CODE</th>
      <th>NERC.REGION</th>
      <th>CLIMATE.REGION</th>
      <th>ANOMALY.LEVEL</th>
      <th>CLIMATE.CATEGORY</th>
      <th>OUTAGE.START.DATE</th>
      <th>OUTAGE.START.TIME</th>
      <th>...</th>
      <th>POPPCT_URBAN</th>
      <th>POPPCT_UC</th>
      <th>POPDEN_URBAN</th>
      <th>POPDEN_UC</th>
      <th>POPDEN_RURAL</th>
      <th>AREAPCT_URBAN</th>
      <th>AREAPCT_UC</th>
      <th>PCT_LAND</th>
      <th>PCT_WATER_TOT</th>
      <th>PCT_WATER_INLAND</th>
    </tr>
    <tr>
      <th>OBS</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2011</td>
      <td>7</td>
      <td>Minnesota</td>
      <td>MN</td>
      <td>MRO</td>
      <td>East North Central</td>
      <td>-0.3</td>
      <td>normal</td>
      <td>2011-07-01 00:00:00</td>
      <td>17:00:00</td>
      <td>...</td>
      <td>73.27</td>
      <td>15.28</td>
      <td>2279</td>
      <td>1700.5</td>
      <td>18.2</td>
      <td>2.14</td>
      <td>0.6</td>
      <td>91.592666</td>
      <td>8.407334</td>
      <td>5.478743</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2014</td>
      <td>5</td>
      <td>Minnesota</td>
      <td>MN</td>
      <td>MRO</td>
      <td>East North Central</td>
      <td>-0.1</td>
      <td>normal</td>
      <td>2014-05-11 00:00:00</td>
      <td>18:38:00</td>
      <td>...</td>
      <td>73.27</td>
      <td>15.28</td>
      <td>2279</td>
      <td>1700.5</td>
      <td>18.2</td>
      <td>2.14</td>
      <td>0.6</td>
      <td>91.592666</td>
      <td>8.407334</td>
      <td>5.478743</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2010</td>
      <td>10</td>
      <td>Minnesota</td>
      <td>MN</td>
      <td>MRO</td>
      <td>East North Central</td>
      <td>-1.5</td>
      <td>cold</td>
      <td>2010-10-26 00:00:00</td>
      <td>20:00:00</td>
      <td>...</td>
      <td>73.27</td>
      <td>15.28</td>
      <td>2279</td>
      <td>1700.5</td>
      <td>18.2</td>
      <td>2.14</td>
      <td>0.6</td>
      <td>91.592666</td>
      <td>8.407334</td>
      <td>5.478743</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2012</td>
      <td>6</td>
      <td>Minnesota</td>
      <td>MN</td>
      <td>MRO</td>
      <td>East North Central</td>
      <td>-0.1</td>
      <td>normal</td>
      <td>2012-06-19 00:00:00</td>
      <td>04:30:00</td>
      <td>...</td>
      <td>73.27</td>
      <td>15.28</td>
      <td>2279</td>
      <td>1700.5</td>
      <td>18.2</td>
      <td>2.14</td>
      <td>0.6</td>
      <td>91.592666</td>
      <td>8.407334</td>
      <td>5.478743</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2015</td>
      <td>7</td>
      <td>Minnesota</td>
      <td>MN</td>
      <td>MRO</td>
      <td>East North Central</td>
      <td>1.2</td>
      <td>warm</td>
      <td>2015-07-18 00:00:00</td>
      <td>02:00:00</td>
      <td>...</td>
      <td>73.27</td>
      <td>15.28</td>
      <td>2279</td>
      <td>1700.5</td>
      <td>18.2</td>
      <td>2.14</td>
      <td>0.6</td>
      <td>91.592666</td>
      <td>8.407334</td>
      <td>5.478743</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1530</th>
      <td>2011</td>
      <td>12</td>
      <td>North Dakota</td>
      <td>ND</td>
      <td>MRO</td>
      <td>West North Central</td>
      <td>-0.9</td>
      <td>cold</td>
      <td>2011-12-06 00:00:00</td>
      <td>08:00:00</td>
      <td>...</td>
      <td>59.9</td>
      <td>19.9</td>
      <td>2192.2</td>
      <td>1868.2</td>
      <td>3.9</td>
      <td>0.27</td>
      <td>0.1</td>
      <td>97.599649</td>
      <td>2.401765</td>
      <td>2.401765</td>
    </tr>
    <tr>
      <th>1531</th>
      <td>2006</td>
      <td>NaN</td>
      <td>North Dakota</td>
      <td>ND</td>
      <td>MRO</td>
      <td>West North Central</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>59.9</td>
      <td>19.9</td>
      <td>2192.2</td>
      <td>1868.2</td>
      <td>3.9</td>
      <td>0.27</td>
      <td>0.1</td>
      <td>97.599649</td>
      <td>2.401765</td>
      <td>2.401765</td>
    </tr>
    <tr>
      <th>1532</th>
      <td>2009</td>
      <td>8</td>
      <td>South Dakota</td>
      <td>SD</td>
      <td>RFC</td>
      <td>West North Central</td>
      <td>0.5</td>
      <td>warm</td>
      <td>2009-08-29 00:00:00</td>
      <td>22:54:00</td>
      <td>...</td>
      <td>56.65</td>
      <td>26.73</td>
      <td>2038.3</td>
      <td>1905.4</td>
      <td>4.7</td>
      <td>0.3</td>
      <td>0.15</td>
      <td>98.307744</td>
      <td>1.692256</td>
      <td>1.692256</td>
    </tr>
    <tr>
      <th>1533</th>
      <td>2009</td>
      <td>8</td>
      <td>South Dakota</td>
      <td>SD</td>
      <td>MRO</td>
      <td>West North Central</td>
      <td>0.5</td>
      <td>warm</td>
      <td>2009-08-29 00:00:00</td>
      <td>11:00:00</td>
      <td>...</td>
      <td>56.65</td>
      <td>26.73</td>
      <td>2038.3</td>
      <td>1905.4</td>
      <td>4.7</td>
      <td>0.3</td>
      <td>0.15</td>
      <td>98.307744</td>
      <td>1.692256</td>
      <td>1.692256</td>
    </tr>
    <tr>
      <th>1534</th>
      <td>2000</td>
      <td>NaN</td>
      <td>Alaska</td>
      <td>AK</td>
      <td>ASCC</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>66.02</td>
      <td>21.56</td>
      <td>1802.6</td>
      <td>1276</td>
      <td>0.4</td>
      <td>0.05</td>
      <td>0.02</td>
      <td>85.761154</td>
      <td>14.238846</td>
      <td>2.901182</td>
    </tr>
  </tbody>
</table>
<p>1534 rows × 55 columns</p>
</div>



Looking at the data, I want to investigate **which factors affect the duration and intensity of a power outage, and if we can preemptively predict and detect large scale power outages in the United States**.

## Step 2: Data Cleaning and Exploratory Data Analysis

Since the raw DataFrame has multiple irrelevant columns, I eliminate most and retain the ones relevant for the purpose of this project, replacing unavailable values whenever encountered.


```python
cols_to_retain = [
    "YEAR", "MONTH", "U.S._STATE", "NERC.REGION", "CLIMATE.REGION",
    "ANOMALY.LEVEL", "CLIMATE.CATEGORY", "OUTAGE.START.DATE", "OUTAGE.START.TIME",
    "OUTAGE.RESTORATION.DATE", "OUTAGE.RESTORATION.TIME", "CAUSE.CATEGORY",
    "CAUSE.CATEGORY.DETAIL", "HURRICANE.NAMES", "OUTAGE.DURATION",
    "CUSTOMERS.AFFECTED", "POPULATION", "POPPCT_URBAN", "POPDEN_URBAN",
    "POPDEN_RURAL", "TOTAL.CUSTOMERS", "PC.REALGSP.STATE"
]
cleaned_df = raw_df[cols_to_retain].copy()
cleaned_df = cleaned_df.replace('NA', np.nan)
cleaned_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>YEAR</th>
      <th>MONTH</th>
      <th>U.S._STATE</th>
      <th>NERC.REGION</th>
      <th>CLIMATE.REGION</th>
      <th>ANOMALY.LEVEL</th>
      <th>CLIMATE.CATEGORY</th>
      <th>OUTAGE.START.DATE</th>
      <th>OUTAGE.START.TIME</th>
      <th>OUTAGE.RESTORATION.DATE</th>
      <th>...</th>
      <th>CAUSE.CATEGORY.DETAIL</th>
      <th>HURRICANE.NAMES</th>
      <th>OUTAGE.DURATION</th>
      <th>CUSTOMERS.AFFECTED</th>
      <th>POPULATION</th>
      <th>POPPCT_URBAN</th>
      <th>POPDEN_URBAN</th>
      <th>POPDEN_RURAL</th>
      <th>TOTAL.CUSTOMERS</th>
      <th>PC.REALGSP.STATE</th>
    </tr>
    <tr>
      <th>OBS</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2011</td>
      <td>7</td>
      <td>Minnesota</td>
      <td>MRO</td>
      <td>East North Central</td>
      <td>-0.3</td>
      <td>normal</td>
      <td>2011-07-01 00:00:00</td>
      <td>17:00:00</td>
      <td>2011-07-03 00:00:00</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3060</td>
      <td>70000</td>
      <td>5348119</td>
      <td>73.27</td>
      <td>2279</td>
      <td>18.2</td>
      <td>2595696</td>
      <td>51268</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2014</td>
      <td>5</td>
      <td>Minnesota</td>
      <td>MRO</td>
      <td>East North Central</td>
      <td>-0.1</td>
      <td>normal</td>
      <td>2014-05-11 00:00:00</td>
      <td>18:38:00</td>
      <td>2014-05-11 00:00:00</td>
      <td>...</td>
      <td>vandalism</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>5457125</td>
      <td>73.27</td>
      <td>2279</td>
      <td>18.2</td>
      <td>2640737</td>
      <td>53499</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2010</td>
      <td>10</td>
      <td>Minnesota</td>
      <td>MRO</td>
      <td>East North Central</td>
      <td>-1.5</td>
      <td>cold</td>
      <td>2010-10-26 00:00:00</td>
      <td>20:00:00</td>
      <td>2010-10-28 00:00:00</td>
      <td>...</td>
      <td>heavy wind</td>
      <td>NaN</td>
      <td>3000</td>
      <td>70000</td>
      <td>5310903</td>
      <td>73.27</td>
      <td>2279</td>
      <td>18.2</td>
      <td>2586905</td>
      <td>50447</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2012</td>
      <td>6</td>
      <td>Minnesota</td>
      <td>MRO</td>
      <td>East North Central</td>
      <td>-0.1</td>
      <td>normal</td>
      <td>2012-06-19 00:00:00</td>
      <td>04:30:00</td>
      <td>2012-06-20 00:00:00</td>
      <td>...</td>
      <td>thunderstorm</td>
      <td>NaN</td>
      <td>2550</td>
      <td>68200</td>
      <td>5380443</td>
      <td>73.27</td>
      <td>2279</td>
      <td>18.2</td>
      <td>2606813</td>
      <td>51598</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2015</td>
      <td>7</td>
      <td>Minnesota</td>
      <td>MRO</td>
      <td>East North Central</td>
      <td>1.2</td>
      <td>warm</td>
      <td>2015-07-18 00:00:00</td>
      <td>02:00:00</td>
      <td>2015-07-19 00:00:00</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1740</td>
      <td>250000</td>
      <td>5489594</td>
      <td>73.27</td>
      <td>2279</td>
      <td>18.2</td>
      <td>2673531</td>
      <td>54431</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1530</th>
      <td>2011</td>
      <td>12</td>
      <td>North Dakota</td>
      <td>MRO</td>
      <td>West North Central</td>
      <td>-0.9</td>
      <td>cold</td>
      <td>2011-12-06 00:00:00</td>
      <td>08:00:00</td>
      <td>2011-12-06 00:00:00</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>720</td>
      <td>34500</td>
      <td>685326</td>
      <td>59.9</td>
      <td>2192.2</td>
      <td>3.9</td>
      <td>394394</td>
      <td>57012</td>
    </tr>
    <tr>
      <th>1531</th>
      <td>2006</td>
      <td>NaN</td>
      <td>North Dakota</td>
      <td>MRO</td>
      <td>West North Central</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>Coal</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>649422</td>
      <td>59.9</td>
      <td>2192.2</td>
      <td>3.9</td>
      <td>366037</td>
      <td>42913</td>
    </tr>
    <tr>
      <th>1532</th>
      <td>2009</td>
      <td>8</td>
      <td>South Dakota</td>
      <td>RFC</td>
      <td>West North Central</td>
      <td>0.5</td>
      <td>warm</td>
      <td>2009-08-29 00:00:00</td>
      <td>22:54:00</td>
      <td>2009-08-29 00:00:00</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>59</td>
      <td>NaN</td>
      <td>807067</td>
      <td>56.65</td>
      <td>2038.3</td>
      <td>4.7</td>
      <td>436229</td>
      <td>45230</td>
    </tr>
    <tr>
      <th>1533</th>
      <td>2009</td>
      <td>8</td>
      <td>South Dakota</td>
      <td>MRO</td>
      <td>West North Central</td>
      <td>0.5</td>
      <td>warm</td>
      <td>2009-08-29 00:00:00</td>
      <td>11:00:00</td>
      <td>2009-08-29 00:00:00</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>181</td>
      <td>NaN</td>
      <td>807067</td>
      <td>56.65</td>
      <td>2038.3</td>
      <td>4.7</td>
      <td>436229</td>
      <td>45230</td>
    </tr>
    <tr>
      <th>1534</th>
      <td>2000</td>
      <td>NaN</td>
      <td>Alaska</td>
      <td>ASCC</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>failure</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>14273</td>
      <td>627963</td>
      <td>66.02</td>
      <td>1802.6</td>
      <td>0.4</td>
      <td>273530</td>
      <td>57401</td>
    </tr>
  </tbody>
</table>
<p>1534 rows × 22 columns</p>
</div>



Next, I convert certain relevant columns to numeric data, and the date columns to datetime objects.


```python
cleaned_df["OUTAGE.DURATION"] = pd.to_numeric(cleaned_df["OUTAGE.DURATION"], errors="coerce")
cleaned_df["CUSTOMERS.AFFECTED"] = pd.to_numeric(cleaned_df["CUSTOMERS.AFFECTED"], errors="coerce")
cleaned_df["OUTAGE.START.DATE"] = pd.to_datetime(cleaned_df["OUTAGE.START.DATE"], errors="coerce")
cleaned_df["OUTAGE.RESTORATION.DATE"] = pd.to_datetime(cleaned_df["OUTAGE.RESTORATION.DATE"], 
                                                       errors="coerce")
cleaned_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>YEAR</th>
      <th>MONTH</th>
      <th>U.S._STATE</th>
      <th>NERC.REGION</th>
      <th>CLIMATE.REGION</th>
      <th>ANOMALY.LEVEL</th>
      <th>CLIMATE.CATEGORY</th>
      <th>OUTAGE.START.DATE</th>
      <th>OUTAGE.START.TIME</th>
      <th>OUTAGE.RESTORATION.DATE</th>
      <th>...</th>
      <th>CAUSE.CATEGORY.DETAIL</th>
      <th>HURRICANE.NAMES</th>
      <th>OUTAGE.DURATION</th>
      <th>CUSTOMERS.AFFECTED</th>
      <th>POPULATION</th>
      <th>POPPCT_URBAN</th>
      <th>POPDEN_URBAN</th>
      <th>POPDEN_RURAL</th>
      <th>TOTAL.CUSTOMERS</th>
      <th>PC.REALGSP.STATE</th>
    </tr>
    <tr>
      <th>OBS</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2011</td>
      <td>7</td>
      <td>Minnesota</td>
      <td>MRO</td>
      <td>East North Central</td>
      <td>-0.3</td>
      <td>normal</td>
      <td>2011-07-01</td>
      <td>17:00:00</td>
      <td>2011-07-03</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3060.0</td>
      <td>70000.0</td>
      <td>5348119</td>
      <td>73.27</td>
      <td>2279</td>
      <td>18.2</td>
      <td>2595696</td>
      <td>51268</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2014</td>
      <td>5</td>
      <td>Minnesota</td>
      <td>MRO</td>
      <td>East North Central</td>
      <td>-0.1</td>
      <td>normal</td>
      <td>2014-05-11</td>
      <td>18:38:00</td>
      <td>2014-05-11</td>
      <td>...</td>
      <td>vandalism</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>5457125</td>
      <td>73.27</td>
      <td>2279</td>
      <td>18.2</td>
      <td>2640737</td>
      <td>53499</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2010</td>
      <td>10</td>
      <td>Minnesota</td>
      <td>MRO</td>
      <td>East North Central</td>
      <td>-1.5</td>
      <td>cold</td>
      <td>2010-10-26</td>
      <td>20:00:00</td>
      <td>2010-10-28</td>
      <td>...</td>
      <td>heavy wind</td>
      <td>NaN</td>
      <td>3000.0</td>
      <td>70000.0</td>
      <td>5310903</td>
      <td>73.27</td>
      <td>2279</td>
      <td>18.2</td>
      <td>2586905</td>
      <td>50447</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2012</td>
      <td>6</td>
      <td>Minnesota</td>
      <td>MRO</td>
      <td>East North Central</td>
      <td>-0.1</td>
      <td>normal</td>
      <td>2012-06-19</td>
      <td>04:30:00</td>
      <td>2012-06-20</td>
      <td>...</td>
      <td>thunderstorm</td>
      <td>NaN</td>
      <td>2550.0</td>
      <td>68200.0</td>
      <td>5380443</td>
      <td>73.27</td>
      <td>2279</td>
      <td>18.2</td>
      <td>2606813</td>
      <td>51598</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2015</td>
      <td>7</td>
      <td>Minnesota</td>
      <td>MRO</td>
      <td>East North Central</td>
      <td>1.2</td>
      <td>warm</td>
      <td>2015-07-18</td>
      <td>02:00:00</td>
      <td>2015-07-19</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1740.0</td>
      <td>250000.0</td>
      <td>5489594</td>
      <td>73.27</td>
      <td>2279</td>
      <td>18.2</td>
      <td>2673531</td>
      <td>54431</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1530</th>
      <td>2011</td>
      <td>12</td>
      <td>North Dakota</td>
      <td>MRO</td>
      <td>West North Central</td>
      <td>-0.9</td>
      <td>cold</td>
      <td>2011-12-06</td>
      <td>08:00:00</td>
      <td>2011-12-06</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>720.0</td>
      <td>34500.0</td>
      <td>685326</td>
      <td>59.9</td>
      <td>2192.2</td>
      <td>3.9</td>
      <td>394394</td>
      <td>57012</td>
    </tr>
    <tr>
      <th>1531</th>
      <td>2006</td>
      <td>NaN</td>
      <td>North Dakota</td>
      <td>MRO</td>
      <td>West North Central</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>...</td>
      <td>Coal</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>649422</td>
      <td>59.9</td>
      <td>2192.2</td>
      <td>3.9</td>
      <td>366037</td>
      <td>42913</td>
    </tr>
    <tr>
      <th>1532</th>
      <td>2009</td>
      <td>8</td>
      <td>South Dakota</td>
      <td>RFC</td>
      <td>West North Central</td>
      <td>0.5</td>
      <td>warm</td>
      <td>2009-08-29</td>
      <td>22:54:00</td>
      <td>2009-08-29</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>59.0</td>
      <td>NaN</td>
      <td>807067</td>
      <td>56.65</td>
      <td>2038.3</td>
      <td>4.7</td>
      <td>436229</td>
      <td>45230</td>
    </tr>
    <tr>
      <th>1533</th>
      <td>2009</td>
      <td>8</td>
      <td>South Dakota</td>
      <td>MRO</td>
      <td>West North Central</td>
      <td>0.5</td>
      <td>warm</td>
      <td>2009-08-29</td>
      <td>11:00:00</td>
      <td>2009-08-29</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>181.0</td>
      <td>NaN</td>
      <td>807067</td>
      <td>56.65</td>
      <td>2038.3</td>
      <td>4.7</td>
      <td>436229</td>
      <td>45230</td>
    </tr>
    <tr>
      <th>1534</th>
      <td>2000</td>
      <td>NaN</td>
      <td>Alaska</td>
      <td>ASCC</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>...</td>
      <td>failure</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>14273.0</td>
      <td>627963</td>
      <td>66.02</td>
      <td>1802.6</td>
      <td>0.4</td>
      <td>273530</td>
      <td>57401</td>
    </tr>
  </tbody>
</table>
<p>1534 rows × 22 columns</p>
</div>



### Feature Engineering



Based on the existing columns of the given data, I engineered additional relevant and related columns to work with in subsequent steps of the project.


```python
cleaned_df["OUTAGE_SEASON"] = cleaned_df["OUTAGE.START.DATE"].dt.month % 12 // 3 + 1
season_map = {1: "Winter", 2: "Spring", 3: "Summer", 4: "Fall"}
cleaned_df["OUTAGE_SEASON"] = cleaned_df["OUTAGE_SEASON"].map(season_map)
cleaned_df["OUTAGE_DAYOFWEEK"] = cleaned_df["OUTAGE.START.DATE"].dt.dayofweek
cleaned_df["IS_WEEKEND"] = cleaned_df["OUTAGE_DAYOFWEEK"].isin([5, 6])
cleaned_df["CUSTOMER_DENSITY"] = cleaned_df["TOTAL.CUSTOMERS"] / cleaned_df["POPULATION"]
cleaned_df["URBANIZATION_RATIO"] = cleaned_df["POPPCT_URBAN"] / 100
cleaned_df["POPULATION_DENSITY"] = cleaned_df["POPDEN_URBAN"] * cleaned_df["URBANIZATION_RATIO"] 
                        + cleaned_df["POPDEN_RURAL"] * (1 - cleaned_df["URBANIZATION_RATIO"])
cleaned_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>YEAR</th>
      <th>MONTH</th>
      <th>U.S._STATE</th>
      <th>NERC.REGION</th>
      <th>CLIMATE.REGION</th>
      <th>ANOMALY.LEVEL</th>
      <th>CLIMATE.CATEGORY</th>
      <th>OUTAGE.START.DATE</th>
      <th>OUTAGE.START.TIME</th>
      <th>OUTAGE.RESTORATION.DATE</th>
      <th>...</th>
      <th>POPDEN_URBAN</th>
      <th>POPDEN_RURAL</th>
      <th>TOTAL.CUSTOMERS</th>
      <th>PC.REALGSP.STATE</th>
      <th>OUTAGE_SEASON</th>
      <th>OUTAGE_DAYOFWEEK</th>
      <th>IS_WEEKEND</th>
      <th>CUSTOMER_DENSITY</th>
      <th>URBANIZATION_RATIO</th>
      <th>POPULATION_DENSITY</th>
    </tr>
    <tr>
      <th>OBS</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2011</td>
      <td>7</td>
      <td>Minnesota</td>
      <td>MRO</td>
      <td>East North Central</td>
      <td>-0.3</td>
      <td>normal</td>
      <td>2011-07-01</td>
      <td>17:00:00</td>
      <td>2011-07-03</td>
      <td>...</td>
      <td>2279</td>
      <td>18.2</td>
      <td>2595696</td>
      <td>51268</td>
      <td>Summer</td>
      <td>4.0</td>
      <td>False</td>
      <td>0.485347</td>
      <td>0.7327</td>
      <td>1674.68816</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2014</td>
      <td>5</td>
      <td>Minnesota</td>
      <td>MRO</td>
      <td>East North Central</td>
      <td>-0.1</td>
      <td>normal</td>
      <td>2014-05-11</td>
      <td>18:38:00</td>
      <td>2014-05-11</td>
      <td>...</td>
      <td>2279</td>
      <td>18.2</td>
      <td>2640737</td>
      <td>53499</td>
      <td>Spring</td>
      <td>6.0</td>
      <td>True</td>
      <td>0.483906</td>
      <td>0.7327</td>
      <td>1674.68816</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2010</td>
      <td>10</td>
      <td>Minnesota</td>
      <td>MRO</td>
      <td>East North Central</td>
      <td>-1.5</td>
      <td>cold</td>
      <td>2010-10-26</td>
      <td>20:00:00</td>
      <td>2010-10-28</td>
      <td>...</td>
      <td>2279</td>
      <td>18.2</td>
      <td>2586905</td>
      <td>50447</td>
      <td>Fall</td>
      <td>1.0</td>
      <td>False</td>
      <td>0.487093</td>
      <td>0.7327</td>
      <td>1674.68816</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2012</td>
      <td>6</td>
      <td>Minnesota</td>
      <td>MRO</td>
      <td>East North Central</td>
      <td>-0.1</td>
      <td>normal</td>
      <td>2012-06-19</td>
      <td>04:30:00</td>
      <td>2012-06-20</td>
      <td>...</td>
      <td>2279</td>
      <td>18.2</td>
      <td>2606813</td>
      <td>51598</td>
      <td>Summer</td>
      <td>1.0</td>
      <td>False</td>
      <td>0.484498</td>
      <td>0.7327</td>
      <td>1674.68816</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2015</td>
      <td>7</td>
      <td>Minnesota</td>
      <td>MRO</td>
      <td>East North Central</td>
      <td>1.2</td>
      <td>warm</td>
      <td>2015-07-18</td>
      <td>02:00:00</td>
      <td>2015-07-19</td>
      <td>...</td>
      <td>2279</td>
      <td>18.2</td>
      <td>2673531</td>
      <td>54431</td>
      <td>Summer</td>
      <td>5.0</td>
      <td>True</td>
      <td>0.487018</td>
      <td>0.7327</td>
      <td>1674.68816</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1530</th>
      <td>2011</td>
      <td>12</td>
      <td>North Dakota</td>
      <td>MRO</td>
      <td>West North Central</td>
      <td>-0.9</td>
      <td>cold</td>
      <td>2011-12-06</td>
      <td>08:00:00</td>
      <td>2011-12-06</td>
      <td>...</td>
      <td>2192.2</td>
      <td>3.9</td>
      <td>394394</td>
      <td>57012</td>
      <td>Winter</td>
      <td>1.0</td>
      <td>False</td>
      <td>0.575484</td>
      <td>0.599</td>
      <td>1314.6917</td>
    </tr>
    <tr>
      <th>1531</th>
      <td>2006</td>
      <td>NaN</td>
      <td>North Dakota</td>
      <td>MRO</td>
      <td>West North Central</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>...</td>
      <td>2192.2</td>
      <td>3.9</td>
      <td>366037</td>
      <td>42913</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.563635</td>
      <td>0.599</td>
      <td>1314.6917</td>
    </tr>
    <tr>
      <th>1532</th>
      <td>2009</td>
      <td>8</td>
      <td>South Dakota</td>
      <td>RFC</td>
      <td>West North Central</td>
      <td>0.5</td>
      <td>warm</td>
      <td>2009-08-29</td>
      <td>22:54:00</td>
      <td>2009-08-29</td>
      <td>...</td>
      <td>2038.3</td>
      <td>4.7</td>
      <td>436229</td>
      <td>45230</td>
      <td>Summer</td>
      <td>5.0</td>
      <td>True</td>
      <td>0.540512</td>
      <td>0.5665</td>
      <td>1156.7344</td>
    </tr>
    <tr>
      <th>1533</th>
      <td>2009</td>
      <td>8</td>
      <td>South Dakota</td>
      <td>MRO</td>
      <td>West North Central</td>
      <td>0.5</td>
      <td>warm</td>
      <td>2009-08-29</td>
      <td>11:00:00</td>
      <td>2009-08-29</td>
      <td>...</td>
      <td>2038.3</td>
      <td>4.7</td>
      <td>436229</td>
      <td>45230</td>
      <td>Summer</td>
      <td>5.0</td>
      <td>True</td>
      <td>0.540512</td>
      <td>0.5665</td>
      <td>1156.7344</td>
    </tr>
    <tr>
      <th>1534</th>
      <td>2000</td>
      <td>NaN</td>
      <td>Alaska</td>
      <td>ASCC</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>...</td>
      <td>1802.6</td>
      <td>0.4</td>
      <td>273530</td>
      <td>57401</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.435583</td>
      <td>0.6602</td>
      <td>1190.21244</td>
    </tr>
  </tbody>
</table>
<p>1534 rows × 28 columns</p>
</div>




```python
def categorize_severity(customers_affected):
    if pd.isna(customers_affected) or customers_affected == 0:
        return "Unknown/Minor"
    elif customers_affected < 10000:
        return "Small"
    elif customers_affected < 100000:
        return "Medium"
    else:
        return "Large"

cleaned_df["SEVERITY_CATEGORY"] = cleaned_df["CUSTOMERS.AFFECTED"].apply(categorize_severity)
cleaned_df["IS_EXTREME_WEATHER"] = cleaned_df["CAUSE.CATEGORY"].isin(["severe weather", 
                                                                      "hurricanes", "winter storms", "tornadoes"])
cleaned_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>YEAR</th>
      <th>MONTH</th>
      <th>U.S._STATE</th>
      <th>NERC.REGION</th>
      <th>CLIMATE.REGION</th>
      <th>ANOMALY.LEVEL</th>
      <th>CLIMATE.CATEGORY</th>
      <th>OUTAGE.START.DATE</th>
      <th>OUTAGE.START.TIME</th>
      <th>OUTAGE.RESTORATION.DATE</th>
      <th>...</th>
      <th>TOTAL.CUSTOMERS</th>
      <th>PC.REALGSP.STATE</th>
      <th>OUTAGE_SEASON</th>
      <th>OUTAGE_DAYOFWEEK</th>
      <th>IS_WEEKEND</th>
      <th>CUSTOMER_DENSITY</th>
      <th>URBANIZATION_RATIO</th>
      <th>POPULATION_DENSITY</th>
      <th>SEVERITY_CATEGORY</th>
      <th>IS_EXTREME_WEATHER</th>
    </tr>
    <tr>
      <th>OBS</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2011</td>
      <td>7</td>
      <td>Minnesota</td>
      <td>MRO</td>
      <td>East North Central</td>
      <td>-0.3</td>
      <td>normal</td>
      <td>2011-07-01</td>
      <td>17:00:00</td>
      <td>2011-07-03</td>
      <td>...</td>
      <td>2595696</td>
      <td>51268</td>
      <td>Summer</td>
      <td>4.0</td>
      <td>False</td>
      <td>0.485347</td>
      <td>0.7327</td>
      <td>1674.68816</td>
      <td>Medium</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2014</td>
      <td>5</td>
      <td>Minnesota</td>
      <td>MRO</td>
      <td>East North Central</td>
      <td>-0.1</td>
      <td>normal</td>
      <td>2014-05-11</td>
      <td>18:38:00</td>
      <td>2014-05-11</td>
      <td>...</td>
      <td>2640737</td>
      <td>53499</td>
      <td>Spring</td>
      <td>6.0</td>
      <td>True</td>
      <td>0.483906</td>
      <td>0.7327</td>
      <td>1674.68816</td>
      <td>Unknown/Minor</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2010</td>
      <td>10</td>
      <td>Minnesota</td>
      <td>MRO</td>
      <td>East North Central</td>
      <td>-1.5</td>
      <td>cold</td>
      <td>2010-10-26</td>
      <td>20:00:00</td>
      <td>2010-10-28</td>
      <td>...</td>
      <td>2586905</td>
      <td>50447</td>
      <td>Fall</td>
      <td>1.0</td>
      <td>False</td>
      <td>0.487093</td>
      <td>0.7327</td>
      <td>1674.68816</td>
      <td>Medium</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2012</td>
      <td>6</td>
      <td>Minnesota</td>
      <td>MRO</td>
      <td>East North Central</td>
      <td>-0.1</td>
      <td>normal</td>
      <td>2012-06-19</td>
      <td>04:30:00</td>
      <td>2012-06-20</td>
      <td>...</td>
      <td>2606813</td>
      <td>51598</td>
      <td>Summer</td>
      <td>1.0</td>
      <td>False</td>
      <td>0.484498</td>
      <td>0.7327</td>
      <td>1674.68816</td>
      <td>Medium</td>
      <td>True</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2015</td>
      <td>7</td>
      <td>Minnesota</td>
      <td>MRO</td>
      <td>East North Central</td>
      <td>1.2</td>
      <td>warm</td>
      <td>2015-07-18</td>
      <td>02:00:00</td>
      <td>2015-07-19</td>
      <td>...</td>
      <td>2673531</td>
      <td>54431</td>
      <td>Summer</td>
      <td>5.0</td>
      <td>True</td>
      <td>0.487018</td>
      <td>0.7327</td>
      <td>1674.68816</td>
      <td>Large</td>
      <td>True</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1530</th>
      <td>2011</td>
      <td>12</td>
      <td>North Dakota</td>
      <td>MRO</td>
      <td>West North Central</td>
      <td>-0.9</td>
      <td>cold</td>
      <td>2011-12-06</td>
      <td>08:00:00</td>
      <td>2011-12-06</td>
      <td>...</td>
      <td>394394</td>
      <td>57012</td>
      <td>Winter</td>
      <td>1.0</td>
      <td>False</td>
      <td>0.575484</td>
      <td>0.599</td>
      <td>1314.6917</td>
      <td>Medium</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1531</th>
      <td>2006</td>
      <td>NaN</td>
      <td>North Dakota</td>
      <td>MRO</td>
      <td>West North Central</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>...</td>
      <td>366037</td>
      <td>42913</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.563635</td>
      <td>0.599</td>
      <td>1314.6917</td>
      <td>Unknown/Minor</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1532</th>
      <td>2009</td>
      <td>8</td>
      <td>South Dakota</td>
      <td>RFC</td>
      <td>West North Central</td>
      <td>0.5</td>
      <td>warm</td>
      <td>2009-08-29</td>
      <td>22:54:00</td>
      <td>2009-08-29</td>
      <td>...</td>
      <td>436229</td>
      <td>45230</td>
      <td>Summer</td>
      <td>5.0</td>
      <td>True</td>
      <td>0.540512</td>
      <td>0.5665</td>
      <td>1156.7344</td>
      <td>Unknown/Minor</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1533</th>
      <td>2009</td>
      <td>8</td>
      <td>South Dakota</td>
      <td>MRO</td>
      <td>West North Central</td>
      <td>0.5</td>
      <td>warm</td>
      <td>2009-08-29</td>
      <td>11:00:00</td>
      <td>2009-08-29</td>
      <td>...</td>
      <td>436229</td>
      <td>45230</td>
      <td>Summer</td>
      <td>5.0</td>
      <td>True</td>
      <td>0.540512</td>
      <td>0.5665</td>
      <td>1156.7344</td>
      <td>Unknown/Minor</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1534</th>
      <td>2000</td>
      <td>NaN</td>
      <td>Alaska</td>
      <td>ASCC</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>...</td>
      <td>273530</td>
      <td>57401</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.435583</td>
      <td>0.6602</td>
      <td>1190.21244</td>
      <td>Medium</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>1534 rows × 30 columns</p>
</div>



I also applied a logarithmic transformation using 1 plus the input value, to account for the fact that the logarithm of 0 is undefined.


```python
cleaned_df["LOG_DURATION"] = np.log1p(cleaned_df["OUTAGE.DURATION"].fillna(0))
cleaned_df["LOG_CUSTOMERS_AFFECTED"] = np.log1p(cleaned_df["CUSTOMERS.AFFECTED"].fillna(0))
cleaned_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>YEAR</th>
      <th>MONTH</th>
      <th>U.S._STATE</th>
      <th>NERC.REGION</th>
      <th>CLIMATE.REGION</th>
      <th>ANOMALY.LEVEL</th>
      <th>CLIMATE.CATEGORY</th>
      <th>OUTAGE.START.DATE</th>
      <th>OUTAGE.START.TIME</th>
      <th>OUTAGE.RESTORATION.DATE</th>
      <th>...</th>
      <th>OUTAGE_SEASON</th>
      <th>OUTAGE_DAYOFWEEK</th>
      <th>IS_WEEKEND</th>
      <th>CUSTOMER_DENSITY</th>
      <th>URBANIZATION_RATIO</th>
      <th>POPULATION_DENSITY</th>
      <th>SEVERITY_CATEGORY</th>
      <th>IS_EXTREME_WEATHER</th>
      <th>LOG_DURATION</th>
      <th>LOG_CUSTOMERS_AFFECTED</th>
    </tr>
    <tr>
      <th>OBS</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2011</td>
      <td>7</td>
      <td>Minnesota</td>
      <td>MRO</td>
      <td>East North Central</td>
      <td>-0.3</td>
      <td>normal</td>
      <td>2011-07-01</td>
      <td>17:00:00</td>
      <td>2011-07-03</td>
      <td>...</td>
      <td>Summer</td>
      <td>4.0</td>
      <td>False</td>
      <td>0.485347</td>
      <td>0.7327</td>
      <td>1674.68816</td>
      <td>Medium</td>
      <td>True</td>
      <td>8.026497</td>
      <td>11.156265</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2014</td>
      <td>5</td>
      <td>Minnesota</td>
      <td>MRO</td>
      <td>East North Central</td>
      <td>-0.1</td>
      <td>normal</td>
      <td>2014-05-11</td>
      <td>18:38:00</td>
      <td>2014-05-11</td>
      <td>...</td>
      <td>Spring</td>
      <td>6.0</td>
      <td>True</td>
      <td>0.483906</td>
      <td>0.7327</td>
      <td>1674.68816</td>
      <td>Unknown/Minor</td>
      <td>False</td>
      <td>0.693147</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2010</td>
      <td>10</td>
      <td>Minnesota</td>
      <td>MRO</td>
      <td>East North Central</td>
      <td>-1.5</td>
      <td>cold</td>
      <td>2010-10-26</td>
      <td>20:00:00</td>
      <td>2010-10-28</td>
      <td>...</td>
      <td>Fall</td>
      <td>1.0</td>
      <td>False</td>
      <td>0.487093</td>
      <td>0.7327</td>
      <td>1674.68816</td>
      <td>Medium</td>
      <td>True</td>
      <td>8.006701</td>
      <td>11.156265</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2012</td>
      <td>6</td>
      <td>Minnesota</td>
      <td>MRO</td>
      <td>East North Central</td>
      <td>-0.1</td>
      <td>normal</td>
      <td>2012-06-19</td>
      <td>04:30:00</td>
      <td>2012-06-20</td>
      <td>...</td>
      <td>Summer</td>
      <td>1.0</td>
      <td>False</td>
      <td>0.484498</td>
      <td>0.7327</td>
      <td>1674.68816</td>
      <td>Medium</td>
      <td>True</td>
      <td>7.844241</td>
      <td>11.130215</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2015</td>
      <td>7</td>
      <td>Minnesota</td>
      <td>MRO</td>
      <td>East North Central</td>
      <td>1.2</td>
      <td>warm</td>
      <td>2015-07-18</td>
      <td>02:00:00</td>
      <td>2015-07-19</td>
      <td>...</td>
      <td>Summer</td>
      <td>5.0</td>
      <td>True</td>
      <td>0.487018</td>
      <td>0.7327</td>
      <td>1674.68816</td>
      <td>Large</td>
      <td>True</td>
      <td>7.462215</td>
      <td>12.429220</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1530</th>
      <td>2011</td>
      <td>12</td>
      <td>North Dakota</td>
      <td>MRO</td>
      <td>West North Central</td>
      <td>-0.9</td>
      <td>cold</td>
      <td>2011-12-06</td>
      <td>08:00:00</td>
      <td>2011-12-06</td>
      <td>...</td>
      <td>Winter</td>
      <td>1.0</td>
      <td>False</td>
      <td>0.575484</td>
      <td>0.599</td>
      <td>1314.6917</td>
      <td>Medium</td>
      <td>False</td>
      <td>6.580639</td>
      <td>10.448744</td>
    </tr>
    <tr>
      <th>1531</th>
      <td>2006</td>
      <td>NaN</td>
      <td>North Dakota</td>
      <td>MRO</td>
      <td>West North Central</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.563635</td>
      <td>0.599</td>
      <td>1314.6917</td>
      <td>Unknown/Minor</td>
      <td>False</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1532</th>
      <td>2009</td>
      <td>8</td>
      <td>South Dakota</td>
      <td>RFC</td>
      <td>West North Central</td>
      <td>0.5</td>
      <td>warm</td>
      <td>2009-08-29</td>
      <td>22:54:00</td>
      <td>2009-08-29</td>
      <td>...</td>
      <td>Summer</td>
      <td>5.0</td>
      <td>True</td>
      <td>0.540512</td>
      <td>0.5665</td>
      <td>1156.7344</td>
      <td>Unknown/Minor</td>
      <td>False</td>
      <td>4.094345</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1533</th>
      <td>2009</td>
      <td>8</td>
      <td>South Dakota</td>
      <td>MRO</td>
      <td>West North Central</td>
      <td>0.5</td>
      <td>warm</td>
      <td>2009-08-29</td>
      <td>11:00:00</td>
      <td>2009-08-29</td>
      <td>...</td>
      <td>Summer</td>
      <td>5.0</td>
      <td>True</td>
      <td>0.540512</td>
      <td>0.5665</td>
      <td>1156.7344</td>
      <td>Unknown/Minor</td>
      <td>False</td>
      <td>5.204007</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1534</th>
      <td>2000</td>
      <td>NaN</td>
      <td>Alaska</td>
      <td>ASCC</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.435583</td>
      <td>0.6602</td>
      <td>1190.21244</td>
      <td>Medium</td>
      <td>False</td>
      <td>0.000000</td>
      <td>9.566195</td>
    </tr>
  </tbody>
</table>
<p>1534 rows × 32 columns</p>
</div>



Finally, as my investigation primarily involves the number of customers affected by a power outage, I chose to fill missing values for this column with 0. Here, I use the underlying assumption that if an outage was not overly significant, the number of customers affected by it was also insignificant/not reported due to its small-scale impact.


```python
cleaned_df["CUSTOMERS.AFFECTED"] = cleaned_df["CUSTOMERS.AFFECTED"].fillna(0)
cleaned_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>YEAR</th>
      <th>MONTH</th>
      <th>U.S._STATE</th>
      <th>NERC.REGION</th>
      <th>CLIMATE.REGION</th>
      <th>ANOMALY.LEVEL</th>
      <th>CLIMATE.CATEGORY</th>
      <th>OUTAGE.START.DATE</th>
      <th>OUTAGE.START.TIME</th>
      <th>OUTAGE.RESTORATION.DATE</th>
      <th>...</th>
      <th>OUTAGE_SEASON</th>
      <th>OUTAGE_DAYOFWEEK</th>
      <th>IS_WEEKEND</th>
      <th>CUSTOMER_DENSITY</th>
      <th>URBANIZATION_RATIO</th>
      <th>POPULATION_DENSITY</th>
      <th>SEVERITY_CATEGORY</th>
      <th>IS_EXTREME_WEATHER</th>
      <th>LOG_DURATION</th>
      <th>LOG_CUSTOMERS_AFFECTED</th>
    </tr>
    <tr>
      <th>OBS</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2011</td>
      <td>7</td>
      <td>Minnesota</td>
      <td>MRO</td>
      <td>East North Central</td>
      <td>-0.3</td>
      <td>normal</td>
      <td>2011-07-01</td>
      <td>17:00:00</td>
      <td>2011-07-03</td>
      <td>...</td>
      <td>Summer</td>
      <td>4.0</td>
      <td>False</td>
      <td>0.485347</td>
      <td>0.7327</td>
      <td>1674.68816</td>
      <td>Medium</td>
      <td>True</td>
      <td>8.026497</td>
      <td>11.156265</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2014</td>
      <td>5</td>
      <td>Minnesota</td>
      <td>MRO</td>
      <td>East North Central</td>
      <td>-0.1</td>
      <td>normal</td>
      <td>2014-05-11</td>
      <td>18:38:00</td>
      <td>2014-05-11</td>
      <td>...</td>
      <td>Spring</td>
      <td>6.0</td>
      <td>True</td>
      <td>0.483906</td>
      <td>0.7327</td>
      <td>1674.68816</td>
      <td>Unknown/Minor</td>
      <td>False</td>
      <td>0.693147</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2010</td>
      <td>10</td>
      <td>Minnesota</td>
      <td>MRO</td>
      <td>East North Central</td>
      <td>-1.5</td>
      <td>cold</td>
      <td>2010-10-26</td>
      <td>20:00:00</td>
      <td>2010-10-28</td>
      <td>...</td>
      <td>Fall</td>
      <td>1.0</td>
      <td>False</td>
      <td>0.487093</td>
      <td>0.7327</td>
      <td>1674.68816</td>
      <td>Medium</td>
      <td>True</td>
      <td>8.006701</td>
      <td>11.156265</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2012</td>
      <td>6</td>
      <td>Minnesota</td>
      <td>MRO</td>
      <td>East North Central</td>
      <td>-0.1</td>
      <td>normal</td>
      <td>2012-06-19</td>
      <td>04:30:00</td>
      <td>2012-06-20</td>
      <td>...</td>
      <td>Summer</td>
      <td>1.0</td>
      <td>False</td>
      <td>0.484498</td>
      <td>0.7327</td>
      <td>1674.68816</td>
      <td>Medium</td>
      <td>True</td>
      <td>7.844241</td>
      <td>11.130215</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2015</td>
      <td>7</td>
      <td>Minnesota</td>
      <td>MRO</td>
      <td>East North Central</td>
      <td>1.2</td>
      <td>warm</td>
      <td>2015-07-18</td>
      <td>02:00:00</td>
      <td>2015-07-19</td>
      <td>...</td>
      <td>Summer</td>
      <td>5.0</td>
      <td>True</td>
      <td>0.487018</td>
      <td>0.7327</td>
      <td>1674.68816</td>
      <td>Large</td>
      <td>True</td>
      <td>7.462215</td>
      <td>12.429220</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1530</th>
      <td>2011</td>
      <td>12</td>
      <td>North Dakota</td>
      <td>MRO</td>
      <td>West North Central</td>
      <td>-0.9</td>
      <td>cold</td>
      <td>2011-12-06</td>
      <td>08:00:00</td>
      <td>2011-12-06</td>
      <td>...</td>
      <td>Winter</td>
      <td>1.0</td>
      <td>False</td>
      <td>0.575484</td>
      <td>0.599</td>
      <td>1314.6917</td>
      <td>Medium</td>
      <td>False</td>
      <td>6.580639</td>
      <td>10.448744</td>
    </tr>
    <tr>
      <th>1531</th>
      <td>2006</td>
      <td>NaN</td>
      <td>North Dakota</td>
      <td>MRO</td>
      <td>West North Central</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.563635</td>
      <td>0.599</td>
      <td>1314.6917</td>
      <td>Unknown/Minor</td>
      <td>False</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1532</th>
      <td>2009</td>
      <td>8</td>
      <td>South Dakota</td>
      <td>RFC</td>
      <td>West North Central</td>
      <td>0.5</td>
      <td>warm</td>
      <td>2009-08-29</td>
      <td>22:54:00</td>
      <td>2009-08-29</td>
      <td>...</td>
      <td>Summer</td>
      <td>5.0</td>
      <td>True</td>
      <td>0.540512</td>
      <td>0.5665</td>
      <td>1156.7344</td>
      <td>Unknown/Minor</td>
      <td>False</td>
      <td>4.094345</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1533</th>
      <td>2009</td>
      <td>8</td>
      <td>South Dakota</td>
      <td>MRO</td>
      <td>West North Central</td>
      <td>0.5</td>
      <td>warm</td>
      <td>2009-08-29</td>
      <td>11:00:00</td>
      <td>2009-08-29</td>
      <td>...</td>
      <td>Summer</td>
      <td>5.0</td>
      <td>True</td>
      <td>0.540512</td>
      <td>0.5665</td>
      <td>1156.7344</td>
      <td>Unknown/Minor</td>
      <td>False</td>
      <td>5.204007</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1534</th>
      <td>2000</td>
      <td>NaN</td>
      <td>Alaska</td>
      <td>ASCC</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.435583</td>
      <td>0.6602</td>
      <td>1190.21244</td>
      <td>Medium</td>
      <td>False</td>
      <td>0.000000</td>
      <td>9.566195</td>
    </tr>
  </tbody>
</table>
<p>1534 rows × 32 columns</p>
</div>



### Univariate Analysis

Before delving into univariate analysis, here are some summary statistics to better gauge the data I am working with:


```python
print(f"Average outage duration: {cleaned_df['OUTAGE.DURATION'].mean()}")
print(f"Median customers affected: {cleaned_df['CUSTOMERS.AFFECTED'].median()}")
print(f"Most common cause: {cleaned_df['CAUSE.CATEGORY'].mode().iloc[0]}")
print(f"Most affected season: {cleaned_df['OUTAGE_SEASON'].mode().iloc[0]}")
```

    Average outage duration: 2625.39837398374
    Median customers affected: 30534.0
    Most common cause: severe weather
    Most affected season: Summer
    

With this understanding, I plot the distribution of outage durations. To ensure better bin values, I excluded those power outages exceeding or equal to 10000 minutes as they will be outliers for this plot. Note that I only took this step of filtering outliers for this plot, and the remainder of the project has data unaltered to retain its quality.


```python
median_duration = cleaned_df["OUTAGE.DURATION"].median()
# I used the original, unfiltered DataFrame to ensure the median is correct.
fig1 = px.histogram(
    cleaned_df[cleaned_df["OUTAGE.DURATION"] < 10000],
    x="OUTAGE.DURATION",
    nbins=50,
    title="Distribution of Outage Durations",
    labels={"OUTAGE.DURATION": "Duration (minutes)", "count": "Number of Outages"},
    color_discrete_sequence=["#2E86AB"]
)

fig1.add_shape(
    type="line",
    x0=median_duration,
    x1=median_duration,
    y0=0,
    y1=cleaned_df["OUTAGE.DURATION"].value_counts().max(),
    line=dict(color="red", width=3, dash="dash")
)

fig1.add_annotation(
    x=median_duration,
    y=cleaned_df["OUTAGE.DURATION"].value_counts().max(),
    text=f"Median: {median_duration:.0f} min",
)

fig1.update_layout(
    width=900,
    height=500
)

fig1.show()
```


<iframe
    scrolling="no"
    width="920px"
    height="520"
    src="iframe_figures/figure_13.html"
    frameborder="0"
    allowfullscreen
></iframe>



Next, I plotted a pie chart depicting the severity of each power outage based on the previous feature engineering I conducted.


```python
severity_counts = cleaned_df["SEVERITY_CATEGORY"].value_counts()
fig2 = px.pie(
    values=severity_counts.values,
    names=severity_counts.index,
    title="Distribution of Outage Severity",
    color_discrete_sequence=px.colors.qualitative.Set3
)

fig2.update_layout(
    width=800,
    height=500
)

fig2.show()
```


<iframe
    scrolling="no"
    width="820px"
    height="520"
    src="iframe_figures/figure_34.html"
    frameborder="0"
    allowfullscreen
></iframe>



Finally, I attempted to uncover seasonal patterns by looking at the average number of customers affected in each seasons, and the average duration of a power outage in each season.


```python
seasonal_stats = cleaned_df.groupby("OUTAGE_SEASON").agg({
    "CUSTOMERS.AFFECTED": ["mean", "count"],
    "OUTAGE.DURATION": "mean"
}).round(2)

fig3 = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Average Customers Affected by Season", "Average Duration by Season"),
    specs=[[{"secondary_y": False}, {"secondary_y": False}]]
)
fig3.add_trace(
    go.Bar(x=seasonal_stats.index,
           y=seasonal_stats[("CUSTOMERS.AFFECTED", "mean")],
           name="Customers Affected",
           marker_color="#F18F01"),
    row=1, col=1
)
fig3.add_trace(
    go.Bar(x=seasonal_stats.index,
           y=seasonal_stats[("OUTAGE.DURATION", "mean")],
           name="Duration (min)",
           marker_color="#C73E1D"),
    row=1, col=2
)
fig3.update_layout(
    title_text="Seasonal Patterns in Power Outages",
    showlegend=False,
    width=800,
    height=500,
    margin=dict(l=50, r=50, t=80, b=50)
)

fig3.show()
```


<iframe
    scrolling="no"
    width="820px"
    height="520"
    src="iframe_figures/figure_35.html"
    frameborder="0"
    allowfullscreen
></iframe>



### Bivariate Analysis

First, I looked at a box plot representing the outage duration distibution based on the cause category, again filtering outliers exceeding 10000 minutes.


```python
fig4 = px.box(
    cleaned_df[cleaned_df["OUTAGE.DURATION"] < 10000], # filtering for outliers
    x="CAUSE.CATEGORY",
    y="OUTAGE.DURATION",
    title="Outage Duration Distribution by Cause",
    labels={"OUTAGE.DURATION": "Duration (minutes)", "CAUSE.CATEGORY": "Cause Category"}
)
fig4.update_xaxes(tickangle=45)
fig4.update_layout(
    width=800,
    height=500
)
fig4.show()
```


<iframe
    scrolling="no"
    width="820px"
    height="520"
    src="iframe_figures/figure_36.html"
    frameborder="0"
    allowfullscreen
></iframe>



Next, I looked at the spread of outage durations considering the number of customers affected. This time, I only considered data when the number of customers affected exceeded 10000, again to account for skew/outliers.


```python
subset_df = cleaned_df[cleaned_df["CUSTOMERS.AFFECTED"] > 10000]

fig5 = px.scatter(
    subset_df,
    x="CUSTOMERS.AFFECTED",
    y="OUTAGE.DURATION",
    size="CUSTOMERS.AFFECTED",
    opacity=0.6,
    title="Outage Duration vs Customers Affected",
    labels={
        "CUSTOMERS.AFFECTED": "Customers Affected",
        "OUTAGE.DURATION": "Duration (hours)"
    },
)

fig5.update_layout(
    width=800,
    height=500
)

fig5.show()
```


<iframe
    scrolling="no"
    width="820px"
    height="520"
    src="iframe_figures/figure_37.html"
    frameborder="0"
    allowfullscreen
></iframe>



### Interesting Aggregations

The first aggregation I attempted was to look at the data state-by-state, considering the mean of 3 columns as given below:


```python
state_summary = cleaned_df.groupby("U.S._STATE").agg({
    "CUSTOMERS.AFFECTED": "mean",
    "OUTAGE.DURATION": "mean",
    "POPULATION": "mean"
}).round(2)

state_summary = state_summary.sort_values(("CUSTOMERS.AFFECTED"), ascending=False)
state_summary.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CUSTOMERS.AFFECTED</th>
      <th>OUTAGE.DURATION</th>
      <th>POPULATION</th>
    </tr>
    <tr>
      <th>U.S._STATE</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Florida</th>
      <td>282938.67</td>
      <td>4094.67</td>
      <td>18095994.422222</td>
    </tr>
    <tr>
      <th>South Carolina</th>
      <td>251913.12</td>
      <td>3135.00</td>
      <td>4453812.625</td>
    </tr>
    <tr>
      <th>Illinois</th>
      <td>198025.98</td>
      <td>1602.45</td>
      <td>12761171.413043</td>
    </tr>
    <tr>
      <th>District of Columbia</th>
      <td>175238.30</td>
      <td>4303.60</td>
      <td>621908.2</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>165226.91</td>
      <td>2704.82</td>
      <td>25170665.212598</td>
    </tr>
    <tr>
      <th>Pennsylvania</th>
      <td>150796.11</td>
      <td>3811.70</td>
      <td>12678678.508772</td>
    </tr>
    <tr>
      <th>Hawaii</th>
      <td>147237.20</td>
      <td>845.40</td>
      <td>1327926.6</td>
    </tr>
    <tr>
      <th>Michigan</th>
      <td>144832.02</td>
      <td>5302.98</td>
      <td>9942253.505263</td>
    </tr>
    <tr>
      <th>New Jersey</th>
      <td>141906.31</td>
      <td>4450.91</td>
      <td>8843437.657143</td>
    </tr>
    <tr>
      <th>Virginia</th>
      <td>137313.19</td>
      <td>1051.19</td>
      <td>7920286.783784</td>
    </tr>
  </tbody>
</table>
</div>



Next, I attemped a similar aggregation for the customers affected by each cause category of power outage, considering mean and median as aggregation functions.


```python
cause_analysis = cleaned_df.groupby("CAUSE.CATEGORY").agg({
    "OUTAGE.DURATION": ["mean", "median"],
    "CUSTOMERS.AFFECTED": ["mean", "median"]
}).round(2)

cause_analysis
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">OUTAGE.DURATION</th>
      <th colspan="2" halign="left">CUSTOMERS.AFFECTED</th>
    </tr>
    <tr>
      <th></th>
      <th>mean</th>
      <th>median</th>
      <th>mean</th>
      <th>median</th>
    </tr>
    <tr>
      <th>CAUSE.CATEGORY</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>equipment failure</th>
      <td>1816.91</td>
      <td>221.0</td>
      <td>50967.78</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>fuel supply emergency</th>
      <td>13484.03</td>
      <td>3960.0</td>
      <td>0.02</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>intentional attack</th>
      <td>429.98</td>
      <td>56.0</td>
      <td>852.43</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>islanding</th>
      <td>200.55</td>
      <td>77.5</td>
      <td>4559.76</td>
      <td>280.0</td>
    </tr>
    <tr>
      <th>public appeal</th>
      <td>1468.45</td>
      <td>455.0</td>
      <td>2318.75</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>severe weather</th>
      <td>3883.99</td>
      <td>2460.0</td>
      <td>177205.94</td>
      <td>105000.0</td>
    </tr>
    <tr>
      <th>system operability disruption</th>
      <td>728.87</td>
      <td>215.0</td>
      <td>137940.79</td>
      <td>25000.0</td>
    </tr>
  </tbody>
</table>
</div>



Finally, an aggregation for the average duration of power outages for each season given a specific cause category:


```python
seas_categ_pivot = cleaned_df.pivot_table(
    values="OUTAGE.DURATION",
    index="CAUSE.CATEGORY",
    columns="OUTAGE_SEASON",
    aggfunc="mean"
)
seas_categ_pivot
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>OUTAGE_SEASON</th>
      <th>Fall</th>
      <th>Spring</th>
      <th>Summer</th>
      <th>Winter</th>
    </tr>
    <tr>
      <th>CAUSE.CATEGORY</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>equipment failure</th>
      <td>389.500000</td>
      <td>4959.764706</td>
      <td>292.809524</td>
      <td>648.000000</td>
    </tr>
    <tr>
      <th>fuel supply emergency</th>
      <td>793.666667</td>
      <td>16503.285714</td>
      <td>7626.166667</td>
      <td>18935.937500</td>
    </tr>
    <tr>
      <th>intentional attack</th>
      <td>279.683544</td>
      <td>693.219512</td>
      <td>249.709677</td>
      <td>395.351852</td>
    </tr>
    <tr>
      <th>islanding</th>
      <td>69.000000</td>
      <td>88.444444</td>
      <td>231.882353</td>
      <td>385.000000</td>
    </tr>
    <tr>
      <th>public appeal</th>
      <td>244.000000</td>
      <td>2152.285714</td>
      <td>1187.625000</td>
      <td>2592.636364</td>
    </tr>
    <tr>
      <th>severe weather</th>
      <td>5646.116883</td>
      <td>3287.683333</td>
      <td>3180.321429</td>
      <td>3869.321053</td>
    </tr>
    <tr>
      <th>system operability disruption</th>
      <td>371.466667</td>
      <td>501.243243</td>
      <td>1152.093023</td>
      <td>571.178571</td>
    </tr>
  </tbody>
</table>
</div>



## Step 3: Assessment of Missingness

I first looked at how much percentage of values the data is missing in each column, which also serves as a way to cross-validate any null replacement steps I have taken above.


```python
missingness = cleaned_df.isnull().sum().sort_values(ascending=False)
missingness_pct = (missingness / len(cleaned_df) * 100).round(2)
for col, pct in missingness_pct[missingness_pct > 0].items():
    print(f"{col}: {pct}% data missing")
```

    HURRICANE.NAMES: 95.31% data missing
    CAUSE.CATEGORY.DETAIL: 30.7% data missing
    OUTAGE.RESTORATION.TIME: 3.78% data missing
    OUTAGE.RESTORATION.DATE: 3.78% data missing
    OUTAGE.DURATION: 3.78% data missing
    POPDEN_RURAL: 0.65% data missing
    POPULATION_DENSITY: 0.65% data missing
    MONTH: 0.59% data missing
    ANOMALY.LEVEL: 0.59% data missing
    OUTAGE.START.TIME: 0.59% data missing
    CLIMATE.CATEGORY: 0.59% data missing
    OUTAGE.START.DATE: 0.59% data missing
    OUTAGE_DAYOFWEEK: 0.59% data missing
    OUTAGE_SEASON: 0.59% data missing
    CLIMATE.REGION: 0.39% data missing
    

The missing column I chose to examine was 'CAUSE.CATEGORY.DETAIL', which I essentially one-hot encoded, using an additional column.


```python
missing_col = 'CAUSE.CATEGORY.DETAIL'
cleaned_df['missing_detail'] = cleaned_df[missing_col].isna().astype(int)
cleaned_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>YEAR</th>
      <th>MONTH</th>
      <th>U.S._STATE</th>
      <th>NERC.REGION</th>
      <th>CLIMATE.REGION</th>
      <th>ANOMALY.LEVEL</th>
      <th>CLIMATE.CATEGORY</th>
      <th>OUTAGE.START.DATE</th>
      <th>OUTAGE.START.TIME</th>
      <th>OUTAGE.RESTORATION.DATE</th>
      <th>...</th>
      <th>OUTAGE_DAYOFWEEK</th>
      <th>IS_WEEKEND</th>
      <th>CUSTOMER_DENSITY</th>
      <th>URBANIZATION_RATIO</th>
      <th>POPULATION_DENSITY</th>
      <th>SEVERITY_CATEGORY</th>
      <th>IS_EXTREME_WEATHER</th>
      <th>LOG_DURATION</th>
      <th>LOG_CUSTOMERS_AFFECTED</th>
      <th>missing_detail</th>
    </tr>
    <tr>
      <th>OBS</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2011</td>
      <td>7</td>
      <td>Minnesota</td>
      <td>MRO</td>
      <td>East North Central</td>
      <td>-0.3</td>
      <td>normal</td>
      <td>2011-07-01</td>
      <td>17:00:00</td>
      <td>2011-07-03</td>
      <td>...</td>
      <td>4.0</td>
      <td>False</td>
      <td>0.485347</td>
      <td>0.7327</td>
      <td>1674.68816</td>
      <td>Medium</td>
      <td>True</td>
      <td>8.026497</td>
      <td>11.156265</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2014</td>
      <td>5</td>
      <td>Minnesota</td>
      <td>MRO</td>
      <td>East North Central</td>
      <td>-0.1</td>
      <td>normal</td>
      <td>2014-05-11</td>
      <td>18:38:00</td>
      <td>2014-05-11</td>
      <td>...</td>
      <td>6.0</td>
      <td>True</td>
      <td>0.483906</td>
      <td>0.7327</td>
      <td>1674.68816</td>
      <td>Unknown/Minor</td>
      <td>False</td>
      <td>0.693147</td>
      <td>0.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2010</td>
      <td>10</td>
      <td>Minnesota</td>
      <td>MRO</td>
      <td>East North Central</td>
      <td>-1.5</td>
      <td>cold</td>
      <td>2010-10-26</td>
      <td>20:00:00</td>
      <td>2010-10-28</td>
      <td>...</td>
      <td>1.0</td>
      <td>False</td>
      <td>0.487093</td>
      <td>0.7327</td>
      <td>1674.68816</td>
      <td>Medium</td>
      <td>True</td>
      <td>8.006701</td>
      <td>11.156265</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2012</td>
      <td>6</td>
      <td>Minnesota</td>
      <td>MRO</td>
      <td>East North Central</td>
      <td>-0.1</td>
      <td>normal</td>
      <td>2012-06-19</td>
      <td>04:30:00</td>
      <td>2012-06-20</td>
      <td>...</td>
      <td>1.0</td>
      <td>False</td>
      <td>0.484498</td>
      <td>0.7327</td>
      <td>1674.68816</td>
      <td>Medium</td>
      <td>True</td>
      <td>7.844241</td>
      <td>11.130215</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2015</td>
      <td>7</td>
      <td>Minnesota</td>
      <td>MRO</td>
      <td>East North Central</td>
      <td>1.2</td>
      <td>warm</td>
      <td>2015-07-18</td>
      <td>02:00:00</td>
      <td>2015-07-19</td>
      <td>...</td>
      <td>5.0</td>
      <td>True</td>
      <td>0.487018</td>
      <td>0.7327</td>
      <td>1674.68816</td>
      <td>Large</td>
      <td>True</td>
      <td>7.462215</td>
      <td>12.429220</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1530</th>
      <td>2011</td>
      <td>12</td>
      <td>North Dakota</td>
      <td>MRO</td>
      <td>West North Central</td>
      <td>-0.9</td>
      <td>cold</td>
      <td>2011-12-06</td>
      <td>08:00:00</td>
      <td>2011-12-06</td>
      <td>...</td>
      <td>1.0</td>
      <td>False</td>
      <td>0.575484</td>
      <td>0.599</td>
      <td>1314.6917</td>
      <td>Medium</td>
      <td>False</td>
      <td>6.580639</td>
      <td>10.448744</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1531</th>
      <td>2006</td>
      <td>NaN</td>
      <td>North Dakota</td>
      <td>MRO</td>
      <td>West North Central</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>...</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.563635</td>
      <td>0.599</td>
      <td>1314.6917</td>
      <td>Unknown/Minor</td>
      <td>False</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1532</th>
      <td>2009</td>
      <td>8</td>
      <td>South Dakota</td>
      <td>RFC</td>
      <td>West North Central</td>
      <td>0.5</td>
      <td>warm</td>
      <td>2009-08-29</td>
      <td>22:54:00</td>
      <td>2009-08-29</td>
      <td>...</td>
      <td>5.0</td>
      <td>True</td>
      <td>0.540512</td>
      <td>0.5665</td>
      <td>1156.7344</td>
      <td>Unknown/Minor</td>
      <td>False</td>
      <td>4.094345</td>
      <td>0.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1533</th>
      <td>2009</td>
      <td>8</td>
      <td>South Dakota</td>
      <td>MRO</td>
      <td>West North Central</td>
      <td>0.5</td>
      <td>warm</td>
      <td>2009-08-29</td>
      <td>11:00:00</td>
      <td>2009-08-29</td>
      <td>...</td>
      <td>5.0</td>
      <td>True</td>
      <td>0.540512</td>
      <td>0.5665</td>
      <td>1156.7344</td>
      <td>Unknown/Minor</td>
      <td>False</td>
      <td>5.204007</td>
      <td>0.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1534</th>
      <td>2000</td>
      <td>NaN</td>
      <td>Alaska</td>
      <td>ASCC</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>...</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.435583</td>
      <td>0.6602</td>
      <td>1190.21244</td>
      <td>Medium</td>
      <td>False</td>
      <td>0.000000</td>
      <td>9.566195</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1534 rows × 33 columns</p>
</div>



Subsequently, I wrote a helper function for permutation testing, which can be used to test different columns for whether or not they depend on missingness of 'CAUSE.CATEGORY.DETAIL'.


```python
def permutation_test(df, col, target_missing, n_permutations=1000):
    observed_diff = df.groupby(col)[target_missing].mean().max() - df.groupby(col)[target_missing].mean().min()
    diffs = []
    
    for _ in range(n_permutations):
        shuffled = df[target_missing].sample(frac=1, replace=False).values
        df_shuffled = df.copy()
        df_shuffled[target_missing] = shuffled
        diff = df_shuffled.groupby(col)[target_missing].mean().max() - df_shuffled.groupby(col)[target_missing].mean().min()
        diffs.append(diff)
    
    p_value = np.mean([d >= observed_diff for d in diffs])
    return observed_diff, p_value
```


```python
observed_diff, p_value = permutation_test(cleaned_df, 'CAUSE.CATEGORY', 'missing_detail')
print(f"Observed difference: {observed_diff}, p-value: {p_value}")
```

    Observed difference: 0.8851674641148325, p-value: 0.0
    

'CAUSE.CATEGORY' has a large observed difference in missingness of close to 0.89. The negligible p-value validates this, and we can safely conclude that the missingness of 'CAUSE.CATEGORY.DETAIL' **does depend** on 'CAUSE.CATEGORY', which also makes intuitive sense.


```python
observed_diff, p_value = permutation_test(cleaned_df, 'OUTAGE_DAYOFWEEK', 'missing_detail')
print(f"Observed difference: {observed_diff}, p-value: {p_value}")
```

    Observed difference: 0.12177419354838714, p-value: 0.088
    

In contrast, 'OUTAGE_DAYOFWEEK' seems to show little to no difference (approximately 0.12) in missingness when compared with 'CAUSE.CATEGORY.DETAIL', and the p-value exceeds 0.05, suggesting that missingness **does not depend** on day of the week.

## Step 4: Hypothesis Testing

For this hypothesis test, I chose to investigate whether mean outage duration is the same for severe weather and equipment failure, or if severe weather outages last longer than equipment failure outages:

**Null Hypothesis (H₀):** The mean outage duration is the same for severe weather and equipment failure outages.

**Alternative Hypothesis (H₁)**: Severe weather outages last longer, on average, than equipment failure outages.


```python
df = cleaned_df.copy()
df_sub = df[df["CAUSE.CATEGORY"].isin(["severe weather", "equipment failure"])].copy()
weather = df_sub[df_sub["CAUSE.CATEGORY"] == "severe weather"]["OUTAGE.DURATION"].dropna()
equipment = df_sub[df_sub["CAUSE.CATEGORY"] == "equipment failure"]["OUTAGE.DURATION"].dropna()
obs_diff = weather.mean() - equipment.mean()
print("Observed difference (Weather - Equipment):", obs_diff)
```

    Observed difference (Weather - Equipment): 2067.0761241446726
    

Once I computed the observed difference, I conducted a permutation test to assess whether the said difference could be attributed to randomness, or whether there was an actual association present.


```python
all_durations = np.concatenate([weather.values, equipment.values])
labels = np.array(["Weather"] * len(weather) + ["Equipment"] * len(equipment))

diffs = []
for _ in range(10000):
    shuffled_labels = np.random.permutation(labels)
    weather_mean = all_durations[shuffled_labels == "Weather"].mean()
    equip_mean = all_durations[shuffled_labels == "Equipment"].mean()
    diffs.append(weather_mean - equip_mean)

diffs = np.array(diffs)

p_value = np.mean(diffs >= obs_diff)
print("Permutation test p-value:", p_value)
```

    Permutation test p-value: 0.0
    

Since our p-value is much lesser than 0.05, we can safely **reject the null hypothesis**, and conclude that **outages due to severe weather conditions do last longer than equipment failure outages**, on average. I also added a plot for better visualization of the result of this test.


```python
plt.hist(diffs, bins=50, alpha=0.7, color="steelblue")
plt.axvline(obs_diff, color="red", linestyle="dashed", linewidth=2, label=f"Observed diff = {obs_diff}")
plt.legend()
plt.xlabel("Difference in Means (Weather - Equipment)")
plt.ylabel("Frequency")
plt.title("Permutation Test Null Distribution")
plt.show()
```


    
![png](output_62_0.png)
    


## Step 5: Framing a Prediction Problem

The goal of this project is to **predict the category of a major power outage's duration (DURATION_CLASS), classified as Short or Long**, based on factors such as location, time, cause, and demographic characteristics.

**Target Variable:** DURATION_CLASS (derived from LOG_DURATION)

## Step 6: Baseline Model

First, I dropped the column I had previously added from the missingness analysis section.


```python
processed_df = cleaned_df.drop(columns='missing_detail').copy()
processed_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>YEAR</th>
      <th>MONTH</th>
      <th>U.S._STATE</th>
      <th>NERC.REGION</th>
      <th>CLIMATE.REGION</th>
      <th>ANOMALY.LEVEL</th>
      <th>CLIMATE.CATEGORY</th>
      <th>OUTAGE.START.DATE</th>
      <th>OUTAGE.START.TIME</th>
      <th>OUTAGE.RESTORATION.DATE</th>
      <th>...</th>
      <th>OUTAGE_SEASON</th>
      <th>OUTAGE_DAYOFWEEK</th>
      <th>IS_WEEKEND</th>
      <th>CUSTOMER_DENSITY</th>
      <th>URBANIZATION_RATIO</th>
      <th>POPULATION_DENSITY</th>
      <th>SEVERITY_CATEGORY</th>
      <th>IS_EXTREME_WEATHER</th>
      <th>LOG_DURATION</th>
      <th>LOG_CUSTOMERS_AFFECTED</th>
    </tr>
    <tr>
      <th>OBS</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2011</td>
      <td>7</td>
      <td>Minnesota</td>
      <td>MRO</td>
      <td>East North Central</td>
      <td>-0.3</td>
      <td>normal</td>
      <td>2011-07-01</td>
      <td>17:00:00</td>
      <td>2011-07-03</td>
      <td>...</td>
      <td>Summer</td>
      <td>4.0</td>
      <td>False</td>
      <td>0.485347</td>
      <td>0.7327</td>
      <td>1674.68816</td>
      <td>Medium</td>
      <td>True</td>
      <td>8.026497</td>
      <td>11.156265</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2014</td>
      <td>5</td>
      <td>Minnesota</td>
      <td>MRO</td>
      <td>East North Central</td>
      <td>-0.1</td>
      <td>normal</td>
      <td>2014-05-11</td>
      <td>18:38:00</td>
      <td>2014-05-11</td>
      <td>...</td>
      <td>Spring</td>
      <td>6.0</td>
      <td>True</td>
      <td>0.483906</td>
      <td>0.7327</td>
      <td>1674.68816</td>
      <td>Unknown/Minor</td>
      <td>False</td>
      <td>0.693147</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2010</td>
      <td>10</td>
      <td>Minnesota</td>
      <td>MRO</td>
      <td>East North Central</td>
      <td>-1.5</td>
      <td>cold</td>
      <td>2010-10-26</td>
      <td>20:00:00</td>
      <td>2010-10-28</td>
      <td>...</td>
      <td>Fall</td>
      <td>1.0</td>
      <td>False</td>
      <td>0.487093</td>
      <td>0.7327</td>
      <td>1674.68816</td>
      <td>Medium</td>
      <td>True</td>
      <td>8.006701</td>
      <td>11.156265</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2012</td>
      <td>6</td>
      <td>Minnesota</td>
      <td>MRO</td>
      <td>East North Central</td>
      <td>-0.1</td>
      <td>normal</td>
      <td>2012-06-19</td>
      <td>04:30:00</td>
      <td>2012-06-20</td>
      <td>...</td>
      <td>Summer</td>
      <td>1.0</td>
      <td>False</td>
      <td>0.484498</td>
      <td>0.7327</td>
      <td>1674.68816</td>
      <td>Medium</td>
      <td>True</td>
      <td>7.844241</td>
      <td>11.130215</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2015</td>
      <td>7</td>
      <td>Minnesota</td>
      <td>MRO</td>
      <td>East North Central</td>
      <td>1.2</td>
      <td>warm</td>
      <td>2015-07-18</td>
      <td>02:00:00</td>
      <td>2015-07-19</td>
      <td>...</td>
      <td>Summer</td>
      <td>5.0</td>
      <td>True</td>
      <td>0.487018</td>
      <td>0.7327</td>
      <td>1674.68816</td>
      <td>Large</td>
      <td>True</td>
      <td>7.462215</td>
      <td>12.429220</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1530</th>
      <td>2011</td>
      <td>12</td>
      <td>North Dakota</td>
      <td>MRO</td>
      <td>West North Central</td>
      <td>-0.9</td>
      <td>cold</td>
      <td>2011-12-06</td>
      <td>08:00:00</td>
      <td>2011-12-06</td>
      <td>...</td>
      <td>Winter</td>
      <td>1.0</td>
      <td>False</td>
      <td>0.575484</td>
      <td>0.599</td>
      <td>1314.6917</td>
      <td>Medium</td>
      <td>False</td>
      <td>6.580639</td>
      <td>10.448744</td>
    </tr>
    <tr>
      <th>1531</th>
      <td>2006</td>
      <td>NaN</td>
      <td>North Dakota</td>
      <td>MRO</td>
      <td>West North Central</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.563635</td>
      <td>0.599</td>
      <td>1314.6917</td>
      <td>Unknown/Minor</td>
      <td>False</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1532</th>
      <td>2009</td>
      <td>8</td>
      <td>South Dakota</td>
      <td>RFC</td>
      <td>West North Central</td>
      <td>0.5</td>
      <td>warm</td>
      <td>2009-08-29</td>
      <td>22:54:00</td>
      <td>2009-08-29</td>
      <td>...</td>
      <td>Summer</td>
      <td>5.0</td>
      <td>True</td>
      <td>0.540512</td>
      <td>0.5665</td>
      <td>1156.7344</td>
      <td>Unknown/Minor</td>
      <td>False</td>
      <td>4.094345</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1533</th>
      <td>2009</td>
      <td>8</td>
      <td>South Dakota</td>
      <td>MRO</td>
      <td>West North Central</td>
      <td>0.5</td>
      <td>warm</td>
      <td>2009-08-29</td>
      <td>11:00:00</td>
      <td>2009-08-29</td>
      <td>...</td>
      <td>Summer</td>
      <td>5.0</td>
      <td>True</td>
      <td>0.540512</td>
      <td>0.5665</td>
      <td>1156.7344</td>
      <td>Unknown/Minor</td>
      <td>False</td>
      <td>5.204007</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1534</th>
      <td>2000</td>
      <td>NaN</td>
      <td>Alaska</td>
      <td>ASCC</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.435583</td>
      <td>0.6602</td>
      <td>1190.21244</td>
      <td>Medium</td>
      <td>False</td>
      <td>0.000000</td>
      <td>9.566195</td>
    </tr>
  </tbody>
</table>
<p>1534 rows × 32 columns</p>
</div>



Next, I used the data and classified each outage as either 'Short' or 'Long', based on the logarithmic duration. After excluding the duration columns from the training data (to prevent any biases and possible redundancies), I preprocessed the features, trained a **Decision Tree** classifier, and finally evaluated how well it predicts outage duration categories using a Classification Report and a Confusion Matrix. 


```python
df_model = processed_df.copy()

def classify_duration(log_d):
    if log_d < 6:
        return "Short"
    else:
        return "Long"

df_model["DURATION_CLASS"] = df_model["LOG_DURATION"].apply(classify_duration)
df_model = df_model.dropna(subset=["OUTAGE.DURATION", "LOG_DURATION"])

features = [
    "MONTH", "ANOMALY.LEVEL", "CUSTOMERS.AFFECTED",
    "POPPCT_URBAN", "CUSTOMER_DENSITY", "POPULATION_DENSITY", "CLIMATE.CATEGORY",
    "CAUSE.CATEGORY", "SEVERITY_CATEGORY", "OUTAGE_SEASON", "IS_EXTREME_WEATHER"
]

X = df_model[features]
y = df_model["DURATION_CLASS"]

categorical_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean"))
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols)
])

model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```

    Classification Report:
                  precision    recall  f1-score   support
    
            Long       0.80      0.78      0.79       170
           Short       0.71      0.73      0.72       126
    
        accuracy                           0.76       296
       macro avg       0.75      0.76      0.76       296
    weighted avg       0.76      0.76      0.76       296
    
    Confusion Matrix:
    [[133  37]
     [ 34  92]]
    

## Step 7: Final Model

To improve my baseline model, I pivoted to a **Random Forest** classifier, introducing Grid Search for more effective hyperparameter tuning, specifically:

**Number of Trees:** While more trees generally reduce variance, they do increase computation time about which I am vary.

**Maximum Depth of Each Tree:** To reduce overfitting but ensure enough depth to capture salient features.

**Minimum Samples Needed to Split Node:** To prevent splitting nodes on noisy partitions, which can contribute to overfitting.

**Minimum Samples Required at Leaf Node:** To ensure leaf nodes are not too small, i.e. again to prevent overfitting and decrease sensitivity to noise.

Finally, I evaluated model performance using the same Classification Report and Confusion Matrix metrics, and fortunately, the final model showed a slight but significant increase in almost every metric!


```python
df_model = processed_df.copy()
df_model["DURATION_CLASS"] = df_model["LOG_DURATION"].apply(lambda x: "Short" if x < 6 else "Long")
df_model = df_model.dropna(subset=["OUTAGE.DURATION", "LOG_DURATION"])

features = [
    "MONTH", "ANOMALY.LEVEL", "CUSTOMERS.AFFECTED",
    "POPPCT_URBAN", "CUSTOMER_DENSITY", "POPULATION_DENSITY", "CLIMATE.CATEGORY",
    "CAUSE.CATEGORY", "SEVERITY_CATEGORY", "OUTAGE_SEASON", "IS_EXTREME_WEATHER"
]

X = df_model[features]
y = df_model["DURATION_CLASS"]

categorical_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
log_transformer = FunctionTransformer(np.log1p, validate=False)
quantile_transformer = QuantileTransformer(output_distribution='uniform', random_state=42)
numeric_base_cols = [col for col in numeric_cols if col not in ["CUSTOMER_DENSITY"]]

numeric_transformer = ColumnTransformer(transformers=[
    ("num_base", Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ]), numeric_base_cols),
    
    ("customer_density_log", Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("log", FunctionTransformer(np.log1p))
    ]), ["CUSTOMER_DENSITY"]),
])
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_cols + ["CUSTOMER_DENSITY"]),
    ("cat", categorical_transformer, categorical_cols)
])
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])
param_grid = {
    "classifier__n_estimators": [90, 100, 200, 300],
    "classifier__max_depth": [None, 10, 20, 30, 40],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4, 8]
}
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1, scoring='accuracy')
grid_search.fit(X_train, y_train)
y_pred = grid_search.predict(X_test)
print("Best Hyperparameters:", grid_search.best_params_)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```

    Fitting 5 folds for each of 240 candidates, totalling 1200 fits
    Best Hyperparameters: {'classifier__max_depth': 20, 'classifier__min_samples_leaf': 1, 'classifier__min_samples_split': 5, 'classifier__n_estimators': 300}
    
    Classification Report:
                  precision    recall  f1-score   support
    
            Long       0.82      0.82      0.82       170
           Short       0.76      0.76      0.76       126
    
        accuracy                           0.80       296
       macro avg       0.79      0.79      0.79       296
    weighted avg       0.80      0.80      0.80       296
    
    Confusion Matrix:
    [[140  30]
     [ 30  96]]
    

## Step 8: Fairness Analysis

Assuming we partition the data into 2 groups, specifically:

**GROUP X:** High percentage urban areas (POPPCT_URBAN >= 50%)

**GROUP Y:** Low percentage urban areas (POPPCT_URBAN < 50%)

Using the metric of **precision scores for 'Long' outages**, we have the following hypotheses:

**Null Hypothesis (H₀):** The model is fair across groups (GROUP X = GROUP Y), as the precision scores are the same for 'Long' outages regardless of group chosen.

**Alternative Hypothesis (H₁):** The model is unfair across groups (GROUP X != GROUP Y), as the precision scores are not the same for 'Long' outages, and probably depend on group chosen.


```python
group_mask = X_test['POPPCT_URBAN'] >= 50
group_X = y_test[group_mask]         # High percentage urban areas
group_Y = y_test[~group_mask]        # Low percentage urban areas

pred_X = y_pred[group_mask]
pred_Y = y_pred[~group_mask]

precision_X = precision_score(group_X, pred_X, pos_label="Long", zero_division=1)
precision_Y = precision_score(group_Y, pred_Y, pos_label="Long", zero_division=1)
obs_diff = precision_X - precision_Y
print(f"Observed difference in precision (Group X - Group Y): {obs_diff:.4f}")
```

    Observed difference in precision (Group X - Group Y): -0.1796
    

Once I computed the observed difference, I conducted a permutation test to assess whether the said difference could be attributed to randomness, or whether there was an actual association present.


```python
combined_y = np.concatenate([group_X, group_Y])
combined_pred = np.concatenate([pred_X, pred_Y])
size_X = len(group_X)
perm_diffs = []

for _ in range(10000):
    permuted_indices = np.random.permutation(len(combined_y))
    perm_y_X = combined_y[permuted_indices[:size_X]]
    perm_pred_X = combined_pred[permuted_indices[:size_X]]
    perm_y_Y = combined_y[permuted_indices[size_X:]]
    perm_pred_Y = combined_pred[permuted_indices[size_X:]]
    
    perm_precision_X = precision_score(perm_y_X, perm_pred_X, pos_label="Long", zero_division=1)
    perm_precision_Y = precision_score(perm_y_Y, perm_pred_Y, pos_label="Long", zero_division=1)
    perm_diffs.append(perm_precision_X - perm_precision_Y)

perm_diffs = np.array(perm_diffs)
p_value = np.mean(perm_diffs >= obs_diff)
print(f"Permutation test p-value: {p_value}")
```

    Permutation test p-value: 0.7018
    

Since our p-value is much greater than 0.05, we can safely **fail to reject the null hypothesis**, and conclude that the model is fair across groups.
