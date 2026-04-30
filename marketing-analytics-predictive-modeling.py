"""
PROJECT: Marketing Analytics-Customer Behavior, Campaign Efficacy, and Predictive Modeling.
OBJECTIVE: Clean, analyze, and model customer data to identify purchase drivers,
           campaign effectiveness, and regional differences.
LIBRARIES: Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn, SciPy.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # Or 'Qt5Agg'
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import date
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_regression, RFE
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import norm
import itertools
import warnings
import re

# Global Visual & Display Configuration
sns.set()
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

# Ignore non-critical warnings
warnings.filterwarnings("ignore")

# Verification of environment setup
print(f"Seaborn version: {sns.__version__}")

# -----DATA LOADING-----
data = pd.read_csv('data/marketing_data.csv', index_col='ID', parse_dates=['Dt_Customer'])

print(data.head())

print('Number of columns :',data.shape[1])
print('Number of records :',data.shape[0])

# --- 1. MODEL DEFINITION ---

model_pipeline = Pipeline([
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', StandardScaler()),
    ('regressor', Lasso(alpha=1.0))
])

# --- 2. DATA CLEANING & SANITIZATION ---

# Rename columns

data = data.rename(columns={
    'Dt_Customer':'Enrollment date',
    'Recency':'Days since last purchase',
    ' Income ':'Income'})

# Creating new column to fill existing value for dataframe

data['ID'] = data.index

# Replace the income column (string) to (Float) and replacing 'commas', ',', '$'

data['Income'] = data['Income'].str.replace(r'[\$,]', '', regex=True)
print(data['Income'].head())

def extract(x):
    if x is np.nan: return np.nan
    return float(re.sub(r'[$,]', "", str(x)))

data['Income'] = data['Income'].apply(extract)

# --- 3. DATA SEGMENTATION ---

#  Create table for each section of data
# Store customer's information
Customers = data.loc[:,:'Days since last purchase'].join(data[['Country']])

# Store product's information
Products = data.loc[:,'MntWines':'MntGoldProds']

# Store Purchases' information
Purchases = data.loc[:,'NumDealsPurchases':'NumWebVisitsMonth']

# Store campaign's information
Campaigns = data.loc[:,'AcceptedCmp3':'AcceptedCmp2']
Misc = data.loc[:,['Response','Complain']]


# Create table for each type of data

category = data.select_dtypes(include='object')
numeric = data.select_dtypes(exclude='object')

# --- 4. EXPLORATORY DATA ANALYSIS (EDA) ---

# Quantitative check for missing values across the dataset

print(pd.DataFrame(data.isnull().sum(), columns=['#Null values']).T)

# Frequency distribution for categorical columns

for f in category.columns:
    print(category[f].value_counts())
    print('***********************************')

# Statistical summary with custom styling for boundary values

df = numeric.describe()

def custom_style(row):
    color = 'white'
    if row.name == 'min' or row.name == 'max':
        color = 'darkkhaki'

    return ['background-color: %s' % color] * len(row.values)


df.style.apply(custom_style, axis=1)

print(df)

# --- 5. OUTLIER DETECTION & FEATURE ENGINEERING ---

def report(feature):
    fig, ax = plt.subplots(1,2)
    fig.set_size_inches(16,4)
    fig.suptitle(feature, fontsize=16)
    sns.histplot(data=numeric, x=feature, kde=True, ax=ax[0])
    sns.boxplot(data=numeric, x=feature, ax=ax[1])
    plt.show()

    print(numeric[feature].describe())

# Feature Engineering: Calculate Current Age and Age at Enrollment

Age = date.today().year-numeric['Year_Birth']
numeric.insert(1, 'Age', Age,)

Enroll_at_age = numeric['Enrollment date'].dt.year - numeric['Year_Birth']
numeric.insert(6, 'Enroll_at_age', Enroll_at_age)

# Iterative outlier reporting across numerical columns

for col in numeric.columns:
    if col in ['Year_Birth','Enrollment date']: continue
    if col == 'AcceptedCmp3' : break
    report(col)

plt.show()

# Monitoring Age Outlier

print(numeric[numeric['Age']>80])

# Correcting age anomalies (logical boundaries: 120 years)

temp = numeric[numeric['Age']>120].index
numeric.loc[temp, ["Year_Birth", "Age", "Enroll_at_age"]] = np.nan

# Monitoring Income Outlier

print(numeric[numeric['Income']>160000])

# --- 6. CORE TREND ANALYSIS ---

# Annual Enrollment Analysis

plt.bar(height = numeric['Enrollment date'].dt.year.value_counts()[[2012,2013,2014]], x=['2012','2013','2014'])
plt.title('Number of enrollment in each year')
plt.show()

# Time-series analysis of enrollments with a 10-period rolling mean

df = pd.pivot_table(numeric, values='ID', index='Enrollment date', aggfunc='count')
df['count'] = df['ID'].rolling(10).mean()
df['Year'] = df.index.year.astype('category')

fig, ax = plt.subplots()
fig.set_size_inches(20,6)

sns.lineplot(data=df, x='Enrollment date', y='count', ax=ax, hue='Year')
ax.set_ylabel('Average enrollments')
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y - %m'))
plt.xticks(rotation=40)
plt.show()

# Financial Deep-Dive: Income by Country (excluding extreme outliers before calculating any statistics)
# Income

# Mean and Median of Income in each country

# This time, we need to exclude the outlier in "Income" column before calculating any statistics.
numeric_analysis = numeric[numeric['Income']!=666666]

df = pd.pivot_table(numeric_analysis.join(category[['Country']]),
                     values='Income',
                     index='Country',
                     aggfunc={'Income':['mean','median']})
df.plot(kind='bar')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.ylabel('Income')
plt.show()

# Correlation between Age and Income segmented by Region

temp = numeric_analysis[['Income','Age']].join(category[['Country']])
sns.lmplot(data=temp, y='Income', x='Age', col='Country', col_wrap=4, line_kws={'color': 'darkorange'}, scatter_kws={'color':'teal'})
plt.ylim(0,200000)
plt.show()

# Education-based Income analysis

df = pd.pivot_table(numeric_analysis.join(category[['Education']]),
               values='Income',
               index='Education',
               aggfunc={'Income':['count','mean','median']})

df[['mean','median']].plot(kind='bar')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.ylabel('Income')
plt.xlabel('')
plt.title('Income in each education level')
plt.show()

print(df)

del numeric_analysis

# --- 7. PURCHASE CHANNEL & PRODUCT PERFORMANCE ---

# create new column "Total products amount" And Calculate Total expenditure aggregation

numeric['Total products amount'] = np.sum(Products, axis=1)

# Channel efficacy: Proportion of total purchases by platform

total_purchase_each = np.sum(Purchases.iloc[:, :-1], axis=0)

percent_purchase_each = total_purchase_each / np.sum(total_purchase_each) * 100


def plot_pie_chart(labels, sizes: pd.Series, title):
    fig, ax = plt.subplots()
    fig.suptitle(title, fontsize=16)
    fig.set_size_inches(5, 5)
    ax.pie(sizes, labels=labels, autopct='%1.1f%%',
           shadow=True, startangle=90)
    ax.axis('equal')
    plt.show()
    print(sizes.sort_values(ascending=False))


plot_pie_chart(labels=['Deal', 'Web', 'Catalog', 'Store'], sizes=percent_purchase_each, title='Purchases in each channel')

# Looking into total number of purchases in each country.

if 'Total purchase' not in Purchases.columns:
    Purchases['Total purchase'] = np.sum(Purchases, axis=1)

Purchase_category = Purchases.join(category)

Purchase_country_summary = pd.pivot_table(Purchase_category,
                                          values='Total purchase',
                                          index='Country',
                                          aggfunc={'Total purchase':['sum']})

Purchase_country_summary.plot(kind='bar')
plt.title('Total purchases in each country')
plt.show()

# Product popularity benchmarks

sum_each_product = np.sum(Products, axis=0)

plot_pie_chart(sizes=sum_each_product/np.sum(sum_each_product)*100,
               labels=['Wine','Fruit','Meat','Fish','Sweet','Gold'],
              title='Overall %Products')

# Normalized analysis of individual purchase behavior

if 'Total' not in Products.columns:
    Products['Total'] = np.sum(Products, axis=1)

Each_ID_Products = Products.apply(lambda x: x / x.iloc[-1] * 100, axis=1)

Avg_Each_ID_Products = np.mean(Each_ID_Products, axis=0)

plot_pie_chart(sizes = Avg_Each_ID_Products[:-1], labels=['Wine','Fruit','Meat','Fish','Sweet','Gold'], title='Average Product for each ID')

# --- 8. MARKETING CAMPAIGN ANALYSIS ---

# Evaluating campaign success counts and conversion rates

each_campaign = np.sum(Campaigns,axis=0)

CR_each_campaign = each_campaign/len(Campaigns)*100

fig, ax = plt.subplots(1,2)
fig.set_size_inches(15,4)

cam_color = ['steelblue','peru','olivedrab','teal','sienna']

ax[0].barh(y=['Campaign 3','Campaign 4','Campaign 5','Campaign 1','Campaign 2'],
           width=each_campaign.values, color=cam_color)
ax[0].set_xlabel('# success')

ax[1].pie(x=CR_each_campaign, labels=['Campaign 3','Campaign 4','Campaign 5','Campaign 1','Campaign 2'],
         autopct='%1.1f%%', shadow=True, startangle=90, colors=cam_color)
ax[1].axis('equal')
ax[1].set_xlabel('Overall each campaign\'s success rate')

plt.show()

# Demographic influence on campaign acceptance (Age distribution)

fig, ax = plt.subplots(1, len(Campaigns.columns), sharey=True)
i = 0
fig.set_size_inches(20, 6)

# We want to know if 'Age' have some noticeable effect on campaign acceptance.

for campaign in Campaigns.columns:
    sns.histplot(data=numeric[numeric[campaign] == 1], x='Age', ax=ax[i])
    ax[i].set_ylim(0, 40)
    ax[i].set_title(campaign)
    i += 1

plt.show()

# Conversion Rate (CR) benchmarks by Country

Campaigns_category = Campaigns.join(category)
Campaigns_category['Total accept'] = np.sum(Campaigns, axis=1)

summary_country = pd.pivot_table(Campaigns_category,
                                   values='Total accept',
                                   index='Country',
                                   aggfunc={'Total accept':['sum','count']})

summary_country['CR'] = summary_country['sum']/summary_country['count']
summary_country.rename(columns={'count':'#customers', 'sum':'Total accept'}).style.background_gradient(sns.light_palette('khaki', as_cmap=True),
                                                                                                       subset=pd.IndexSlice[:, ['CR']])
print(summary_country)

# Average Regional performance of each campaign in each country.

temp = Campaigns_category.groupby(by='Country').mean(numeric_only=True)

# 2.Drop the row 'ME' as shown in your screenshot
if 'ME' in temp.index:
    temp.drop(['ME'], axis=0, inplace=True)

# 3. Apply the green styling
# In a Jupyter Notebook, simply ending with this line will "print" it
temp.style.background_gradient(cmap=sns.light_palette('green', as_cmap=True))

print(temp)

# --- 9. STATISTICAL ANALYSIS & PREDICTIVE MODELING ---

# Filtering for valid analysis set and defining target variable: 'NumStorePurchases'.

# Building analysis dataset
numeric_analysis = numeric[numeric['Income']!=666666]

# Let's clear redundant features
X_numeric = numeric_analysis.drop(['NumStorePurchases','Enrollment date','Year_Birth','ID','Total products amount'], axis=1)

# Focus on number of store purchases
y = numeric_analysis[['NumStorePurchases']]

# Preprocess

# Pipeline construction for Imputation and Standardization

# Impute null values.
Imputer = KNNImputer(missing_values=np.nan)

# Standardize numeric columns
Scaler = StandardScaler()

numeric_pipe = Pipeline([("Impute", Imputer),
                         ("Scale", Scaler)])

# Data transformation and Model Fitting (Lasso, Ridge, and F-Regression)

X_numeric_preprocessed = numeric_pipe.fit_transform(X_numeric)
lasso = Lasso(alpha=0.01).fit(X_numeric_preprocessed, y)
ridge = Ridge().fit(X_numeric_preprocessed, y)
F, p = f_regression(X_numeric_preprocessed, y)

# Consolidating statistical results into a MultiIndex DataFrame

index = pd.MultiIndex.from_tuples([
    ('Features', ''),
    ('Variance', ''),
    ('Lasso', 'Coef'),
    ('Ridge', 'Coef'),
    ('F-test', 'F-value'),
    ('F-test', 'p-value')
])

# Building the DataFrame using your defined lowercase variables
StorePurchases_effect_num = pd.DataFrame(list(zip(
    X_numeric.columns,
    X_numeric.var() / np.nanmean(X_numeric, axis=0),
    lasso.coef_,
    ridge.coef_[0] if ridge.coef_.ndim > 1 else ridge.coef_,
    F,
    p
)), columns=index)

# Setting 'Features' as the index
StorePurchases_effect_num = StorePurchases_effect_num.set_index(('Features', ''))
StorePurchases_effect_num.index.name = 'Features'
print(StorePurchases_effect_num.sort_values(by=('F-test', 'p-value')))

# Sorting by p-value and printing the raw table to the console
final_table = StorePurchases_effect_num.sort_values(by=('F-test', 'p-value'))

print(final_table)

# For simplicity, we plot the coefficient of LASSO and F-value of F-test in the bar graphs below.

plt.figure(figsize=(7,7))
StorePurchases_effect_num['Lasso']['Coef'].sort_values().plot(kind='barh')
plt.xlabel('Lasso coef')
plt.title('LASSO analysis')
plt.show()

plt.figure(figsize=(7,7))
StorePurchases_effect_num['F-test']['F-value'].sort_values().plot(kind='barh')
plt.xlabel('F-value')
plt.title('F-test')
plt.show()

# Machine Learning Feature Ranking using RFE (Recursive Feature Elimination)

rf = RandomForestRegressor()

rfe = RFE(estimator = rf, n_features_to_select = 1, verbose=1).fit(X_numeric_preprocessed, y)

# 1. Create the ranking series using the RFE results
# 'X_numeric.columns' are your features, 'rfe.ranking_' are the RFE output scores

ranking_dict = dict(zip(X_numeric.columns, rfe.ranking_))

ranking_df = pd.Series(ranking_dict, name='Rank').to_frame().sort_values(by='Rank')

print(ranking_df)


# --- 10. CATEGORICAL DEEP-DIVES ---

category_NumStore = numeric[['NumStorePurchases']].join(category)
print(category_NumStore.head())


# Education vs Store Purchases across Countries

sns.catplot(kind='box', data=category_NumStore.query("Country!='ME'"), col='Country', x='Education', y='NumStorePurchases',  col_wrap=4)
plt.show()

# Marital Status vs Store Purchases across Countries

sns.catplot(kind='bar', data=category_NumStore.query("(Country!='ME') and (Marital_Status not in ['YOLO','Alone','Absurd'])"),
            col='Country', x='Marital_Status', y='NumStorePurchases',  col_wrap=4)
plt.show()

# Checking for sample bias across categorical segments

fig, ax = plt.subplots(1,3, sharey=True)
fig.set_size_inches(15,5)

for i,col in list(enumerate(['Education', 'Marital_Status', 'Country'])):
    sns.countplot(data=data, x='Education',  ax=ax[i])

for ax_i in ax:
    ax_i.xaxis.set_label_coords(0.5, 1.05)
    plt.setp( ax_i.xaxis.get_majorticklabels(), rotation=40 )

plt.show()

# Regional Purchase Distribution and Statistical Significance testing (US Mean Comparison)

temp = Purchases.join(data[['Country','ID']])

fig, ax = plt.subplots(2,4)
fig.set_size_inches(20,7)
i=0

axes = list(itertools.chain(ax[0],ax[1]))

for c in temp.Country.unique():
    sns.histplot(data=temp[temp['Country'] == c], x='Total purchase', label=c, ax=axes[i], bins=15)
    axes[i].legend()
    i+=1
plt.show()

summary_purchase_country = np.sum(Purchases, axis=1).to_frame('Total purchases').join(category[['Country']]).groupby(by='Country').agg(['count','sum','mean','median','std'])

print(summary_purchase_country)

# Comparative P-Value analysis for the US market
# from scipy.stats import norm

US_mean = summary_purchase_country.loc['US', ('Total purchases','mean')]
US_std = summary_purchase_country.loc['US', ('Total purchases','std')]
US_n = summary_purchase_country.loc['US', ('Total purchases','count')]

for country in summary_purchase_country.index:
    if country == 'US':break
    other_mean = summary_purchase_country.loc[country, ('Total purchases','mean')]
    x = (US_mean-other_mean)/(US_std/US_n**0.5)
    print('mean "US" > mean "'+country.upper()+'" p-value = '+str(norm.sf(x)))

# Gold Consumer Segment analysis: Impact on In-Store behavior

Gold_avg = np.mean(Products['MntGoldProds'])

mask = Products['MntGoldProds']>=Gold_avg

Above_gold = Products[mask].index
Below_gold = Products[~mask].index

fig,ax = plt.subplots(1,2)
fig.set_size_inches(15,5)
sns.histplot(ax=ax[0], data=Purchases[mask], x='NumStorePurchases', kde=True, label='Purchase Gold Above Avg.', color='indigo',element='step')
sns.histplot(ax=ax[0], data=Purchases[~mask], x='NumStorePurchases', kde=True, label='Below', color='darkorange', element='step')

temp = Purchases.join(mask)
temp['MntGoldProds'] = temp['MntGoldProds'].replace({True:'Gold above avg.', False:'Gold below avg.'})
sns.boxplot(ax=ax[1], data=temp, y='NumStorePurchases', x='MntGoldProds', palette={'Gold above avg.':'indigo','Gold below avg.':'darkorange'})

ax[0].legend()
plt.show()

# --- 11. PRODUCT SPECIFIC HYPOTHESIS TESTING (MntFishProducts) ---

# Analysis of Married PhD candidates and Fish expenditure

PhD_Married = (category['Education']=='PhD') & (category['Marital_Status']=='Married')

summary_table = numeric.loc[PhD_Married,['MntFishProducts']].describe().join(numeric.loc[~PhD_Married,['MntFishProducts']].describe(),
                                                            lsuffix='_PhD&Married',
                                                            rsuffix='_Not')
print(summary_table)

# Correlation matrix for Fish Products expenditure

corr_table = (Customers.drop(['Year_Birth','Enrollment date'], axis=1)\
.join([Purchases, Products, Campaigns])\
.corr(numeric_only=True)[['MntFishProducts']])

print(corr_table)

# Demographic visualizations for Fish Expenditure

fig, ax = plt.subplots(1,3, sharey=True)
fig.set_size_inches(20,7)
sns.boxplot(data=data[data['Country']!='ME'],
            x='Country', y='MntFishProducts',
            ax=ax[0])

sns.boxplot(data=data.query('Marital_Status not in  ["YOLO","Alone","Absurd"]'),
            x='Marital_Status', y='MntFishProducts',
            ax=ax[1])

sns.boxplot(data=data, x='Education', y='MntFishProducts',ax=ax[2])

fig.tight_layout()
fig.suptitle('Relationship between each categorical variable\nand amount spent on Fish', fontsize=18, y=1.07)
plt.show()

# Multi-dimensional Segment analysis for Fish Products

sns.catplot(kind='box', data = data.query('(Marital_Status not in ["YOLO","Absurd","Alone"]) and (Country != "ME")'),
            x ='Marital_Status',
            y ='MntFishProducts',
            col='Country', col_wrap=4)
plt.suptitle('In each country, \nAmount spent on Fish in each marital status', fontsize=18, y=1.07)

plt.show()

# 'Education' insight in each country

sns.catplot(kind='box', data = data.query('(Marital_Status not in ["YOLO","Absurd","Alone"]) and (Country != "ME")'),
            x ='Education',
            y ='MntFishProducts',
            col='Country', col_wrap=4)
plt.suptitle('In each country, \nAmount spent on Fish in each education level', fontsize=18, y=1.07)

plt.show()









