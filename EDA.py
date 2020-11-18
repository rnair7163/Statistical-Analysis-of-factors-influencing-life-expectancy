# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 19:55:16 2020

@author: Cheng Jiang
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.stats.mstats import winsorize
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv("Life Expectancy Data.csv")
df.head()

df.describe()
df.shape
df.columns


# Rename column names to uniform the format
df.rename(columns = {" BMI " :"BMI",
                     "Life expectancy ": "Life_Expectancy",
                     "Adult Mortality":"Adult_Mortality",
                     "infant deaths":"Infant_Deaths",
                     "percentage expenditure":"Percentage_Expenditure",
                     "Hepatitis B":"HepatitisB",
                     "Measles ":"Measles",
                     "under-five deaths ": "Under_Five_Deaths",
                     "Total expenditure":"Total_Expenditure",
                     "Diphtheria ": "Diphtheria",
                     " thinness  1-19 years":"Thinness_1-19_Years",
                     " thinness 5-9 years":"Thinness_5-9_Years",
                     " HIV/AIDS":"HIV/AIDS",
                     "Income composition of resources":"Income_Composition_Of_Resources"}, inplace = True)
df.columns
# =============================================================================
# Dealing with Outliers
# =============================================================================
# Check outliers
col_dict = {'Life_Expectancy':1,
            'Adult_Mortality':2,
            'Infant_Deaths':3,
            'Alcohol':4,
            'Percentage_Expenditure':5,
            'HepatitisB':6,
            'Measles':7,
            'BMI':8,
            'Under_Five_Deaths':9,
            'Polio':10,
            'Total_Expenditure':11,
            'Diphtheria':12,
            'HIV/AIDS':13,
            'GDP':14,
            'Population':15,
            'Thinness_1-19_Years':16,
            'Thinness_5-9_Years':17,
            'Income_Composition_Of_Resources':18,
            'Schooling':19}

plt.figure(figsize=(20,30))
for col, i in col_dict.items():
    plt.subplot(4,5,i)
    sns.boxplot(df[col]).set(xlabel=None)
    plt.title(col)
plt.show()

# Dealing with outliers
    #Delete unrealistic data
df = df[df['Infant_Deaths'] < 1001]
df = df[df['Under_Five_Deaths'] < 1001]
df = df[df['Measles'] < 1001]

    #Count Outliers
def OutlierCounter(col):
    print(15*'-' + col + 15*'-')
    q25, q75 = np.nanquantile(df[col], [.25,.75])
    IQR = q75-q25
    minval = q25 - (1.5*IQR)
    maxval = q75 + (1.5*IQR)
    outlier_count = len(np.where((df[col] > maxval) | (df[col] < minval))[0])
    outlier_percent = round(outlier_count / len(df[col]*100),2)
    print("Outlier Counts: {}".format(outlier_count))
    print("Outlier Percentage: {}".format(outlier_percent))

cont_vars = df.columns[3:]  
for cols in cont_vars:
    OutlierCounter(cols)
    
    #winsorize the data
df_clean = pd.DataFrame()
df_clean = df[["Country", "Year", "Status"]].copy()
df_clean[cont_vars[0]] = winsorize(df[cont_vars[0]], (0.03,0))  
df_clean[cont_vars[1]] = winsorize(df[cont_vars[1]], (0,0.04))  
df_clean[cont_vars[2]] = winsorize(df[cont_vars[2]], (0,0.15))  
df_clean[cont_vars[3]] = winsorize(df[cont_vars[3]], (0,0))  
df_clean[cont_vars[4]] = winsorize(df[cont_vars[4]], (0,0.14))  
df_clean[cont_vars[5]] = winsorize(df[cont_vars[5]], (0.1,0))  
df_clean[cont_vars[6]] = winsorize(df[cont_vars[6]], (0,0.17))  
df_clean[cont_vars[7]] = winsorize(df[cont_vars[7]], (0,0))  
df_clean[cont_vars[8]] = winsorize(df[cont_vars[8]], (0,0.16))  
df_clean[cont_vars[9]] = winsorize(df[cont_vars[9]], (0.11,0)) 
df_clean[cont_vars[10]] = winsorize(df[cont_vars[10]], (0.01,0))
df_clean[cont_vars[11]] = winsorize(df[cont_vars[11]], (0.11,0))
df_clean[cont_vars[12]] = winsorize(df[cont_vars[12]], (0,0.21))   
df_clean[cont_vars[13]] = winsorize(df[cont_vars[13]], (0,0.12))  
df_clean[cont_vars[14]] = winsorize(df[cont_vars[14]], (0,0.1))  
df_clean[cont_vars[15]] = winsorize(df[cont_vars[15]], (0,0.04))  
df_clean[cont_vars[16]] = winsorize(df[cont_vars[16]], (0,0.04))
df_clean[cont_vars[17]] = winsorize(df[cont_vars[17]], (0.05,0))
df_clean[cont_vars[18]] = winsorize(df[cont_vars[18]], (0.017,0.05))
    #plot the data after winsorized 
plt.figure(figsize=(20,30))
for col, i in col_dict.items():
    plt.subplot(4,5,i)
    sns.boxplot(df_clean[col]).set(xlabel=None)
    plt.title(col)
plt.show()

    #Run log transformation on Total_Expenditure, Population and GDP
df_clean['Total_Expenditure'] = np.log(df_clean['Total_Expenditure'])
df_clean['GDP'] = np.log(df_clean['GDP'])
df_clean['Population'] = np.log(df_clean['Population']) 
plt.figure(figsize=(20,30))
for col, i in col_dict.items():
    plt.subplot(4,5,i)
    sns.boxplot(df_clean[col]).set(xlabel=None)
    plt.title(col)
plt.show()


# =============================================================================
# Dealing with missing values
# =============================================================================

print(df_clean.isnull().sum())

imputed_data = []
for year in list(df_clean.Year.unique()):
    year_data = df_clean[df_clean.Year == year].copy()
    for col in list(year_data.columns)[4:]:
        year_data[col] = year_data[col].fillna(year_data[col].dropna().median()).copy()
    imputed_data.append(year_data)
df_clean = pd.concat(imputed_data).copy()

df_clean = df_clean[df_clean.Life_Expectancy.notnull()]

print(df_clean.isnull().sum())



# =============================================================================
# EDA
# =============================================================================
# Histogram
plt.figure(figsize=(15, 20))
for i, col in enumerate(cont_vars, 1):
    plt.subplot(5, 4, i)
    plt.hist(df_clean.iloc[:,3:][col])
    plt.title(col)

# Status
plt.figure(figsize=(15, 8))
plt.subplot(1,2,1)
df_clean.Status.value_counts().plot(kind='pie', autopct='%.2f')
plt.title('Country Status Pie Chart')
plt.show()
plt.subplot(1,2,2)
sns.barplot(x="Status", y="Life_Expectancy", data=df_clean)
plt.title("Country Status vs. Life Expectancy")
plt.show()

#Correlations
corrmat = df_clean.iloc[:,3:].corr()
mask = np.triu(df_clean.iloc[:,3:].corr())
plt.figure(figsize=(15,6))
sns.heatmap(corrmat, annot=True, fmt='.2g', vmin=-1, vmax=1, center=0, cmap='coolwarm', mask=mask)
plt.title('Correlation Matrix Heatmap')
plt.show()