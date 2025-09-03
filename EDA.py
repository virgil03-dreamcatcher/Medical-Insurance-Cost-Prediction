import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_med=pd.read_csv('medical_insurance.csv')
# print(df_med.isnull().sum())
# print(df_med.dtypes)
# print(df_med['sex'].unique())
# print(df_med.describe(include="object")) 
# print(df_med['sex'].value_counts())
# print(df_med['smoker'].value_counts())
# print(df_med['region'].value_counts())
print(df_med.info())
print(df_med.head())

plt.figure(figsize=(8,5))
sns.histplot(df_med['charges'], kde=True)
plt.title('Distribution of Medical Insurance Charges')
plt.show()

# Charges vs Age scatter plot
plt.figure(figsize=(8,5))
sns.scatterplot(x='age', y='charges', data=df_med)
plt.title('Charges vs Age')
plt.show()

# Charges vs BMI grouped by smoker status
plt.figure(figsize=(8,5))
sns.scatterplot(x='bmi', y='charges', hue='smoker', data=df_med)
plt.title('Charges vs BMI by Smoker Status')
plt.show()

