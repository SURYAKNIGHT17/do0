
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from the CSV file
df = pd.read_csv('Software Engineer Salaries.csv')

# 1. Diagnose the data:
print("**Data Diagnosis**")
print(df.info())
print(df.describe())

# 2. Preprocess the data:
print("\n**Data Preprocessing**")

# 2.1 Extract Salary Range:
df['Salary_Min'] = df['Salary'].str.extract(r'(\$\d+K) - (\$\d+K)')[0].str.replace('$', '').str.replace('K', '').astype(float) * 1000
df['Salary_Max'] = df['Salary'].str.extract(r'(\$\d+K) - (\$\d+K)')[1].str.replace('$', '').str.replace('K', '').astype(float) * 1000

# 2.2 Clean up Job Title:
df['Job Title'] = df['Job Title'].str.replace(',', '')
df['Job Title'] = df['Job Title'].str.replace(r'\d+ Years of Experience', '')

# 2.3 Extract Location City and State:
df['City'] = df['Location'].str.split(',').str[0]
df['State'] = df['Location'].str.split(',').str[1]

# 2.4 Convert Date to Numeric:
df = df[~df['Date'].str.contains('\+')]  # Remove rows with '+' in 'Date' (escaped plus sign)
df['Date'] = df['Date'].str.replace('d', '').astype(int)  # Convert to integer

# 3. Data Analysis:
print("\n**Data Analysis**")

# 3.1 Salary Distribution:
print("\nSalary Distribution:")
print(df['Salary_Min'].describe())
print(df['Salary_Max'].describe())

# 3.2 Average Salary by Company Score:
print("\nAverage Salary by Company Score:")
print(df.groupby('Company Score')['Salary_Min'].mean())

# 3.3 Average Salary by Location:
print("\nAverage Salary by Location:")
print(df.groupby('City')['Salary_Min'].mean())

# 3.4 Salary by Job Title:
print("\nAverage Salary by Job Title:")
print(df.groupby('Job Title')['Salary_Min'].mean())

# 4. Visualizations:
print("\n**Visualizations**")

# 4.1 Histogram of Salary Distribution:
plt.figure(figsize=(8, 6))
sns.histplot(df['Salary_Min'], kde=True)
plt.title('Distribution of Minimum Salary')
plt.xlabel('Minimum Salary')
plt.ylabel('Frequency')
plt.show()

# 4.2 Boxplot of Salary by Company Score:
plt.figure(figsize=(8, 6))
sns.boxplot(x='Company Score', y='Salary_Min', data=df)
plt.title('Salary Distribution by Company Score')
plt.xlabel('Company Score')
plt.ylabel('Minimum Salary')
plt.show()

# 4.3 Scatter Plot of Salary vs. Company Score:
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Company Score', y='Salary_Min', data=df)
plt.title('Salary vs. Company Score')
plt.xlabel('Company Score')
plt.ylabel('Minimum Salary')
plt.show()

# 5. Further Analysis (Optional):

# 5.1 Regression Analysis:
# (Use statsmodels or scikit-learn to build a linear regression model)
# from statsmodels.formula.api import ols
# model = ols('Salary_Min ~ Company Score + Date', data=df)
# results = model.fit()
# print(results.summary()) 

# 5.2 Clustering:
# (Use scikit-learn to cluster the data based on salary and other features)
# from sklearn.cluster import KMeans
# kmeans = KMeans(n_clusters=3, random_state=0)
# kmeans.fit(df[['Salary_Min', 'Company Score']])
# df['Cluster'] = kmeans.labels_

# 5.3 Association Rule Mining:
# (Use mlxtend library to discover association rules between job titles, location, and salary)
# from mlxtend.frequent_patterns import apriori
# from mlxtend.association_rules import association_rules
# frequent_itemsets = apriori(df[['Job Title', 'City', 'State']], min_support=0.1)
# rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1)

# 6. Save the cleaned data:
df.to_csv('Software_Engineer_Salaries_Cleaned.csv', index=False)
