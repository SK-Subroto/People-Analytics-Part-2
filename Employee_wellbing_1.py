import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
file_path = 'data/FAU_Bank_Employee_Wellbeing.csv'
df = pd.read_csv(file_path)

# Check for missing values
missing_values = df.isnull().sum()
print("Missing values:\n", missing_values)

# Drop irrelevant columns (e.g., Employee_ID as it is likely just an identifier)
df.drop(columns=['Employee_ID'], inplace=True)

# Convert string datatypes to numeric
# For Age: map age ranges to numerical values
age_mapping = {'Less than 20': 1, '21 to 35': 2, '36 to 50': 3, '51 or more': 4}
df['AGE'] = df['AGE'].map(age_mapping)

# For Gender: map genders to numerical values
gender_mapping = {'Male': 0, 'Female': 1}
df['GENDER'] = df['GENDER'].map(gender_mapping)

# Plot bar chart for daily stress according to gender
plt.figure(figsize=(10, 6))
sns.barplot(x='GENDER', y='DAILY_STRESS', data=df, errorbar=None)
plt.xlabel('Gender')
plt.ylabel('Daily Stress')
plt.title('Daily Stress by Gender')
plt.xticks(ticks=[0, 1], labels=['Male', 'Female'])
plt.show()

# Plot bar chart for daily stress according to job role
plt.figure(figsize=(14, 8))
sns.barplot(x='JOB_ROLE', y='DAILY_STRESS', data=df, errorbar=None)
plt.xlabel('Job Role')
plt.ylabel('Daily Stress')
plt.title('Daily Stress by Job Role')
plt.xticks(rotation=45)
plt.show()

# Determine who dedicates more time to their hobbies, men or women
hobby_time_by_gender = df.groupby('GENDER')['TIME_FOR_HOBBY'].mean()
# print("Average time for hobby by gender:\n", hobby_time_by_gender)

# Bar chart for time dedicated to hobbies by gender with numbers on the chart
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=hobby_time_by_gender.index, y=hobby_time_by_gender.values)
plt.xlabel('Gender')
plt.ylabel('Average Time for Hobby')
plt.title('Average Time for Hobby by Gender')
plt.xticks(ticks=[0, 1], labels=['Male', 'Female'])

# Adding numbers on the bar chart
for i in ax.containers:
    ax.bar_label(i, fmt='%.2f')

plt.show()

# Build a heatmap to determine the attributes highly correlated with WLB score

# For Job Role: map Job Role ranges to numerical values
job_role_mapping = {
    'Bank Teller': 1, 'Business Analyst': 2, 'Credit Analyst': 3, 'Customer Service': 4, 'Finance Analyst': 5, 'Human Resources': 6, 'Investment Banker': 7, 'Loan Processor': 8, 'Mortgage Consultant': 9, 'Risk Analyst': 10
    }
df['JOB_ROLE'] = df['JOB_ROLE'].map(job_role_mapping)

# Convert string datatypes to numeric
correlation_matrix = df.corr()
plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Find highly correlated attributes with WLB score
wlb_correlation = correlation_matrix['WORK_LIFE_BALANCE_SCORE'].sort_values(ascending=False)
print("Attributes highly correlated with WLB score:\n", wlb_correlation)
