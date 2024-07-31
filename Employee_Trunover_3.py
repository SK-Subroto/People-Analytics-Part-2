import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'data/FAU_Bank_turnover.csv'
df = pd.read_csv(file_path)

# Define mappings for job_role and salary
job_role_mapping = {
    'bank_teller': 1, 'business_analyst': 2, 'credit_analyst': 3, 'customer_service': 4,
    'finance_analyst': 5, 'hr': 6, 'investment_banker': 7, 'IT': 8, 'loan_analyst': 9, 'mortgage_consultant': 10
}
salary_mapping = {'low': 1, 'medium': 2, 'high': 3}

# Apply the mappings
df['job_role'] = df['job_role'].map(job_role_mapping)
df['salary'] = df['salary'].map(salary_mapping)

# Data binning for job satisfaction and last performance evaluation
df['job_satisfaction_bin'] = pd.cut(df['job_satisfaction_level'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=[1, 2, 3, 4, 5])
df['performance_evaluation_bin'] = pd.cut(df['last_performance_evaluation'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=[1, 2, 3, 4, 5])

# Create a new feature combining number of projects and working hours
df['projects_hours'] = df['completed_projects'] * df['average_working_hours_monthly']

# Display the updated dataframe
print("\nData after Preprocessing:")
print(df.head())

# Calculate the average job satisfaction level of employees who left
avg_job_satisfaction_left = df[df['left'] == 1]['job_satisfaction_level'].mean()
print(f'\nAverage job satisfaction level of employees who left: {round(avg_job_satisfaction_left, 2)}')

# Calculate the average salary satisfaction level of employees who left
avg_salary_satisfaction_left = df[df['left'] == 1]['salary'].map({1: 'low', 2: 'medium', 3: 'high'}).mode()[0]
print(f'Average salary satisfaction level of employees who left: {avg_salary_satisfaction_left}')

# Calculate the average duration employees who have left stayed with FAU Bank
avg_years_spent_left = df[df['left'] == 1]['years_spent_with_company'].mean()
print(f'Average years spent with FAU Bank for employees who left: {round(avg_years_spent_left, 2)}')

# Analyze the effect of salary on employees deciding to quit
salary_effect = df.groupby('salary')['left'].mean()
# Reverse the salary mapping
reverse_salary_mapping = {v: k for k, v in salary_mapping.items()}
# Map the numerical values back to the categories
salary_effect.index = salary_effect.index.map(reverse_salary_mapping)

print(f'\nSalary effect on turnover:\n{salary_effect}')

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Display the correlation matrix
print('\nCorrelation matrix:')
print(correlation_matrix)

# Display the correlations of the 'left' column with other features
print('\nCorrelations with left column:')
print(correlation_matrix['left'].sort_values(ascending=False))

# Create a heatmap of the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()
