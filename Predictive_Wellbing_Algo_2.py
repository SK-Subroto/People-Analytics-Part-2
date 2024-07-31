import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer

# Load the dataset
file_path = 'data/FAU_Bank_Employee_Wellbeing.csv'
df = pd.read_csv(file_path)

# Convert string datatypes to numeric
age_mapping = {'Less than 20': 1, '21 to 35': 2, '36 to 50': 3, '51 or more': 4}
df['AGE'] = df['AGE'].map(age_mapping)
gender_mapping = {'Male': 0, 'Female': 1}
df['GENDER'] = df['GENDER'].map(gender_mapping)
job_role_mapping = {
    'Bank Teller': 1, 'Business Analyst': 2, 'Credit Analyst': 3, 'Customer Service': 4, 
    'Finance Analyst': 5, 'Human Resources': 6, 'Investment Banker': 7, 'Loan Processor': 8, 
    'Mortgage Consultant': 9, 'Risk Analyst': 10
}
df['JOB_ROLE'] = df['JOB_ROLE'].map(job_role_mapping)

# Handle missing values
X = df.drop(columns=['WORK_LIFE_BALANCE_SCORE'])
y = df['WORK_LIFE_BALANCE_SCORE']
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
print(f"R² score: {round(r2, 2)}")

# Present the difference between actual and predicted values in a tabular form
results = pd.DataFrame({
    'Actual': y_test.reset_index(drop=True),
    'Predicted': y_pred,
    'Difference': y_test.reset_index(drop=True) - y_pred
})
print("\nDifference between actual and predicted values:\n", results)

# Create a scatter plot of actual vs predicted values
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Scatter Plot of Actual vs. Predicted Values')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', linewidth=2)  # Add a reference line
plt.show()

# Predict the WLB score for a new employee
new_employee = {
    'Employee_ID': 2222,
    'JOB_ROLE': 1,  # Bank Teller
    'DAILY_STRESS': 2,
    'WORK_TRAVELS': 2,
    'TEAM_SIZE': 5,
    'DAYS_ABSENT': 0,
    'WEEKLY_EXTRA_HOURS': 5,
    'ACHIEVED_BIG_PROJECTS': 2,
    'EXTRA_HOLIDAYS': 0,
    'BMI_RANGE': 1,
    'TODO_COMPLETED': 6,
    'DAILY_STEPS_IN_THOUSAND': 5,
    'SLEEP_HOURS': 7,
    'LOST_VACATION': 5,
    'SUFFICIENT_INCOME': 1,
    'PERSONAL_AWARDS': 4,
    'TIME_FOR_HOBBY': 0,
    'AGE': 2,  # 21 to 35
    'GENDER': 0  # Male
}

# Create DataFrame for the new employee
new_employee_df = pd.DataFrame([new_employee])

# Impute missing values (in this case, none should be present)
new_employee_df_imputed = imputer.transform(new_employee_df)

# Predict the WLB score
predicted_wlb_score = model.predict(new_employee_df_imputed)
print(f"\nPredicted WLB score for the new employee: {round(predicted_wlb_score[0], 2)}")
