# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# Load dataset
df = pd.read_csv('StudentsPerformance.csv')
print("Dataset loaded successfully. Here are the first 5 rows:")
print(df.head())

print("\nData Information:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nStatistical Summary:")
print(df.describe())

# Boxplot: Math Scores by Gender
plt.figure(figsize=(8,6))
sns.boxplot(x='gender', y='math score', data=df)
plt.title('Math Scores by Gender')
plt.xlabel('Gender')
plt.ylabel('Math Score')
plt.show()

# Histogram: Distribution of Math Scores
plt.figure(figsize=(8,6))
sns.histplot(df['math score'], bins=30, kde=True)
plt.title('Distribution of Math Scores')
plt.xlabel('Math Score')
plt.ylabel('Frequency')
plt.show()


# Copy dataset
data = df.copy()

# Encode categorical variables
categorical_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
encoder = LabelEncoder()

for col in categorical_cols:
    data[col] = encoder.fit_transform(data[col])

# Create new target variable
data['average score'] = data[['math score', 'reading score', 'writing score']].mean(axis=1)

# Define features and target
X = data[categorical_cols]
y = data['average score']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nData split into training and testing sets.")


# Train Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

# Evaluate Linear Regression Model
lr_mae = mean_absolute_error(y_test, lr_predictions)
lr_r2 = r2_score(y_test, lr_predictions)
print(f"Linear Regression - MAE: {lr_mae:.2f}, R2 Score: {lr_r2:.2f}")

# Train Random Forest Regressor Model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Evaluate Random Forest Model
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)
print(f"Random Forest - MAE: {rf_mae:.2f}, R2 Score: {rf_r2:.2f}")


# Compare models
print("\nModel Comparison:")
print(f"Linear Regression: MAE = {lr_mae:.2f}, R2 = {lr_r2:.2f}")
print(f"Random Forest:     MAE = {rf_mae:.2f}, R2 = {rf_r2:.2f}")

# Scatter plot: Actual vs Predicted (Random Forest)
plt.figure(figsize=(8,6))
plt.scatter(y_test, rf_predictions, alpha=0.7)
plt.xlabel("Actual Average Scores")
plt.ylabel("Predicted Average Scores")
plt.title("Actual vs Predicted Average Scores (Random Forest)")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.show()


