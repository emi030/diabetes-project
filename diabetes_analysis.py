import pandas as pd

# Load the dataset
df = pd.read_csv('diabetes.csv')

# Preview the first 5 rows
print(df.head())

# Check the shape of the dataset (rows, columns)
print(df.shape)
# Basic statistics of the dataset
print(df.describe())
import matplotlib.pyplot as plt

# Compare glucose levels between diabetic and non-diabetic patients
df.groupby('Outcome')['Glucose'].hist(alpha=0.5, bins=20)
plt.xlabel('Glucose Level')
plt.ylabel('Count')
plt.title('Glucose Distribution by Diabetes Outcome')
plt.legend(['No Diabetes', 'Diabetes'])
plt.savefig('glucose_distribution.png')
print("Plot saved!")

import seaborn as sns

# Compare BMI between diabetic and non-diabetic patients
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
sns.boxplot(x='Outcome', y='BMI', data=df)
plt.xticks([0, 1], ['No Diabetes', 'Diabetes'])
plt.title('BMI by Diabetes Outcome')

plt.subplot(1, 2, 2)
sns.boxplot(x='Outcome', y='Age', data=df)
plt.xticks([0, 1], ['No Diabetes', 'Diabetes'])
plt.title('Age by Diabetes Outcome')

plt.tight_layout()
plt.savefig('bmi_age_analysis.png')
print("Plot saved!")

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap of All Features')
plt.savefig('correlation_heatmap.png')
print("Heatmap saved!")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Split data into features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2%}")