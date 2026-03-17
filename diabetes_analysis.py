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